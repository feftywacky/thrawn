#include "transposition_table.h"
#include "constants.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#elif defined(__linux__)
#include <sys/mman.h>
#endif

namespace {
constexpr int TTAgeBits = 5;
constexpr int TTAgeShift = 59;
constexpr int TTAgeMask = (1 << TTAgeBits) - 1;
constexpr int TTKeyBits = 48;
constexpr int TTStaticEvalShift = TTKeyBits;
constexpr uint64_t TTKeyMask = (UINT64_C(1) << TTKeyBits) - 1;
constexpr int16_t TTStaticEvalNone = std::numeric_limits<int16_t>::min();

// Back the table with 2 MB-aligned memory so Linux can serve it from huge
// pages: one TLB entry then covers 32k clusters instead of 32.
constexpr std::size_t TTHugePageSize = 2ULL * 1024ULL * 1024ULL;

constexpr std::memory_order TTRelaxed = std::memory_order_relaxed;

int relative_age(int currentAge, int entryAge) {
    return (currentAge - entryAge) & TTAgeMask;
}

int packed_depth(uint64_t data) {
    return static_cast<int>((data >> 24) & 0xFFULL);
}

int packed_flag(uint64_t data) {
    return static_cast<int>((data >> 57) & 0x3ULL);
}

int packed_age(uint64_t data) {
    return static_cast<int>((data >> TTAgeShift) & TTAgeMask);
}

int replacement_value(uint64_t data, int currentAge) {
    if (data == 0) {
        return -1000000;
    }

    const int flag = packed_flag(data);
    const int exactBonus = flag == BOUND_EXACT ? 2 : 0;
    return packed_depth(data) + exactBonus - 8 * relative_age(currentAge, packed_age(data));
}

int16_t encode_static_eval(int staticEval) {
    if (staticEval == no_hashmap_entry) {
        return TTStaticEvalNone;
    }

    return static_cast<int16_t>(std::clamp(staticEval,
                                           static_cast<int>(TTStaticEvalNone) + 1,
                                           static_cast<int>(std::numeric_limits<int16_t>::max())));
}

int decode_static_eval(uint64_t key) {
    const auto raw = static_cast<int16_t>(key >> TTStaticEvalShift);
    return raw == TTStaticEvalNone ? no_hashmap_entry : static_cast<int>(raw);
}

uint64_t key_tag(uint64_t zobristKey, uint64_t data) {
    const uint64_t dataSignature = data ^ (data >> 16) ^ (data >> 32) ^ (data >> 48);
    return (zobristKey ^ dataSignature) & TTKeyMask;
}

bool key_matches(uint64_t packedKey, uint64_t zobristKey, uint64_t data) {
    return (packedKey & TTKeyMask) == key_tag(zobristKey, data);
}

uint64_t encode_key(uint64_t zobristKey, uint64_t data, int staticEval) {
    const uint64_t eval = static_cast<uint64_t>(
        static_cast<uint16_t>(encode_static_eval(staticEval)));
    return key_tag(zobristKey, data) | (eval << TTStaticEvalShift);
}

bool update_static_eval_key(TTEntry& entry, uint64_t zobristKey, int staticEval) {
    const uint64_t data = entry.smp_data.load(TTRelaxed);
    const uint64_t key = entry.smp_key.load(TTRelaxed);
    if (!key_matches(key, zobristKey, data)) {
        return false;
    }

    entry.smp_key.store((key & TTKeyMask) |
        (static_cast<uint64_t>(static_cast<uint16_t>(encode_static_eval(staticEval)))
             << TTStaticEvalShift),
        TTRelaxed);
    return true;
}

void* aligned_alloc_bytes(std::size_t bytes, std::size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free_bytes(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// Zero the table in parallel; a 1 GB single-threaded memset costs ~100 ms on
// every `ucinewgame`, which is real time lost at fast time controls.
//
// The worker count is capped rather than taken from hardware_concurrency():
// under a tournament manager running many games at once, every engine process
// would otherwise fan out across every core at the same moment, even one told
// `Threads=1`. Thread creation can also simply fail under that load, and an
// escaping std::system_error would kill the process with no output at all -
// which the manager reports as "engine didn't respond to uciok". So the spawn
// is bounded, and any failure falls back to finishing the job in place.
constexpr std::size_t TTZeroMaxWorkers = 4;

void parallel_zero(void* base, std::size_t bytes) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0)
        hw = 1;
    const std::size_t chunkMin = 8ULL * 1024ULL * 1024ULL;
    std::size_t workers = std::min<std::size_t>(hw, std::max<std::size_t>(1, bytes / chunkMin));
    workers = std::min(workers, TTZeroMaxWorkers);

    auto* bytePtr = static_cast<unsigned char*>(base);
    if (workers <= 1) {
        std::memset(bytePtr, 0, bytes);
        return;
    }

    const std::size_t chunk = bytes / workers;
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);

    // Index of the first chunk no worker has taken; whatever is left when the
    // spawn stops is zeroed by this thread.
    std::size_t spawned = 1;
    for (std::size_t i = 1; i < workers; ++i) {
        const std::size_t begin = i * chunk;
        const std::size_t len = (i == workers - 1) ? bytes - begin : chunk;
        try {
            pool.emplace_back([bytePtr, begin, len] { std::memset(bytePtr + begin, 0, len); });
        } catch (const std::system_error&) {
            break;
        }
        ++spawned;
    }

    std::memset(bytePtr, 0, chunk);
    if (spawned < workers)
        std::memset(bytePtr + spawned * chunk, 0, bytes - spawned * chunk);

    for (std::thread& worker : pool)
        worker.join();
}
} // namespace

TranspositionTable::TranspositionTable()
    : table(nullptr), tableAlloc(nullptr), allocBytes(0), numClusters(0), clusterMask(0),
      currentAge(0)
{
}

TranspositionTable::~TranspositionTable()
{
    freeTable();
}

void TranspositionTable::freeTable()
{
    if (tableAlloc) {
        aligned_free_bytes(tableAlloc);
        tableAlloc = nullptr;
    }
    table = nullptr;
    allocBytes = 0;
    numClusters = 0;
    clusterMask = 0;
}

void TranspositionTable::initTable(int mb)
{
    if (mb < 1)
        mb = 1;

    const std::size_t bytes = static_cast<std::size_t>(mb) * 0x100000ULL;  // Convert MB to bytes
    std::size_t clusterCapacity = bytes / sizeof(TTCluster);

    freeTable();

    if (clusterCapacity < 1) {
        std::cout << "info string TT init: table too small, forcing 4 MB\n";
        initTable(4);
        return;
    }

#if !defined(__SIZEOF_INT128__)
    // Without a 128-bit multiply the index falls back to masking, which needs a
    // power-of-two cluster count.
    std::size_t pow2 = 1;
    while (pow2 <= clusterCapacity / 2)
        pow2 *= 2;
    clusterCapacity = pow2;
#endif

    numClusters = clusterCapacity;
    clusterMask = numClusters - 1;
    allocBytes = numClusters * sizeof(TTCluster);

    tableAlloc = aligned_alloc_bytes(allocBytes, TTHugePageSize);
    if (!tableAlloc) {
        // Fall back to plain alignment; only huge-page backing is lost.
        tableAlloc = aligned_alloc_bytes(allocBytes, alignof(TTCluster));
    }
    if (!tableAlloc) {
        std::cout << "info string TT init: allocation of " << mb << " MB failed\n";
        numClusters = 0;
        clusterMask = 0;
        allocBytes = 0;
        if (mb > 4)
            initTable(4);
        return;
    }

#if defined(__linux__) && defined(MADV_HUGEPAGE)
    madvise(tableAlloc, allocBytes, MADV_HUGEPAGE);
#endif

    // Lock-free 64-bit atomics are plain words in memory, so a bulk zero is a
    // valid way to bring every entry up as empty.
    table = static_cast<TTCluster*>(tableAlloc);
    reset();

    std::cout << "info string TT allocated " << mb << " MB, entries = "
              << numClusters * TT_CLUSTER_SIZE << "\n";
}

void TranspositionTable::reset()
{
    if (table && numClusters > 0) {
        parallel_zero(table, allocBytes);
    }
    currentAge = 0;
}

std::size_t TranspositionTable::clusterIndex(uint64_t zobristKey) const
{
#if defined(__SIZEOF_INT128__)
    // Multiply-shift maps the key onto [0, numClusters) for any cluster count,
    // so the table can use every byte the user asked for instead of rounding
    // the capacity down to a power of two.
    return static_cast<std::size_t>((static_cast<unsigned __int128>(zobristKey) *
                                     static_cast<uint64_t>(numClusters)) >> 64);
#else
    return static_cast<std::size_t>(zobristKey) & clusterMask;
#endif
}

void TranspositionTable::prefetch(uint64_t zobristKey) const
{
#if defined(__GNUC__) || defined(__clang__)
    if (table && numClusters > 0) {
        // rw=0, locality=3: the TT is re-read constantly, so keep the line in L1.
        __builtin_prefetch(&table[clusterIndex(zobristKey)], 0, 3);
    }
#else
    (void)zobristKey;
#endif
}

int TranspositionTable::hashfull() const
{
    if (!table || numClusters == 0)
        return 0;

    const std::size_t sampleClusters = std::min<std::size_t>(numClusters, 1000);
    int used = 0;
    int seen = 0;
    for (std::size_t i = 0; i < sampleClusters; ++i) {
        for (int j = 0; j < TT_CLUSTER_SIZE; ++j) {
            const uint64_t data = table[i].entries[j].smp_data.load(TTRelaxed);
            ++seen;
            if (data != 0 && packed_age(data) == (currentAge & TTAgeMask))
                ++used;
        }
    }
    return seen ? static_cast<int>((static_cast<long long>(used) * 1000) / seen) : 0;
}

int TranspositionTable::hashfullMb() const
{
    const int sizeMb = hashSizeMb();
    return (hashfull() * sizeMb + 500) / 1000;
}

int TranspositionTable::hashSizeMb() const
{
    return static_cast<int>(allocBytes / 0x100000ULL);
}

bool TranspositionTable::probe(const thrawn::Position* pos, int& depth,
                               int& bestMove, int& score, int& flag, int& staticEval,
                               bool& wasPv)
{
    staticEval = no_hashmap_entry;
    wasPv = false;

    if (!table || numClusters == 0)
        return false;

    const TTCluster& cluster = table[clusterIndex(pos->zobristKey)];

    for (int i = 0; i < TT_CLUSTER_SIZE; i++)
    {
        const TTEntry& entry = cluster.entries[i];
        const uint64_t entry_data = entry.smp_data.load(TTRelaxed);
        if (entry_data == 0)
            continue;

        const uint64_t entry_key = entry.smp_key.load(TTRelaxed);
        if (!key_matches(entry_key, pos->zobristKey, entry_data))
            continue;

        depth = extractTTDepth(entry_data);
        bestMove = extractTTBestMove(entry_data);
        flag = extractTTHashFlag(entry_data);
        wasPv = extractTTWasPv(entry_data);
        staticEval = decode_static_eval(entry_key);

        score = extractTTScore(entry_data);
        // adjusted mate
        if (score < -mateScore)
            score += pos->ply;
        if (score > mateScore)
            score -= pos->ply;

        return true;
    }
    return false;
}

void TranspositionTable::store(const thrawn::Position* pos, int depth, int score, int flag,
                               int bestMove, int staticEval, bool wasPv)
{
    if (!table || numClusters == 0)
        return;

    TTCluster& cluster = table[clusterIndex(pos->zobristKey)];
    TTEntry* replace = &cluster.entries[0];
    uint64_t replace_data = 0;
    uint64_t replace_key = 0;
    const int current = currentAge & TTAgeMask;
    int worst_value = 1000000;
    bool replacingSamePosition = false;

    for (int i = 0; i < TT_CLUSTER_SIZE; i++)
    {
        TTEntry& candidate = cluster.entries[i];
        const uint64_t old_data = candidate.smp_data.load(TTRelaxed);
        const uint64_t old_key = candidate.smp_key.load(TTRelaxed);
        const bool samePosition = old_data != 0 && key_matches(old_key, pos->zobristKey, old_data);

        if (samePosition)
        {
            if (flag != BOUND_EXACT && depth < extractTTDepth(old_data) - 2)
            {
                if (staticEval != no_hashmap_entry)
                    update_static_eval_key(candidate, pos->zobristKey, staticEval);
                return;
            }

            replace = &candidate;
            replace_data = old_data;
            replace_key = old_key;
            replacingSamePosition = true;
            break;
        }

        const int value = replacement_value(old_data, current);
        if (value < worst_value)
        {
            worst_value = value;
            replace = &candidate;
            replace_data = old_data;
            replace_key = old_key;
        }
    }

    if (!replacingSamePosition && replace_data != 0 && flag != BOUND_EXACT && worst_value > depth + 2)
        return;

    // Adjust mate scores consistently:
    if (score < -mateScore)
        score -= pos->ply;
    if (score > mateScore)
        score += pos->ply;

    uint64_t data = encodeTTData(bestMove, depth, score, flag, wasPv);
    data |= (static_cast<uint64_t>(current) << TTAgeShift);

    int eval_to_store = no_hashmap_entry;
    if (staticEval != no_hashmap_entry)
        eval_to_store = staticEval;
    else if (replacingSamePosition)
        eval_to_store = decode_static_eval(replace_key);
    
    const uint64_t key = encode_key(pos->zobristKey, data, eval_to_store);
    // Publish data first, then the tag derived from it. A reader that catches
    // the pair half-written sees a tag that does not match and skips the entry.
    replace->smp_data.store(data, TTRelaxed);
    replace->smp_key.store(key, TTRelaxed);
}

void TranspositionTable::storeStaticEval(const thrawn::Position* pos, int staticEval)
{
    if (!table || numClusters == 0 || staticEval == no_hashmap_entry)
        return;

    TTCluster& cluster = table[clusterIndex(pos->zobristKey)];

    for (int i = 0; i < TT_CLUSTER_SIZE; i++)
    {
        TTEntry& entry = cluster.entries[i];
        if (entry.smp_data.load(TTRelaxed) == 0)
            continue;
        if (update_static_eval_key(entry, pos->zobristKey, staticEval))
            return;
    }
}

// bit allocations:
// best_move: 24 bits (mask: 0xFFFFFF)
// depth:      16 bits (mask: 0xFFFF)
// score:      17 bits (mask: 0x1FFFF) after adding an offset of 50000
// hash_flag:   2 bits (mask: 0x3)
// age:         5 bits (stored by store() in bits 59-63)
//
// Note: Score is encoded as score + SEARCH_INFINITY so that the range
// -50000...+50000 becomes 0...100000.

uint64_t TranspositionTable::encodeTTData(int best_move, int depth, int score, int hash_flag,
                                          bool wasPv) {
    // Offset the score to make it non-negative.
    int encoded_score = score + SEARCH_INFINITY; // now in the range 0 .. 100000

    uint64_t data = 0;
    data |= ((uint64_t)best_move & 0xFFFFFFULL);                // bits 0-23: best_move (24 bits)
    data |= (((uint64_t)std::clamp(depth, 0, 255) & 0xFFULL) << 24); // bits 24-31: depth (8 bits)
    data |= ((uint64_t)wasPv << 32);                              // bit 32: was a PV node
    data |= (((uint64_t)encoded_score & 0x1FFFFULL) << 40);       // bits 40-56: score (17 bits)
    data |= (((uint64_t)hash_flag & 0x3ULL) << 57);               // bits 57-58: hash_flag (2 bits)
    return data;
}

int TranspositionTable::extractTTBestMove(uint64_t data) {
    // Extract bits 0-23.
    return (int)(data & 0xFFFFFFULL);
}

int TranspositionTable::extractTTDepth(uint64_t data) {
    // Extract bits 24-31.
    return (int)((data >> 24) & 0xFFULL);
}

bool TranspositionTable::extractTTWasPv(uint64_t data) {
    return ((data >> 32) & 1ULL) != 0;
}

int TranspositionTable::extractTTScore(uint64_t data) {
    // Extract bits 40-56 then remove the offset.
    int encoded_score = (int)((data >> 40) & 0x1FFFFULL);
    return encoded_score - SEARCH_INFINITY;
}

int TranspositionTable::extractTTHashFlag(uint64_t data) {
    // Extract bits 57-58.
    return (int)((data >> 57) & 0x3ULL);
}
