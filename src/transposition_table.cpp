#include "transposition_table.h"
#include "constants.h"
#include <algorithm>
#include <iostream>
#include <limits>

namespace {
constexpr int TTClusterSize = 4;
constexpr int TTAgeBits = 5;
constexpr int TTAgeShift = 59;
constexpr int TTAgeMask = (1 << TTAgeBits) - 1;
constexpr int TTKeyBits = 48;
constexpr int TTStaticEvalShift = TTKeyBits;
constexpr uint64_t TTKeyMask = (UINT64_C(1) << TTKeyBits) - 1;
constexpr int16_t TTStaticEvalNone = std::numeric_limits<int16_t>::min();

int relative_age(int currentAge, int entryAge) {
    return (currentAge - entryAge) & TTAgeMask;
}

int packed_depth(uint64_t data) {
    return static_cast<int>((data >> 24) & 0xFFFFULL);
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
    return (zobristKey ^ data) & TTKeyMask;
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
    uint64_t data = entry.smp_data.load(std::memory_order_relaxed);
    uint64_t expected = entry.smp_key.load(std::memory_order_relaxed);

    while (key_matches(expected, zobristKey, data)) {
        const uint64_t desired = (expected & TTKeyMask) |
            (static_cast<uint64_t>(static_cast<uint16_t>(encode_static_eval(staticEval))) << TTStaticEvalShift);
        if (entry.smp_key.compare_exchange_weak(expected, desired,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
            return true;
        }
        data = entry.smp_data.load(std::memory_order_relaxed);
    }

    return false;
}
} // namespace

TranspositionTable::TranspositionTable()
    : table(nullptr), numEntries(0), numClusters(0), clusterMask(0), currentAge(0)
{
}

TranspositionTable::~TranspositionTable()
{
    if (table) {
        delete[] table;
        table = nullptr;
    }
}

void TranspositionTable::initTable(int mb)
{
    int bytes = mb * 0x100000;  // Convert MB to bytes
    const int clusterCapacity = bytes / static_cast<int>(sizeof(TTEntry)) / TTClusterSize;

    if (table) {
        delete[] table;
        table = nullptr;
    }

    if (clusterCapacity < 1) {
        std::cerr << "TT init: table too small, forcing 4 MB.\n";
        initTable(4);
        return;
    }

    numClusters = 1;
    while (numClusters <= clusterCapacity / 2) {
        numClusters *= 2;
    }
    clusterMask = numClusters - 1;
    numEntries = numClusters * TTClusterSize;

    table = new TTEntry[numEntries];
    reset();

    std::cout << "TT: allocated " << mb << " MB, entries = " << numEntries << std::endl;
}

void TranspositionTable::reset()
{
    if (table && numEntries > 0) {
        for (int i = 0; i < numEntries; i++) {
            table[i].smp_key.store(0, std::memory_order_relaxed);
            table[i].smp_data.store(0, std::memory_order_relaxed);
        }
    }
    currentAge.store(0, std::memory_order_relaxed);
}

bool TranspositionTable::probe(const thrawn::Position* pos, int& depth, int alpha, int beta,
                               int& bestMove, int& score, int& flag, int& staticEval)
{
    staticEval = no_hashmap_entry;

    if (!table || numClusters <= 0)
        return false;

    int index = static_cast<int>((pos->zobristKey & static_cast<uint64_t>(clusterMask)) * TTClusterSize);

    for (int i = 0; i < TTClusterSize; i++)
    {
        const TTEntry& entry = table[index + i];
        const uint64_t entry_key = entry.smp_key.load(std::memory_order_relaxed);
        const uint64_t entry_data = entry.smp_data.load(std::memory_order_relaxed);

        if (entry_data != 0 && key_matches(entry_key, pos->zobristKey, entry_data))
        {
            depth = extractTTDepth(entry_data);
            bestMove = extractTTBestMove(entry_data);
            flag = extractTTHashFlag(entry_data);
            staticEval = decode_static_eval(entry_key);
            
            score = extractTTScore(entry_data);
            // adjusted mate
            if (score < -mateScore)
                score += pos->ply;
            if (score > mateScore) 
                score -= pos->ply;

            // if (entry_hash_flag == BOUND_EXACT) // pv node
            //     return score;
            // if (entry_hash_flag == BOUND_LOWER && score <= alpha) // fail-low score
            //     return alpha;
            // if (entry_hash_flag == BOUND_UPPER && score >= beta) // fail-high score
            //     return beta;

            return true;
        }
    }
    return false;
}

void TranspositionTable::store(const thrawn::Position* pos, int depth, int score, int flag,
                               int bestMove, int staticEval)
{
    if (!table || numClusters <= 0)
        return;

    int index = static_cast<int>((pos->zobristKey & static_cast<uint64_t>(clusterMask)) * TTClusterSize);
    TTEntry* replace = &table[index];
    uint64_t replace_data = replace->smp_data.load(std::memory_order_relaxed);
    uint64_t replace_key = replace->smp_key.load(std::memory_order_relaxed);
    const int current = currentAge.load(std::memory_order_relaxed);
    int worst_value = 1000000;
    bool replacingSamePosition = false;

    for (int i = 0; i < TTClusterSize; i++)
    {
        TTEntry& candidate = table[index + i];
        const uint64_t old_data = candidate.smp_data.load(std::memory_order_relaxed);
        const uint64_t old_key = candidate.smp_key.load(std::memory_order_relaxed);
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

        const int value = replacement_value(old_data, current & TTAgeMask);
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

    uint64_t data = encodeTTData(bestMove, depth, score, flag);
    data |= (static_cast<uint64_t>(current & TTAgeMask) << TTAgeShift);

    int eval_to_store = no_hashmap_entry;
    if (staticEval != no_hashmap_entry)
        eval_to_store = staticEval;
    else if (replacingSamePosition)
        eval_to_store = decode_static_eval(replace_key);
    
    const uint64_t key = encode_key(pos->zobristKey, data, eval_to_store);
    replace->smp_data.store(data, std::memory_order_relaxed);
    replace->smp_key.store(key, std::memory_order_relaxed);
}

void TranspositionTable::storeStaticEval(const thrawn::Position* pos, int staticEval)
{
    if (!table || numClusters <= 0 || staticEval == no_hashmap_entry)
        return;

    int index = static_cast<int>((pos->zobristKey & static_cast<uint64_t>(clusterMask)) * TTClusterSize);

    for (int i = 0; i < TTClusterSize; i++)
    {
        TTEntry& entry = table[index + i];
        const uint64_t entry_key = entry.smp_key.load(std::memory_order_relaxed);
        const uint64_t entry_data = entry.smp_data.load(std::memory_order_relaxed);
        if (entry_data != 0 && key_matches(entry_key, pos->zobristKey, entry_data))
        {
            update_static_eval_key(entry, pos->zobristKey, staticEval);
            return;
        }
    }

    store(pos, 0, 0, BOUND_NONE, 0, staticEval);
}

// bit allocations:
// best_move: 24 bits (mask: 0xFFFFFF)
// depth:      16 bits (mask: 0xFFFF)
// score:      17 bits (mask: 0x1FFFF) after adding an offset of 50000
// hash_flag:   2 bits (mask: 0x3)
// age:         5 bits (stored by store() in bits 59-63)
//
// Note: Score is encoded as score + INFINITY so that the range -50000...+50000 becomes 0...100000.

uint64_t TranspositionTable::encodeTTData(int best_move, int depth, int score, int hash_flag) {
    // Offset the score to make it non-negative.
    int encoded_score = score + INFINITY; // now in the range 0 .. 100000

    uint64_t data = 0;
    data |= ((uint64_t)best_move & 0xFFFFFFULL);                // bits 0-23: best_move (24 bits)
    data |= (((uint64_t)depth & 0xFFFFULL) << 24);                // bits 24-39: depth (16 bits)
    data |= (((uint64_t)encoded_score & 0x1FFFFULL) << 40);       // bits 40-56: score (17 bits)
    data |= (((uint64_t)hash_flag & 0x3ULL) << 57);               // bits 57-58: hash_flag (2 bits)
    return data;
}

int TranspositionTable::extractTTBestMove(uint64_t data) {
    // Extract bits 0-23.
    return (int)(data & 0xFFFFFFULL);
}

int TranspositionTable::extractTTDepth(uint64_t data) {
    // Extract bits 24-39.
    return (int)((data >> 24) & 0xFFFFULL);
}

int TranspositionTable::extractTTScore(uint64_t data) {
    // Extract bits 40-56 then remove the offset.
    int encoded_score = (int)((data >> 40) & 0x1FFFFULL);
    return encoded_score - INFINITY;
}

int TranspositionTable::extractTTHashFlag(uint64_t data) {
    // Extract bits 57-58.
    return (int)((data >> 57) & 0x3ULL);
}
