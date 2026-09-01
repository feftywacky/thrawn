#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include "position.h"

// Constants for the tt->
static const int no_hashmap_entry = 100000;  // Sentinel for "TT miss"
static const int BOUND_NONE       = 0;
static const int BOUND_LOWER      = 1;
static const int BOUND_UPPER      = 2;
static const int BOUND_EXACT      = 3;
static const int TT_CLUSTER_SIZE  = 4;

// Entries are shared across search threads, so both words are atomic. All
// accesses are relaxed: on every supported target these compile to a plain
// load/store, but they make the concurrent access well-defined and stop the
// compiler from re-reading a word between the tag check and its use.
struct TTEntry 
{
    std::atomic<uint64_t> smp_key;  // 48-bit zobrist/data tag + 16-bit static eval
    std::atomic<uint64_t> smp_data; // Encoding depth, score, hash_flag and best_move into a U64
};

struct alignas(64) TTCluster
{
    TTEntry entries[TT_CLUSTER_SIZE];
};

static_assert(sizeof(TTEntry) == 16, "TTEntry must stay compact");
static_assert(sizeof(TTCluster) == 64, "TTCluster should occupy one cache line");
static_assert(alignof(TTCluster) == 64, "TTCluster should be cache-line aligned");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "TT entries require lock-free 64-bit atomics");

class TranspositionTable
{
public:
    TranspositionTable();
    ~TranspositionTable();

    // Initialize or resize the table to 'mb' megabytes.
    void initTable(int mb);

    // True once a table has actually been allocated. The allocation is deferred
    // until the GUI is done configuring us, so callers that need a live table
    // check this first.
    bool isAllocated() const { return table != nullptr && numClusters > 0; }

    // Clears all entries and resets the current age.
    void reset();

    // Increments the current age (to be called at the start of a new search)
    void incrementAge() { ++currentAge; }

    // Prefetch a cluster before a likely child-node TT probe.
    void prefetch(uint64_t zobristKey) const;

    // Permille of the table holding entries from the current search.
    int hashfull() const;

    int hashfullMb() const;
    int hashSizeMb() const;

    // Lookup a position in the tt
    bool probe(const thrawn::Position* pos, int& depth,
               int& bestMove, int& score, int& flag, int& staticEval, bool& wasPv);

    // Store an entry in the tt
    void store(const thrawn::Position* pos, int depth, int score, int flag,
               int bestMove, int staticEval = no_hashmap_entry, bool wasPv = false);

    // Attach a static eval to an existing entry without creating eval-only TT entries.
    void storeStaticEval(const thrawn::Position* pos, int staticEval);
    
    uint64_t encodeTTData(int bestMove, int depth, int score, int hash_flag, bool wasPv);

    int extractTTBestMove(uint64_t data);
    int extractTTDepth(uint64_t data);
    static bool extractTTWasPv(uint64_t data);
    int extractTTScore(uint64_t data);
    int extractTTHashFlag(uint64_t data);

private:
    std::size_t clusterIndex(uint64_t zobristKey) const;
    void freeTable();

    TTCluster*  table;       // Array of 64-byte TT clusters.
    void*       tableAlloc;  // Base of the (over-aligned) allocation backing `table`.
    std::size_t allocBytes;
    std::size_t numClusters;
    std::size_t clusterMask; // Only used on targets without a 128-bit multiply.
    int         currentAge;  // Current age, updated once per search before workers start.
};

#endif // TRANSPOSITION_TABLE_H
