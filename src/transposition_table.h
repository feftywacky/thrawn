#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <cstdint>
#include "position.h"

// Constants for the tt->
static const int no_hashmap_entry = 100000;  // Sentinel for "TT miss"
static const int BOUND_NONE       = 0;
static const int BOUND_LOWER      = 1;
static const int BOUND_UPPER      = 2;
static const int BOUND_EXACT      = 3;
static const int TT_CLUSTER_SIZE  = 4;

struct TTEntry 
{
    uint64_t smp_key;  // 48-bit zobrist/data tag + 16-bit static eval
    uint64_t smp_data; // Encoding depth, score, hash_flag and best_move into a U64
};

struct alignas(64) TTCluster
{
    TTEntry entries[TT_CLUSTER_SIZE];
};

static_assert(sizeof(TTEntry) == 16, "TTEntry must stay compact");
static_assert(sizeof(TTCluster) == 64, "TTCluster should occupy one cache line");
static_assert(alignof(TTCluster) == 64, "TTCluster should be cache-line aligned");

class TranspositionTable
{
public:
    TranspositionTable();
    ~TranspositionTable();

    // Initialize or resize the table to 'mb' megabytes.
    void initTable(int mb);

    // Clears all entries and resets the current age.
    void reset();

    // Increments the current age (to be called at the start of a new search)
    void incrementAge() { ++currentAge; }

    // Prefetch a cluster before a likely child-node TT probe.
    void prefetch(uint64_t zobristKey) const;

    // Lookup a position in the tt
    bool probe(const thrawn::Position* pos, int& depth, int alpha, int beta,
               int& bestMove, int& score, int& flag, int& staticEval);

    // Store an entry in the tt
    void store(const thrawn::Position* pos, int depth, int score, int flag,
               int bestMove, int staticEval = no_hashmap_entry);

    // Attach a static eval to an existing entry without creating eval-only TT entries.
    void storeStaticEval(const thrawn::Position* pos, int staticEval);
    
    uint64_t encodeTTData(int bestMove, int depth, int score, int hash_flag);

    int extractTTBestMove(uint64_t data);
    int extractTTDepth(uint64_t data);
    int extractTTScore(uint64_t data);
    int extractTTHashFlag(uint64_t data);

private:
    int clusterIndex(uint64_t zobristKey) const;

    TTCluster* table;    // Array of 64-byte TT clusters.
    int      numEntries; // Number of entries in the table.
    int      numClusters;
    int      clusterMask;
    int      currentAge; // Current age, updated once per search before workers start.
};

#endif // TRANSPOSITION_TABLE_H
