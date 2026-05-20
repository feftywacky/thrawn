#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <atomic>
#include <cstdint>
#include "position.h"

// Constants for the tt->
static const int no_hashmap_entry = 100000;  // Sentinel for "TT miss"
static const int BOUND_NONE       = 0;
static const int BOUND_LOWER      = 1;
static const int BOUND_UPPER      = 2;
static const int BOUND_EXACT      = 3;

struct TTEntry 
{
    std::atomic<uint64_t> smp_key{0};  // 48-bit zobrist/data tag + 16-bit static eval
    std::atomic<uint64_t> smp_data{0}; // Encoding depth, score, hash_flag and best_move into a U64
};

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
    void incrementAge() { currentAge.fetch_add(1, std::memory_order_relaxed); }

    // Lookup a position in the tt
    bool probe(const thrawn::Position* pos, int& depth, int alpha, int beta,
               int& bestMove, int& score, int& flag, int& staticEval);

    // Store an entry in the tt
    void store(const thrawn::Position* pos, int depth, int score, int flag,
               int bestMove, int staticEval = no_hashmap_entry);

    // Cache just the static eval when search has not yet produced a bound.
    void storeStaticEval(const thrawn::Position* pos, int staticEval);
    
    uint64_t encodeTTData(int bestMove, int depth, int score, int hash_flag);

    int extractTTBestMove(uint64_t data);
    int extractTTDepth(uint64_t data);
    int extractTTScore(uint64_t data);
    int extractTTHashFlag(uint64_t data);

private:
    TTEntry* table;      // Array of TT entries.
    int      numEntries; // Number of entries in the table.
    int      numClusters;
    int      clusterMask;
    std::atomic<int> currentAge; // Current age, updated once per search.
};

#endif // TRANSPOSITION_TABLE_H
