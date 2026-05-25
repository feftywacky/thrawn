#ifndef THREADING_H
#define THREADING_H

#include <atomic>
#include <array>
#include <mutex>
#include <vector>
#include "position.h"
#include "constants.h"

struct RootMove {
    int move;
    int score;
    int depth;
    int bound;
    int pv_length;
    bool completed;
    std::array<int, MAX_DEPTH> pv;

    RootMove();
};

class ThreadData {
public:
    // PV storage: one array for PV lengths and a 2D array for the PV lines.
    std::array<int, MAX_DEPTH> pv_length;
    std::array<std::array<int, MAX_DEPTH>, MAX_DEPTH> pv_table;

    // Killer, history, and counter-move ordering tables.
    std::array<std::array<int, MAX_DEPTH>, KILLER_MOVES> killer_moves;
    std::array<std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE>, 2> quiet_history;
    std::array<
        std::array<
            std::array<std::array<int, BOARD_SIZE>, HISTORY_SIZE>,
            BOARD_SIZE>,
        HISTORY_SIZE> continuation_history;
    std::array<std::array<std::array<int, HISTORY_SIZE>, BOARD_SIZE>, HISTORY_SIZE> capture_history;
    std::array<std::array<int, SEARCH_CORRECTION_HISTORY_SIZE>, 2> correction_history;
    std::array<std::array<int, BOARD_SIZE>, HISTORY_SIZE> counter_moves;
    std::array<int, MAX_DEPTH> ply_moves;
    std::array<int, MAX_DEPTH> static_eval_stack;

    // Flags used for move ordering and search heuristics.
    bool follow_pv_flag;
    bool score_pv_flag;
    bool allowNullMovePruning;

    long long nodes;

    int thread_id;
    int final_depth;
    int final_score;
    int final_bound;
    std::vector<RootMove> root_moves;
    std::vector<RootMove> iteration_root_moves;

    // Constructor initializes all arrays to zero and flags to false (or true where needed).
    ThreadData();

    // Reset the thread data between searches.
    void resetThreadData();

    void recordRootMove(int move, int score, int depth, int bound);
};

void smp_worker_thread_func(thrawn::Position* pos, int threadID, int maxDepth);

// search position entry point
void search_position_threaded(thrawn::Position* pos, int depth, int numThreads);

struct SearchResult {
    ThreadData* thread;
    int thread_id;
    int move;
    int score;
    int depth;
    int pv_length;
    bool has_move;
    std::array<int, MAX_DEPTH> pv;

    SearchResult();
};

SearchResult select_best_thread(ThreadData threadDatas[], int numThreads);

#endif // THREADING_H
