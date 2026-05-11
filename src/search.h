#ifndef SEARCH_H
#define SEARCH_H

#include <vector>
#include <array>
#include <cstdint>
#include "position.h"
#include "threading.h" // for ThreadData

/*
some notes for negamax
3 types
- fail high: causes beta cut-off
- fail low: don't increase alpha
- pv nodes: increase alpha
*/

// global node counter
extern std::atomic<uint64_t> total_nodes;

/*
 Late Move Reductions or pruning factors
*/
extern std::array<int, 4> LateMovePruning_factors;
extern int RFP_factor;

/*
 * Negamax & Quiescence:
 * Now they accept a reference to ThreadData (td),
 * which holds PV arrays, killer moves, history, etc.
*/
int negamax(thrawn::Position* pos, ThreadData* td, int depth, int alpha, int beta);
int quiescence(thrawn::Position* pos, ThreadData* td, int alpha, int beta);

/*
 * Move ordering utilities
 */
int score_move(thrawn::Position* pos, ThreadData* td, int move);
void sort_moves(thrawn::Position* pos, ThreadData* td, std::vector<int>& moves, int bestMove);
void score_pv(thrawn::Position* pos, std::vector<int>& moves, ThreadData* td);

/*
 * Repetition check
 */
int isRepetition(thrawn::Position* pos);

/*
 * A helper for quicksort-based move ordering
 */
void quicksort_moves(std::vector<int> &moves,
                     std::vector<int> &move_scores,
                     int low, int high);

/*
 * Some futility / LMP helpers
 */
int futility_margin(int depth);
int futility_move_count(int depth);

// some search constants
static const int WindowDepth   = 4;
static const int WindowSize    = 24;

#endif // SEARCH_H
