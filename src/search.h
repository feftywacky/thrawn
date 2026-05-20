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
 * Negamax & Quiescence:
 * Now they accept a reference to ThreadData (td),
 * which holds PV arrays, killer moves, history, etc.
*/
int negamax(thrawn::Position* pos, ThreadData* td, int depth, int alpha, int beta);
int quiescence(thrawn::Position* pos, ThreadData* td, int alpha, int beta);

/*
 * Repetition check
 */
int isRepetition(thrawn::Position* pos);

/*
 * Some futility / LMP helpers
 */
int futility_margin(int depth);
int futility_move_count(int depth);

#endif // SEARCH_H
