#ifndef NNUE_H
#define NNUE_H

#include "position.h"
#include <string>

namespace thrawn {
class Position;
}

void nnue_init(const char* evalFile);
bool nnue_loaded();

int nnue_evaluate(thrawn::Position* pos);                 // engine score unit: cp
float nnue_evaluate_raw(const thrawn::Position* pos);      // Stockfish internal score_stm

void nnue_refresh_root(thrawn::Position* pos);
void nnue_copy_parent_to_child(thrawn::Position* pos, int child_ply);
void nnue_promote_to_root(thrawn::Position* pos, int ply);
void nnue_add_piece(thrawn::Position* pos, int ply, int piece, int square);
void nnue_remove_piece(thrawn::Position* pos, int ply, int piece, int square);

bool nnue_verify_position(const thrawn::Position* pos, std::string* error);
bool nnue_measure_evaluation_parity(const thrawn::Position* pos, float* abs_error_cp, std::string* error);
void nnue_debug_check(const thrawn::Position* pos);

#endif
