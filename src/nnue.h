#ifndef NNUE_H
#define NNUE_H

#include "position.h"

namespace thrawn {
class Position;
}

void nnue_init(const char* evalFile);
bool nnue_loaded();

int nnue_evaluate(thrawn::Position* pos);

void nnue_refresh_root(thrawn::Position* pos);
void nnue_copy_parent_to_child(thrawn::Position* pos, int child_ply);
void nnue_add_piece(thrawn::Position* pos, int ply, int piece, int square);
void nnue_remove_piece(thrawn::Position* pos, int ply, int piece, int square);

void nnue_debug_check(const thrawn::Position* pos);

#endif
