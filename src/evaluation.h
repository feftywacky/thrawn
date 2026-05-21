#ifndef EVALUATION_H
#define EVALUATION_H

#include <array>
#include <cstdint>
#include "position.h"

// constants
enum {opening, endgame, middlegame};
enum {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};

// material
extern const int material_score[2][12];

// position
extern const int position_score[2][6][64];

extern const int mirror_score[128];

// files and rank masks
extern uint64_t file_masks[64];
extern uint64_t rank_masks[64];
extern uint64_t isolated_masks[64];
extern uint64_t wPassedPawn_masks[64];
extern uint64_t bPassedPawn_masks[64];

// penalties and bonuses
extern const int double_pawn_penalty_middlegame;
extern const int double_pawn_penalty_endgame;
extern const int isolated_pawn_penalty_middlegame;
extern const int isolated_pawn_penalty_endgame;

extern const int passed_pawn_bonus[8]; 
extern const int semi_open_file_score;
extern const int open_file_score;

extern const int bishop_mobility_unit;
extern const int queen_mobility_unit;

extern const int bishop_mobility_opening;
extern const int bishop_mobility_endgame;
extern const int queen_mobility_opening;
extern const int queen_mobility_endgame;


extern const int king_shield_bonus;

extern const int opening_phase_score;
extern const int endgame_phase_score;

extern const int get_rank_from_sq[64];

int evaluate_HCE(thrawn::Position* pos);
int evaluate(thrawn::Position* pos); // neural network eval

void init_eval_masks();
uint64_t set_eval_masks(int rankNum, int fileNum);

int get_gamePhase_score(thrawn::Position* pos);

#endif
