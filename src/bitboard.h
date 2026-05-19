#ifndef BITBOARD_H
#define BITBOARD_H

#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include "bitboard_helpers.h"
#include "constants.h"
#include "position.h"
#include <array>

using namespace std;

// get occupancy bitboard by colour
inline uint64_t get_white_occupancy(const thrawn::Position* pos)
{
    return pos->piece_bitboards[P] | pos->piece_bitboards[N] |
           pos->piece_bitboards[B] | pos->piece_bitboards[R] |
           pos->piece_bitboards[Q] | pos->piece_bitboards[K];
}

inline uint64_t get_black_occupancy(const thrawn::Position* pos)
{
    return pos->piece_bitboards[p] | pos->piece_bitboards[n] |
           pos->piece_bitboards[b] | pos->piece_bitboards[r] |
           pos->piece_bitboards[q] | pos->piece_bitboards[k];
}

inline uint64_t get_both_occupancy(const thrawn::Position* pos)
{
    return get_white_occupancy(pos) | get_black_occupancy(pos);
}

// pre-compute all attacks from a square methods

// pawns
uint64_t get_pawn_attacks(int side,const int& square);

// knights
uint64_t get_knight_attacks(const int& sqaure);

// kings
uint64_t get_king_attacks(const int& square);

// bishops
uint64_t get_bishop_mask(const int& square);
uint64_t bishop_attack_runtime_gen(int square, const uint64_t& blockers);
inline uint64_t get_bishop_attacks(const thrawn::Position* pos, int square, uint64_t occupancy)
{
    occupancy &= pos->bishop_masks[square];
    occupancy *= bishop_magic_nums[square];
    occupancy >>= 64 - bishop_relevant_bits[square];
    return pos->bishop_attacks[square][occupancy];
}

// rooks
uint64_t get_rook_mask(const int& square);
uint64_t rook_attack_runtime_gen(int square, uint64_t& blockers);
inline uint64_t get_rook_attacks(const thrawn::Position* pos, int square, uint64_t occupancy)
{
    occupancy &= pos->rook_masks[square];
    occupancy *= rook_magic_nums[square];
    occupancy >>= 64 - rook_relevant_bits[square];
    return pos->rook_attacks[square][occupancy];
}

// queen
inline uint64_t get_queen_attacks(const thrawn::Position* pos, int square, uint64_t occupancy)
{
    return get_bishop_attacks(pos, square, occupancy) |
           get_rook_attacks(pos, square, occupancy);
}

// set occupancy
uint64_t set_occupancy(const int& index, const int& bits_in_mask, uint64_t attack_mask);

bool is_square_under_attack(thrawn::Position* pos,int square, int side);


// MAGIC NUMBERS AND BITBOARDS
uint64_t find_magic_num(const int& square, int relevant_bits, int bishop);
void init_magic_nums();

inline bool noMajorsOrMinorsPieces(const thrawn::Position* pos)
{
    return !(pos->piece_bitboards[N] | pos->piece_bitboards[B] |
             pos->piece_bitboards[R] | pos->piece_bitboards[Q] |
             pos->piece_bitboards[n] | pos->piece_bitboards[b] |
             pos->piece_bitboards[r] | pos->piece_bitboards[q]);
}

// init all piece attacks
void init_leaping_attacks(thrawn::Position* pos);
void init_sliding_attacks(thrawn::Position* pos, int isBishop);
        

#endif
