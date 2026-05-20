#ifndef ENGINE_H
#define ENGINE_H

#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include "bitboard.h"

using namespace std;

vector<int> generate_moves(thrawn::Position* pos);
vector<int> generate_moves(thrawn::Position* pos, int move_type);


int make_move(thrawn::Position* pos, int move, int move_type, int ply);
int make_move_on_board(thrawn::Position* pos, int move, int move_type, int ply);
int make_root_move(thrawn::Position* pos, int move, int move_type);
void unmake_move(thrawn::Position* pos, int ply);
void make_null_move(thrawn::Position* pos, int ply);
void unmake_null_move(thrawn::Position* pos, int ply);
    
    
void parse_white_pawn_moves(thrawn::Position* pos,uint64_t& curr, vector<int>& moves, int move_type = all_moves);
void parse_black_pawn_moves(thrawn::Position* pos,uint64_t& curr, vector<int>& moves, int move_type = all_moves);
void parse_knight_moves(thrawn::Position* pos,uint64_t& curr, const int& piece, vector<int>& moves, int move_type = all_moves);
void parse_bishop_moves(thrawn::Position* pos,uint64_t& curr, const int& piece, vector<int>& moves, int move_type = all_moves);
void parse_rook_moves(thrawn::Position* pos,uint64_t& curr, const int& piece, vector<int>& moves, int move_type = all_moves);
void parse_queen_moves(thrawn::Position* pos,uint64_t& curr, const int& piece, vector<int>& moves, int move_type = all_moves);
void parse_king_moves(thrawn::Position* pos,uint64_t& curr, const int& piece, vector<int>& moves, int move_type = all_moves);
void parse_white_castle_moves(thrawn::Position* pos,vector<int>& moves);
void parse_black_castle_moves(thrawn::Position* pos,vector<int>& moves);

#endif
