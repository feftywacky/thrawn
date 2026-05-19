#ifndef MOVE_HELPERS_H
#define MOVE_HELPERS_H

#include <vector>
  
inline int parse_move(int source, int target, int piece, int promoted_piece, int capture, int double_pawn_move, int enpassant, int castling)
{
    return source | (target << 6) | (piece << 12) | (promoted_piece << 16) |
           (capture << 20) | (double_pawn_move << 21) | (enpassant << 22) |
           (castling << 23);
}

inline int get_move_source(int move)
{
    return move & 0x3f;
}

inline int get_move_target(int move)
{
    return (move >> 6) & 0x3f;
}

inline int get_move_piece(int move)
{
    return (move >> 12) & 0xf;
}

inline int get_promoted_piece(int move)
{
    return (move >> 16) & 0xf;
}

inline int get_is_capture_move(int move)
{
    return move & 0x100000;
}

inline int get_is_double_pawn_move(int move)
{
    return move & 0x200000;
}

inline int get_is_move_enpassant(int move)
{
    return move & 0x400000;
}

inline int get_is_move_castling(int move)
{
    return move & 0x800000;
}

void print_move(const int& move);
void print_move_list(const std::vector<int>& moves);


#endif
