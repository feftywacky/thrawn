#include <iostream>
#include <vector>
#include "move_generator.h"
#include "bitboard.h"
#include "constants.h"
#include "bitboard_helpers.h"
#include "move_helpers.h"
#include "nnue.h"
#include "zobrist_hashing.h"
#include "search.h"
#include "position.h"

using namespace std;

namespace {

inline void append_moves_from_attacks(int source,
                                      int piece,
                                      uint64_t attacks,
                                      uint64_t enemies,
                                      vector<int>& moves,
                                      int move_type)
{
    if (move_type != only_quiets)
    {
        uint64_t captures = attacks & enemies;
        while (captures)
        {
            const int target = pop_lsb(captures);
            moves.push_back(parse_move(source, target, piece, 0, 1, 0, 0, 0));
        }
    }

    if (move_type == all_moves || move_type == only_quiets)
    {
        uint64_t quiets = attacks & ~enemies;
        while (quiets)
        {
            const int target = pop_lsb(quiets);
            moves.push_back(parse_move(source, target, piece, 0, 0, 0, 0, 0));
        }
    }
}

inline int piece_side(int piece)
{
    return piece <= K ? white : black;
}

inline thrawn::UndoData* save_undo(thrawn::Position* pos, int move, int ply)
{
    if (ply < 0 || ply > MAX_DEPTH)
        return nullptr;

    thrawn::UndoData& undo = pos->undo_stack[ply];
    undo.move = move;
    undo.captured_piece = -1;
    undo.captured_square = null_sq;
    undo.castle_rights = pos->castle_rights;
    undo.enpassant = pos->enpassant;
    undo.fifty_move = pos->fifty_move;
    undo.zobristKey = pos->zobristKey;
    return &undo;
}

} // namespace


vector<int> generate_moves(thrawn::Position* pos)
{
    return generate_moves(pos, all_moves);
}

vector<int> generate_moves(thrawn::Position* pos, int move_type)
{
    vector<int> moves;
    moves.reserve(move_type == only_captures ? 32 : 96);

    if (pos->colour_to_move == white)
    {
        uint64_t curr = pos->piece_bitboards[P];
        parse_white_pawn_moves(pos, curr, moves, move_type);

        curr = pos->piece_bitboards[N];
        parse_knight_moves(pos, curr, N, moves, move_type);

        curr = pos->piece_bitboards[B];
        parse_bishop_moves(pos, curr, B, moves, move_type);

        curr = pos->piece_bitboards[R];
        parse_rook_moves(pos, curr, R, moves, move_type);

        curr = pos->piece_bitboards[Q];
        parse_queen_moves(pos, curr, Q, moves, move_type);

        if (move_type == all_moves || move_type == only_quiets)
            parse_white_castle_moves(pos, moves);

        curr = pos->piece_bitboards[K];
        parse_king_moves(pos, curr, K, moves, move_type);
    }
    else
    {
        uint64_t curr = pos->piece_bitboards[p];
        parse_black_pawn_moves(pos, curr, moves, move_type);

        curr = pos->piece_bitboards[n];
        parse_knight_moves(pos, curr, n, moves, move_type);

        curr = pos->piece_bitboards[b];
        parse_bishop_moves(pos, curr, b, moves, move_type);

        curr = pos->piece_bitboards[r];
        parse_rook_moves(pos, curr, r, moves, move_type);

        curr = pos->piece_bitboards[q];
        parse_queen_moves(pos, curr, q, moves, move_type);

        if (move_type == all_moves || move_type == only_quiets)
            parse_black_castle_moves(pos, moves);

        curr = pos->piece_bitboards[k];
        parse_king_moves(pos, curr, k, moves, move_type);
    }

    return moves;
}

void parse_white_pawn_moves(thrawn::Position* pos, uint64_t& curr, vector<int>& moves, int move_type)
{
    while (curr)
    {
        int source = pop_lsb(curr);
        int target = source - 8; // go up one square

        if (target>=a8 && !get_bit(pos->occupancies[both], target))
        {
            // pawn promotion by going up one square (NOT TAKING A PIECE)
            if (source>=a7 && source<=h7)
            {
                if (move_type != only_quiets)
                {
                    moves.push_back(parse_move(source, target, P, Q, 0, 0, 0, 0));
                    moves.push_back(parse_move(source, target, P, R, 0, 0, 0, 0));
                    moves.push_back(parse_move(source, target, P, N, 0, 0, 0, 0));
                    moves.push_back(parse_move(source, target, P, B, 0, 0, 0, 0));
                }
            }

            // one square and two square pawn moves
            else if (move_type == all_moves || move_type == only_quiets)
            {
                // one square
                moves.push_back(parse_move(source, target, P, 0, 0, 0, 0, 0));

                // two square
                if (source>=a2 && source<=h2 && !get_bit(pos->occupancies[both], target-8))
                {
                    moves.push_back(parse_move(source, target-8, P, 0, 0, 1, 0, 0));
                }
            }
        }

        if (move_type != only_quiets)
        {
            uint64_t attacks = pos->pawn_attacks[pos->colour_to_move][source]  & pos->occupancies[black];

            while (attacks) // while attacks squares are present on the board
            {   
                target = pop_lsb(attacks);

                if (source>=a7 && source<=h7) // pawn promotions by capturing a piece
                {
                    moves.push_back(parse_move(source, target, P, Q, 1, 0, 0, 0));
                    moves.push_back(parse_move(source, target, P, R, 1, 0, 0, 0));
                    moves.push_back(parse_move(source, target, P, N, 1, 0, 0, 0));
                    moves.push_back(parse_move(source, target, P, B, 1, 0, 0, 0));
                }

                // diagonal pawn capture
                else
                {
                    moves.push_back(parse_move(source, target, P, 0, 1, 0, 0, 0));
                }

            }

            // enpassant
            if (pos->enpassant!=null_sq)
            {
                
                uint64_t enpassant_attacks = pos->pawn_attacks[pos->colour_to_move][source] & (1ULL << pos->enpassant);
                if (enpassant_attacks)
                {
                    int enpassant_target = get_lsb_index(enpassant_attacks);
                    moves.push_back(parse_move(source, enpassant_target, P, 0, 1, 0, 1, 0));
                }
            }
        }
    }
}

void parse_white_castle_moves(thrawn::Position* pos, vector<int>& moves)
{
    if (pos->castle_rights & wks)
    {
        if (!get_bit(pos->occupancies[both], f1) && !get_bit(pos->occupancies[both], g1))
        {
            // make sure can't castle through check
            // if (!is_square_under_attack(e8, white) && !is_square_under_attack(f1, black) && !is_square_under_attack(g1, black))
            if (!is_square_under_attack(pos,e1, black) && !is_square_under_attack(pos,f1, black))
                moves.push_back(parse_move(e1, g1, K, 0, 0, 0, 0, 1));
        }
    }
    if (pos->castle_rights & wqs)
    {
        if (!get_bit(pos->occupancies[both], b1) && !get_bit(pos->occupancies[both], c1) && !get_bit(pos->occupancies[both], d1))
        {
            // make sure can't castle through check
            // if (!is_square_under_attack(e1, black) && !is_square_under_attack(c1, black) && !is_square_under_attack(d1, black))
            if (!is_square_under_attack(pos,e1, black) && !is_square_under_attack(pos,d1, black))
                moves.push_back(parse_move(e1, c1, K, 0, 0, 0, 0, 1));
        }
    }
}

void parse_black_pawn_moves(thrawn::Position* pos, uint64_t& curr, vector<int>& moves, int move_type)
{
    while(curr) // while white pawns are present on the board
    {
        int source = pop_lsb(curr);
        int target = source + 8; // go down one square

        if (target<=h1 && !get_bit(pos->occupancies[both], target))
        {
            // pawn promotion by going down one square (NOT TAKING A PIECE)
            if (source>=a2 && source<=h2)
            {
                if (move_type != only_quiets)
                {
                    moves.push_back(parse_move(source, target, p, q, 0, 0, 0, 0));
                    moves.push_back(parse_move(source, target, p, r, 0, 0, 0, 0));
                    moves.push_back(parse_move(source, target, p, n, 0, 0, 0, 0));
                    moves.push_back(parse_move(source, target, p, b, 0, 0, 0, 0));
                }
            }

            // one square and two square pawn moves
            else if (move_type == all_moves || move_type == only_quiets)
            {
                // one square
                moves.push_back(parse_move(source, target, p, 0, 0, 0, 0, 0));

                // two square
                if (source>=a7 && source<=h7 && !get_bit(pos->occupancies[both], target+8))
                    moves.push_back(parse_move(source, target+8, p, 0, 0, 1, 0, 0));
            }
        }

        if (move_type != only_quiets)
        {
            uint64_t attacks = pos->pawn_attacks[pos->colour_to_move][source] & pos->occupancies[white];

            while (attacks) // while attacks squares are present on the board
            {   
                target = pop_lsb(attacks);

                if (source>=a2 && source<=h2) // pawn promotion by capturing piece
                {
                    moves.push_back(parse_move(source, target, p, q, 1, 0, 0, 0));
                    moves.push_back(parse_move(source, target, p, r, 1, 0, 0, 0));
                    moves.push_back(parse_move(source, target, p, n, 1, 0, 0, 0));
                    moves.push_back(parse_move(source, target, p, b, 1, 0, 0, 0));
                }

                // diagonal pawn capture
                else
                {
                    moves.push_back(parse_move(source, target, p, 0, 1, 0, 0, 0));
                }

            }

            // enpassant
            if (pos->enpassant!=null_sq)
            {   
                uint64_t enpassant_attacks = pos->pawn_attacks[pos->colour_to_move][source] & (1ULL << pos->enpassant);
                if (enpassant_attacks)
                {
                    int enpassant_target = get_lsb_index(enpassant_attacks);
                    moves.push_back(parse_move(source, enpassant_target, p, 0, 1, 0, 1, 0));
                }
            }
        }
    }
}

void parse_black_castle_moves(thrawn::Position* pos, vector<int>& moves)
{
    if (pos->castle_rights & bks)
    {
        if (!get_bit(pos->occupancies[both], f8) && !get_bit(pos->occupancies[both], g8))
        {
            // pruend by make_move() for g8
            // if (!is_square_under_attack(e8, white) && !is_square_under_attack(f8, white) && !is_square_under_attack(g8, white))
            if (!is_square_under_attack(pos,e8, white) && !is_square_under_attack(pos,f8, white))
                moves.push_back(parse_move(e8, g8, k, 0, 0, 0, 0, 1));
        }
    }
    if (pos->castle_rights & bqs)
    {
        if (!get_bit(pos->occupancies[both], b8) && !get_bit(pos->occupancies[both], c8) && !get_bit(pos->occupancies[both], d8))
        {
            // pruend by make_move() 
            // if (!is_square_under_attack(e8, white) && !is_square_under_attack(c8, white) && !is_square_under_attack(d8, white))
            if (!is_square_under_attack(pos, e8, white) && !is_square_under_attack(pos,d8, white))
                moves.push_back(parse_move(e8, c8, k, 0, 0, 0, 0, 1));
        }
    }
}

void parse_knight_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves, int move_type)
{
    const uint64_t friends = (pos->colour_to_move == white) ? pos->occupancies[white] : pos->occupancies[black];
    const uint64_t enemies = (pos->colour_to_move == white) ? pos->occupancies[black] : pos->occupancies[white];

    while (curr)
    {
        const int source = pop_lsb(curr);
        const uint64_t attacks = pos->knight_attacks[source] & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    }
}

void parse_bishop_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves, int move_type)
{
    const uint64_t friends = (pos->colour_to_move == white) ? pos->occupancies[white] : pos->occupancies[black];
    const uint64_t enemies = (pos->colour_to_move == white) ? pos->occupancies[black] : pos->occupancies[white];

    while(curr)
    {
        const int source = pop_lsb(curr);
        const uint64_t attacks = get_bishop_attacks(pos, source, pos->occupancies[both]) & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    } 
}

void parse_rook_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves, int move_type)
{
    const uint64_t friends = (pos->colour_to_move == white) ? pos->occupancies[white] : pos->occupancies[black];
    const uint64_t enemies = (pos->colour_to_move == white) ? pos->occupancies[black] : pos->occupancies[white];

    while(curr)
    {
        const int source = pop_lsb(curr);
        const uint64_t attacks = get_rook_attacks(pos, source, pos->occupancies[both]) & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    }
}

void parse_queen_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves, int move_type)
{
    const uint64_t friends = (pos->colour_to_move == white) ? pos->occupancies[white] : pos->occupancies[black];
    const uint64_t enemies = (pos->colour_to_move == white) ? pos->occupancies[black] : pos->occupancies[white];

    while (curr)
    {
        const int source = pop_lsb(curr);
        const uint64_t attacks = get_queen_attacks(pos, source, pos->occupancies[both]) & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    }
}

void parse_king_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves, int move_type)
{
    const uint64_t friends = (pos->colour_to_move == white) ? pos->occupancies[white] : pos->occupancies[black];
    const uint64_t enemies = (pos->colour_to_move == white) ? pos->occupancies[black] : pos->occupancies[white];

    while (curr)
    {
        const int source = pop_lsb(curr);
        const uint64_t attacks = pos->king_attacks[source] & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    }
}

int make_move_on_board(thrawn::Position* pos, int move, int move_type, int ply)
{
    if (move_type == only_captures)
    {
        if (get_is_capture_move(move) || get_promoted_piece(move))
            return make_move_on_board(pos, move, all_moves, ply);

        return 0;
    }

    if (move_type != all_moves)
        return 0;

    const int stack_ply = (ply >= 0) ? ply : pos->ply;
    if (stack_ply < 0 || stack_ply > MAX_DEPTH)
        return 0;

    thrawn::UndoData* undo = save_undo(pos, move, stack_ply);

    const int source = get_move_source(move);
    const int target = get_move_target(move);
    const int piece = get_move_piece(move);
    const int promoted_piece = get_promoted_piece(move);
    const bool is_capture_move = get_is_capture_move(move) != 0;
    const bool double_pawn_move = get_is_double_pawn_move(move) != 0;
    const bool enpassant_move = get_is_move_enpassant(move) != 0;
    const bool castling = get_is_move_castling(move) != 0;
    const int moving_side = pos->colour_to_move;
    const int enemy_side = moving_side ^ 1;
    const uint64_t source_bb = square_bb(source);
    const uint64_t target_bb = square_bb(target);
    const int nnue_ply = (stack_ply >= 0) ? stack_ply : pos->ply;
    const bool use_nnue = nnue_loaded();

    const int opponent_king = (moving_side == white) ? k : K;
    if (is_capture_move && get_bit(pos->piece_bitboards[opponent_king], target))
        return 0;

    if (use_nnue)
        nnue_copy_parent_to_child(pos, nnue_ply);

    pop_bit(pos->piece_bitboards[piece], source);
    set_bit(pos->piece_bitboards[piece], target);
    pos->occupancies[moving_side] ^= source_bb | target_bb;
    pos->zobristKey ^= pos->piece_hashkey[piece][source];
    pos->zobristKey ^= pos->piece_hashkey[piece][target];
    if (use_nnue)
        nnue_remove_piece(pos, nnue_ply, piece, source);

    pos->fifty_move++;
    if (piece == P || piece == p)
        pos->fifty_move = 0;

    if (is_capture_move)
    {
        pos->fifty_move = 0;

        const int start_piece = (moving_side == white) ? p : P;
        const int end_piece = (moving_side == white) ? k : K;

        for (int captured = start_piece; captured <= end_piece; captured++)
        {
            if (get_bit(pos->piece_bitboards[captured], target))
            {
                if (undo != nullptr)
                {
                    undo->captured_piece = captured;
                    undo->captured_square = target;
                }

                pop_bit(pos->piece_bitboards[captured], target);
                pos->occupancies[enemy_side] &= ~target_bb;
                pos->zobristKey ^= pos->piece_hashkey[captured][target];
                if (use_nnue)
                    nnue_remove_piece(pos, nnue_ply, captured, target);
                break;
            }
        }
    }

    if (promoted_piece)
    {
        pop_bit(pos->piece_bitboards[piece], target);
        set_bit(pos->piece_bitboards[promoted_piece], target);
        pos->zobristKey ^= pos->piece_hashkey[piece][target];
        pos->zobristKey ^= pos->piece_hashkey[promoted_piece][target];
        if (use_nnue)
            nnue_add_piece(pos, nnue_ply, promoted_piece, target);
    }
    else if (use_nnue)
    {
        nnue_add_piece(pos, nnue_ply, piece, target);
    }

    if (enpassant_move)
    {
        const int captured_piece = (moving_side == white) ? p : P;
        const int captured_square = (moving_side == white) ? target + 8 : target - 8;
        const uint64_t captured_bb = square_bb(captured_square);

        if (undo != nullptr)
        {
            undo->captured_piece = captured_piece;
            undo->captured_square = captured_square;
        }

        pop_bit(pos->piece_bitboards[captured_piece], captured_square);
        pos->occupancies[enemy_side] &= ~captured_bb;
        pos->zobristKey ^= pos->piece_hashkey[captured_piece][captured_square];
        if (use_nnue)
            nnue_remove_piece(pos, nnue_ply, captured_piece, captured_square);
    }

    if (pos->enpassant != null_sq)
        pos->zobristKey ^= pos->enpassant_hashkey[pos->enpassant];
    pos->enpassant = null_sq;

    if (double_pawn_move)
    {
        pos->enpassant = (moving_side == white) ? target + 8 : target - 8;
        pos->zobristKey ^= pos->enpassant_hashkey[pos->enpassant];
    }

    if (castling)
    {
        if (target == g1)
        {
            pop_bit(pos->piece_bitboards[R], h1);
            set_bit(pos->piece_bitboards[R], f1);
            pos->occupancies[white] ^= square_bb(h1) | square_bb(f1);
            pos->zobristKey ^= pos->piece_hashkey[R][h1];
            pos->zobristKey ^= pos->piece_hashkey[R][f1];
            if (use_nnue)
            {
                nnue_remove_piece(pos, nnue_ply, R, h1);
                nnue_add_piece(pos, nnue_ply, R, f1);
            }
        }
        else if (target == c1)
        {
            pop_bit(pos->piece_bitboards[R], a1);
            set_bit(pos->piece_bitboards[R], d1);
            pos->occupancies[white] ^= square_bb(a1) | square_bb(d1);
            pos->zobristKey ^= pos->piece_hashkey[R][a1];
            pos->zobristKey ^= pos->piece_hashkey[R][d1];
            if (use_nnue)
            {
                nnue_remove_piece(pos, nnue_ply, R, a1);
                nnue_add_piece(pos, nnue_ply, R, d1);
            }
        }
        else if (target == g8)
        {
            pop_bit(pos->piece_bitboards[r], h8);
            set_bit(pos->piece_bitboards[r], f8);
            pos->occupancies[black] ^= square_bb(h8) | square_bb(f8);
            pos->zobristKey ^= pos->piece_hashkey[r][h8];
            pos->zobristKey ^= pos->piece_hashkey[r][f8];
            if (use_nnue)
            {
                nnue_remove_piece(pos, nnue_ply, r, h8);
                nnue_add_piece(pos, nnue_ply, r, f8);
            }
        }
        else if (target == c8)
        {
            pop_bit(pos->piece_bitboards[r], a8);
            set_bit(pos->piece_bitboards[r], d8);
            pos->occupancies[black] ^= square_bb(a8) | square_bb(d8);
            pos->zobristKey ^= pos->piece_hashkey[r][a8];
            pos->zobristKey ^= pos->piece_hashkey[r][d8];
            if (use_nnue)
            {
                nnue_remove_piece(pos, nnue_ply, r, a8);
                nnue_add_piece(pos, nnue_ply, r, d8);
            }
        }
    }

    pos->zobristKey ^= pos->castling_hashkey[pos->castle_rights];
    pos->castle_rights &= update_castling_right_values[source];
    pos->castle_rights &= update_castling_right_values[target];
    pos->zobristKey ^= pos->castling_hashkey[pos->castle_rights];

    pos->occupancies[both] = pos->occupancies[white] | pos->occupancies[black];

    pos->colour_to_move ^= 1;
    pos->zobristKey ^= pos->colour_to_move_hashkey;

    const int moved_king_square = (moving_side == white)
        ? get_lsb_index(pos->piece_bitboards[K])
        : get_lsb_index(pos->piece_bitboards[k]);

    if (is_square_under_attack(pos, moved_king_square, enemy_side))
    {
        if (undo != nullptr)
            unmake_move(pos, stack_ply);
        return 0;
    }

    nnue_debug_check(pos);
    return 1;
}

void unmake_move(thrawn::Position* pos, int ply)
{
    if (ply < 0 || ply > MAX_DEPTH)
        return;

    const thrawn::UndoData& undo = pos->undo_stack[ply];
    const int move = undo.move;

    if (move == 0)
    {
        pos->colour_to_move ^= 1;
        pos->castle_rights = undo.castle_rights;
        pos->enpassant = undo.enpassant;
        pos->fifty_move = undo.fifty_move;
        pos->zobristKey = undo.zobristKey;
        return;
    }

    const int source = get_move_source(move);
    const int target = get_move_target(move);
    const int piece = get_move_piece(move);
    const int promoted_piece = get_promoted_piece(move);
    const bool castling = get_is_move_castling(move) != 0;
    const int moving_side = piece_side(piece);
    const int enemy_side = moving_side ^ 1;
    const uint64_t source_bb = square_bb(source);
    const uint64_t target_bb = square_bb(target);

    pos->colour_to_move = moving_side;

    if (promoted_piece)
        pop_bit(pos->piece_bitboards[promoted_piece], target);
    else
        pop_bit(pos->piece_bitboards[piece], target);

    set_bit(pos->piece_bitboards[piece], source);
    pos->occupancies[moving_side] ^= source_bb | target_bb;

    if (undo.captured_piece != -1)
    {
        set_bit(pos->piece_bitboards[undo.captured_piece], undo.captured_square);
        pos->occupancies[enemy_side] |= square_bb(undo.captured_square);
    }

    if (castling)
    {
        if (target == g1)
        {
            pop_bit(pos->piece_bitboards[R], f1);
            set_bit(pos->piece_bitboards[R], h1);
            pos->occupancies[white] ^= square_bb(f1) | square_bb(h1);
        }
        else if (target == c1)
        {
            pop_bit(pos->piece_bitboards[R], d1);
            set_bit(pos->piece_bitboards[R], a1);
            pos->occupancies[white] ^= square_bb(d1) | square_bb(a1);
        }
        else if (target == g8)
        {
            pop_bit(pos->piece_bitboards[r], f8);
            set_bit(pos->piece_bitboards[r], h8);
            pos->occupancies[black] ^= square_bb(f8) | square_bb(h8);
        }
        else if (target == c8)
        {
            pop_bit(pos->piece_bitboards[r], d8);
            set_bit(pos->piece_bitboards[r], a8);
            pos->occupancies[black] ^= square_bb(d8) | square_bb(a8);
        }
    }

    pos->occupancies[both] = pos->occupancies[white] | pos->occupancies[black];
    pos->castle_rights = undo.castle_rights;
    pos->enpassant = undo.enpassant;
    pos->fifty_move = undo.fifty_move;
    pos->zobristKey = undo.zobristKey;
}

void make_null_move(thrawn::Position* pos, int ply)
{
    if (ply < 0 || ply > MAX_DEPTH)
        return;

    save_undo(pos, 0, ply);

    nnue_copy_parent_to_child(pos, ply);

    if (pos->enpassant != null_sq)
        pos->zobristKey ^= pos->enpassant_hashkey[pos->enpassant];
    pos->enpassant = null_sq;

    pos->colour_to_move ^= 1;
    pos->zobristKey ^= pos->colour_to_move_hashkey;
    nnue_debug_check(pos);
}

void unmake_null_move(thrawn::Position* pos, int ply)
{
    unmake_move(pos, ply);
}


int make_move(thrawn::Position* pos, int move, int move_type, int ply)
{
    return make_move_on_board(pos, move, move_type, ply);
}

int make_root_move(thrawn::Position* pos, int move, int move_type)
{
    pos->ply = 1;
    if (!make_move(pos, move, move_type, pos->ply))
    {
        pos->ply = 0;
        return 0;
    }

    if (nnue_loaded())
        nnue_promote_to_root(pos, 1);
    else
        pos->nnue_stack[0].valid = false;
    pos->ply = 0;
    nnue_debug_check(pos);
    return 1;
}
