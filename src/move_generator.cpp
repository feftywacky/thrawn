#include "move_generator.h"
#include "bitboard.h"
#include "constants.h"
#include "bitboard_helpers.h"
#include "move_helpers.h"
#include "nnue.h"
#include "position.h"

namespace {

constexpr uint64_t rank_1 = 0xFF00000000000000ULL;
constexpr uint64_t rank_3 = 0x0000FF0000000000ULL;
constexpr uint64_t rank_6 = 0x0000000000FF0000ULL;
constexpr uint64_t rank_8 = 0x00000000000000FFULL;

void parse_white_pawn_moves(const thrawn::Position* pos, uint64_t pawns, MoveList& moves, int move_type);
void parse_black_pawn_moves(const thrawn::Position* pos, uint64_t pawns, MoveList& moves, int move_type);
void parse_knight_moves(const thrawn::Position* pos, uint64_t pieces, int piece, uint64_t friends, uint64_t enemies, MoveList& moves, int move_type);
void parse_bishop_moves(const thrawn::Position* pos, uint64_t pieces, int piece, uint64_t friends, uint64_t enemies, MoveList& moves, int move_type);
void parse_rook_moves(const thrawn::Position* pos, uint64_t pieces, int piece, uint64_t friends, uint64_t enemies, MoveList& moves, int move_type);
void parse_queen_moves(const thrawn::Position* pos, uint64_t pieces, int piece, uint64_t friends, uint64_t enemies, MoveList& moves, int move_type);
void parse_king_moves(const thrawn::Position* pos, uint64_t pieces, int piece, uint64_t friends, uint64_t enemies, MoveList& moves, int move_type);
void parse_white_castle_moves(const thrawn::Position* pos, MoveList& moves);
void parse_black_castle_moves(const thrawn::Position* pos, MoveList& moves);
int make_move_impl(thrawn::Position* pos, int move, int move_type, int ply, bool update_nnue);

inline void append_moves_from_attacks(int source,
                                      int piece,
                                      uint64_t attacks,
                                      uint64_t enemies,
                                      MoveList& moves,
                                      int move_type)
{
    if (move_type == only_captures || move_type == only_checks)
    {
        uint64_t captures = attacks & enemies;
        while (captures)
        {
            const int target = pop_lsb(captures);
            moves.push_back(parse_move(source, target, piece, 0, 1, 0, 0, 0));
        }
        return;
    }

    if (move_type == only_quiets)
    {
        uint64_t quiets = attacks & ~enemies;
        while (quiets)
        {
            const int target = pop_lsb(quiets);
            moves.push_back(parse_move(source, target, piece, 0, 0, 0, 0, 0));
        }
        return;
    }

    uint64_t captures = attacks & enemies;
    uint64_t quiets = attacks ^ captures;
    while (captures)
    {
        const int target = pop_lsb(captures);
        moves.push_back(parse_move(source, target, piece, 0, 1, 0, 0, 0));
    }
    while (quiets)
    {
        const int target = pop_lsb(quiets);
        moves.push_back(parse_move(source, target, piece, 0, 0, 0, 0, 0));
    }
}

inline void append_promotions(int source, int target, int piece, int queen, int rook_piece,
                              int knight, int bishop_piece, int capture, MoveList& moves)
{
    moves.push_back(parse_move(source, target, piece, queen, capture, 0, 0, 0));
    moves.push_back(parse_move(source, target, piece, rook_piece, capture, 0, 0, 0));
    moves.push_back(parse_move(source, target, piece, knight, capture, 0, 0, 0));
    moves.push_back(parse_move(source, target, piece, bishop_piece, capture, 0, 0, 0));
}

inline void append_white_pawn_capture(int source, int target, MoveList& moves)
{
    if (target <= h8)
        append_promotions(source, target, P, Q, R, N, B, 1, moves);
    else
        moves.push_back(parse_move(source, target, P, 0, 1, 0, 0, 0));
}

inline void append_black_pawn_capture(int source, int target, MoveList& moves)
{
    if (target >= a1)
        append_promotions(source, target, p, q, r, n, b, 1, moves);
    else
        moves.push_back(parse_move(source, target, p, 0, 1, 0, 0, 0));
}

inline int captured_piece_on(const thrawn::Position* pos, int side, uint64_t square)
{
    if (side == white)
    {
        if (pos->piece_bitboards[P] & square) return P;
        if (pos->piece_bitboards[N] & square) return N;
        if (pos->piece_bitboards[B] & square) return B;
        if (pos->piece_bitboards[R] & square) return R;
        if (pos->piece_bitboards[Q] & square) return Q;
        if (pos->piece_bitboards[K] & square) return K;
    }
    else
    {
        if (pos->piece_bitboards[p] & square) return p;
        if (pos->piece_bitboards[n] & square) return n;
        if (pos->piece_bitboards[b] & square) return b;
        if (pos->piece_bitboards[r] & square) return r;
        if (pos->piece_bitboards[q] & square) return q;
        if (pos->piece_bitboards[k] & square) return k;
    }

    return -1;
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


void generate_moves(thrawn::Position* pos, int move_type, MoveList& moves)
{
    moves.clear();

    const int side = pos->colour_to_move;
    const uint64_t friends = pos->occupancies[side];
    const uint64_t enemies = pos->occupancies[side ^ 1];

    if (pos->colour_to_move == white)
    {
        uint64_t curr = pos->piece_bitboards[P];
        parse_white_pawn_moves(pos, curr, moves, move_type);

        curr = pos->piece_bitboards[N];
        parse_knight_moves(pos, curr, N, friends, enemies, moves, move_type);

        curr = pos->piece_bitboards[B];
        parse_bishop_moves(pos, curr, B, friends, enemies, moves, move_type);

        curr = pos->piece_bitboards[R];
        parse_rook_moves(pos, curr, R, friends, enemies, moves, move_type);

        curr = pos->piece_bitboards[Q];
        parse_queen_moves(pos, curr, Q, friends, enemies, moves, move_type);

        if (move_type == all_moves || move_type == only_quiets)
            parse_white_castle_moves(pos, moves);

        curr = pos->piece_bitboards[K];
        parse_king_moves(pos, curr, K, friends, enemies, moves, move_type);
    }
    else
    {
        uint64_t curr = pos->piece_bitboards[p];
        parse_black_pawn_moves(pos, curr, moves, move_type);

        curr = pos->piece_bitboards[n];
        parse_knight_moves(pos, curr, n, friends, enemies, moves, move_type);

        curr = pos->piece_bitboards[b];
        parse_bishop_moves(pos, curr, b, friends, enemies, moves, move_type);

        curr = pos->piece_bitboards[r];
        parse_rook_moves(pos, curr, r, friends, enemies, moves, move_type);

        curr = pos->piece_bitboards[q];
        parse_queen_moves(pos, curr, q, friends, enemies, moves, move_type);

        if (move_type == all_moves || move_type == only_quiets)
            parse_black_castle_moves(pos, moves);

        curr = pos->piece_bitboards[k];
        parse_king_moves(pos, curr, k, friends, enemies, moves, move_type);
    }
}

namespace {

void parse_white_pawn_moves(const thrawn::Position* pos, uint64_t pawns, MoveList& moves, int move_type)
{
    const uint64_t empty = ~pos->occupancies[both];

    if (move_type == all_moves || move_type == only_quiets)
    {
        uint64_t pushes = (pawns >> 8) & empty;
        uint64_t quiets = pushes & ~rank_8;
        while (quiets)
        {
            const int target = pop_lsb(quiets);
            moves.push_back(parse_move(target + 8, target, P, 0, 0, 0, 0, 0));
        }

        uint64_t doubles = ((pushes & rank_3) >> 8) & empty;
        while (doubles)
        {
            const int target = pop_lsb(doubles);
            moves.push_back(parse_move(target + 16, target, P, 0, 0, 1, 0, 0));
        }
    }

    if (move_type == only_quiets)
        return;

    uint64_t promotions = ((pawns >> 8) & empty) & rank_8;
    while (promotions)
    {
        const int target = pop_lsb(promotions);
        append_promotions(target + 8, target, P, Q, R, N, B, 0, moves);
    }

    uint64_t captures = ((pawns >> 7) & not_a_file) & pos->occupancies[black];
    while (captures)
    {
        const int target = pop_lsb(captures);
        append_white_pawn_capture(target + 7, target, moves);
    }

    captures = ((pawns >> 9) & not_h_file) & pos->occupancies[black];
    while (captures)
    {
        const int target = pop_lsb(captures);
        append_white_pawn_capture(target + 9, target, moves);
    }

    if (pos->enpassant != null_sq)
    {
        uint64_t attackers = pos->pawn_attacks[black][pos->enpassant] & pawns;
        while (attackers)
        {
            const int source = pop_lsb(attackers);
            moves.push_back(parse_move(source, pos->enpassant, P, 0, 1, 0, 1, 0));
        }
    }
}

void parse_white_castle_moves(const thrawn::Position* pos, MoveList& moves)
{
    if (pos->castle_rights & wks)
    {
        if (!get_bit(pos->occupancies[both], f1) && !get_bit(pos->occupancies[both], g1))
        {
            if (!is_square_under_attack(pos, e1, black) && !is_square_under_attack(pos, f1, black))
                moves.push_back(parse_move(e1, g1, K, 0, 0, 0, 0, 1));
        }
    }
    if (pos->castle_rights & wqs)
    {
        if (!get_bit(pos->occupancies[both], b1) && !get_bit(pos->occupancies[both], c1) && !get_bit(pos->occupancies[both], d1))
        {
            if (!is_square_under_attack(pos, e1, black) && !is_square_under_attack(pos, d1, black))
                moves.push_back(parse_move(e1, c1, K, 0, 0, 0, 0, 1));
        }
    }
}

void parse_black_pawn_moves(const thrawn::Position* pos, uint64_t pawns, MoveList& moves, int move_type)
{
    const uint64_t empty = ~pos->occupancies[both];

    if (move_type == all_moves || move_type == only_quiets)
    {
        uint64_t pushes = (pawns << 8) & empty;
        uint64_t quiets = pushes & ~rank_1;
        while (quiets)
        {
            const int target = pop_lsb(quiets);
            moves.push_back(parse_move(target - 8, target, p, 0, 0, 0, 0, 0));
        }

        uint64_t doubles = ((pushes & rank_6) << 8) & empty;
        while (doubles)
        {
            const int target = pop_lsb(doubles);
            moves.push_back(parse_move(target - 16, target, p, 0, 0, 1, 0, 0));
        }
    }

    if (move_type == only_quiets)
        return;

    uint64_t promotions = ((pawns << 8) & empty) & rank_1;
    while (promotions)
    {
        const int target = pop_lsb(promotions);
        append_promotions(target - 8, target, p, q, r, n, b, 0, moves);
    }

    uint64_t captures = ((pawns << 7) & not_h_file) & pos->occupancies[white];
    while (captures)
    {
        const int target = pop_lsb(captures);
        append_black_pawn_capture(target - 7, target, moves);
    }

    captures = ((pawns << 9) & not_a_file) & pos->occupancies[white];
    while (captures)
    {
        const int target = pop_lsb(captures);
        append_black_pawn_capture(target - 9, target, moves);
    }

    if (pos->enpassant != null_sq)
    {
        uint64_t attackers = pos->pawn_attacks[white][pos->enpassant] & pawns;
        while (attackers)
        {
            const int source = pop_lsb(attackers);
            moves.push_back(parse_move(source, pos->enpassant, p, 0, 1, 0, 1, 0));
        }
    }
}

void parse_black_castle_moves(const thrawn::Position* pos, MoveList& moves)
{
    if (pos->castle_rights & bks)
    {
        if (!get_bit(pos->occupancies[both], f8) && !get_bit(pos->occupancies[both], g8))
        {
            if (!is_square_under_attack(pos, e8, white) && !is_square_under_attack(pos, f8, white))
                moves.push_back(parse_move(e8, g8, k, 0, 0, 0, 0, 1));
        }
    }
    if (pos->castle_rights & bqs)
    {
        if (!get_bit(pos->occupancies[both], b8) && !get_bit(pos->occupancies[both], c8) && !get_bit(pos->occupancies[both], d8))
        {
            if (!is_square_under_attack(pos, e8, white) && !is_square_under_attack(pos, d8, white))
                moves.push_back(parse_move(e8, c8, k, 0, 0, 0, 0, 1));
        }
    }
}

void parse_knight_moves(const thrawn::Position* pos, uint64_t pieces, int piece,
                        uint64_t friends, uint64_t enemies, MoveList& moves, int move_type)
{
    while (pieces)
    {
        const int source = pop_lsb(pieces);
        const uint64_t attacks = pos->knight_attacks[source] & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    }
}

void parse_bishop_moves(const thrawn::Position* pos, uint64_t pieces, int piece,
                        uint64_t friends, uint64_t enemies, MoveList& moves, int move_type)
{
    while (pieces)
    {
        const int source = pop_lsb(pieces);
        const uint64_t attacks = get_bishop_attacks(pos, source, pos->occupancies[both]) & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    } 
}

void parse_rook_moves(const thrawn::Position* pos, uint64_t pieces, int piece,
                      uint64_t friends, uint64_t enemies, MoveList& moves, int move_type)
{
    while (pieces)
    {
        const int source = pop_lsb(pieces);
        const uint64_t attacks = get_rook_attacks(pos, source, pos->occupancies[both]) & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    }
}

void parse_queen_moves(const thrawn::Position* pos, uint64_t pieces, int piece,
                       uint64_t friends, uint64_t enemies, MoveList& moves, int move_type)
{
    while (pieces)
    {
        const int source = pop_lsb(pieces);
        const uint64_t attacks = get_queen_attacks(pos, source, pos->occupancies[both]) & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    }
}

void parse_king_moves(const thrawn::Position* pos, uint64_t pieces, int piece,
                      uint64_t friends, uint64_t enemies, MoveList& moves, int move_type)
{
    while (pieces)
    {
        const int source = pop_lsb(pieces);
        const uint64_t attacks = pos->king_attacks[source] & ~friends;
        append_moves_from_attacks(source, piece, attacks, enemies, moves, move_type);
    }
}

int make_move_impl(thrawn::Position* pos, int move, int move_type, int ply, bool update_nnue)
{
    if (move_type == only_captures)
    {
        if (get_is_capture_move(move) || get_promoted_piece(move))
            return make_move_impl(pos, move, all_moves, ply, update_nnue);

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
    const bool use_nnue = update_nnue && nnue_loaded();

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

        const int captured = enpassant_move ? -1 : captured_piece_on(pos, enemy_side, target_bb);
        if (captured != -1)
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

    const int moved_king_square = (piece == K || piece == k)
        ? target
        : ((moving_side == white)
            ? get_lsb_index(pos->piece_bitboards[K])
            : get_lsb_index(pos->piece_bitboards[k]));

    if (is_square_under_attack(pos, moved_king_square, enemy_side))
    {
        if (undo != nullptr)
            unmake_move(pos, stack_ply);
        return 0;
    }

    if (use_nnue)
        nnue_debug_check(pos);
    return 1;
}

} // namespace

int make_move_on_board(thrawn::Position* pos, int move, int move_type, int ply)
{
    return make_move_impl(pos, move, move_type, ply, true);
}

int make_move_for_perft(thrawn::Position* pos, int move, int ply)
{
    return make_move_impl(pos, move, all_moves, ply, false);
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
