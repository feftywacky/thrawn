#include "search.h"
#include "evaluation.h"
#include "move_generator.h"
#include "move_helpers.h"
#include "zobrist_hashing.h"
#include "transposition_table.h"
#include "bitboard.h"
#include "nnue.h"
#include "uci.h" // for 'stopped' and 'communicate()'
#include "globals.h"
#include "constants.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <atomic>

/*
some notes for negamax
3 types
- fail high: causes beta cut-off
- fail low: don't increase alpha
- pv nodes: increase alpha
*/

std::atomic<uint64_t> total_nodes(0);

namespace {


bool is_quiet_move(int move) {
    return !get_is_capture_move(move) && !get_promoted_piece(move);
}

bool is_mate_score(int score) {
    return score <= -mateScore || score >= mateScore;
}

int floor_log2_int(int value) {
    int result = 0;
    value = std::max(1, value);
    while (value >>= 1) {
        ++result;
    }
    return result;
}

int reverse_futility_margin(int depth) {
    if (depth <= 0) {
        return 0;
    }
    if (depth == 1) {
        return SEARCH_REVERSE_FUTILITY_MARGIN_1;
    }
    if (depth == 2) {
        return SEARCH_REVERSE_FUTILITY_MARGIN_2;
    }
    return SEARCH_REVERSE_FUTILITY_DEPTH_FACTOR * depth;
}

int razor_margin(int depth) {
    if (depth <= 0) {
        return 0;
    }
    if (depth == 1) {
        return SEARCH_RAZOR_MARGIN_1;
    }
    if (depth == 2) {
        return SEARCH_RAZOR_MARGIN_2;
    }
    return SEARCH_RAZOR_MARGIN_DEPTH_N;
}

int null_move_reduction(int depth, int static_eval, int beta) {
    const int eval_margin = std::max(0, static_eval - beta);
    const int depth_divisor = std::max(1, SEARCH_NULL_MOVE_DEPTH_DIVISOR);
    const int eval_divisor = std::max(1, SEARCH_NULL_MOVE_EVAL_DIVISOR);
    int reduction = SEARCH_NULL_MOVE_BASE_REDUCTION + depth / depth_divisor;
    reduction += std::min(SEARCH_NULL_MOVE_EVAL_BONUS_MAX, eval_margin / eval_divisor);
    return std::clamp(reduction, 1, std::max(1, depth - 1));
}

struct StaticEvalContext {
    bool improving = false;
    bool opponentWorsening = false;
    bool cutNode = false;
    bool allNode = false;
};

int reverse_futility_margin(int depth, const StaticEvalContext& context) {
    int margin = reverse_futility_margin(depth);
    if (context.improving)
        margin -= 24;
    else
        margin += 24;
    if (context.opponentWorsening)
        margin -= 16;
    // RFP only fires when static_eval - margin >= beta, which implies
    // static_eval >= beta and (with the !isPvNode guard at the call site) makes
    // context.cutNode necessarily true here. The cutNode term is therefore a
    // constant -16 for RFP rather than a discriminating signal; apply it
    // unconditionally so its effect is explicit.
    margin -= 16;
    return std::max(80, margin);
}

int null_move_reduction(int depth, int static_eval, int beta,
                        const StaticEvalContext& context) {
    int reduction = null_move_reduction(depth, static_eval, beta);
    if (context.cutNode && static_eval >= beta + 160)
        ++reduction;
    if (context.opponentWorsening)
        ++reduction;
    if (!context.improving && static_eval < beta + 80)
        --reduction;
    return std::clamp(reduction, 1, std::max(1, depth - 1));
}

int futility_margin_for_context(int depth, const StaticEvalContext& context) {
    int margin = futility_margin(depth);
    if (context.improving)
        margin += 32;
    if (context.opponentWorsening)
        margin -= 16;
    if (context.allNode)
        margin += 16;
    return std::max(80, margin);
}

int futility_move_count_for_context(int depth, const StaticEvalContext& context) {
    int count = futility_move_count(depth);
    if (context.improving)
        count += 4;
    if (context.opponentWorsening)
        count -= 2;
    if (context.cutNode)
        count -= 2;
    return std::max(1, count);
}

int probcut_margin_for_context(const StaticEvalContext& context) {
    int margin = SEARCH_PROBCUT_MARGIN;
    if (context.improving)
        margin -= 16;
    if (context.opponentWorsening)
        margin -= 16;
    if (context.allNode)
        margin += 16;
    return std::max(80, margin);
}

int piece_value(int piece) {
    return PIECE_VALUES[piece % 6];
}

void count_node(ThreadData* td) {
    td->nodes++;
    if ((td->nodes & (NODE_COUNTER_BATCH - 1)) == 0) {
        total_nodes.fetch_add(NODE_COUNTER_BATCH, std::memory_order_relaxed);
    }
}

int evaluate_static(thrawn::Position* pos, int cachedStaticEval = no_hashmap_entry) {
    if (cachedStaticEval != no_hashmap_entry) {
        return cachedStaticEval;
    }

    const int value = evaluate(pos);
    tt->storeStaticEval(pos, value);
    return value;
}

int correction_history_index(const thrawn::Position* pos) {
    static_assert((SEARCH_CORRECTION_HISTORY_SIZE & (SEARCH_CORRECTION_HISTORY_SIZE - 1)) == 0,
                  "correction history size must be a power of two");
    return static_cast<int>(pos->zobristKey) & (SEARCH_CORRECTION_HISTORY_SIZE - 1);
}

int corrected_static_eval(ThreadData* td, thrawn::Position* pos, int rawStaticEval) {
    const int correction = td->correction_history[pos->colour_to_move]
                                                 [correction_history_index(pos)];
    const int adjusted = rawStaticEval + correction / SEARCH_CORRECTION_HISTORY_GRAIN;
    return std::clamp(adjusted, -mateScore + MAX_DEPTH, mateScore - MAX_DEPTH);
}

StaticEvalContext make_static_eval_context(ThreadData* td, int ply, int staticEval,
                                           bool isPvNode, int beta) {
    StaticEvalContext context;
    const int trendMargin = 12;

    if (ply >= 2 && td->static_eval_stack[ply - 2] != no_hashmap_entry) {
        context.improving = staticEval > td->static_eval_stack[ply - 2] + trendMargin;
    }

    if (ply >= 1 && td->static_eval_stack[ply - 1] != no_hashmap_entry) {
        context.opponentWorsening = -staticEval < td->static_eval_stack[ply - 1] - trendMargin;
    }

    context.cutNode = !isPvNode && staticEval >= beta;
    context.allNode = !isPvNode && !context.cutNode;
    return context;
}

void update_correction_history(ThreadData* td, thrawn::Position* pos, int rawStaticEval,
                               int score, int depth, int bound) {
    if (rawStaticEval == no_hashmap_entry || is_mate_score(score)) {
        return;
    }

    const bool usefulBound =
        bound == BOUND_EXACT ||
        (bound == BOUND_LOWER && score > rawStaticEval) ||
        (bound == BOUND_UPPER && score < rawStaticEval);
    if (!usefulBound) {
        return;
    }

    int& entry = td->correction_history[pos->colour_to_move]
                                       [correction_history_index(pos)];
    const int target = std::clamp((score - rawStaticEval) *
                                      SEARCH_CORRECTION_HISTORY_GRAIN,
                                  -SEARCH_CORRECTION_HISTORY_MAX,
                                  SEARCH_CORRECTION_HISTORY_MAX);
    const int weight = std::clamp(depth + 1, 1, 16);
    entry += (target - entry) * weight / SEARCH_CORRECTION_HISTORY_WEIGHT_SCALE;
    entry = std::clamp(entry, -SEARCH_CORRECTION_HISTORY_MAX,
                       SEARCH_CORRECTION_HISTORY_MAX);
}

bool is_slider_piece(int piece) {
    const int type = piece % 6;
    return type == BISHOP || type == ROOK || type == QUEEN;
}

bool move_gives_check(thrawn::Position* pos, int move) {
    const int side = pos->colour_to_move;
    const int source = get_move_source(move);
    const int target = get_move_target(move);
    const int piece = get_move_piece(move);
    const int promoted = get_promoted_piece(move);
    const int checking_piece = promoted ? promoted : piece;
    const uint64_t enemy_king = pos->piece_bitboards[side == white ? k : K];
    if (!enemy_king) {
        return false;
    }

    const int king_square = get_lsb_index(enemy_king);
    const uint64_t source_bb = square_bb(source);
    const uint64_t target_bb = square_bb(target);

    uint64_t occupancy_after = pos->occupancies[both];
    occupancy_after &= ~source_bb;
    if (get_is_move_enpassant(move)) {
        const int captured_square = side == white ? target + 8 : target - 8;
        occupancy_after &= ~square_bb(captured_square);
    }
    occupancy_after |= target_bb;

    switch (checking_piece % 6) {
        case PAWN:
            if (pos->pawn_attacks[side][target] & enemy_king)
                return true;
            break;
        case KNIGHT:
            if (pos->knight_attacks[target] & enemy_king)
                return true;
            break;
        case BISHOP:
            if (get_bishop_attacks(pos, target, occupancy_after) & enemy_king)
                return true;
            break;
        case ROOK:
            if (get_rook_attacks(pos, target, occupancy_after) & enemy_king)
                return true;
            break;
        case QUEEN:
            if (get_queen_attacks(pos, target, occupancy_after) & enemy_king)
                return true;
            break;
        default:
            if (pos->king_attacks[target] & enemy_king)
                return true;
            break;
    }

    uint64_t diagonal_sliders = side == white
        ? (pos->piece_bitboards[B] | pos->piece_bitboards[Q])
        : (pos->piece_bitboards[b] | pos->piece_bitboards[q]);
    uint64_t orthogonal_sliders = side == white
        ? (pos->piece_bitboards[R] | pos->piece_bitboards[Q])
        : (pos->piece_bitboards[r] | pos->piece_bitboards[q]);

    if (is_slider_piece(piece)) {
        diagonal_sliders &= ~source_bb;
        orthogonal_sliders &= ~source_bb;
    }

    switch (checking_piece % 6) {
        case BISHOP:
            diagonal_sliders |= target_bb;
            break;
        case ROOK:
            orthogonal_sliders |= target_bb;
            break;
        case QUEEN:
            diagonal_sliders |= target_bb;
            orthogonal_sliders |= target_bb;
            break;
        default:
            break;
    }

    return (get_bishop_attacks(pos, king_square, occupancy_after) & diagonal_sliders) ||
           (get_rook_attacks(pos, king_square, occupancy_after) & orthogonal_sliders);
}

int captured_piece(thrawn::Position* pos, int move) {
    if (!get_is_capture_move(move)) {
        return -1;
    }

    if (get_is_move_enpassant(move)) {
        return pos->colour_to_move == white ? p : P;
    }

    // The victim of a (non-en-passant) capture is exactly the occupant of the
    // target square, which the mailbox holds in O(1) — identical to the old
    // 6-way piece-bitboard scan, but this runs on the hot SEE/move-ordering path.
    return pos->mailbox[get_move_target(move)];
}

int promotion_gain(int move) {
    const int promoted = get_promoted_piece(move);
    if (!promoted) {
        return 0;
    }

    return piece_value(promoted) - piece_value(get_move_piece(move));
}

int qsearch_move_gain_upper_bound(thrawn::Position* pos, int move) {
    const int victim = captured_piece(pos, move);
    return (victim == -1 ? 0 : piece_value(victim)) + promotion_gain(move);
}

uint64_t side_pieces(thrawn::Position* pos, int side, int white_piece, int black_piece) {
    return pos->piece_bitboards[side == white ? white_piece : black_piece];
}

int least_valuable_attacker(thrawn::Position* pos, int target, int side,
                            uint64_t occupancy, int& from)
{
    uint64_t attackers = (side == white ? pos->pawn_attacks[black][target]
                                        : pos->pawn_attacks[white][target]) &
                         side_pieces(pos, side, P, p) & occupancy;
    if (attackers) {
        from = get_lsb_index(attackers);
        return side == white ? P : p;
    }

    attackers = pos->knight_attacks[target] &
                side_pieces(pos, side, N, n) & occupancy;
    if (attackers) {
        from = get_lsb_index(attackers);
        return side == white ? N : n;
    }

    const uint64_t bishopAttacks = get_bishop_attacks(pos, target, occupancy);
    attackers = bishopAttacks & side_pieces(pos, side, B, b) & occupancy;
    if (attackers) {
        from = get_lsb_index(attackers);
        return side == white ? B : b;
    }

    const uint64_t rookAttacks = get_rook_attacks(pos, target, occupancy);
    attackers = rookAttacks & side_pieces(pos, side, R, r) & occupancy;
    if (attackers) {
        from = get_lsb_index(attackers);
        return side == white ? R : r;
    }

    // get_queen_attacks(target, occ) == bishop|rook attacks; reuse the sets
    // already computed above instead of issuing two more slider-table lookups.
    attackers = (bishopAttacks | rookAttacks) & side_pieces(pos, side, Q, q) & occupancy;
    if (attackers) {
        from = get_lsb_index(attackers);
        return side == white ? Q : q;
    }

    attackers = pos->king_attacks[target] &
                side_pieces(pos, side, K, k) & occupancy;
    if (attackers) {
        from = get_lsb_index(attackers);
        return side == white ? K : k;
    }

    from = null_sq;
    return -1;
}

int static_exchange_eval(thrawn::Position* pos, int move) {
    const int source = get_move_source(move);
    const int target = get_move_target(move);
    const int moving_piece = get_move_piece(move);
    const int victim = captured_piece(pos, move);
    const int promoted_piece = get_promoted_piece(move);

    if (victim == -1 && !promoted_piece) {
        return 0;
    }

    // Uninitialized: the swap-off loop writes gains[0] then each gains[d] before
    // it is ever read, and only [0, depth] is touched. Zero-init would memset
    // 128 bytes on every SEE call (run per capture during move ordering/pruning).
    std::array<int, 32> gains;
    int depth = 0;
    gains[0] = (victim == -1 ? 0 : piece_value(victim)) + promotion_gain(move);

    uint64_t occupancy = pos->occupancies[both];
    occupancy &= ~(1ULL << source);

    if (get_is_move_enpassant(move)) {
        const int captured_square = pos->colour_to_move == white ? target + 8 : target - 8;
        occupancy &= ~(1ULL << captured_square);
        occupancy |= (1ULL << target);
    } else if (victim == -1) {
        // victim == -1 here (non-en-passant branch) is exactly !get_is_capture_move:
        // captured_piece() returns -1 iff the move is not a capture.
        occupancy |= (1ULL << target);
    }

    int side_to_move = pos->colour_to_move ^ 1;
    int captured_value = promoted_piece ? piece_value(promoted_piece) : piece_value(moving_piece);

    while (depth + 1 < static_cast<int>(gains.size())) {
        int from = null_sq;
        const int attacker = least_valuable_attacker(pos, target, side_to_move, occupancy, from);
        if (attacker == -1) {
            break;
        }

        depth++;
        gains[depth] = captured_value - gains[depth - 1];
        occupancy &= ~(1ULL << from);
        captured_value = piece_value(attacker);
        side_to_move ^= 1;

        if (std::max(-gains[depth - 1], gains[depth]) < 0) {
            break;
        }
    }

    while (depth > 0) {
        gains[depth - 1] = -std::max(-gains[depth - 1], gains[depth]);
        depth--;
    }

    return gains[0];
}

bool qsearch_delta_prune(thrawn::Position* pos, int move, int static_eval, int alpha,
                         bool pawnOnlyEndgame) {
    if (is_mate_score(alpha) || pawnOnlyEndgame) {
        return false;
    }

    return static_eval + qsearch_move_gain_upper_bound(pos, move) +
           SEARCH_QSEARCH_DELTA_MARGIN <= alpha;
}

int history_bonus(int depth) {
    const int history_max = std::max(1, SEARCH_HISTORY_MAX);
    const int bonus = SEARCH_HISTORY_BONUS_DEPTH_SQUARED * depth * depth +
                      SEARCH_HISTORY_BONUS_DEPTH_LINEAR * depth;
    return std::min(history_max / 2, bonus);
}

int previous_ply_move(ThreadData* td, int ply);

void update_history_entry(int& entry, int bonus) {
    const int history_max = std::max(1, SEARCH_HISTORY_MAX);
    bonus = std::clamp(bonus, -history_max, history_max);
    const int gravity = bonus >= 0 ? bonus : -bonus;
    entry += bonus - entry * gravity / history_max;
    entry = std::clamp(entry, -history_max, history_max);
}

int continuation_history_score(ThreadData* td, int ply, int move) {
    const int previousMove = previous_ply_move(td, ply);
    if (previousMove == 0) {
        return 0;
    }

    return td->continuation_history[get_move_piece(previousMove)]
                                   [get_move_target(previousMove)]
                                   [get_move_piece(move)]
                                   [get_move_target(move)];
}

int quiet_history_score(ThreadData* td, int side, int ply, int move) {
    const int fromTo = td->quiet_history[side][get_move_source(move)]
                                        [get_move_target(move)];
    return 2 * fromTo + continuation_history_score(td, ply, move);
}

void update_continuation_history(ThreadData* td, int ply, int move, int bonus) {
    const int previousMove = previous_ply_move(td, ply);
    if (previousMove == 0) {
        return;
    }

    int& entry = td->continuation_history[get_move_piece(previousMove)]
                                         [get_move_target(previousMove)]
                                         [get_move_piece(move)]
                                         [get_move_target(move)];
    update_history_entry(entry, bonus);
}

void update_quiet_history(ThreadData* td, int side, int ply, int move, int depth) {
    const int bonus = history_bonus(depth);
    update_history_entry(td->quiet_history[side][get_move_source(move)]
                                           [get_move_target(move)], bonus);
    update_continuation_history(td, ply, move, bonus * SEARCH_CONTINUATION_HISTORY_NUMERATOR / SEARCH_CONTINUATION_HISTORY_DENOMINATOR);
}

template <typename MoveContainer>
void penalize_quiet_history(ThreadData* td, int side, int ply,
                            const MoveContainer& moves, int depth) {
    const int penalty = -history_bonus(depth);
    for (int move : moves) {
        update_history_entry(td->quiet_history[side][get_move_source(move)]
                                               [get_move_target(move)], penalty);
        update_continuation_history(td, ply, move, penalty * SEARCH_CONTINUATION_HISTORY_NUMERATOR / SEARCH_CONTINUATION_HISTORY_DENOMINATOR);
    }
}

int capture_history_victim(thrawn::Position* pos, int move) {
    int victim = captured_piece(pos, move);
    if (victim == -1) {
        victim = pos->colour_to_move == white ? p : P;
    }
    return victim;
}

int capture_history_score(ThreadData* td, thrawn::Position* pos, int move) {
    if (!get_is_capture_move(move)) {
        return 0;
    }

    const int victim = capture_history_victim(pos, move);
    return td->capture_history[get_move_piece(move)]
                              [get_move_target(move)]
                              [victim];
}

void update_capture_history(ThreadData* td, thrawn::Position* pos, int move, int depth) {
    if (!get_is_capture_move(move)) {
        return;
    }

    const int victim = capture_history_victim(pos, move);
    update_history_entry(td->capture_history[get_move_piece(move)]
                                            [get_move_target(move)]
                                            [victim],
                         history_bonus(depth));
}

template <typename MoveContainer>
void penalize_capture_history(ThreadData* td, thrawn::Position* pos,
                              const MoveContainer& moves, int depth) {
    const int penalty = -history_bonus(depth);
    for (int move : moves) {
        if (!get_is_capture_move(move)) {
            continue;
        }

        const int victim = capture_history_victim(pos, move);
        update_history_entry(td->capture_history[get_move_piece(move)]
                                                [get_move_target(move)]
                                                [victim],
                             penalty);
    }
}

int previous_ply_move(ThreadData* td, int ply) {
    if (ply <= 0 || ply > MAX_DEPTH - 1) {
        return 0;
    }
    return td->ply_moves[ply - 1];
}

bool is_counter_move(ThreadData* td, int ply, int move) {
    const int previousMove = previous_ply_move(td, ply);
    if (previousMove == 0) {
        return false;
    }
    return td->counter_moves[get_move_piece(previousMove)][get_move_target(previousMove)] == move;
}

void update_counter_move(ThreadData* td, int ply, int move) {
    const int previousMove = previous_ply_move(td, ply);
    if (previousMove == 0) {
        return;
    }
    td->counter_moves[get_move_piece(previousMove)][get_move_target(previousMove)] = move;
}

int late_move_reduction(int depth, int move_number, bool is_pv_node,
                        bool is_counter, int quiet_history,
                        const StaticEvalContext& context) {
    int reduction = SEARCH_LMR_BASE_REDUCTION +
                    floor_log2_int(depth) * floor_log2_int(move_number) / SEARCH_LMR_LOG_DIVISOR;
    if (!is_pv_node && depth >= SEARCH_LMR_NON_PV_DEPTH) {
        ++reduction;
    } else if (is_pv_node) {
        --reduction;
    }
    if (context.cutNode) {
        ++reduction;
    }
    if (context.improving) {
        --reduction;
    }
    if (context.opponentWorsening && !is_pv_node) {
        ++reduction;
    }
    if (depth >= SEARCH_LMR_MOVE_DEPTH_1 &&
        move_number > SEARCH_LMR_MOVE_NUMBER_1) {
        ++reduction;
    }
    if (depth >= SEARCH_LMR_MOVE_DEPTH_2 &&
        move_number > SEARCH_LMR_MOVE_NUMBER_2) {
        ++reduction;
    }

    const int history_max = std::max(1, SEARCH_HISTORY_MAX);
    const int good_history_divisor = std::max(1, SEARCH_LMR_GOOD_HISTORY_DIVISOR);
    const int bad_history_divisor = std::max(1, SEARCH_LMR_BAD_HISTORY_DIVISOR);
    const int good_threshold = history_max / good_history_divisor;
    const int bad_threshold = history_max / bad_history_divisor;

    if (is_counter || quiet_history > good_threshold) {
        --reduction;
    }
    if (quiet_history > history_max) {
        --reduction;
    }
    if (!is_counter && quiet_history < -bad_threshold) {
        ++reduction;
    }
    if (!is_counter && quiet_history < -history_max) {
        ++reduction;
    }

    return std::clamp(reduction, 1, std::max(1, depth - 2));
}

bool is_side_piece(int piece, int side) {
    return side == white ? piece >= P && piece <= K : piece >= p && piece <= k;
}

bool valid_promotion_piece(int side, int promoted) {
    if (!promoted) {
        return true;
    }

    if (!is_side_piece(promoted, side)) {
        return false;
    }

    const int type = promoted % 6;
    return type == KNIGHT || type == BISHOP || type == ROOK || type == QUEEN;
}

bool is_promotion_rank(int side, int source, int target) {
    return side == white ? (source >= a7 && source <= h7 && target >= a8 && target <= h8)
                         : (source >= a2 && source <= h2 && target >= a1 && target <= h1);
}

bool castle_move_is_pseudo_legal(thrawn::Position* pos, int move) {
    if (get_is_capture_move(move) || get_promoted_piece(move) ||
        get_is_double_pawn_move(move) || get_is_move_enpassant(move)) {
        return false;
    }

    const int source = get_move_source(move);
    const int target = get_move_target(move);
    const int side = pos->colour_to_move;
    const int enemy = side ^ 1;

    if (side == white && get_move_piece(move) == K && source == e1) {
        if (target == g1 && (pos->castle_rights & wks)) {
            return !get_bit(pos->occupancies[both], f1) &&
                   !get_bit(pos->occupancies[both], g1) &&
                   !is_square_under_attack(pos, e1, enemy) &&
                   !is_square_under_attack(pos, f1, enemy);
        }
        if (target == c1 && (pos->castle_rights & wqs)) {
            return !get_bit(pos->occupancies[both], b1) &&
                   !get_bit(pos->occupancies[both], c1) &&
                   !get_bit(pos->occupancies[both], d1) &&
                   !is_square_under_attack(pos, e1, enemy) &&
                   !is_square_under_attack(pos, d1, enemy);
        }
    }

    if (side == black && get_move_piece(move) == k && source == e8) {
        if (target == g8 && (pos->castle_rights & bks)) {
            return !get_bit(pos->occupancies[both], f8) &&
                   !get_bit(pos->occupancies[both], g8) &&
                   !is_square_under_attack(pos, e8, enemy) &&
                   !is_square_under_attack(pos, f8, enemy);
        }
        if (target == c8 && (pos->castle_rights & bqs)) {
            return !get_bit(pos->occupancies[both], b8) &&
                   !get_bit(pos->occupancies[both], c8) &&
                   !get_bit(pos->occupancies[both], d8) &&
                   !is_square_under_attack(pos, e8, enemy) &&
                   !is_square_under_attack(pos, d8, enemy);
        }
    }

    return false;
}

bool pawn_move_is_pseudo_legal(thrawn::Position* pos, int move) {
    const int source = get_move_source(move);
    const int target = get_move_target(move);
    const int side = pos->colour_to_move;
    const int promoted = get_promoted_piece(move);
    const bool capture = get_is_capture_move(move);
    const bool doublePush = get_is_double_pawn_move(move);
    const bool enPassant = get_is_move_enpassant(move);
    const uint64_t targetBb = square_bb(target);

    if (!valid_promotion_piece(side, promoted)) {
        return false;
    }
    if (promoted && !is_promotion_rank(side, source, target)) {
        return false;
    }
    if (!promoted && (side == white ? target >= a8 && target <= h8
                                    : target >= a1 && target <= h1)) {
        return false;
    }

    if (capture) {
        if (!(pos->pawn_attacks[side][source] & targetBb)) {
            return false;
        }
        if (doublePush) {
            return false;
        }
        if (enPassant) {
            if (target != pos->enpassant || promoted) {
                return false;
            }
            const int capturedSquare = side == white ? target + 8 : target - 8;
            const int capturedPawn = side == white ? p : P;
            return capturedSquare >= a8 && capturedSquare <= h1 &&
                   get_bit(pos->piece_bitboards[capturedPawn], capturedSquare);
        }
        return get_bit(pos->occupancies[side ^ 1], target);
    }

    if (enPassant) {
        return false;
    }

    const int singleTarget = side == white ? source - 8 : source + 8;
    if (target == singleTarget && !get_bit(pos->occupancies[both], target)) {
        return !doublePush;
    }

    const bool onStartRank = side == white ? (source >= a2 && source <= h2)
                                           : (source >= a7 && source <= h7);
    const int doubleTarget = side == white ? source - 16 : source + 16;
    if (doublePush && onStartRank && target == doubleTarget) {
        return !get_bit(pos->occupancies[both], singleTarget) &&
               !get_bit(pos->occupancies[both], target);
    }

    return false;
}

bool move_type_allows(int moveType, int move) {
    if (moveType == all_moves) {
        return true;
    }
    if (moveType == only_captures) {
        return get_is_capture_move(move) || get_promoted_piece(move);
    }
    if (moveType == only_quiets) {
        return is_quiet_move(move);
    }
    return true;
}

bool move_is_pseudo_legal(thrawn::Position* pos, int move, int moveType) {
    if (!move || !move_type_allows(moveType, move)) {
        return false;
    }

    const int source = get_move_source(move);
    const int target = get_move_target(move);
    const int piece = get_move_piece(move);
    if (source < a8 || source > h1 || target < a8 || target > h1 ||
        piece < P || piece > k) {
        return false;
    }

    const int side = pos->colour_to_move;
    if (!is_side_piece(piece, side) || !get_bit(pos->piece_bitboards[piece], source)) {
        return false;
    }

    const uint64_t targetBb = square_bb(target);
    if (pos->occupancies[side] & targetBb) {
        return false;
    }

    const bool capture = get_is_capture_move(move);
    const bool enPassant = get_is_move_enpassant(move);
    const bool castling = get_is_move_castling(move);
    if (castling) {
        return castle_move_is_pseudo_legal(pos, move);
    }

    if (get_is_double_pawn_move(move) && piece % 6 != PAWN) {
        return false;
    }
    if (enPassant && (piece % 6 != PAWN || !capture)) {
        return false;
    }
    if (get_promoted_piece(move) && piece % 6 != PAWN) {
        return false;
    }

    if (capture && !enPassant && !(pos->occupancies[side ^ 1] & targetBb)) {
        return false;
    }
    if (!capture && (pos->occupancies[side ^ 1] & targetBb)) {
        return false;
    }

    switch (piece % 6) {
        case PAWN:
            return pawn_move_is_pseudo_legal(pos, move);
        case KNIGHT:
            return !get_is_double_pawn_move(move) && !enPassant &&
                   !get_promoted_piece(move) &&
                   (pos->knight_attacks[source] & targetBb);
        case BISHOP:
            return !get_is_double_pawn_move(move) && !enPassant &&
                   !get_promoted_piece(move) &&
                   (get_bishop_attacks(pos, source, pos->occupancies[both]) & targetBb);
        case ROOK:
            return !get_is_double_pawn_move(move) && !enPassant &&
                   !get_promoted_piece(move) &&
                   (get_rook_attacks(pos, source, pos->occupancies[both]) & targetBb);
        case QUEEN:
            return !get_is_double_pawn_move(move) && !enPassant &&
                   !get_promoted_piece(move) &&
                   (get_queen_attacks(pos, source, pos->occupancies[both]) & targetBb);
        case KING:
            return !get_is_double_pawn_move(move) && !enPassant &&
                   !get_promoted_piece(move) &&
                   (pos->king_attacks[source] & targetBb);
        default:
            return false;
    }
}

int quiet_move_score(ThreadData* td, int side, int ply, int move) {
    if (td->killer_moves[0][ply] == move)
        return SEARCH_KILLER_MOVE_SCORE_1;
    if (td->killer_moves[1][ply] == move)
        return SEARCH_KILLER_MOVE_SCORE_2;
    if (is_counter_move(td, ply, move))
        return SEARCH_COUNTER_MOVE_SCORE +
               std::clamp(quiet_history_score(td, side, ply, move) /
                              std::max(1, SEARCH_COUNTER_MOVE_HISTORY_DIVISOR),
                          -SEARCH_COUNTER_MOVE_HISTORY_CAP,
                          SEARCH_COUNTER_MOVE_HISTORY_CAP);

    return std::clamp(quiet_history_score(td, side, ply, move),
                      -SEARCH_HISTORY_SCORE_CAP,
                      SEARCH_HISTORY_SCORE_CAP);
}

int tactical_move_score(thrawn::Position* pos, ThreadData* td, int move) {
    const int promotedPiece = get_promoted_piece(move);
    if (get_is_capture_move(move)) {
        int target = captured_piece(pos, move);
        if (target == -1)
            target = pos->colour_to_move == white ? p : P;

        int score = SEARCH_TACTICAL_CAPTURE_BASE_SCORE +
                    SEARCH_TACTICAL_VICTIM_MULTIPLIER * piece_value(target) -
                    piece_value(get_move_piece(move)) / SEARCH_TACTICAL_ATTACKER_DIVISOR +
                    capture_history_score(td, pos, move) / SEARCH_TACTICAL_CAPTURE_HISTORY_DIVISOR;
        if (promotedPiece == Q || promotedPiece == q)
            score += SEARCH_TACTICAL_QUEEN_PROMOTION_BONUS;
        else if (promotedPiece)
            score += piece_value(promotedPiece) / 2;
        return score;
    }

    if (promotedPiece == Q || promotedPiece == q)
        return SEARCH_QUEEN_PROMOTION_SCORE;
    if (promotedPiece)
        return SEARCH_QUEEN_PROMOTION_SCORE - SEARCH_TACTICAL_UNDERPROMOTION_OFFSET + piece_value(promotedPiece);

    return 0;
}

int bad_capture_score(thrawn::Position* pos, ThreadData* td, int move, int seeScore) {
    int target = captured_piece(pos, move);
    if (target == -1)
        target = pos->colour_to_move == white ? p : P;

    return SEARCH_BAD_CAPTURE_BASE_SCORE +
           piece_value(target) / SEARCH_BAD_CAPTURE_VICTIM_DIVISOR +
           capture_history_score(td, pos, move) / SEARCH_BAD_CAPTURE_HISTORY_DIVISOR +
           std::clamp(seeScore, SEARCH_BAD_CAPTURE_SEE_FLOOR, 0);
}

bool should_classify_capture_with_see(int move) {
    const int piece = get_move_piece(move);
    return get_is_capture_move(move) && !get_promoted_piece(move) &&
           piece != K && piece != k;
}

struct PickedMove {
    int move = 0;
    int seeScore = 0;
    bool seeKnown = false;
};

// No default member initializers: this keeps ScoredMove trivially default-
// constructible so a FixedBuffer<ScoredMove, 256> can be left uninitialized on
// construction (no per-element ctor loop). Every push_back below fully
// brace-initializes all four fields, and only [0, count) is ever read.
struct ScoredMove {
    int move;
    int score;
    int seeScore;
    bool seeKnown;
};

template <typename T, std::size_t Capacity>
struct FixedBuffer {
    // Backing storage is deliberately uninitialized: every access goes through
    // count (push_back / operator[] / begin()..end()), so only [0, count) is
    // touched. Value-initializing all Capacity (256) slots would memset several
    // KB per buffer on every node — these buffers live on the negamax hot path.
    std::array<T, Capacity> values;
    std::size_t count = 0;

    void clear() { count = 0; }
    void push_back(const T& value) {
        if (count < Capacity)
            values[count++] = value;
    }
    std::size_t size() const { return count; }
    bool empty() const { return count == 0; }
    T& operator[](std::size_t index) { return values[index]; }
    const T& operator[](std::size_t index) const { return values[index]; }
    T* begin() { return values.data(); }
    T* end() { return values.data() + count; }
    const T* begin() const { return values.data(); }
    const T* end() const { return values.data() + count; }
};

int next_square_on_step(int square, int step) {
    const int next = square + step;
    if (next < a8 || next > h1)
        return null_sq;

    const int fileDelta = std::abs((next % 8) - (square % 8));
    const int rankDelta = std::abs((next / 8) - (square / 8));
    if (fileDelta > 1 || rankDelta > 1)
        return null_sq;

    return next;
}

int direction_between_squares(int from, int to) {
    const int fromFile = from % 8;
    const int fromRank = from / 8;
    const int toFile = to % 8;
    const int toRank = to / 8;
    const int fileDelta = toFile - fromFile;
    const int rankDelta = toRank - fromRank;

    if (fileDelta == 0 && rankDelta != 0)
        return rankDelta > 0 ? 8 : -8;
    if (rankDelta == 0 && fileDelta != 0)
        return fileDelta > 0 ? 1 : -1;
    if (std::abs(fileDelta) == std::abs(rankDelta) && fileDelta != 0)
        return (rankDelta > 0 ? 8 : -8) + (fileDelta > 0 ? 1 : -1);

    return 0;
}

bool is_diagonal_step(int step) {
    return step == -9 || step == -7 || step == 7 || step == 9;
}

bool is_orthogonal_step(int step) {
    return step == -8 || step == -1 || step == 1 || step == 8;
}

bool slider_matches_pin_ray(int piece, int step) {
    const int type = piece % 6;
    return type == QUEEN ||
           (is_diagonal_step(step) && type == BISHOP) ||
           (is_orthogonal_step(step) && type == ROOK);
}

int first_occupied_square_on_ray(thrawn::Position* pos, int from, int step) {
    for (int sq = next_square_on_step(from, step); sq != null_sq;
         sq = next_square_on_step(sq, step)) {
        if (get_bit(pos->occupancies[both], sq))
            return sq;
    }
    return null_sq;
}

bool target_stays_between_king_and_slider(int kingSquare, int target,
                                          int sliderSquare, int step) {
    for (int sq = next_square_on_step(kingSquare, step); sq != null_sq;
         sq = next_square_on_step(sq, step)) {
        if (sq == target)
            return true;
        if (sq == sliderSquare)
            return false;
    }
    return false;
}

uint64_t enemy_piece_bb_after_king_move(thrawn::Position* pos, int enemy,
                                        int whitePiece, int blackPiece,
                                        uint64_t capturedTarget) {
    const int piece = enemy == white ? whitePiece : blackPiece;
    return pos->piece_bitboards[piece] & ~capturedTarget;
}

bool king_target_attacked_after_move(thrawn::Position* pos, int move) {
    const int side = pos->colour_to_move;
    const int enemy = side ^ 1;
    const int source = get_move_source(move);
    const int target = get_move_target(move);
    const uint64_t sourceBb = square_bb(source);
    const uint64_t targetBb = square_bb(target);
    const uint64_t capturedTarget =
        get_is_capture_move(move) && !get_is_move_enpassant(move) ? targetBb : 0ULL;

    uint64_t occupancy = pos->occupancies[both];
    occupancy &= ~sourceBb;
    occupancy |= targetBb;

    if (get_is_move_castling(move)) {
        if (target == g1)
            occupancy = (occupancy & ~square_bb(h1)) | square_bb(f1);
        else if (target == c1)
            occupancy = (occupancy & ~square_bb(a1)) | square_bb(d1);
        else if (target == g8)
            occupancy = (occupancy & ~square_bb(h8)) | square_bb(f8);
        else if (target == c8)
            occupancy = (occupancy & ~square_bb(a8)) | square_bb(d8);
    }

    const uint64_t enemyPawns =
        enemy_piece_bb_after_king_move(pos, enemy, P, p, capturedTarget);
    const uint64_t pawnAttackers = enemy == white
        ? pos->pawn_attacks[black][target] & enemyPawns
        : pos->pawn_attacks[white][target] & enemyPawns;
    if (pawnAttackers)
        return true;

    if (pos->knight_attacks[target] &
        enemy_piece_bb_after_king_move(pos, enemy, N, n, capturedTarget))
        return true;

    if (pos->king_attacks[target] &
        enemy_piece_bb_after_king_move(pos, enemy, K, k, capturedTarget))
        return true;

    const uint64_t enemyBishops =
        enemy_piece_bb_after_king_move(pos, enemy, B, b, capturedTarget);
    const uint64_t enemyRooks =
        enemy_piece_bb_after_king_move(pos, enemy, R, r, capturedTarget);
    const uint64_t enemyQueens =
        enemy_piece_bb_after_king_move(pos, enemy, Q, q, capturedTarget);

    if (get_bishop_attacks(pos, target, occupancy) & (enemyBishops | enemyQueens))
        return true;
    if (get_rook_attacks(pos, target, occupancy) & (enemyRooks | enemyQueens))
        return true;

    return false;
}

bool move_respects_absolute_pin(thrawn::Position* pos, int move) {
    if (get_is_move_enpassant(move))
        return true;

    const int side = pos->colour_to_move;
    const int piece = get_move_piece(move);
    if (piece % 6 == KING)
        return !king_target_attacked_after_move(pos, move);

    const uint64_t kingBb = pos->piece_bitboards[side == white ? K : k];
    if (!kingBb)
        return true;

    const int kingSquare = get_lsb_index(kingBb);
    const int source = get_move_source(move);
    const int target = get_move_target(move);
    const int step = direction_between_squares(kingSquare, source);
    if (!step)
        return true;

    if (first_occupied_square_on_ray(pos, kingSquare, step) != source)
        return true;

    const int sliderSquare = first_occupied_square_on_ray(pos, source, step);
    if (sliderSquare == null_sq)
        return true;

    const int enemyStart = side == white ? p : P;
    const int enemyEnd = side == white ? k : K;
    // sliderSquare is occupied (first_occupied_square_on_ray returned non-null),
    // so its occupant is in the mailbox; the old loop matched only an enemy piece
    // there, so reproduce that by keeping the occupant only when it is enemy-side.
    const int occupant = pos->mailbox[sliderSquare];
    const int slider = (occupant >= enemyStart && occupant <= enemyEnd) ? occupant : -1;

    if (slider == -1 || !slider_matches_pin_ray(slider, step))
        return true;

    return direction_between_squares(kingSquare, target) == step &&
           target_stays_between_king_and_slider(kingSquare, target, sliderSquare, step);
}

bool move_passes_fast_legal_filter(thrawn::Position* pos, int move, bool inCheck) {
    const int piece = get_move_piece(move);
    if (piece % 6 == KING)
        return !king_target_attacked_after_move(pos, move);
    if (inCheck)
        return true;
    return move_respects_absolute_pin(pos, move);
}

class MovePicker {
public:
    MovePicker(thrawn::Position* pos, ThreadData* td, int ttMove,
               bool followPv, int moveType, bool inCheck)
        : pos(pos),
          td(td),
          ttMove(ttMove),
          pvMove(followPv && pos->ply < MAX_DEPTH ? td->pv_table[0][pos->ply] : 0),
          moveType(moveType),
          inCheck(inCheck),
          stage(Stage::TtMove) {}

    bool next(PickedMove& picked) {
        while (stage != Stage::Done) {
            switch (stage) {
                case Stage::TtMove:
                    stage = Stage::PvMove;
                    if (try_special(ttMove, picked)) {
                        return true;
                    }
                    break;
                case Stage::PvMove:
                    stage = moveType == only_quiets ? Stage::GenerateQuiets
                                                    : Stage::GenerateTacticals;
                    if (try_special(pvMove, picked)) {
                        return true;
                    }
                    break;
                case Stage::GenerateTacticals:
                    generate_tacticals();
                    stage = Stage::GoodTacticals;
                    break;
                case Stage::GoodTacticals:
                    if (next_good_tactical(picked)) {
                        return true;
                    }
                    stage = includes_quiets() ? Stage::Killer1 : Stage::BadTacticals;
                    break;
                case Stage::Killer1:
                    stage = Stage::Killer2;
                    if (try_special_quiet(td->killer_moves[0][pos->ply], picked)) {
                        return true;
                    }
                    break;
                case Stage::Killer2:
                    stage = Stage::CounterMove;
                    if (try_special_quiet(td->killer_moves[1][pos->ply], picked)) {
                        return true;
                    }
                    break;
                case Stage::CounterMove:
                    stage = Stage::GenerateQuiets;
                    if (try_special_quiet(counter_move(), picked)) {
                        return true;
                    }
                    break;
                case Stage::GenerateQuiets:
                    generate_quiets();
                    stage = Stage::Quiets;
                    break;
                case Stage::Quiets:
                    if (next_scored(quietMoves, quietIndex, picked)) {
                        return true;
                    }
                    stage = Stage::BadTacticals;
                    break;
                case Stage::BadTacticals:
                    if (next_scored(badTacticals, badIndex, picked)) {
                        return true;
                    }
                    stage = Stage::Done;
                    break;
                case Stage::Done:
                    break;
            }
        }

        return false;
    }

private:
    enum class Stage {
        TtMove,
        PvMove,
        GenerateTacticals,
        GoodTacticals,
        Killer1,
        Killer2,
        CounterMove,
        GenerateQuiets,
        Quiets,
        BadTacticals,
        Done
    };

    thrawn::Position* pos;
    ThreadData* td;
    int ttMove;
    int pvMove;
    int moveType;
    bool inCheck;
    Stage stage;
    FixedBuffer<ScoredMove, MAX_GENERATED_MOVES> tacticals;
    FixedBuffer<ScoredMove, MAX_GENERATED_MOVES> badTacticals;
    FixedBuffer<ScoredMove, MAX_GENERATED_MOVES> quietMoves;
    std::size_t tacticalIndex = 0;
    std::size_t badIndex = 0;
    std::size_t quietIndex = 0;
    std::array<int, 8> triedMoves;  // only [0, triedCount) is ever read
    int triedCount = 0;

    bool includes_quiets() const {
        return moveType == all_moves || moveType == only_quiets;
    }

    bool already_tried(int move) const {
        for (int i = 0; i < triedCount; ++i) {
            if (triedMoves[i] == move) {
                return true;
            }
        }
        return false;
    }

    void mark_tried(int move) {
        if (move && triedCount < static_cast<int>(triedMoves.size()) &&
            !already_tried(move)) {
            triedMoves[triedCount++] = move;
        }
    }

    bool try_special(int move, PickedMove& picked) {
        if (!move || already_tried(move) ||
            !move_is_pseudo_legal(pos, move, moveType) ||
            !move_passes_fast_legal_filter(pos, move, inCheck)) {
            return false;
        }

        mark_tried(move);
        picked = {move, 0, false};
        return true;
    }

    bool try_special_quiet(int move, PickedMove& picked) {
        if (!includes_quiets() || !is_quiet_move(move)) {
            return false;
        }
        return try_special(move, picked);
    }

    int counter_move() const {
        const int previousMove = previous_ply_move(td, pos->ply);
        if (previousMove == 0) {
            return 0;
        }
        return td->counter_moves[get_move_piece(previousMove)]
                                [get_move_target(previousMove)];
    }

    static std::size_t select_best(FixedBuffer<ScoredMove, MAX_GENERATED_MOVES>& moves,
                                   std::size_t first) {
        std::size_t best = first;
        for (std::size_t i = first + 1; i < moves.size(); ++i) {
            if (moves[i].score > moves[best].score) {
                best = i;
            }
        }
        if (best != first) {
            std::swap(moves[best], moves[first]);
        }
        return first;
    }

    bool next_scored(FixedBuffer<ScoredMove, MAX_GENERATED_MOVES>& moves, std::size_t& index,
                     PickedMove& picked) {
        while (index < moves.size()) {
            const std::size_t best = select_best(moves, index);
            ScoredMove scored = moves[best];
            ++index;
            if (already_tried(scored.move)) {
                continue;
            }

            mark_tried(scored.move);
            picked = {scored.move, scored.seeScore, scored.seeKnown};
            return true;
        }
        return false;
    }

    static void score_tactical_move(int move, void* context) {
        static_cast<MovePicker*>(context)->add_tactical(move);
    }

    static void score_quiet_move(int move, void* context) {
        static_cast<MovePicker*>(context)->add_quiet(move);
    }

    void add_tactical(int move) {
        if (already_tried(move) ||
            !move_passes_fast_legal_filter(pos, move, inCheck)) {
            return;
        }
        tacticals.push_back({move, tactical_move_score(pos, td, move), 0, false});
    }

    void add_quiet(int move) {
        if (already_tried(move) ||
            !move_passes_fast_legal_filter(pos, move, inCheck)) {
            return;
        }
        quietMoves.push_back({move, quiet_move_score(td, pos->colour_to_move, pos->ply, move), 0, false});
    }

    void generate_tacticals() {
        MoveList moves(score_tactical_move, this);
        generate_moves(pos, only_captures, moves);
    }

    void generate_quiets() {
        if (!includes_quiets()) {
            return;
        }

        MoveList moves(score_quiet_move, this);
        generate_moves(pos, only_quiets, moves);
    }

    bool next_good_tactical(PickedMove& picked) {
        while (tacticalIndex < tacticals.size()) {
            const std::size_t best = select_best(tacticals, tacticalIndex);
            ScoredMove scored = tacticals[best];
            ++tacticalIndex;

            if (already_tried(scored.move)) {
                continue;
            }

            if (should_classify_capture_with_see(scored.move)) {
                scored.seeScore = static_exchange_eval(pos, scored.move);
                scored.seeKnown = true;
                if (scored.seeScore < 0) {
                    scored.score = bad_capture_score(pos, td, scored.move, scored.seeScore);
                    badTacticals.push_back(scored);
                    continue;
                }
            }

            mark_tried(scored.move);
            picked = {scored.move, scored.seeScore, scored.seeKnown};
            return true;
        }
        return false;
    }
};

int negamax_impl(thrawn::Position* pos, ThreadData* td, int depth, int alpha,
                 int beta, int excludedMove);

} // namespace

int negamax(thrawn::Position* pos, ThreadData* td, int depth, int alpha, int beta)
{
    return negamax_impl(pos, td, depth, alpha, beta, 0);
}

namespace {

int negamax_impl(thrawn::Position* pos, ThreadData* td, int depth, int alpha,
                 int beta, int excludedMove)
{
    int score = 0;
    int bestScore = -SEARCH_INFINITY;
    int bestMove = 0;
    int hashFlag = BOUND_UPPER;
    int static_eval = 0;
    int raw_static_eval = no_hashmap_entry;
    StaticEvalContext evalContext;
    const bool excludedNode = excludedMove != 0;

    if (stopped.load(std::memory_order_relaxed) == 1)
        return alpha;

    // Only the main worker consumes stdin; helpers just observe the shared stop flag.
    if (td->thread_id == 0 && (td->nodes & 1023) == 0)
    {
        communicate();
        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;
    }

    // Keep PV and move-ordering arrays inside their fixed search horizon.
    if (pos->ply >= MAX_DEPTH - 1)
    {
        return evaluate(pos);
    }

    // init local pv
    td->pv_length[pos->ply] = pos->ply;

    // 1) Check repetition or 50-move draw
    if ((pos->ply && isRepetition(pos)) || pos->fifty_move >= 100)
    {
        return 0;
    }

    if (pos->ply)
    {
        alpha = std::max(alpha, -mateVal + pos->ply);
        beta = std::min(beta, mateVal - pos->ply - 1);
        if (alpha >= beta)
            return alpha;
    }

    int isPvNode = ((beta - alpha) > 1);

    bool inCheck = is_square_under_attack(
        pos,
        (pos->colour_to_move == white ?
            get_lsb_index(pos->piece_bitboards[K]) :
            get_lsb_index(pos->piece_bitboards[k])),
        pos->colour_to_move ^ 1
    );

    // Check extension: search check evasions, and replies to checking moves,
    // one ply deeper than an ordinary node. Do this before TT cutoffs so the
    // stored entry is deep enough for the actual node depth being searched.
    if (inCheck)
    {
        depth += SEARCH_CHECK_EXTENSION;
    }

    if (depth <= 0)
    {
        return quiescence(pos, td, alpha, beta);
    }

    // Transposition Table lookup
    int ttHit=0;
    int ttDepth = 0;
    int ttMove = 0;
    int ttFlag = BOUND_NONE;
    int ttScore = 0;
    int ttStaticEval = no_hashmap_entry;
    if ((ttHit = tt->probe(pos, ttDepth, alpha, beta, ttMove, ttScore, ttFlag, ttStaticEval)))
    {
        if (!excludedNode && ttDepth >= depth && !isPvNode)
        {
            // Table is exact or produces a cutoff
            if ( ttFlag == BOUND_EXACT 
                || (ttFlag == BOUND_LOWER && ttScore >= beta) 
                || (ttFlag == BOUND_UPPER && ttScore <= alpha))
                return ttScore;
        }

        // if (!isPvNode && ttDepth >= depth - 1 &&
        //     (ttFlag & BOUND_UPPER) &&
        //     ttScore <= alpha &&
        //     ttScore + 141 <= alpha)
        //     return alpha;
    }

    // Increment node counter
    count_node(td);

    // Compute static evaluation
    if (!inCheck)
    {
        raw_static_eval = evaluate_static(pos, ttStaticEval);
        static_eval = corrected_static_eval(td, pos, raw_static_eval);
        td->static_eval_stack[pos->ply] = static_eval;
        evalContext = make_static_eval_context(td, pos->ply, static_eval, isPvNode, beta);
    }
    else
    {
        td->static_eval_stack[pos->ply] = no_hashmap_entry;
    }
    const bool pawnOnlyEndgame = noMajorsOrMinorsPieces(pos);

    // --------------------------------------
    // Razoring (shallow depth, not in check, non-PV)
    // --------------------------------------
    if (!excludedNode && !inCheck && !isPvNode && !pawnOnlyEndgame &&
        depth <= SEARCH_RAZOR_MAX_DEPTH && pos->ply > 0 &&
        !is_mate_score(alpha))
    {
        score = static_eval + razor_margin(depth);
        int razor_score;
        // If the position is very likely losing (or not better) vs beta, do a quick check
        if (score <= alpha)
        {
            if (depth == 1)
            {
                razor_score = quiescence(pos, td, alpha, beta);
                return (razor_score > score) ? razor_score : score;
            }
            if (depth <= 2)
            {
                razor_score = quiescence(pos,td, alpha, beta);
                if (razor_score < beta) // quiescence says score fail-low node
                    return (razor_score > score) ? razor_score : score;
            }
        }
    }

    // --------------------------------------
    // Reverse Futility Pruning (RFP)
    //    Often called "static-nullmove" or "futility"
    // --------------------------------------
    if (!excludedNode && !inCheck && !isPvNode && !pawnOnlyEndgame &&
        depth <= SEARCH_REVERSE_FUTILITY_MAX_DEPTH &&
        !is_mate_score(beta) && !is_mate_score(static_eval))
    {
        int eval_margin = reverse_futility_margin(depth, evalContext);
        // if static eval already big enough to exceed beta
        if (static_eval - eval_margin >= beta)
        {
            return static_eval - eval_margin;
        }
    }

    // --------------------------------------
    // Null-move pruning
    // --------------------------------------
    if (!excludedNode && !inCheck && depth >= SEARCH_NULL_MOVE_MIN_DEPTH && !isPvNode && static_eval >= beta &&
        !pawnOnlyEndgame && td->allowNullMovePruning &&
        !is_mate_score(beta) && !is_mate_score(static_eval))
    {
        pos->ply++;
        td->ply_moves[pos->ply - 1] = 0;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;
        make_null_move(pos, pos->ply);
        tt->prefetch(pos->zobristKey);

        // Null-move search with reduced depth
        int reduction = null_move_reduction(depth, static_eval, beta, evalContext);
        td->allowNullMovePruning = false;
        score = -negamax_impl(pos, td, depth - 1 - reduction, -beta, -beta + 1, 0);
        td->allowNullMovePruning = true;
        
        unmake_null_move(pos, pos->ply);
        pos->ply--;
        pos->repetition_index--;

        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;

        // If this "fake pass" search fails high, then cut
        if (score >= beta && !is_mate_score(score))
        {
            if (depth >= SEARCH_NULL_MOVE_VERIFICATION_DEPTH)
            {
                const bool savedNullMoveState = td->allowNullMovePruning;
                td->allowNullMovePruning = false;
                const int verificationDepth = std::max(1, depth - reduction);
                const int verification = negamax_impl(pos, td, verificationDepth, beta - 1, beta, 0);
                td->allowNullMovePruning = savedNullMoveState;

                if (stopped.load(std::memory_order_relaxed) == 1)
                    return alpha;

                if (verification >= beta)
                    return beta;
            }
            else
            {
                return beta;
            }
        }
    }

    // --------------------------------------
    // ProbCut
    // --------------------------------------
    const int probCutDepth = depth - 1 - SEARCH_PROBCUT_REDUCTION;
    if (!excludedNode && !inCheck && !isPvNode && !pawnOnlyEndgame &&
        depth >= SEARCH_PROBCUT_MIN_DEPTH && probCutDepth > 0 &&
        !is_mate_score(beta) && beta < mateScore - probcut_margin_for_context(evalContext))
    {
        const int probCutBeta = beta + probcut_margin_for_context(evalContext);
        const bool ttRefutesProbCut =
            ttHit && ttDepth >= depth - SEARCH_PROBCUT_REDUCTION &&
            (ttFlag == BOUND_EXACT || ttFlag == BOUND_UPPER) &&
            ttScore < probCutBeta;

        if (!ttRefutesProbCut)
        {
            MovePicker probCutPicker(pos, td, ttMove, false, only_captures, inCheck);
            PickedMove probCutPicked;
            while (probCutPicker.next(probCutPicked))
            {
                const int move = probCutPicked.move;
                if (!get_promoted_piece(move))
                {
                    const int seeScore = probCutPicked.seeKnown
                        ? probCutPicked.seeScore
                        : static_exchange_eval(pos, move);
                    if (seeScore < SEARCH_PROBCUT_SEE_MARGIN)
                        continue;
                }

                const int parentPly = pos->ply;
                pos->ply++;
                pos->repetition_index++;
                pos->repetition_table[pos->repetition_index] = pos->zobristKey;

                if (!make_move_on_board(pos, move, all_moves, pos->ply))
                {
                    pos->ply--;
                    pos->repetition_index--;
                    continue;
                }
                tt->prefetch(pos->zobristKey);
                td->ply_moves[parentPly] = move;

                const bool savedFollowPv = td->follow_pv_flag;
                td->follow_pv_flag = false;

                score = -quiescence(pos, td, -probCutBeta, -probCutBeta + 1);
                if (score >= probCutBeta)
                {
                    score = -negamax_impl(pos, td, probCutDepth,
                                          -probCutBeta, -probCutBeta + 1, 0);
                }

                td->follow_pv_flag = savedFollowPv;

                unmake_move(pos, pos->ply);
                pos->ply--;
                pos->repetition_index--;

                if (stopped.load(std::memory_order_relaxed) == 1)
                    return alpha;

                if (score >= probCutBeta)
                {
                    update_correction_history(td, pos, raw_static_eval, beta, depth,
                                              BOUND_LOWER);
                    // The fail-high was only proven by a search at probCutDepth
                    // (plus the qsearch screen), so record the bound at that
                    // verified depth, not the full node depth. Storing it at the
                    // unreduced `depth` would let a later node take a TT cutoff as
                    // though a full-depth search had confirmed beta.
                    tt->store(pos, probCutDepth + 1, beta, BOUND_LOWER, move, raw_static_eval);
                    return beta;
                }
            }
        }
    }

    // --------------------------------------
    // Internal Iterative Reductions (IIR)
    // No usable TT move means move ordering at this node is poor, so a full-depth
    // search would waste effort. Reduce one ply; the shallower search populates
    // the TT with a best move that guides a later (re-)search of this node.
    // Mutually exclusive with the singular extension below (which requires a TT
    // move), and placed after all eval-based pruning so those use the true depth.
    // --------------------------------------
    if (!excludedNode && ttMove == 0 && depth >= SEARCH_IIR_MIN_DEPTH)
    {
        depth -= SEARCH_IIR_REDUCTION;
    }

    const bool nodeFollowPv = td->follow_pv_flag;
    MovePicker movePicker(pos, td, ttMove, nodeFollowPv, all_moves, inCheck);

    // We are about to search each move
    int valid_moves = 0;
    int moves_searched = 0;
    int quiet_moves_seen = 0;
    FixedBuffer<int, MAX_GENERATED_MOVES> failed_quiet_moves;
    FixedBuffer<int, MAX_GENERATED_MOVES> failed_capture_moves;

    // Search each move (LMR, LMP, PVS logic, etc.)
    PickedMove picked;
    while (movePicker.next(picked))
    {
        const int move = picked.move;
        if (move == excludedMove)
            continue;

        const int parentPly = pos->ply;
        const int parentSide = pos->colour_to_move;
        const bool quietMove = is_quiet_move(move);
        const bool captureMove = get_is_capture_move(move);
        const bool pawnMove = get_move_piece(move) == P || get_move_piece(move) == p;
        const bool firstMove = moves_searched == 0;
        bool givesCheck = false;
        bool givesCheckKnown = false;

        auto gives_check = [&]() {
            if (!givesCheckKnown) {
                givesCheck = !inCheck && move_gives_check(pos, move);
                givesCheckKnown = true;
            }
            return givesCheck;
        };

        if (quietMove)
        {
            quiet_moves_seen++;
        }

        if (!excludedNode && !firstMove && pos->ply && !isPvNode && !inCheck && !pawnOnlyEndgame)
        {
            const bool castleMove = get_is_move_castling(move);

            if (quietMove && !pawnMove && !castleMove &&
                depth <= SEARCH_FUTILITY_MAX_DEPTH &&
                static_eval + futility_margin_for_context(depth, evalContext) <= alpha &&
                !gives_check())
            {
                continue;
            }

            if (quietMove && !pawnMove && !castleMove &&
                depth <= SEARCH_LATE_MOVE_PRUNING_MAX_DEPTH &&
                quiet_moves_seen > futility_move_count_for_context(depth, evalContext) &&
                !gives_check())
            {
                continue;
            }

            // History pruning: a quiet move with clearly bad history at shallow
            // depth is very unlikely to raise alpha; skip it. Threshold scales
            // linearly with depth, matching the futility/LMP gating above.
            if (quietMove && !pawnMove && !castleMove &&
                depth <= SEARCH_HISTORY_PRUNING_MAX_DEPTH &&
                quiet_history_score(td, parentSide, parentPly, move) <
                    -SEARCH_HISTORY_PRUNING_DEPTH_MARGIN * depth &&
                !gives_check())
            {
                continue;
            }

            if (get_is_capture_move(move) && !get_promoted_piece(move) &&
                depth <= SEARCH_SEE_PRUNE_MAX_DEPTH)
            {
                const int seeScore = picked.seeKnown
                    ? picked.seeScore
                    : static_exchange_eval(pos, move);
                if (seeScore < -SEARCH_SEE_PRUNE_DEPTH_MARGIN * depth && !gives_check())
                {
                    continue;
                }
            }
        }

        const bool needsLmrCheckInfo =
            !firstMove && quietMove && !get_is_move_castling(move) &&
            depth >= SEARCH_LMR_REDUCTION_DEPTH_LIMIT && !inCheck;
        if (needsLmrCheckInfo)
            gives_check();

        int extension = 0;
        if (!excludedNode && pos->ply > 0 && move == ttMove &&
            SEARCH_SINGULAR_EXTENSION > 0 &&
            depth >= SEARCH_SINGULAR_EXTENSION_MIN_DEPTH &&
            ttHit && ttDepth >= depth - SEARCH_SINGULAR_EXTENSION_DEPTH_MARGIN &&
            (ttFlag == BOUND_LOWER || ttFlag == BOUND_EXACT) &&
            ttScore > alpha && !is_mate_score(ttScore) && !inCheck)
        {
            const int singularMargin =
                SEARCH_SINGULAR_EXTENSION_BASE_MARGIN +
                SEARCH_SINGULAR_EXTENSION_DEPTH_FACTOR * depth;
            const int singularBeta =
                std::max(-SEARCH_INFINITY + 1, std::min(SEARCH_INFINITY - 1,
                                                 ttScore - singularMargin));
            const int singularDepth = std::max(1, (depth - 1) / 2);

            const bool savedFollowPv = td->follow_pv_flag;
            const int savedPvLength = td->pv_length[pos->ply];
            // The singular re-search runs at this same ply and clobbers
            // td->pv_table[ply]/pv_length[ply]. Only the valid prefix
            // [ply, savedPvLength) is ever read afterwards (reads are bounded by
            // pv_length, and a later alpha-raise rewrites the row from `ply`),
            // so save/restore just that slice instead of memcpy-ing all 64 ints.
            std::array<int, MAX_DEPTH> savedPvRow;
            for (int i = pos->ply; i < savedPvLength; ++i)
                savedPvRow[i] = td->pv_table[pos->ply][i];
            td->follow_pv_flag = false;

            const int singularScore = negamax_impl(pos, td, singularDepth,
                                                   singularBeta - 1,
                                                   singularBeta, move);

            td->follow_pv_flag = savedFollowPv;
            td->pv_length[pos->ply] = savedPvLength;
            for (int i = pos->ply; i < savedPvLength; ++i)
                td->pv_table[pos->ply][i] = savedPvRow[i];

            if (stopped.load(std::memory_order_relaxed) == 1)
                return alpha;

            if (singularScore < singularBeta)
                extension = SEARCH_SINGULAR_EXTENSION;
            // Negative extension: the TT move is not singular and the table
            // already proves a fail-high here, so other moves are likely just as
            // good. Spend one ply less on this subtree.
            else if (ttScore >= beta)
                extension = -SEARCH_SINGULAR_EXTENSION;
        }

        pos->ply++;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;

        if (!make_move_on_board(pos, move, all_moves, pos->ply))
        {
            pos->ply--;
            pos->repetition_index--;
            continue;
        }
        tt->prefetch(pos->zobristKey);
        valid_moves++;
        td->ply_moves[parentPly] = move;
        const int childDepth = depth - 1 + extension;

        auto search_child = [&](int searchDepth, int childAlpha, int childBeta) {
            const bool savedFollowPv = td->follow_pv_flag;
            td->follow_pv_flag = nodeFollowPv && td->pv_table[0][parentPly] == move;
            const int childScore = -negamax_impl(pos, td, searchDepth, childAlpha,
                                                 childBeta, 0);
            td->follow_pv_flag = savedFollowPv;
            return childScore;
        };

        // -------------------------------------------
        // Principal Variation Search logic
        // -------------------------------------------
        if (moves_searched == 0)
        {
            // First move: full-window search
            score = search_child(childDepth, -beta, -alpha);
        }
        else
        {
            // -----------------------------
            // Late Move Reductions (LMR)
            // -----------------------------
            const int quietHistory = quietMove
                ? quiet_history_score(td, parentSide, parentPly, move)
                : 0;
            const bool counterMove = quietMove && is_counter_move(td, parentPly, move);

            if (quiet_moves_seen >= SEARCH_LMR_FULL_DEPTH_MOVES &&
                depth >= SEARCH_LMR_REDUCTION_DEPTH_LIMIT &&
                !inCheck &&
                !givesCheck &&
                quietMove &&
                !get_is_move_castling(move))
            {
                // Reduced search
                int reduction = late_move_reduction(depth, moves_searched + 1,
                                                    isPvNode, counterMove, quietHistory,
                                                    evalContext);
                int reducedDepth = std::max(1, childDepth - reduction);
                score = search_child(reducedDepth, -alpha - 1, -alpha);
            }
            else if (captureMove && picked.seeKnown && picked.seeScore < 0 &&
                     !get_promoted_piece(move) &&
                     depth >= SEARCH_LMR_REDUCTION_DEPTH_LIMIT &&
                     !inCheck && !gives_check())
            {
                // Late losing captures (negative SEE) are very unlikely to be
                // best; reduce them like late quiets. A PVS re-search below
                // restores full depth if the reduced search still beats alpha.
                int reduction = late_move_reduction(depth, moves_searched + 1,
                                                    isPvNode, false, 0, evalContext);
                int reducedDepth = std::max(1, childDepth - reduction);
                score = search_child(reducedDepth, -alpha - 1, -alpha);
            }
            else
            {
                // Force a normal re-search with a null window to see if it fails high
                score = alpha + 1;
            }

            // If we got a score above alpha after LMR attempt...
            if (score > alpha)
            {
                // Do a PVS re-search with a narrow window
                score = search_child(childDepth, -alpha - 1, -alpha);

                // If it's still above alpha but not >= beta, do a full re-search
                if (score > alpha && score < beta)
                {
                    score = search_child(childDepth, -beta, -alpha);
                }
            }
        }

        unmake_move(pos, pos->ply);
        pos->ply--;
        pos->repetition_index--;
        moves_searched++;

        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;

        if (score > bestScore)
        {
            bestScore = score;
            bestMove = move;
        }

        // Check if this move improved alpha
        const int oldAlpha = alpha;
        if (pos->ply == 0) {
            const int rootBound = score > oldAlpha ? (score >= beta ? BOUND_LOWER : BOUND_EXACT)
                                                   : BOUND_UPPER;
            td->recordRootMove(move, score, depth, rootBound);
        }

        if (score > alpha)
        {
            hashFlag = BOUND_EXACT; // PV node

            if (!excludedNode && quietMove)
            {
                update_quiet_history(td, parentSide, pos->ply, move, depth);
            }
            else if (!excludedNode && captureMove)
            {
                update_capture_history(td, pos, move, depth);
            }

            alpha = score;

            // Update PV
            td->pv_table[pos->ply][pos->ply] = move;
            for (int nextPly = pos->ply + 1; nextPly < td->pv_length[pos->ply + 1]; nextPly++)
            {
                td->pv_table[pos->ply][nextPly] = td->pv_table[pos->ply + 1][nextPly];
            }
            td->pv_length[pos->ply] = std::max(td->pv_length[pos->ply + 1], pos->ply + 1);

            // Fail-hard beta cutoff
            if (alpha >= beta)
            {
                if (!excludedNode)
                {
                    update_correction_history(td, pos, raw_static_eval, beta, depth,
                                              BOUND_LOWER);
                    tt->store(pos, depth, beta, BOUND_LOWER, bestMove,
                              inCheck ? no_hashmap_entry : raw_static_eval);
                }

                if (!excludedNode && quietMove)
                {
                    td->killer_moves[1][pos->ply] = td->killer_moves[0][pos->ply];
                    td->killer_moves[0][pos->ply] = move;
                    update_counter_move(td, pos->ply, move);
                }
                if (!excludedNode)
                {
                    penalize_quiet_history(td, parentSide, pos->ply,
                                           failed_quiet_moves, depth);
                    penalize_capture_history(td, pos, failed_capture_moves, depth);
                }
                return beta;
            }
        }

        if (quietMove && score <= oldAlpha)
        {
            failed_quiet_moves.push_back(move);
        }
        else if (captureMove && score <= oldAlpha)
        {
            failed_capture_moves.push_back(move);
        }
    } // end of move loop

    // if no valid moves
    if (valid_moves == 0)
    {
        if (excludedNode)
            return alpha;

        // Note: pruning (futility/LMP/SEE) is gated on !firstMove, so it never
        // fires before a legal move has been searched. Reaching valid_moves == 0
        // therefore means a genuine stalemate/checkmate, handled below.
        if (inCheck)
            return -mateVal + pos->ply;
        else
            return 0;
    }

    // Store in TT and return
    if (!excludedNode)
    {
        update_correction_history(td, pos, raw_static_eval, alpha, depth, hashFlag);
        tt->store(pos, depth, alpha, hashFlag, bestMove,
                  inCheck ? no_hashmap_entry : raw_static_eval);
    }
    return alpha;
}

} // namespace

int quiescence(thrawn::Position* pos, ThreadData* td,
               int alpha, int beta)
{
    if (stopped.load(std::memory_order_relaxed) == 1)
        return alpha;

    if (td->thread_id == 0 && (td->nodes & 1023) == 0)
    {
        communicate();
        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;
    }

    if ((pos->ply && isRepetition(pos)) || pos->fifty_move >= 100)
        return 0;

    if (pos->ply)
    {
        alpha = std::max(alpha, -mateVal + pos->ply);
        beta = std::min(beta, mateVal - pos->ply - 1);
        if (alpha >= beta)
            return alpha;
    }

    count_node(td);

    const bool inCheck = is_square_under_attack(
        pos,
        (pos->colour_to_move == white)
            ? get_lsb_index(pos->piece_bitboards[K])
            : get_lsb_index(pos->piece_bitboards[k]),
        pos->colour_to_move ^ 1
    );

    // safety check for array bounds
    if (pos->ply >= MAX_DEPTH - 1)
    {
        return inCheck ? -mateVal + pos->ply : evaluate(pos);
    }

    const int oldAlpha = alpha;
    const bool isPvNode = ((beta - alpha) > 1);

    int ttDepth = 0;
    int ttMove = 0;
    int ttFlag = BOUND_NONE;
    int ttScore = 0;
    int ttStaticEval = no_hashmap_entry;
    if (tt->probe(pos, ttDepth, alpha, beta, ttMove, ttScore, ttFlag, ttStaticEval))
    {
        if (!isPvNode || ttFlag == BOUND_EXACT)
        {
            if (ttFlag == BOUND_EXACT ||
                (ttFlag == BOUND_LOWER && ttScore >= beta) ||
                (ttFlag == BOUND_UPPER && ttScore <= alpha))
            {
                return ttScore;
            }
        }
    }

    int static_eval = 0;
    int raw_static_eval = no_hashmap_entry;
    if (!inCheck)
    {
        raw_static_eval = evaluate_static(pos, ttStaticEval);
        static_eval = corrected_static_eval(td, pos, raw_static_eval);
        td->static_eval_stack[pos->ply] = static_eval;

        // fail-hard beta cutoff
        if (static_eval >= beta)
        {
            tt->store(pos, 0, beta, BOUND_LOWER, 0, raw_static_eval);
            return beta; // fails high
        }

        // found better move
        if (static_eval > alpha)
            alpha = static_eval; // principal variation PV node (best move)
    }
    else
    {
        td->static_eval_stack[pos->ply] = no_hashmap_entry;
    }

    const int move_type = inCheck ? all_moves : only_captures;
    MovePicker movePicker(pos, td, ttMove, false, move_type, inCheck);

    int valid_moves = 0;
    int bestMove = 0;
    int bestScore = inCheck ? -SEARCH_INFINITY : alpha;

    // Both are invariant across this node's move loop (each move is made then
    // unmade, leaving pos unchanged), so compute them once instead of per move.
    const bool pawnOnlyEndgame = noMajorsOrMinorsPieces(pos);
    const uint64_t enemyKing =
        pos->piece_bitboards[pos->colour_to_move == white ? k : K];

    PickedMove picked;
    while (movePicker.next(picked))
    {
        const int move = picked.move;
        const bool promotionMove = get_promoted_piece(move) != 0;
        const bool deltaPruned = !inCheck && !promotionMove &&
                                 qsearch_delta_prune(pos, move, static_eval, alpha,
                                                     pawnOnlyEndgame);
        bool seePruned = false;
        if (!inCheck && !promotionMove &&
            get_is_capture_move(move) &&
            get_move_piece(move) != K && get_move_piece(move) != k)
        {
            const int target = get_move_target(move);
            if (!(pos->king_attacks[target] & enemyKing))
            {
                const int seeScore = picked.seeKnown
                    ? picked.seeScore
                    : static_exchange_eval(pos, move);
                seePruned = seeScore < 0;
            }
        }

        if ((deltaPruned || seePruned) && !move_gives_check(pos, move))
        {
            continue;
        }

        const int parentPly = pos->ply;
        pos->ply++;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;

        if (!make_move_on_board(pos, move, move_type, pos->ply))
        {
            pos->ply--;
            pos->repetition_index--;
            continue;
        }
        tt->prefetch(pos->zobristKey);
        valid_moves++;
        td->ply_moves[parentPly] = move;

        int score = -quiescence(pos, td, -beta, -alpha);

        unmake_move(pos, pos->ply);
        pos->ply--;
        pos->repetition_index--;

        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;

        if (score > bestScore)
        {
            bestScore = score;
            bestMove = move;
        }

        // found better move
        if (score > alpha)
        {
            alpha = score; // principal variation PV node (best move)

            // fail-hard beta cutoff
            if (score >= beta)
            {
                tt->store(pos, 0, beta, BOUND_LOWER, bestMove,
                          inCheck ? no_hashmap_entry : raw_static_eval);
                return beta; // fails high
            }
        }
    }

    if (inCheck && valid_moves == 0)
    {
        const int mate = -mateVal + pos->ply;
        tt->store(pos, 0, mate, BOUND_EXACT, 0);
        return mate;
    }

    // move fails low (<= alpha)
    tt->store(pos, 0, alpha, alpha > oldAlpha ? BOUND_EXACT : BOUND_UPPER,
              bestMove, inCheck ? no_hashmap_entry : raw_static_eval);
    return alpha;
}

// repetition check
int isRepetition(thrawn::Position* pos)
{
    if (pos->fifty_move < 4)
        return 0;

    const int oldest_reversible = std::max(0, pos->repetition_index - pos->fifty_move);
    for (int i = pos->repetition_index - 1; i >= oldest_reversible; i -= 2)
    {
        if (pos->repetition_table[i] == pos->zobristKey)
            return 1;
    }
    return 0;
}

int futility_margin(int depth)
{
    if (depth <= 0) {
        return 0;
    }
    if (depth == 1) {
        return SEARCH_FUTILITY_MARGIN_1;
    }
    if (depth == 2) {
        return SEARCH_FUTILITY_MARGIN_2;
    }
    return SEARCH_FUTILITY_MARGIN_3;
}

int futility_move_count(int depth)
{
    if (depth <= 0) {
        return 0;
    }
    if (depth == 1) {
        return SEARCH_LATE_MOVE_PRUNING_DEPTH_1;
    }
    if (depth == 2) {
        return SEARCH_LATE_MOVE_PRUNING_DEPTH_2;
    }
    return SEARCH_LATE_MOVE_PRUNING_DEPTH_3;
}
