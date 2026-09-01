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
#include <cmath>

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

using LmrTable = std::array<std::array<int8_t, SEARCH_LMR_TABLE_MOVES>, MAX_DEPTH + 1>;

// floor(base + log(depth) * log(move) / divisor), the reduction curve every
// modern engine uses.
LmrTable build_lmr_table() {
    LmrTable table{};
    for (int depth = 1; depth <= MAX_DEPTH; ++depth) {
        for (int move = 1; move < SEARCH_LMR_TABLE_MOVES; ++move) {
            const double reduction = SEARCH_LMR_TABLE_BASE +
                                     std::log(static_cast<double>(depth)) *
                                     std::log(static_cast<double>(move)) /
                                     SEARCH_LMR_TABLE_DIVISOR;
            table[depth][move] = static_cast<int8_t>(reduction);
        }
    }
    return table;
}

// Const and filled before main(), so every search thread reads it race-free.
const LmrTable lmr_table = build_lmr_table();

int lmr_table_reduction(int depth, int move_number) {
    const int d = std::clamp(depth, 0, MAX_DEPTH);
    const int m = std::clamp(move_number, 0, SEARCH_LMR_TABLE_MOVES - 1);
    return lmr_table[d][m];
}

// Node type as the parent decided it, not as the static eval guesses it.
struct NodeContext {
    bool improving = false;
    bool opponentWorsening = false;
    bool cutNode = false;
};

// A rising trend for us, or a falling one for the opponent, is worth one depth
// step off the margin so the cutoff fires sooner.
int reverse_futility_margin(int depth, const NodeContext& context) {
    const int trend = context.improving + context.opponentWorsening;
    return std::max(SEARCH_REVERSE_FUTILITY_DEPTH_MUL * (depth - trend),
                    SEARCH_REVERSE_FUTILITY_MIN);
}

int razor_margin(int depth) {
    return depth <= 0 ? 0 : SEARCH_RAZOR_DEPTH_FACTOR * depth;
}

int null_move_reduction(int depth, int static_eval, int beta,
                        const NodeContext& context) {
    const int eval_margin = std::max(0, static_eval - beta);
    int reduction = SEARCH_NULL_MOVE_BASE_REDUCTION +
                    depth / SEARCH_NULL_MOVE_DEPTH_DIVISOR;
    reduction += std::min(SEARCH_NULL_MOVE_EVAL_BONUS_MAX,
                          eval_margin / SEARCH_NULL_MOVE_EVAL_DIVISOR);
    if (context.opponentWorsening)
        ++reduction;
    if (!context.improving)
        --reduction;
    return std::clamp(reduction, 1, std::max(1, depth - 1));
}

int futility_margin(int lmr_depth) {
    if (lmr_depth < 0) {
        lmr_depth = 0;
    }
    return SEARCH_FUTILITY_BASE_MARGIN + SEARCH_FUTILITY_DEPTH_FACTOR * lmr_depth;
}

int late_move_pruning_count(int depth, bool improving) {
    if (depth <= 0) {
        return 0;
    }
    return std::max(1, (SEARCH_LATE_MOVE_PRUNING_BASE + depth * depth) / (improving ? 1 : 2));
}

int probcut_margin(const NodeContext& context) {
    return SEARCH_PROBCUT_MARGIN - SEARCH_PROBCUT_IMPROVING_MARGIN * context.improving;
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

int evaluate_static(thrawn::Position* pos, int cachedStaticEval, bool ttHit) {
    if (cachedStaticEval != no_hashmap_entry) {
        return cachedStaticEval;
    }

    const int value = evaluate(pos);
    // Without a TT entry for this position there is nothing to attach the eval
    // to, so skip the cluster walk; the node's own store carries it instead.
    if (ttHit)
        tt->storeStaticEval(pos, value);
    return value;
}

// Keyed on pawn structure, not the full position key: a correction learned in
// one position then applies to every position sharing that structure, which is
// the whole point of the table.
int correction_history_index(const thrawn::Position* pos) {
    static_assert((SEARCH_CORRECTION_HISTORY_SIZE & (SEARCH_CORRECTION_HISTORY_SIZE - 1)) == 0,
                  "correction history size must be a power of two");
    uint64_t key = pos->piece_bitboards[P] * 0x9E3779B97F4A7C15ULL ^
                   pos->piece_bitboards[p] * 0xC2B2AE3D27D4EB4FULL;
    key ^= key >> 29;
    return static_cast<int>(key) & (SEARCH_CORRECTION_HISTORY_SIZE - 1);
}

int correction_value(ThreadData* td, thrawn::Position* pos) {
    const int entry = td->correction_history[pos->colour_to_move]
                                            [correction_history_index(pos)];
    return SEARCH_CORRECTION_HISTORY_WEIGHT * entry / 512;
}

int corrected_static_eval(int rawStaticEval, int correction) {
    return std::clamp(rawStaticEval + correction, -mateScore + MAX_DEPTH,
                      mateScore - MAX_DEPTH);
}

NodeContext make_node_context(ThreadData* td, int ply, int staticEval, bool cutNode) {
    NodeContext context;
    context.cutNode = cutNode;
    if (staticEval == no_hashmap_entry) {
        return context;
    }

    if (ply >= 2 && td->static_eval_stack[ply - 2] != no_hashmap_entry) {
        context.improving = staticEval > td->static_eval_stack[ply - 2];
    } else if (ply >= 4 && td->static_eval_stack[ply - 4] != no_hashmap_entry) {
        context.improving = staticEval > td->static_eval_stack[ply - 4];
    }

    if (ply >= 1 && td->static_eval_stack[ply - 1] != no_hashmap_entry) {
        context.opponentWorsening = staticEval > -td->static_eval_stack[ply - 1];
    }

    return context;
}

void update_correction_history(ThreadData* td, thrawn::Position* pos, int staticEval,
                               int score, int depth, int bound) {
    if (staticEval == no_hashmap_entry || is_mate_score(score)) {
        return;
    }

    const bool usefulBound =
        bound == BOUND_EXACT ||
        (bound == BOUND_LOWER && score > staticEval) ||
        (bound == BOUND_UPPER && score < staticEval);
    if (!usefulBound) {
        return;
    }

    int& entry = td->correction_history[pos->colour_to_move]
                                       [correction_history_index(pos)];
    const int bonus = std::clamp((score - staticEval) * depth /
                                     SEARCH_CORRECTION_HISTORY_DEPTH_DIVISOR,
                                 -SEARCH_CORRECTION_HISTORY_MAX / 4,
                                 SEARCH_CORRECTION_HISTORY_MAX / 4);
    entry += bonus - entry * std::abs(bonus) / SEARCH_CORRECTION_HISTORY_MAX;
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

    // The victim of a non-en-passant capture is the occupant of the target
    // square, which the mailbox holds in O(1).
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

    // Uninitialized: the swap-off loop writes gains[0] then each gains[d] before
    // it is ever read, and only [0, depth] is touched. Zero-init would memset
    // 128 bytes on every SEE call, which runs per move during pruning.
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
        // Quiet move: nothing is taken off the target square, the mover lands
        // on it and the swap-off loop measures who can take it back.
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

bool qsearch_delta_prune(thrawn::Position* pos, int move, int static_eval, int alpha) {
    if (is_mate_score(alpha)) {
        return false;
    }

    return static_eval + qsearch_move_gain_upper_bound(pos, move) +
           SEARCH_QSEARCH_DELTA_MARGIN <= alpha;
}

int history_bonus(int depth) {
    return std::min(SEARCH_HISTORY_BONUS_DEPTH * depth + SEARCH_HISTORY_BONUS_BIAS,
                    SEARCH_HISTORY_BONUS_MAX);
}

int history_malus(int depth) {
    return std::min(SEARCH_HISTORY_MALUS_DEPTH * depth + SEARCH_HISTORY_MALUS_BIAS,
                    SEARCH_HISTORY_MALUS_MAX);
}

int previous_ply_move(ThreadData* td, int ply, int pliesBack = 1);

template <typename Entry>
void update_history_entry(Entry& entry, int bonus) {
    const int history_max = std::max(1, SEARCH_HISTORY_MAX);
    bonus = std::clamp(bonus, -history_max, history_max);
    const int gravity = bonus >= 0 ? bonus : -bonus;
    int value = entry;
    value += bonus - value * gravity / history_max;
    entry = static_cast<Entry>(std::clamp(value, -history_max, history_max));
}

int continuation_history_score(ThreadData* td, int ply, int move) {
    int score = 0;
    for (int back = 1; back <= SEARCH_CONTINUATION_HISTORY_PLIES; ++back) {
        const int previousMove = previous_ply_move(td, ply, back);
        if (previousMove == 0) {
            continue;
        }
        score += td->continuation_history[back - 1]
                                         [get_move_piece(previousMove)]
                                         [get_move_target(previousMove)]
                                         [get_move_piece(move)]
                                         [get_move_target(move)];
    }
    return score;
}

int quiet_history_score(ThreadData* td, int side, int ply, int move) {
    const int fromTo = td->quiet_history[side][get_move_source(move)]
                                        [get_move_target(move)];
    return 2 * fromTo + continuation_history_score(td, ply, move);
}

void update_continuation_history(ThreadData* td, int ply, int move, int bonus) {
    // The weight depends only on how far back the ply is, so a null move at one
    // ply does not change what the ply behind it receives.
    int weight = bonus;
    for (int back = 1; back <= SEARCH_CONTINUATION_HISTORY_PLIES;
         ++back, weight = weight * SEARCH_CONTINUATION_HISTORY_DECAY_NUMERATOR /
                          SEARCH_CONTINUATION_HISTORY_DECAY_DENOMINATOR) {
        const int previousMove = previous_ply_move(td, ply, back);
        if (previousMove == 0) {
            continue;
        }
        update_history_entry(td->continuation_history[back - 1]
                                                     [get_move_piece(previousMove)]
                                                     [get_move_target(previousMove)]
                                                     [get_move_piece(move)]
                                                     [get_move_target(move)],
                             weight);
    }
}

void update_quiet_history(ThreadData* td, int side, int ply, int move, int bonus) {
    update_history_entry(td->quiet_history[side][get_move_source(move)]
                                           [get_move_target(move)], bonus);
    update_continuation_history(td, ply, move,
                                bonus * SEARCH_CONTINUATION_HISTORY_NUMERATOR /
                                    SEARCH_CONTINUATION_HISTORY_DENOMINATOR);
}

template <typename MoveContainer>
void penalize_quiet_history(ThreadData* td, int side, int ply,
                            const MoveContainer& moves, int depth) {
    const int penalty = -history_malus(depth);
    for (int move : moves) {
        update_quiet_history(td, side, ply, move, penalty);
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

void update_capture_history(ThreadData* td, thrawn::Position* pos, int move, int bonus) {
    if (!get_is_capture_move(move)) {
        return;
    }

    const int victim = capture_history_victim(pos, move);
    update_history_entry(td->capture_history[get_move_piece(move)]
                                            [get_move_target(move)]
                                            [victim],
                         bonus);
}

template <typename MoveContainer>
void penalize_capture_history(ThreadData* td, thrawn::Position* pos,
                              const MoveContainer& moves, int depth) {
    const int penalty = -history_malus(depth);
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

int previous_ply_move(ThreadData* td, int ply, int pliesBack) {
    const int index = ply - pliesBack;
    if (index < 0 || index > MAX_DEPTH - 1) {
        return 0;
    }
    return td->ply_moves[index];
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

// Base reduction from the log table, then the usual node-type and history
// adjustments. History is scaled continuously rather than in fixed steps so a
// move with a strong record can buy most of its depth back.
int late_move_reduction(int depth, int move_number, bool is_pv_node, bool tt_pv,
                        bool is_counter, bool gives_check, bool is_capture,
                        int history, const NodeContext& context) {
    int reduction = lmr_table_reduction(depth, move_number);

    // A former PV node pays a smaller cut-node penalty: the line has been worth
    // searching before, so it should not be thrown away as cheaply.
    if (context.cutNode)
        reduction += SEARCH_LMR_CUT_NODE_REDUCTION - tt_pv;
    if (is_pv_node)
        --reduction;
    if (!context.improving)
        ++reduction;
    if (gives_check)
        --reduction;
    if (is_counter)
        --reduction;

    const int divisor = is_capture ? SEARCH_LMR_CAPTURE_HISTORY_DIVISOR
                                   : SEARCH_LMR_QUIET_HISTORY_DIVISOR;
    reduction -= std::clamp(history / divisor, -SEARCH_LMR_HISTORY_CAP,
                            SEARCH_LMR_HISTORY_CAP);

    // Floor of 0 so a well-scoring move can keep its full depth; the caller
    // skips the null-window re-search when the reduction comes back 0.
    return std::clamp(reduction, 0, std::max(0, depth - 1));
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

// Killers and the counter move have their own picker stages ahead of the quiet
// buffer, so anything scored here is an ordinary quiet: history alone orders it,
// unclamped, or every well established quiet would tie at the clamp.
int quiet_move_score(ThreadData* td, int side, int ply, int move) {
    return quiet_history_score(td, side, ply, move);
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
    // KB per buffer on every node, and they live on the negamax hot path.
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
    // sliderSquare is occupied, so the mailbox holds its occupant. Only an
    // enemy slider can pin, so anything else clears the move.
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

    // Once the node has decided quiets are hopeless, skip straight past the
    // killer/counter/quiet stages instead of picking and discarding each one.
    void skip_quiets() {
        if (stage == Stage::Killer1 || stage == Stage::Killer2 ||
            stage == Stage::CounterMove || stage == Stage::GenerateQuiets ||
            stage == Stage::Quiets) {
            stage = Stage::BadTacticals;
        }
        quietsSkipped = true;
    }

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
    bool quietsSkipped = false;
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
        if (quietsSkipped || !includes_quiets() || !is_quiet_move(move)) {
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
        // No already_tried()/mark_tried() here: quietMoves and badTacticals are
        // both built after every special move (tt/pv/killers/counter) has been
        // marked tried, and add_quiet()/add_tactical()/next_good_tactical() all
        // exclude tried moves at insertion time. So nothing in these buffers is
        // ever a tried move, and no later stage reads the tried set for them.
        if (index < moves.size()) {
            const std::size_t best = select_best(moves, index);
            const ScoredMove scored = moves[best];
            ++index;
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
        if (quietsSkipped || !includes_quiets()) {
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
                 int beta, bool cutNode, int excludedMove);

} // namespace

int negamax(thrawn::Position* pos, ThreadData* td, int depth, int alpha, int beta)
{
    return negamax_impl(pos, td, depth, alpha, beta, false, 0);
}

namespace {

int negamax_impl(thrawn::Position* pos, ThreadData* td, int depth, int alpha,
                 int beta, bool cutNode, int excludedMove)
{
    int score = 0;
    int bestScore = -SEARCH_INFINITY;
    int bestMove = 0;
    int hashFlag = BOUND_UPPER;
    // raw: what goes in the TT. static_eval: correction-adjusted, what trains
    // correction history. eval: static_eval refined by the TT bound, what the
    // pruning decisions read.
    int raw_static_eval = no_hashmap_entry;
    int static_eval = 0;
    int eval = 0;
    NodeContext nodeContext;
    const bool excludedNode = excludedMove != 0;

    if (stopped.load(std::memory_order_relaxed) == 1)
        return alpha;

    // Only the main worker consumes stdin; helpers just observe the shared stop flag.
    // Counted off a dedicated tick rather than td->nodes: TT cutoffs return
    // before count_node(), which would otherwise pin nodes on a multiple of the
    // batch and fire an ioctl at every node in a cutoff-heavy region.
    if (td->thread_id == 0 && --td->check_counter <= 0)
    {
        td->check_counter = NODE_COUNTER_BATCH;
        communicate();
        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;
    }

    // Reset before any early return. A parent copies this node's PV row using
    // pv_length[ply], so a node that returns without searching has to leave an
    // empty row behind rather than whatever an unrelated line left there.
    td->pv_length[pos->ply] = pos->ply;

    // Keep PV and move-ordering arrays inside their fixed search horizon.
    if (pos->ply >= MAX_DEPTH - 1)
        return evaluate(pos);

    // Killers are indexed by ply, so without this a subtree inherits whatever
    // an unrelated sibling line left at ply+2 and orders on a refutation that
    // has nothing to do with the current position.
    if (pos->ply + 2 < MAX_DEPTH)
    {
        td->killer_moves[0][pos->ply + 2] = 0;
        td->killer_moves[1][pos->ply + 2] = 0;
    }

    if ((pos->ply && isRepetition(pos)) || pos->fifty_move >= 100)
        return 0;

    // Mate distance pruning
    if (pos->ply)
    {
        alpha = std::max(alpha, -mateVal + pos->ply);
        beta = std::min(beta, mateVal - pos->ply - 1);
        if (alpha >= beta)
            return alpha;
    }

    const bool isPvNode = (beta - alpha) > 1;

    const bool inCheck = is_square_under_attack(
        pos,
        (pos->colour_to_move == white ?
            get_lsb_index(pos->piece_bitboards[K]) :
            get_lsb_index(pos->piece_bitboards[k])),
        pos->colour_to_move ^ 1
    );

    // Check extension, applied before the TT probe so the stored entry matches
    // the depth actually searched.
    if (inCheck)
        depth += SEARCH_CHECK_EXTENSION;

    if (depth <= 0)
        return quiescence(pos, td, alpha, beta);

    // Transposition Table lookup
    int ttDepth = 0;
    int ttMove = 0;
    int ttFlag = BOUND_NONE;
    int ttScore = 0;
    int ttStaticEval = no_hashmap_entry;
    bool ttWasPv = false;
    const int ttHit = tt->probe(pos, ttDepth, ttMove, ttScore, ttFlag, ttStaticEval, ttWasPv);
    // Sticky PV marking: a node that has ever been on a PV keeps a wider
    // margin and a smaller reduction for the rest of the search.
    const bool ttPv = isPvNode || (ttHit && ttWasPv);
    if (ttHit && !excludedNode && ttDepth >= depth && !isPvNode)
    {
        if (ttFlag == BOUND_EXACT
            || (ttFlag == BOUND_LOWER && ttScore >= beta)
            || (ttFlag == BOUND_UPPER && ttScore <= alpha))
            return ttScore;
    }

    count_node(td);

    if (!inCheck)
    {
        raw_static_eval = evaluate_static(pos, ttStaticEval, ttHit != 0);
        static_eval = corrected_static_eval(raw_static_eval, correction_value(td, pos));
    }
    else
    {
        // Carry the eval from two plies back through checks so the improving
        // trend stays defined instead of resetting at every check in the line.
        static_eval = pos->ply >= 2 ? td->static_eval_stack[pos->ply - 2]
                                    : no_hashmap_entry;
    }
    eval = static_eval;
    td->static_eval_stack[pos->ply] = static_eval;

    nodeContext = make_node_context(td, pos->ply, static_eval, cutNode);

    // A TT score whose bound points the same way is a better estimate of this
    // node than the static eval, so pruning uses it. static_eval keeps the
    // unrefined value: it is the correction-history training target, and
    // feeding a search result back in would train the table on its own output.
    if (!inCheck && ttHit && ttFlag != BOUND_NONE && !is_mate_score(ttScore) &&
        (ttFlag == BOUND_EXACT ||
         (ttFlag == BOUND_LOWER && ttScore > eval) ||
         (ttFlag == BOUND_UPPER && ttScore < eval)))
    {
        eval = ttScore;
    }

    const bool ourNonPawnMaterial = hasNonPawnMaterial(pos, pos->colour_to_move);

    // Razoring
    if (!excludedNode && !inCheck && !isPvNode && pos->ply > 0 &&
        !is_mate_score(alpha) && eval < alpha - razor_margin(depth))
    {
        return quiescence(pos, td, alpha, beta);
    }

    // Reverse futility pruning
    if (!excludedNode && !inCheck && !ttPv &&
        depth <= SEARCH_REVERSE_FUTILITY_MAX_DEPTH && eval >= beta &&
        !is_mate_score(beta) && !is_mate_score(eval) &&
        eval - reverse_futility_margin(depth, nodeContext) >= beta)
    {
        return (eval + beta) / 2;
    }

    // Null-move pruning.
    // The only structural restriction is "no two null moves in a row": a pass
    // answered by a pass is the same position two tempi later. ply_moves[ply-1]
    // is the move that reached this node and is 0 for a null move, so the
    // previous-move test states that rule directly.
    if (!excludedNode && !inCheck && cutNode && depth >= SEARCH_NULL_MOVE_MIN_DEPTH &&
        eval >= beta && ourNonPawnMaterial && previous_ply_move(td, pos->ply) != 0 &&
        pos->ply >= td->nmpMinPly &&
        !is_mate_score(beta) && !is_mate_score(eval))
    {
        pos->ply++;
        td->ply_moves[pos->ply - 1] = 0;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;
        make_null_move(pos, pos->ply);
        tt->prefetch(pos->zobristKey);

        const int reduction = null_move_reduction(depth, eval, beta, nodeContext);
        score = -negamax_impl(pos, td, depth - 1 - reduction, -beta, -beta + 1, false, 0);

        unmake_null_move(pos, pos->ply);
        pos->ply--;
        pos->repetition_index--;

        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;

        if (score >= beta && !is_mate_score(score))
        {
            // Verification is what catches zugzwang, so it must not be able to
            // reproduce the fail-high with the trick it is checking. A non-zero
            // nmpMinPly means this node already sits inside a verification
            // search, where recursing again costs an exponential number of
            // re-searches for nothing.
            if (depth < SEARCH_NULL_MOVE_VERIFICATION_DEPTH || td->nmpMinPly != 0)
                return score;

            const int verificationDepth = std::max(1, depth - reduction);
            // Suppress null moves over the shallow part of the verification
            // subtree only; deeper down the position has changed enough that a
            // pass is informative again.
            td->nmpMinPly = pos->ply + 3 * verificationDepth / 4;
            const int verification =
                negamax_impl(pos, td, verificationDepth, beta - 1, beta, false, 0);
            td->nmpMinPly = 0;

            if (stopped.load(std::memory_order_relaxed) == 1)
                return alpha;

            if (verification >= beta)
                return score;
        }
    }

    // ProbCut
    const int probCutDepth = depth - 1 - SEARCH_PROBCUT_REDUCTION;
    const int probCutMargin = probcut_margin(nodeContext);
    if (!excludedNode && !inCheck && !isPvNode && ourNonPawnMaterial &&
        depth >= SEARCH_PROBCUT_MIN_DEPTH && probCutDepth > 0 &&
        !is_mate_score(beta) && beta < mateScore - probCutMargin)
    {
        const int probCutBeta = beta + probCutMargin;
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
                    score = -negamax_impl(pos, td, probCutDepth, -probCutBeta,
                                          -probCutBeta + 1, !cutNode, 0);
                }

                td->follow_pv_flag = savedFollowPv;

                unmake_move(pos, pos->ply);
                pos->ply--;
                pos->repetition_index--;

                if (stopped.load(std::memory_order_relaxed) == 1)
                    return alpha;

                if (score >= probCutBeta)
                {
                    update_correction_history(td, pos, static_eval, score, depth,
                                              BOUND_LOWER);
                    // The fail-high was only proven at probCutDepth plus the
                    // qsearch screen, so record the bound at that verified depth.
                    tt->store(pos, probCutDepth + 1, score, BOUND_LOWER, move,
                              raw_static_eval, ttPv);
                    return score;
                }
            }
        }
    }

    // Internal iterative reductions.
    // No usable TT move means ordering at this node is poor, so a full-depth
    // search would waste effort. Reduce one ply; the shallower search seeds the
    // TT with a move for the re-search. Placed after all eval-based pruning so
    // those see the true depth.
    if (!excludedNode && ttMove == 0 && depth >= SEARCH_IIR_MIN_DEPTH &&
        (isPvNode || cutNode))
    {
        depth -= SEARCH_IIR_REDUCTION;
    }

    const bool nodeFollowPv = td->follow_pv_flag;
    MovePicker movePicker(pos, td, ttMove, nodeFollowPv, all_moves, inCheck);

    int moves_searched = 0;
    int move_count = 0;
    FixedBuffer<int, MAX_GENERATED_MOVES> failed_quiet_moves;
    FixedBuffer<int, MAX_GENERATED_MOVES> failed_capture_moves;
    const int parentSide = pos->colour_to_move;
    const int parentPly = pos->ply;

    PickedMove picked;
    while (movePicker.next(picked))
    {
        const int move = picked.move;
        if (move == excludedMove)
            continue;

        move_count++;

        const bool quietMove = is_quiet_move(move);
        const bool captureMove = get_is_capture_move(move);
        bool givesCheck = false;
        bool givesCheckKnown = false;

        auto gives_check = [&]() {
            if (!givesCheckKnown) {
                givesCheck = !inCheck && move_gives_check(pos, move);
                givesCheckKnown = true;
            }
            return givesCheck;
        };

        const int history = quietMove
            ? quiet_history_score(td, parentSide, parentPly, move)
            : capture_history_score(td, pos, move);

        // Depth the reduced search would run at, which is what the shallow
        // pruning margins are calibrated against.
        const int historyDivisor = quietMove ? SEARCH_LMR_QUIET_HISTORY_DIVISOR
                                             : SEARCH_LMR_CAPTURE_HISTORY_DIVISOR;
        const int estimatedReduction =
            lmr_table_reduction(depth, move_count) + !nodeContext.improving -
            std::clamp(history / historyDivisor, -SEARCH_LMR_HISTORY_CAP,
                       SEARCH_LMR_HISTORY_CAP);
        const int lmrDepth = std::max(0, depth - estimatedReduction);

        if (!excludedNode && pos->ply && !inCheck && ourNonPawnMaterial &&
            moves_searched > 0 && !is_mate_score(bestScore))
        {
            // Late move pruning: stop generating quiets once the move count is
            // past what this depth can justify. The current move still runs.
            if (move_count >= late_move_pruning_count(depth, nodeContext.improving))
                movePicker.skip_quiets();

            // Forcing moves (captures, and quiets that give check) are only
            // screened by the lenient SEE margin. A quiet sacrifice that checks
            // is exactly the kind of move the quiet margins would throw away.
            if (!quietMove || gives_check())
            {
                if (depth <= SEARCH_SEE_PRUNE_MAX_DEPTH)
                {
                    const int seeScore = picked.seeKnown
                        ? picked.seeScore
                        : static_exchange_eval(pos, move);
                    if (seeScore < -SEARCH_SEE_PRUNE_CAPTURE_MARGIN * depth)
                        continue;
                }
            }
            else
            {
                if (lmrDepth <= SEARCH_FUTILITY_MAX_DEPTH &&
                    eval + futility_margin(lmrDepth) <= alpha)
                {
                    movePicker.skip_quiets();
                    continue;
                }

                if (depth <= SEARCH_HISTORY_PRUNING_MAX_DEPTH &&
                    history < -SEARCH_HISTORY_PRUNING_DEPTH_MARGIN * depth)
                {
                    continue;
                }

                if (depth <= SEARCH_SEE_PRUNE_MAX_DEPTH &&
                    static_exchange_eval(pos, move) <
                        -SEARCH_SEE_PRUNE_QUIET_MARGIN * lmrDepth * lmrDepth)
                {
                    continue;
                }
            }
        }

        // Singular extensions
        int extension = 0;
        if (!excludedNode && pos->ply > 0 && move == ttMove && !inCheck &&
            depth >= SEARCH_SINGULAR_EXTENSION_MIN_DEPTH + ttPv &&
            pos->ply < 2 * td->root_depth &&
            ttHit && ttDepth >= depth - SEARCH_SINGULAR_EXTENSION_DEPTH_MARGIN &&
            (ttFlag == BOUND_LOWER || ttFlag == BOUND_EXACT) &&
            !is_mate_score(ttScore))
        {
            const int singularBeta =
                std::max(-SEARCH_INFINITY + 1,
                         ttScore - SEARCH_SINGULAR_BETA_DEPTH_MARGIN * depth);
            const int singularDepth = std::max(1, (depth - 1) / 2);

            const bool savedFollowPv = td->follow_pv_flag;
            const int savedPvLength = td->pv_length[pos->ply];
            // The singular re-search runs at this same ply and clobbers
            // td->pv_table[ply]. Only [ply, savedPvLength) is ever read after,
            // so save and restore just that slice.
            std::array<int, MAX_DEPTH> savedPvRow;
            for (int i = pos->ply; i < savedPvLength; ++i)
                savedPvRow[i] = td->pv_table[pos->ply][i];
            td->follow_pv_flag = false;

            const int singularScore = negamax_impl(pos, td, singularDepth,
                                                   singularBeta - 1, singularBeta,
                                                   cutNode, move);

            td->follow_pv_flag = savedFollowPv;
            td->pv_length[pos->ply] = savedPvLength;
            for (int i = pos->ply; i < savedPvLength; ++i)
                td->pv_table[pos->ply][i] = savedPvRow[i];

            if (stopped.load(std::memory_order_relaxed) == 1)
                return alpha;

            if (singularScore < singularBeta)
            {
                extension = 1;
                // Failing far below singular beta means the TT move is the only
                // move here by a wide margin, so spend a second ply on it.
                if (!isPvNode && singularScore < singularBeta - SEARCH_SINGULAR_DOUBLE_MARGIN)
                    extension = 2;
            }
            // Multi-cut: some move other than the TT move already reaches
            // singularBeta, and the TT move is worth at least ttScore. Two moves
            // at or above beta is enough to cut without searching anything else.
            else if (singularBeta >= beta && !is_mate_score(singularBeta))
                return singularBeta;
            // The TT move is not singular and the table already proves a
            // fail-high, so other moves are likely just as good.
            else if (ttScore >= beta || cutNode)
                extension = -SEARCH_SINGULAR_NEGATIVE_EXTENSION;
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
        td->ply_moves[parentPly] = move;
        const int childDepth = depth - 1 + extension;

        auto search_child = [&](int searchDepth, int childAlpha, int childBeta, bool childCut) {
            const bool savedFollowPv = td->follow_pv_flag;
            td->follow_pv_flag = nodeFollowPv && td->pv_table[0][parentPly] == move;
            const int childScore = -negamax_impl(pos, td, searchDepth, childAlpha,
                                                 childBeta, childCut, 0);
            td->follow_pv_flag = savedFollowPv;
            return childScore;
        };

        // Principal variation search with late move reductions
        if (moves_searched == 0)
        {
            score = search_child(childDepth, -beta, -alpha, isPvNode ? false : !cutNode);
        }
        else
        {
            // Depth the first (null-window) pass ran at, and whether it ran at
            // all. Both feed the re-search guard below.
            int firstPassDepth = childDepth;
            bool firstPassSearched = true;

            const int lmrFirstMove = SEARCH_LMR_FIRST_MOVE +
                                     (parentPly == 0 ? SEARCH_LMR_ROOT_EXTRA_MOVES : 0);

            if (moves_searched >= lmrFirstMove && depth >= SEARCH_LMR_MIN_DEPTH)
            {
                const bool childInCheck = is_square_under_attack(
                    pos,
                    (pos->colour_to_move == white ?
                        get_lsb_index(pos->piece_bitboards[K]) :
                        get_lsb_index(pos->piece_bitboards[k])),
                    pos->colour_to_move ^ 1);
                const bool counterMove = quietMove && is_counter_move(td, parentPly, move);

                const int reduction =
                    late_move_reduction(depth, move_count, isPvNode, ttPv, counterMove,
                                        childInCheck, captureMove, history, nodeContext);
                firstPassDepth = std::max(1, childDepth - reduction);
                score = search_child(firstPassDepth, -alpha - 1, -alpha, true);
            }
            else
            {
                score = alpha + 1;
                firstPassSearched = false;
            }

            if (score > alpha)
            {
                // Re-search at full depth unless the first pass already was that
                // search (reduction 0, or no first pass), in which case repeating
                // it would re-walk an identical subtree for an identical answer.
                if (!firstPassSearched || firstPassDepth < childDepth)
                    score = search_child(childDepth, -alpha - 1, -alpha, !cutNode);

                if (score > alpha && score < beta)
                    score = search_child(childDepth, -beta, -alpha, false);
            }
        }

        unmake_move(pos, pos->ply);
        pos->ply--;
        pos->repetition_index--;
        moves_searched++;

        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;

        if (pos->ply == 0)
        {
            const int rootBound = score > alpha ? (score >= beta ? BOUND_LOWER : BOUND_EXACT)
                                                : BOUND_UPPER;
            td->recordRootMove(move, score, depth, rootBound);
        }

        if (score > bestScore)
        {
            bestScore = score;
            bestMove = move;

            if (score > alpha)
            {
                hashFlag = BOUND_EXACT;
                alpha = score;

                td->pv_table[pos->ply][pos->ply] = move;
                for (int nextPly = pos->ply + 1; nextPly < td->pv_length[pos->ply + 1]; nextPly++)
                    td->pv_table[pos->ply][nextPly] = td->pv_table[pos->ply + 1][nextPly];
                td->pv_length[pos->ply] = std::max(td->pv_length[pos->ply + 1], pos->ply + 1);

                if (score >= beta)
                {
                    hashFlag = BOUND_LOWER;
                    break;
                }
            }
        }

        if (move != bestMove)
        {
            if (quietMove)
                failed_quiet_moves.push_back(move);
            else if (captureMove)
                failed_capture_moves.push_back(move);
        }
    } // end of move loop

    if (moves_searched == 0)
    {
        if (excludedNode)
            return alpha;

        // Pruning is gated on moves_searched > 0, so it never fires before a
        // legal move has been searched; reaching zero here is a real terminal.
        return inCheck ? -mateVal + pos->ply : 0;
    }

    if (excludedNode)
        return bestScore;

    // One bonus for the move that caused the cutoff, one malus for every move
    // that was tried and failed. Applying a bonus at each alpha raise instead
    // would reward moves that were later beaten.
    if (hashFlag == BOUND_LOWER && bestMove)
    {
        const int bonus = history_bonus(depth);
        if (is_quiet_move(bestMove))
        {
            update_quiet_history(td, parentSide, parentPly, bestMove, bonus);
            penalize_quiet_history(td, parentSide, parentPly, failed_quiet_moves, depth);

            td->killer_moves[1][parentPly] = td->killer_moves[0][parentPly];
            td->killer_moves[0][parentPly] = bestMove;
            update_counter_move(td, parentPly, bestMove);
        }
        else
        {
            update_capture_history(td, pos, bestMove, bonus);
        }
        penalize_capture_history(td, pos, failed_capture_moves, depth);
    }

    // Correction history learns the gap between the adjusted static eval and
    // what the search actually found. A capture best move says more about
    // material than about the evaluation, so it teaches nothing here.
    if (!inCheck && !(bestMove && get_is_capture_move(bestMove)))
        update_correction_history(td, pos, static_eval, bestScore, depth, hashFlag);

    // On a fail-low bestScore is below alpha and is the real upper bound on this
    // node; returning alpha instead would tell the aspiration loop nothing about
    // how far the search actually failed.
    tt->store(pos, depth, bestScore, hashFlag, bestMove,
              inCheck ? no_hashmap_entry : raw_static_eval, ttPv);
    return bestScore;
}

} // namespace

int quiescence(thrawn::Position* pos, ThreadData* td,
               int alpha, int beta)
{
    if (stopped.load(std::memory_order_relaxed) == 1)
        return alpha;

    if (td->thread_id == 0 && --td->check_counter <= 0)
    {
        td->check_counter = NODE_COUNTER_BATCH;
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

    if (pos->ply >= MAX_DEPTH - 1)
        return inCheck ? -mateVal + pos->ply : evaluate(pos);

    const int oldAlpha = alpha;
    const bool isPvNode = ((beta - alpha) > 1);

    int ttDepth = 0;
    int ttMove = 0;
    int ttFlag = BOUND_NONE;
    int ttScore = 0;
    int ttStaticEval = no_hashmap_entry;
    bool ttWasPv = false;
    const bool ttHit = tt->probe(pos, ttDepth, ttMove, ttScore, ttFlag, ttStaticEval, ttWasPv);
    // Qsearch only carries an existing PV marking forward; it never creates one,
    // or every capture sequence off a PV node would be marked.
    const bool ttPv = ttHit && ttWasPv;
    if (ttHit && !isPvNode &&
        (ttFlag == BOUND_EXACT ||
         (ttFlag == BOUND_LOWER && ttScore >= beta) ||
         (ttFlag == BOUND_UPPER && ttScore <= alpha)))
    {
        return ttScore;
    }

    int raw_static_eval = no_hashmap_entry;
    int static_eval = 0;
    // Not `alpha`: the stand pat is a real lower bound on this node, and when
    // it sits below alpha the node still has to report how far below.
    int bestScore = -SEARCH_INFINITY;

    if (!inCheck)
    {
        raw_static_eval = evaluate_static(pos, ttStaticEval, ttHit);
        static_eval = corrected_static_eval(raw_static_eval, correction_value(td, pos));
        bestScore = static_eval;
        td->static_eval_stack[pos->ply] = static_eval;

        // A TT score whose bound points the right way is a better stand pat.
        if (ttHit && ttFlag != BOUND_NONE && !is_mate_score(ttScore) &&
            (ttFlag == BOUND_EXACT ||
             (ttFlag == BOUND_LOWER && ttScore > bestScore) ||
             (ttFlag == BOUND_UPPER && ttScore < bestScore)))
        {
            bestScore = ttScore;
        }

        if (bestScore >= beta)
        {
            tt->store(pos, 0, bestScore, BOUND_LOWER, 0, raw_static_eval, ttPv);
            return bestScore;
        }

        if (bestScore > alpha)
            alpha = bestScore;
    }
    else
    {
        td->static_eval_stack[pos->ply] =
            pos->ply >= 2 ? td->static_eval_stack[pos->ply - 2] : no_hashmap_entry;
    }

    const int move_type = inCheck ? all_moves : only_captures;
    MovePicker movePicker(pos, td, ttMove, false, move_type, inCheck);

    int valid_moves = 0;
    int bestMove = 0;

    PickedMove picked;
    while (movePicker.next(picked))
    {
        const int move = picked.move;

        if (!inCheck && !get_promoted_piece(move) && !is_mate_score(bestScore))
        {
            const bool deltaPruned =
                qsearch_delta_prune(pos, move, static_eval, alpha);
            const int seeScore = picked.seeKnown
                ? picked.seeScore
                : static_exchange_eval(pos, move);
            const bool seePruned = seeScore < SEARCH_QSEARCH_SEE_MARGIN;

            if ((deltaPruned || seePruned) && !move_gives_check(pos, move))
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

        const int score = -quiescence(pos, td, -beta, -alpha);

        unmake_move(pos, pos->ply);
        pos->ply--;
        pos->repetition_index--;

        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;

        if (score > bestScore)
        {
            bestScore = score;
            bestMove = move;

            if (score > alpha)
            {
                alpha = score;

                if (score >= beta)
                {
                    tt->store(pos, 0, bestScore, BOUND_LOWER, bestMove,
                              inCheck ? no_hashmap_entry : raw_static_eval, ttPv);
                    return bestScore;
                }
            }
        }
    }

    if (inCheck && valid_moves == 0)
    {
        const int mate = -mateVal + pos->ply;
        tt->store(pos, 0, mate, BOUND_EXACT, 0, no_hashmap_entry, ttPv);
        return mate;
    }

    tt->store(pos, 0, bestScore, bestScore > oldAlpha ? BOUND_EXACT : BOUND_UPPER,
              bestMove, inCheck ? no_hashmap_entry : raw_static_eval, ttPv);
    return bestScore;
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

