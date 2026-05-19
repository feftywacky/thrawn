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
#include "search_params.h"
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

constexpr std::array<int, 6> PieceValues = {100, 320, 330, 500, 900, 20000};
constexpr int NodeCounterBatch = 1024;
constexpr int SeePruneMaxDepth = 4;
constexpr int SeePruneDepthMargin = 200;

bool is_quiet_move(int move) {
    return !get_is_capture_move(move) && !get_promoted_piece(move);
}

bool is_mate_score(int score) {
    return score <= -mateScore || score >= mateScore;
}

int reverse_futility_margin(int depth) {
    if (depth <= 0) {
        return 0;
    }
    if (depth == 1) {
        return searchParams.reverseFutilityMargin1;
    }
    if (depth == 2) {
        return searchParams.reverseFutilityMargin2;
    }
    return searchParams.reverseFutilityDepthFactor * depth;
}

int razor_margin(int depth) {
    if (depth <= 0) {
        return 0;
    }
    if (depth == 1) {
        return searchParams.razorMargin1;
    }
    if (depth == 2) {
        return searchParams.razorMargin2;
    }
    return searchParams.razorMarginDepthN;
}

int null_move_reduction(int depth, int static_eval, int beta) {
    const int eval_margin = std::max(0, static_eval - beta);
    const int depth_divisor = std::max(1, searchParams.nullMoveDepthDivisor);
    const int eval_divisor = std::max(1, searchParams.nullMoveEvalDivisor);
    int reduction = searchParams.nullMoveBaseReduction + depth / depth_divisor;
    reduction += std::min(searchParams.nullMoveEvalBonusMax, eval_margin / eval_divisor);
    return std::clamp(reduction, 1, std::max(1, depth - 1));
}

int piece_value(int piece) {
    return PieceValues[piece % 6];
}

void count_node(ThreadData* td) {
    td->nodes++;
    if ((td->nodes & (NodeCounterBatch - 1)) == 0) {
        total_nodes.fetch_add(NodeCounterBatch, std::memory_order_relaxed);
    }
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

    const int target = get_move_target(move);
    const int start_piece = pos->colour_to_move == white ? p : P;
    const int end_piece = pos->colour_to_move == white ? k : K;
    for (int piece = start_piece; piece <= end_piece; piece++) {
        if (get_bit(pos->piece_bitboards[piece], target)) {
            return piece;
        }
    }

    return -1;
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

    attackers = get_bishop_attacks(pos, target, occupancy) &
                side_pieces(pos, side, B, b) & occupancy;
    if (attackers) {
        from = get_lsb_index(attackers);
        return side == white ? B : b;
    }

    attackers = get_rook_attacks(pos, target, occupancy) &
                side_pieces(pos, side, R, r) & occupancy;
    if (attackers) {
        from = get_lsb_index(attackers);
        return side == white ? R : r;
    }

    attackers = get_queen_attacks(pos, target, occupancy) &
                side_pieces(pos, side, Q, q) & occupancy;
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

    std::array<int, 32> gains{};
    int depth = 0;
    gains[0] = (victim == -1 ? 0 : piece_value(victim)) + promotion_gain(move);

    uint64_t occupancy = pos->occupancies[both];
    occupancy &= ~(1ULL << source);

    if (get_is_move_enpassant(move)) {
        const int captured_square = pos->colour_to_move == white ? target + 8 : target - 8;
        occupancy &= ~(1ULL << captured_square);
        occupancy |= (1ULL << target);
    } else if (!get_is_capture_move(move)) {
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
    if (is_mate_score(alpha) || noMajorsOrMinorsPieces(pos)) {
        return false;
    }

    return static_eval + qsearch_move_gain_upper_bound(pos, move) +
           searchParams.qsearchDeltaMargin <= alpha;
}

bool qsearch_see_prune(thrawn::Position* pos, int move) {
    if (!get_is_capture_move(move) || get_promoted_piece(move) ||
        get_move_piece(move) == K || get_move_piece(move) == k) {
        return false;
    }

    const int target = get_move_target(move);
    const uint64_t enemyKing = pos->piece_bitboards[pos->colour_to_move == white ? k : K];
    if (pos->king_attacks[target] & enemyKing) {
        return false;
    }

    return static_exchange_eval(pos, move) < 0;
}

int history_bonus(int depth) {
    const int history_max = std::max(1, searchParams.historyMax);
    const int bonus = searchParams.historyBonusDepthSquared * depth * depth +
                      searchParams.historyBonusDepthLinear * depth;
    return std::min(history_max / 2, bonus);
}

int history_score(ThreadData* td, int move) {
    return td->history_moves[get_move_piece(move)][get_move_target(move)];
}

void update_history(ThreadData* td, int move, int bonus) {
    int& entry = td->history_moves[get_move_piece(move)][get_move_target(move)];
    const int history_max = std::max(1, searchParams.historyMax);
    bonus = std::clamp(bonus, -history_max, history_max);
    const int gravity = bonus >= 0 ? bonus : -bonus;
    entry += bonus - entry * gravity / history_max;
    entry = std::clamp(entry, -history_max, history_max);
}

void update_quiet_history(ThreadData* td, int move, int depth) {
    update_history(td, move, history_bonus(depth));
}

void penalize_quiet_history(ThreadData* td, const std::vector<int>& moves, int depth) {
    const int penalty = -history_bonus(depth);
    for (int move : moves) {
        update_history(td, move, penalty);
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
                        bool is_counter, int quiet_history) {
    int reduction = searchParams.lmrBaseReduction;
    if (!is_pv_node && depth >= searchParams.lmrNonPvDepth) {
        ++reduction;
    }
    if (depth >= searchParams.lmrMoveDepth1 &&
        move_number > searchParams.lmrMoveNumber1) {
        ++reduction;
    }
    if (depth >= searchParams.lmrMoveDepth2 &&
        move_number > searchParams.lmrMoveNumber2) {
        ++reduction;
    }

    const int history_max = std::max(1, searchParams.historyMax);
    const int good_history_divisor = std::max(1, searchParams.lmrGoodHistoryDivisor);
    const int bad_history_divisor = std::max(1, searchParams.lmrBadHistoryDivisor);
    if (is_pv_node || is_counter ||
        quiet_history > history_max / good_history_divisor) {
        --reduction;
    }
    if (!is_counter && quiet_history < -history_max / bad_history_divisor) {
        ++reduction;
    }

    return std::clamp(reduction, 1, std::max(1, depth - 2));
}

void quicksort_scored_moves(std::vector<int>& moves, int* move_scores,
                            int low, int high)
{
    if (low < high)
    {
        int pivot = move_scores[high];
        int i = low - 1;

        for (int j = low; j <= high - 1; j++)
        {
            if (move_scores[j] > pivot)
            {
                i++;
                std::swap(move_scores[i], move_scores[j]);
                std::swap(moves[i], moves[j]);
            }
        }

        std::swap(move_scores[i + 1], move_scores[high]);
        std::swap(moves[i + 1], moves[high]);

        int pi = i + 1;

        quicksort_scored_moves(moves, move_scores, low, pi - 1);
        quicksort_scored_moves(moves, move_scores, pi + 1, high);
    }
}

} // namespace

int negamax(thrawn::Position* pos, ThreadData* td, int depth, int alpha, int beta)
{
    int score = 0;
    int bestScore = -INFINITY;
    int bestMove = 0;
    int hashFlag = BOUND_UPPER;
    int static_eval = 0;

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
        depth += searchParams.checkExtension;
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
    if ((ttHit = tt->probe(pos, ttDepth, alpha, beta, ttMove, ttScore, ttFlag)))
    {
        if (ttDepth >= depth && !isPvNode)
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
    static_eval = evaluate(pos);
    const bool pawnOnlyEndgame = noMajorsOrMinorsPieces(pos);

    // --------------------------------------
    // Razoring (shallow depth, not in check, non-PV)
    // --------------------------------------
    if (!inCheck && !isPvNode && !pawnOnlyEndgame &&
        depth <= searchParams.razorMaxDepth && pos->ply > 0 &&
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
    if (!inCheck && !isPvNode && !pawnOnlyEndgame &&
        depth <= searchParams.reverseFutilityMaxDepth &&
        !is_mate_score(beta) && !is_mate_score(static_eval))
    {
        int eval_margin = reverse_futility_margin(depth);
        // if static eval already big enough to exceed beta
        if (static_eval - eval_margin >= beta)
        {
            return static_eval - eval_margin;
        }
    }

    // --------------------------------------
    // Null-move pruning
    // --------------------------------------
    if (!inCheck && depth >= searchParams.nullMoveMinDepth && !isPvNode && static_eval >= beta &&
        !pawnOnlyEndgame && td->allowNullMovePruning &&
        !is_mate_score(beta) && !is_mate_score(static_eval))
    {
        // we make a null move
        copyBoard(pos);

        pos->ply++;
        td->ply_moves[pos->ply - 1] = 0;
        nnue_copy_parent_to_child(pos, pos->ply);
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;

        // Remove en-passant possibility from the key
        if (pos->enpassant != null_sq)
            pos->zobristKey ^= pos->enpassant_hashkey[pos->enpassant];
        pos->enpassant = null_sq;

        // Switch side
        pos->colour_to_move ^= 1;
        pos->zobristKey ^= pos->colour_to_move_hashkey;
        nnue_debug_check(pos);

        // Null-move search with reduced depth
        int reduction = null_move_reduction(depth, static_eval, beta);
        td->allowNullMovePruning = false;
        score = -negamax(pos, td, depth - 1 - reduction, -beta, -beta + 1);
        td->allowNullMovePruning = true;
        
        pos->ply--;
        pos->repetition_index--;
        
        restoreBoard(pos);

        if (stopped.load(std::memory_order_relaxed) == 1)
            return alpha;

        // If this "fake pass" search fails high, then cut
        if (score >= beta && !is_mate_score(score))
        {
            if (depth >= searchParams.nullMoveVerificationDepth)
            {
                const bool savedNullMoveState = td->allowNullMovePruning;
                td->allowNullMovePruning = false;
                const int verificationDepth = std::max(1, depth - reduction);
                const int verification = negamax(pos, td, verificationDepth, beta - 1, beta);
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

    // Generate moves
    std::vector<int> moves = generate_moves(pos);

    // If we had a best move from TT, ensure it gets sorted first
    const bool nodeFollowPv = td->follow_pv_flag;
    if (nodeFollowPv)
        score_pv(pos, moves, td);
    else
        td->score_pv_flag = false;

    sort_moves(pos, td, moves, ttMove);

    // We are about to search each move
    int valid_moves = 0;
    int moves_searched = 0;
    int quiet_moves_seen = 0;
    bool pruned_moves = false;
    std::vector<int> failed_quiet_moves;
    failed_quiet_moves.reserve(moves.size());

    // Search each move (LMR, LMP, PVS logic, etc.)
    for (int move : moves)
    {
        const int parentPly = pos->ply;
        const bool quietMove = is_quiet_move(move);
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

        if (!firstMove && pos->ply && !isPvNode && !inCheck && !pawnOnlyEndgame)
        {
            const bool castleMove = get_is_move_castling(move);

            if (quietMove && !pawnMove && !castleMove &&
                depth <= searchParams.futilityMaxDepth &&
                static_eval + futility_margin(depth) <= alpha &&
                !gives_check())
            {
                pruned_moves = true;
                continue;
            }

            if (quietMove && !pawnMove && !castleMove &&
                depth <= searchParams.lateMovePruningMaxDepth &&
                quiet_moves_seen > futility_move_count(depth) &&
                !gives_check())
            {
                pruned_moves = true;
                continue;
            }

            if (get_is_capture_move(move) && !get_promoted_piece(move) &&
                depth <= SeePruneMaxDepth &&
                static_exchange_eval(pos, move) < -SeePruneDepthMargin * depth &&
                !gives_check())
            {
                pruned_moves = true;
                continue;
            }
        }

        const bool needsLmrCheckInfo =
            !firstMove && quietMove && !get_is_move_castling(move) &&
            depth >= searchParams.lmrReductionDepthLimit && !inCheck;
        if (needsLmrCheckInfo)
            gives_check();

        copyBoard(pos);
        pos->ply++;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;

        if (!make_move_on_board(pos, move, all_moves, pos->ply))
        {
            restoreBoard(pos);
            pos->ply--;
            pos->repetition_index--;
            continue;
        }
        valid_moves++;
        td->ply_moves[parentPly] = move;

        auto search_child = [&](int childDepth, int childAlpha, int childBeta) {
            const bool savedFollowPv = td->follow_pv_flag;
            const bool savedScorePv = td->score_pv_flag;
            td->follow_pv_flag = nodeFollowPv && td->pv_table[0][parentPly] == move;
            td->score_pv_flag = false;
            const int childScore = -negamax(pos, td, childDepth, childAlpha, childBeta);
            td->follow_pv_flag = savedFollowPv;
            td->score_pv_flag = savedScorePv;
            return childScore;
        };

        // -------------------------------------------
        // Principal Variation Search logic
        // -------------------------------------------
        if (moves_searched == 0)
        {
            // First move: full-window search
            score = search_child(depth - 1, -beta, -alpha);
        }
        else
        {
            // -----------------------------
            // Late Move Reductions (LMR)
            // -----------------------------
            const int quietHistory = quietMove ? history_score(td, move) : 0;
            const bool counterMove = quietMove && is_counter_move(td, parentPly, move);

            if (quiet_moves_seen >= searchParams.lmrFullDepthMoves &&
                depth >= searchParams.lmrReductionDepthLimit &&
                !inCheck &&
                !givesCheck &&
                quietMove &&
                !get_is_move_castling(move))
            {
                // Reduced search
                int reduction = late_move_reduction(depth, moves_searched + 1,
                                                    isPvNode, counterMove, quietHistory);
                int reducedDepth = std::max(1, depth - 1 - reduction);
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
                score = search_child(depth - 1, -alpha - 1, -alpha);

                // If it's still above alpha but not >= beta, do a full re-search
                if (score > alpha && score < beta)
                {
                    score = search_child(depth - 1, -beta, -alpha);
                }
            }
        }

        pos->ply--;
        pos->repetition_index--;
        restoreBoard(pos);
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

            // Update history for quiet moves
            if (quietMove)
            {
                update_quiet_history(td, move, depth);
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
                tt->store(pos, depth, beta, BOUND_LOWER, bestMove);

                // quiet move ordering updates
                if (quietMove)
                {
                    td->killer_moves[1][pos->ply] = td->killer_moves[0][pos->ply];
                    td->killer_moves[0][pos->ply] = move;
                    update_counter_move(td, pos->ply, move);
                    penalize_quiet_history(td, failed_quiet_moves, depth);
                }
                return beta;
            }
        }

        if (quietMove && score <= oldAlpha)
        {
            failed_quiet_moves.push_back(move);
        }
    } // end of move loop

    // if no valid moves
    if (valid_moves == 0)
    {
        if (pruned_moves)
            return alpha;

        // (should not happen because we handle moves.empty() above,
        //  but just in case)
        if (inCheck)
            return -mateVal + pos->ply;
        else
            return 0;
    }

    // Store in TT and return
    tt->store(pos, depth, alpha, hashFlag, bestMove);
    return alpha;
}

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
    if (tt->probe(pos, ttDepth, alpha, beta, ttMove, ttScore, ttFlag))
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
    if (!inCheck)
    {
        static_eval = evaluate(pos);

        // fail-hard beta cutoff
        if (static_eval >= beta)
        {
            tt->store(pos, 0, beta, BOUND_LOWER, 0);
            return beta; // fails high
        }

        // found better move
        if (static_eval > alpha)
            alpha = static_eval; // principal variation PV node (best move)
    }

    const int move_type = inCheck ? all_moves : only_captures;
    std::vector<int> moves = generate_moves(pos, move_type);
    sort_moves(pos, td, moves, ttMove);

    int valid_moves = 0;
    int bestMove = 0;
    int bestScore = inCheck ? -INFINITY : alpha;

    for (int move : moves)
    {
        const bool promotionMove = get_promoted_piece(move) != 0;
        const bool deltaPruned = !inCheck && !promotionMove &&
                                 qsearch_delta_prune(pos, move, static_eval, alpha);
        const bool seePruned = !inCheck && !promotionMove &&
                               qsearch_see_prune(pos, move);

        if ((deltaPruned || seePruned) && !move_gives_check(pos, move))
        {
            continue;
        }

        const int parentPly = pos->ply;
        copyBoard(pos);
        pos->ply++;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;

        if (!make_move_on_board(pos, move, move_type, pos->ply))
        {
            restoreBoard(pos);
            pos->ply--;
            pos->repetition_index--;
            continue;
        }
        valid_moves++;
        td->ply_moves[parentPly] = move;

        int score = -quiescence(pos, td, -beta, -alpha);

        pos->ply--;
        pos->repetition_index--;
        restoreBoard(pos);

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
                tt->store(pos, 0, beta, BOUND_LOWER, bestMove);
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
    tt->store(pos, 0, alpha, alpha > oldAlpha ? BOUND_EXACT : BOUND_UPPER, bestMove);
    return alpha;
}

int score_move(thrawn::Position* pos, ThreadData* td, int move)
{
    // scoring pv
    if (td->score_pv_flag)
    {
        if (td->pv_table[0][pos->ply] == move)
        {
            td->score_pv_flag = false;
            return searchParams.pvMoveScore; // give pv move priority
        }
    }

    const int promotedPiece = get_promoted_piece(move);

    // Captures: start with MVV-LVA, then push SEE-losing captures behind
    // useful quiet moves. This mirrors Stockfish's good-capture / quiet /
    // bad-capture staging without replacing the current full sort.
    if (get_is_capture_move(move))
    {
        int target = captured_piece(pos, move);
        if (target == -1)
            target = pos->colour_to_move == white ? p : P;

        int captureScore = mvv_lva[get_move_piece(move)][target] + 10000;
        if (promotedPiece == Q || promotedPiece == q)
            captureScore += 800;

        if (!promotedPiece)
        {
            const int seeScore = static_exchange_eval(pos, move);
            if (seeScore < 0)
                return -5000 + (mvv_lva[get_move_piece(move)][target] / 10) +
                       std::clamp(seeScore, -2000, 0);

            captureScore += std::min(seeScore, 1000);
        }

        return captureScore;
    }
    else // quiet moves
    {
        if (promotedPiece == Q || promotedPiece == q)
        {
            return searchParams.queenPromotionScore;
        }

        if (td->killer_moves[0][pos->ply] == move)
            return searchParams.killerMoveScore1;
        else if (td->killer_moves[1][pos->ply] == move)
            return searchParams.killerMoveScore2;
        else if (is_counter_move(td, pos->ply, move))
            return searchParams.counterMoveScore +
                   std::clamp(history_score(td, move) /
                                  std::max(1, searchParams.counterMoveHistoryDivisor),
                              -searchParams.counterMoveHistoryCap,
                              searchParams.counterMoveHistoryCap);
        else
            return std::clamp(history_score(td, move),
                              -searchParams.historyScoreCap,
                              searchParams.historyScoreCap);
    }
    return 0;
}

void score_pv(thrawn::Position* pos, std::vector<int> &moves, ThreadData* td)
{
    td->follow_pv_flag = false;
    for (int move : moves)
    {
        if (td->pv_table[0][pos->ply] == move)
        {
            td->score_pv_flag = true;
            td->follow_pv_flag = true;
        }
    }
}

void sort_moves(thrawn::Position* pos, ThreadData* td,
                std::vector<int> &moves, int bestMove)
{
    const int n = static_cast<int>(moves.size());
    if (n < 2)
        return;

    constexpr int MaxStackMoveScores = 256;
    if (n <= MaxStackMoveScores)
    {
        std::array<int, MaxStackMoveScores> scores{};
        for (int i = 0; i < n; i++)
        {
            // If a move is the bestMove from TT, give it a high score
            if (moves[i] == bestMove)
                scores[i] = searchParams.ttMoveScore;
            else
                scores[i] = score_move(pos, td, moves[i]);
        }
        quicksort_scored_moves(moves, scores.data(), 0, n - 1);
        return;
    }

    std::vector<int> scores(n);
    for (int i = 0; i < n; i++)
    {
        if (moves[i] == bestMove)
            scores[i] = searchParams.ttMoveScore;
        else
            scores[i] = score_move(pos, td, moves[i]);
    }
    quicksort_moves(moves, scores, 0, n - 1);
}

// standard quicksort helper
void quicksort_moves(std::vector<int> &moves, std::vector<int> &move_scores,
                     int low, int high)
{
    if (low < high)
    {
        int pivot = move_scores[high];
        int i = low - 1;

        for (int j = low; j <= high - 1; j++)
        {
            if (move_scores[j] > pivot)
            {
                i++;
                std::swap(move_scores[i], move_scores[j]);
                std::swap(moves[i], moves[j]);
            }
        }

        std::swap(move_scores[i + 1], move_scores[high]);
        std::swap(moves[i + 1], moves[high]);

        int pi = i + 1;

        quicksort_moves(moves, move_scores, low, pi - 1);
        quicksort_moves(moves, move_scores, pi + 1, high);
    }
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
        return searchParams.futilityMargin1;
    }
    if (depth == 2) {
        return searchParams.futilityMargin2;
    }
    return searchParams.futilityMargin3;
}

int futility_move_count(int depth)
{
    if (depth <= 0) {
        return 0;
    }
    if (depth == 1) {
        return searchParams.lateMovePruningDepth1;
    }
    if (depth == 2) {
        return searchParams.lateMovePruningDepth2;
    }
    return searchParams.lateMovePruningDepth3;
}
