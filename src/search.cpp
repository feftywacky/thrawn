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

std::array<int, 4> LateMovePruning_factors = {0, 8, 12, 24};
int RFP_factor = 64;

int negamax(thrawn::Position* pos, ThreadData* td, int depth, int alpha, int beta)
{
    int score = 0;
    int bestMove = 0;
    int hashFlag = BOUND_UPPER;
    int static_eval = 0;

    // init local pv
    td->pv_length[pos->ply] = pos->ply;

    // 1) Check repetition or 50-move draw
    if (pos->ply && isRepetition(pos) || pos->fifty_move >= 100)
    {
        return 0;
    }

    int isPvNode = ((beta - alpha) > 1);

    // 2) Transposition Table lookup
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

    // For periodic UCI output / time check
    if ((td->nodes & 2047) == 0)
    {
        communicate();
    }

    // 5) Increment node counter
    td->nodes++;
    total_nodes.fetch_add(1, std::memory_order_relaxed);

    // 6) Are we in check?
    bool inCheck = is_square_under_attack(
        pos,
        (pos->colour_to_move == white ?
            get_lsb_index(pos->piece_bitboards[K]) :
            get_lsb_index(pos->piece_bitboards[k])),
        pos->colour_to_move ^ 1
    );

    // If in check, extend depth by 1
    if (inCheck)
    {
        depth++;
        goto full_search;
    }

    // 3) Quiescence
    if (depth == 0)
    {
        // call quiescence with local pos->ply
        return quiescence(pos, td, alpha, beta);
    }

    // 4) Check for max depth overflow
    if (pos->ply > MAX_DEPTH - 1)
    {
        // fallback to a static evaluation
        return evaluate(pos);
    }

    // 7) Compute static evaluation
    static_eval = evaluate(pos);

    // --------------------------------------
    // 8) Reverse Futility Pruning (RFP)
    //    Often called "static-nullmove" or "futility"
    // --------------------------------------
    if (!inCheck && !isPvNode && depth < 3)
    {
        int eval_margin = 120 * depth;
        // if static eval already big enough to exceed beta
        if (static_eval - eval_margin >= beta)
        {
            return static_eval - eval_margin;
        }
    }

    // --------------------------------------
    // 9) Razoring (shallow depth, not in check, non-PV)
    // --------------------------------------
    if (!inCheck && !isPvNode && depth <= 3)
    {
        score = evaluate(pos) + 125;
        int razor_score;
        // If the position is very likely losing (or not better) vs beta, do a quick check
        if (score < beta)
        {
            if (depth == 1)
            {
                razor_score = quiescence(pos, td, alpha, beta);
                return (razor_score > score) ? razor_score : score;
            }
            score += 175;
            if (score < beta && depth <= 2)
            {
                razor_score = quiescence(pos,td, alpha, beta);
                if (razor_score < beta) // quiescence says score fail-low node
                    return (razor_score > score) ? razor_score : score;
            }
        }
    }

    // --------------------------------------
    // 10) Null-move pruning
    // --------------------------------------
    if (!inCheck && depth >= 3 && !isPvNode && !noMajorsOrMinorsPieces(pos) && td->allowNullMovePruning)
    {
        // we make a null move
        copyBoard(pos);

        pos->ply++;
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
        int reduction = 2;
        td->allowNullMovePruning = false;
        score = -negamax(pos, td, depth - 1 - reduction, -beta, -beta + 1);
        td->allowNullMovePruning = true;
        
        pos->ply--;
        pos->repetition_index--;
        
        restoreBoard(pos);

        if (stopped == 1)
            return alpha;

        // If this "fake pass" search fails high, then cut
        if (score >= beta)
        {
            return beta;
        }
    }

    // No-hashmove reduction
    // bestMove==0 means not in tt
    if (!inCheck && !isPvNode && depth >= 3 && ttMove==0)
        depth--;

full_search:

    // 11) Generate moves
    std::vector<int> moves = generate_moves(pos);

    // 12) If we had a best move from TT, ensure it gets sorted first
    if (td->follow_pv_flag)
        score_pv(moves, td);

    sort_moves(pos, td, moves, ttMove);

    // We are about to search each move
    int valid_moves = 0;
    int moves_searched = 0;

    // 13) Search each move (LMR, LMP, PVS logic, etc.)
    for (int move : moves)
    {
        copyBoard(pos);
        pos->ply++;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;

        if (!make_move(pos, move, all_moves, pos->ply))
        {
            pos->ply--;
            pos->repetition_index--;
            continue;
        }
        valid_moves++;

        // -------------------------------------------
        // Principal Variation Search logic
        // -------------------------------------------
        if (moves_searched == 0)
        {
            // First move: full-window search
            score = -negamax(pos, td, depth - 1, -beta, -alpha);
        }
        else
        {
            // -----------------------------
            // (A) Basic Futility on quiet moves
            //     e.g. if depth is small, not giving check, not a capture, etc.
            // -----------------------------
            bool givesCheck = is_square_under_attack(
                                 pos,
                                 (pos->colour_to_move == white) ? get_lsb_index(pos->piece_bitboards[K])
                                                               : get_lsb_index(pos->piece_bitboards[k]),
                                 pos->colour_to_move ^ 1);

            bool allowFutilityPrune = false;
            if (pos->ply && !isPvNode && (depth <= 3))
            {
                if ((static_eval + futility_margin(depth)) <= alpha)
                {
                    allowFutilityPrune = true;
                }
            }

            if (allowFutilityPrune && !givesCheck &&
                (get_move_piece(move) != P) && (get_move_piece(move) != p) &&
                !get_promoted_piece(move) && !get_is_move_castling(move) &&
                !get_is_capture_move(move))
            {
                restoreBoard(pos);
                pos->ply--;
                pos->repetition_index--;
                continue;
            }

            // -----------------------------
            // (B) Late Move Pruning (LMP)
            //     If depth is small, not in check, not a capture,
            //     and we've already searched "too many" quiet moves
            // -----------------------------
            if (pos->ply && depth <= 3 && !isPvNode && !inCheck &&
                !get_is_capture_move(move) &&
                (valid_moves > LateMovePruning_factors[depth]))
            {
                restoreBoard(pos);
                pos->ply--;
                pos->repetition_index--;
                //unmake_move(pos,pos->ply);
                continue;
            }

            // -----------------------------
            // (C) Late Move Reductions (LMR)
            // -----------------------------
            if (valid_moves >= full_depth_moves &&
                depth >= reduction_limit &&
                !inCheck &&
                get_is_capture_move(move) == 0 &&
                get_promoted_piece(move) == 0)
            {
                // Reduced search
                int reduction = 2;
                score = -negamax(pos, td, depth - reduction, -alpha - 1, -alpha);
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
                score = -negamax(pos, td, depth - 1, -alpha - 1, -alpha);

                // If it's still above alpha but not >= beta, do a full re-search
                if (score > alpha && score < beta)
                {
                    score = -negamax(pos, td, depth - 1, -beta, -alpha);
                }
            }
        }

        pos->ply--;
        pos->repetition_index--;
        restoreBoard(pos);
        moves_searched++;

        if (stopped == 1)
            return alpha;

        // Check if this move improved alpha
        if (score > alpha)
        {
            bestMove = move;
            hashFlag = BOUND_EXACT; // PV node

            // Update history for quiet moves
            if (get_is_capture_move(move) == 0)
            {
                td->history_moves[get_move_piece(move)][get_move_target(move)] += depth;
            }

            alpha = score;

            // Update PV
            td->pv_table[pos->ply][pos->ply] = move;
            for (int nextPly = pos->ply + 1; nextPly < td->pv_length[pos->ply + 1]; nextPly++)
            {
                td->pv_table[pos->ply][nextPly] = td->pv_table[pos->ply + 1][nextPly];
            }
            td->pv_length[pos->ply] = td->pv_length[pos->ply + 1];

            // Fail-hard beta cutoff
            if (alpha >= beta)
            {
                tt->store(pos, depth, beta, BOUND_LOWER, bestMove);

                // killer move
                if (get_is_capture_move(move) == 0)
                {
                    td->killer_moves[1][pos->ply] = td->killer_moves[0][pos->ply];
                    td->killer_moves[0][pos->ply] = move;
                }
                return beta;
            }
        }
    } // end of move loop

    // if no valid moves
    if (valid_moves == 0)
    {
        // (should not happen because we handle moves.empty() above,
        //  but just in case)
        if (inCheck)
            return -mateVal + pos->ply;
        else
            return 0;
    }

    // 14) Store in TT and return
    tt->store(pos, depth, alpha, hashFlag, bestMove);
    return alpha;
}

int quiescence(thrawn::Position* pos, ThreadData* td,
               int alpha, int beta)
{
    if ((td->nodes & 2047) == 0)
    {
        communicate();
    }

    td->nodes++;
    total_nodes.fetch_add(1, std::memory_order_relaxed);

    // safety check for array bounds
    if (pos->ply > MAX_DEPTH-1)
    {
        return evaluate(pos);
    }

    int evaluation = evaluate(pos);

    // fail-hard beta cutoff
    if (evaluation >= beta)
        return beta; // fails high

    // found better move
    if (evaluation > alpha)
        alpha = evaluation; // principal variation PV node (best move)

    std::vector<int> moves = generate_moves(pos);
    sort_moves(pos, td, moves, 0);

    for (int move : moves)
    {
        copyBoard(pos);
        pos->ply++;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;

        if (!make_move(pos, move, only_captures, pos->ply))
        {
            pos->ply--;
            pos->repetition_index--;
            continue;
        }

        int score = -quiescence(pos, td, -beta, -alpha);

        pos->ply--;
        pos->repetition_index--;
        restoreBoard(pos);

        if (stopped == 1)
            return alpha;

        // found better move
        if (score > alpha)
        {
            alpha = score; // principal variation PV node (best move)

            // fail-hard beta cutoff
            if (score >= beta)
                return beta; // fails high
        }
    }

    // move fails low (<= alpha)
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
            return 20000; // give pv move priority
        }
    }

    // handle promotions
    if (get_promoted_piece(move) == Q || get_promoted_piece(move) == q)
    {
        return 10000 + 499;
    }

    // captures: use MVV-LVA
    if (get_is_capture_move(move))
    {
        // find target piece
        int target = P;
        int start_piece;
        int end_piece;
        (pos->colour_to_move == white) ? start_piece = p : start_piece = P;
        (pos->colour_to_move == white) ? end_piece = k : end_piece = K;

        for (int i = start_piece; i <= end_piece; i++)
        {
            if (get_bit(pos->piece_bitboards[i], get_move_target(move)))
            {
                target = i;
                break;
            }
        }
        return mvv_lva[get_move_piece(move)][target] + 10000;
    }
    else // quiet moves
    {
        if (td->killer_moves[0][pos->ply] == move)
            return 9000;
        else if (td->killer_moves[1][pos->ply] == move)
            return 8000;
        else if (get_promoted_piece(move) == Q || get_promoted_piece(move) == q)
            return mvv_lva[get_move_piece(move)][get_move_target(move)] + 100;
        else
            return td->history_moves[get_move_piece(move)][get_move_target(move)];
    }
    return 0;
}

void score_pv(std::vector<int> &moves, ThreadData* td)
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
    std::vector<int> scores(n);
    for (int i = 0; i < n; i++)
    {
        // If a move is the bestMove from TT, give it a high score
        if (moves[i] == bestMove)
            scores[i] = 30000;
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
    for (int i = 0; i <= pos->repetition_index; i++)
    {
        if (pos->repetition_table[i] == pos->zobristKey)
            return 1;
    }
    return 0;
}

int futility_margin(int depth)
{
    int margins[4] = {0, 120, 200, 300};
    return margins[depth];
}
