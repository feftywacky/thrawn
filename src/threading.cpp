#include "threading.h"
#include "search.h"         // for negamax, quiescence, etc.
#include "uci.h"            // for 'stopped', 'communicate()', etc.
#include "move_generator.h"
#include "move_helpers.h"
#include "transposition_table.h"
#include "globals.h"
#include "search_params.h"
#include <thread>
#include <iostream>
#include <atomic>
#include <algorithm>
#include <cstdint>

static std::int64_t globalSearchStartTime = 0;
constexpr int NodeCounterBatch = 1024;

static int uci_mate_score(int score) {
    if (score >= mateScore) {
        return (mateVal - score + 1) / 2;
    }

    if (score <= -mateScore) {
        return -((mateVal + score) / 2);
    }

    return 0;
}

static std::uint64_t exact_node_total(int numThreads) {
    std::uint64_t nodes = 0;
    for (int i = 0; i < numThreads; ++i) {
        nodes += static_cast<std::uint64_t>(threadDatas[i].nodes);
    }
    return nodes;
}

RootMove::RootMove()
    : move(0), score(-INFINITY), depth(0), bound(BOUND_NONE), pv_length(0), completed(false)
{
    pv.fill(0);
}

SearchResult::SearchResult()
    : thread(nullptr), thread_id(0), move(0), score(-INFINITY), depth(0),
      pv_length(0), has_move(false)
{
    pv.fill(0);
}

/*
 * ThreadData constructor:
 * Initializes all fixed‑size arrays to zero and sets flags.
 */
ThreadData::ThreadData() {
    pv_length.fill(0);
    for (auto &row : pv_table)
        row.fill(0);
    for (auto &row : killer_moves)
        row.fill(0);
    quiet_history = {};
    continuation_history = {};
    capture_history = {};
    for (auto &row : counter_moves)
        row.fill(0);
    ply_moves.fill(0);

    follow_pv_flag = false;
    score_pv_flag  = false;
    allowNullMovePruning = true;

    nodes = 0;

    thread_id = 0;
    final_depth = 0;
    final_score = 0;
    final_bound = BOUND_NONE;
    root_moves.clear();
    iteration_root_moves.clear();
}

/*
 * Reset the thread data between searches.
 */
void ThreadData::resetThreadData() {
    pv_length.fill(0);
    for (auto &row : pv_table)
        row.fill(0);
    for (auto &row : killer_moves)
        row.fill(0);
    quiet_history = {};
    continuation_history = {};
    capture_history = {};
    for (auto &row : counter_moves)
        row.fill(0);
    ply_moves.fill(0);

    follow_pv_flag = false;
    score_pv_flag  = false;
    allowNullMovePruning = true;

    nodes = 0;

    thread_id = 0;
    final_depth = 0;
    final_score = 0;
    final_bound = BOUND_NONE;
    root_moves.clear();
    iteration_root_moves.clear();
}

void ThreadData::recordRootMove(int move, int score, int depth, int bound) {
    RootMove* entry = nullptr;
    for (RootMove& rootMove : iteration_root_moves) {
        if (rootMove.move == move) {
            entry = &rootMove;
            break;
        }
    }

    if (entry == nullptr) {
        iteration_root_moves.emplace_back();
        entry = &iteration_root_moves.back();
        entry->move = move;
    }

    entry->score = score;
    entry->depth = depth;
    entry->bound = bound;
    entry->completed = true;
    entry->pv.fill(0);
    entry->pv[0] = move;

    int length = pv_length[1];
    if (length < 1)
        length = 1;
    if (length > MAX_DEPTH)
        length = MAX_DEPTH;

    for (int nextPly = 1; nextPly < length; nextPly++) {
        entry->pv[nextPly] = pv_table[1][nextPly];
    }

    entry->pv_length = length;
}

static int legal_fallback_move(thrawn::Position* rootPos) {
    std::vector<int> moves = generate_moves(rootPos);
    for (int move : moves) {
        thrawn::Position scratch(*rootPos);
        scratch.ply++;
        scratch.repetition_index++;
        scratch.repetition_table[scratch.repetition_index] = scratch.zobristKey;

        if (make_move(&scratch, move, all_moves, scratch.ply))
            return move;
    }

    return 0;
}

static void print_result_info(const SearchResult& result) {
    if (!result.has_move)
        return;

    const std::int64_t currentTime = get_time_ms() - globalSearchStartTime;
    std::cout << "info depth " << result.depth
              << " nodes " << total_nodes.load(std::memory_order_relaxed)
              << " time " << currentTime
              << " score ";

    if (result.score >= mateScore || result.score <= -mateScore) {
        std::cout << "mate " << uci_mate_score(result.score);
    } else {
        std::cout << "cp " << result.score;
    }

    if (result.pv_length > 0) {
        std::cout << " pv ";
        for (int i = 0; i < result.pv_length; i++) {
            print_move(result.pv[i]);
            std::cout << " ";
        }
    } else {
        std::cout << " pv (none)";
    }

    std::cout << "\n";
    std::cout.flush();
}

/**
 * smp_worker_thread_func
 * Each worker thread performw iterative deepening
 * Use the local copy of the position and the thread's search data
 */
void smp_worker_thread_func(thrawn::Position* pos, int threadID, int maxDepth)
{
    ThreadData* td = &threadDatas[threadID];
    td->thread_id = threadID;
    int alpha = -INFINITY;
    int beta  =  INFINITY;
    int score = 0;

    // Perform iterative deepening from depth 1 to maxDepth
    for (int curr_depth = 1; curr_depth <= maxDepth; curr_depth++)
    {
        if (stopped.load(std::memory_order_relaxed) == 1)
            break;
        
        td->follow_pv_flag = true;

        // For configured aspiration depths, derive the window from the previous score.
        const int threadCycle = std::max(1, searchParams.aspirationThreadCycle);
        int delta = searchParams.aspirationWindowSize +
                    (threadID % threadCycle) * searchParams.aspirationThreadDelta;
        if(curr_depth>=searchParams.aspirationWindowDepth)
        {
            // Use the previous iteration’s best score (final_score) to set the window.
            // Here we “clamp” alpha to at least –mateVal and beta to at most mateVal.
            alpha = std::max(-mateVal, td->final_score - delta);
            beta  = std::min(mateVal, td->final_score + delta);
        }
        else
        {
            alpha = -INFINITY;
            beta  = INFINITY;
        }

        // Aspiration loop: adjust the window until search returns a score in (alpha, beta)
        bool completed_iteration = false;
        while (true)
        {
            if(stopped.load(std::memory_order_relaxed)==1)
                break;

            // Backup the current good PV from the previous iteration.
            std::array<int, MAX_DEPTH> backup_pv = td->pv_table[0];
            int backup_pv_length = td->pv_length[0];
            td->iteration_root_moves.clear();
            
            score = negamax(pos, td, curr_depth, alpha, beta);

            if (stopped.load(std::memory_order_relaxed) == 1)
            {
                td->pv_table[0] = backup_pv;
                td->pv_length[0] = backup_pv_length;
                break;
            }
            
            // If the score falls inside the window, then we consider the search good
            if (score > alpha && score < beta)
            {
                completed_iteration = true;
                break;
            }
            
            // If the search fails low, shift the window around the returned score.
            if (score <= alpha) {
                beta = alpha;
                alpha = std::max(-mateVal, score - delta);
                // Restore the previous PV since the current iteration did not complete properly
                td->pv_table[0] = backup_pv;
                td->pv_length[0] = backup_pv_length;
            }
            // If the search fails high, keep some overlap and expand upward.
            else if (score >= beta) {
                alpha = std::max(alpha, beta - delta);
                beta = std::min(mateVal, score + delta);
            }
            
            // Increase delta for the next iteration of the aspiration loop
            delta = delta + delta / 2;
        }

        if (!completed_iteration)
            break;

        // Update the aspiration window.
        alpha = score - searchParams.aspirationWindowSize;
        beta = score + searchParams.aspirationWindowSize;

        td->final_depth = curr_depth;
        td->final_score = score;
        td->root_moves = td->iteration_root_moves;
        td->final_bound = BOUND_NONE;

        if (td->pv_length[0] > 0) {
            const int pvMove = td->pv_table[0][0];
            for (const RootMove& rootMove : td->root_moves) {
                if (rootMove.completed && rootMove.move == pvMove) {
                    td->final_bound = rootMove.bound;
                    break;
                }
            }

            if (td->final_bound == BOUND_NONE)
                td->final_bound = BOUND_EXACT;
        }
        
        // print an info line from the master thread (thread 0)
        if (threadID == 0)
        {   
            const std::int64_t currentTime = get_time_ms() - globalSearchStartTime;
            const std::uint64_t reportedNodes =
                total_nodes.load(std::memory_order_relaxed) +
                static_cast<std::uint64_t>(td->nodes & (NodeCounterBatch - 1));
            std::cout << "info depth " << curr_depth
                      << " nodes " << reportedNodes
                      << " time " << currentTime;
            
            // Determine whether to report a mate score or a centipawn score
            if (score >= mateScore || score <= -mateScore)
            {
                std::cout << " score mate " << uci_mate_score(score);
            }
            else
            {
                std::cout << " score cp " << score;
            }
            
            if (td->pv_length[0] > 0)
            {
                std::cout << " pv ";
                for (int i = 0; i < td->pv_length[0]; i++)
                {
                    print_move(td->pv_table[0][i]);
                    std::cout << " ";
                }
            }
            else
            {
                std::cout << " pv (none)";
            }
            
            std::cout << "\n";
            std::cout.flush();
        }
    }
}

/**
 * entry point to search
 */
void search_position_threaded(thrawn::Position* rootPos, int maxDepth, int numThreads)
{
    // Reset stop flags and counters.
    total_nodes.store(0, std::memory_order_relaxed);
    stopped.store(0, std::memory_order_relaxed);
    globalSearchStartTime = get_time_ms();

    tt->incrementAge();

    // Limit the number of threads to MAX_THREADS.
    if (numThreads > MAX_THREADS)
        numThreads = MAX_THREADS;
    
    for(int i=0;i<numThreads;i++)
    {
        threadDatas[i].resetThreadData();
    }

    // Create position copies for each thread (on the heap)
    std::vector<thrawn::Position*> positionCopies;
    positionCopies.reserve(numThreads);
    for (int i = 0; i < numThreads; i++) {
        positionCopies.push_back(new thrawn::Position(*rootPos));
    }
    
    // Create worker threads
    std::vector<std::thread> workerPool;
    workerPool.reserve(numThreads);
    for (int i = 0; i < numThreads; i++) {
        workerPool.emplace_back(smp_worker_thread_func, positionCopies[i], i, maxDepth);
    }

    // Wait for all worker threads to complete.
    for (int i = 0; i < numThreads; i++)
    {
        workerPool[i].join();
    }
    
    // Delete allocated memory for position copies
    for (int i = 0; i < numThreads; i++) {
        delete positionCopies[i];
    }

    total_nodes.store(exact_node_total(numThreads), std::memory_order_relaxed);

    // After all threads have finished, select the best result.
    SearchResult result = select_best_thread(threadDatas, numThreads);
    int bestMove = result.move;
    if (!result.has_move)
        bestMove = legal_fallback_move(rootPos);
    if (result.has_move && result.thread_id != 0) {
        std::cout << "info string thread " << result.thread_id
                  << " selected by Lazy SMP vote\n";
        print_result_info(result);
    }
    
    // Print the total nodes and best move.
    //std::cout << "total nodes across all threads " << total_nodes << std::endl;
    std::cout << "bestmove ";
    if (bestMove != 0)
        print_move(bestMove);
    else
        std::cout << "0000";
    std::cout << "\n";
    std::cout.flush();
}

SearchResult select_best_thread(ThreadData threadDatas[], int numThreads) {
    struct MoveVote {
        int move = 0;
        long long votes = 0;
        int best_score = -INFINITY;
        int best_depth = 0;
        int thread_id = 0;
        int pv_length = 0;
        std::array<int, MAX_DEPTH> pv{};
    };

    SearchResult result;
    std::vector<MoveVote> moveVotes;
    int worstScore = INFINITY;
    bool foundCandidate = false;
    bool exactResultAvailable = false;

    for (int i = 0; i < numThreads; i++) {
        if (threadDatas[i].final_depth <= 0 || threadDatas[i].pv_length[0] <= 0 ||
            threadDatas[i].pv_table[0][0] == 0)
            continue;

        if (threadDatas[i].final_bound == BOUND_EXACT)
            exactResultAvailable = true;
    }

    for (int i = 0; i < numThreads; i++) {
        if (threadDatas[i].final_depth <= 0 || threadDatas[i].pv_length[0] <= 0 ||
            threadDatas[i].pv_table[0][0] == 0)
            continue;

        if (exactResultAvailable && threadDatas[i].final_bound != BOUND_EXACT)
            continue;

        worstScore = std::min(worstScore, threadDatas[i].final_score);
        foundCandidate = true;
    }

    if (foundCandidate) {
        for (int i = 0; i < numThreads; i++) {
            if (threadDatas[i].final_depth <= 0 || threadDatas[i].pv_length[0] <= 0)
                continue;

            const int move = threadDatas[i].pv_table[0][0];
            if (move == 0)
                continue;

            if (exactResultAvailable && threadDatas[i].final_bound != BOUND_EXACT)
                continue;

            MoveVote* vote = nullptr;
            for (MoveVote& existing : moveVotes) {
                if (existing.move == move) {
                    vote = &existing;
                    break;
                }
            }

            if (vote == nullptr) {
                moveVotes.emplace_back();
                vote = &moveVotes.back();
                vote->move = move;
                vote->pv.fill(0);
            }

            const int scoreGap = std::max(1,
                                          threadDatas[i].final_score - worstScore +
                                              searchParams.smpVoteScoreOffset);
            vote->votes += static_cast<long long>(std::max(1, threadDatas[i].final_depth)) * scoreGap;

            if (threadDatas[i].final_depth > vote->best_depth ||
                (threadDatas[i].final_depth == vote->best_depth &&
                 threadDatas[i].final_score > vote->best_score)) {
                vote->best_score = threadDatas[i].final_score;
                vote->best_depth = threadDatas[i].final_depth;
                vote->thread_id = i;
                vote->pv_length = threadDatas[i].pv_length[0];
                vote->pv = threadDatas[i].pv_table[0];
            }
        }

        const MoveVote* bestVote = nullptr;
        for (const MoveVote& vote : moveVotes) {
            if (bestVote == nullptr ||
                vote.votes > bestVote->votes ||
                (vote.votes == bestVote->votes && vote.best_depth > bestVote->best_depth) ||
                (vote.votes == bestVote->votes && vote.best_depth == bestVote->best_depth &&
                 vote.best_score > bestVote->best_score)) {
                bestVote = &vote;
            }
        }

        if (bestVote != nullptr) {
            result.thread = &threadDatas[bestVote->thread_id];
            result.thread_id = bestVote->thread_id;
            result.move = bestVote->move;
            result.score = bestVote->best_score;
            result.depth = bestVote->best_depth;
            result.pv_length = bestVote->pv_length;
            result.pv = bestVote->pv;
            result.has_move = true;
            return result;
        }
    }

    for (int i = 0; i < numThreads; i++) {
        if (threadDatas[i].final_depth <= 0 || threadDatas[i].pv_length[0] <= 0)
            continue;

        if (exactResultAvailable && threadDatas[i].final_bound != BOUND_EXACT)
            continue;

        if (!result.has_move ||
            threadDatas[i].final_depth > result.depth ||
            (threadDatas[i].final_depth == result.depth && threadDatas[i].final_score > result.score)) {
            result.thread = &threadDatas[i];
            result.thread_id = i;
            result.move = threadDatas[i].pv_table[0][0];
            result.score = threadDatas[i].final_score;
            result.depth = threadDatas[i].final_depth;
            result.pv_length = threadDatas[i].pv_length[0];
            result.pv = threadDatas[i].pv_table[0];
            result.has_move = result.move != 0;
        }
    }

    return result;
}
