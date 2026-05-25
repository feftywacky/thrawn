#include "perft.h"
#include "move_generator.h"
#include "constants.h"
#include "position.h"
#include "fen.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;

// ANSI escape codes for colors
const string RESET   = "\033[0m";
const string GREEN   = "\033[32m";
const string RED     = "\033[31m";
const string YELLOW  = "\033[33m";
const string BLUE    = "\033[34m";
const string MAGENTA = "\033[35m";
const string CYAN    = "\033[36m";

namespace {

std::uint64_t perft_nodes(thrawn::Position* pos, int depth)
{
    if (depth == 0)
        return 1;

    MoveList moves;
    generate_moves(pos, all_moves, moves);
    std::uint64_t nodes = 0;

    for (int move : moves)
    {
        pos->ply++;
        if (!make_move_for_perft(pos, move, pos->ply))
        {
            pos->ply--;
            continue;
        }

        if (depth == 1)
            nodes++;
        else
            nodes += perft_nodes(pos, depth - 1);

        unmake_move(pos, pos->ply);
        pos->ply--;
    }

    return nodes;
}

} // namespace

std::uint64_t perft_search(thrawn::Position* pos, int depth)
{
    return perft_nodes(pos, depth);
}

std::uint64_t perft_test(thrawn::Position* pos, int depth) {
    std::uint64_t leaf_nodes = 0;

    cout << BLUE << "\n===== PERFT TEST (Depth: " << depth << ") =====" << RESET << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    MoveList moves;
    generate_moves(pos, all_moves, moves);
    const size_t total_moves = moves.size();
    size_t moves_processed = 0;

    for (int move : moves) {
        moves_processed++;
        pos->ply++;
        if (!make_move_for_perft(pos, move, pos->ply))
        {
            pos->ply--;
        }
        else
        {
            if (depth <= 1)
                leaf_nodes++;
            else
                leaf_nodes += perft_nodes(pos, depth - 1);
            unmake_move(pos, pos->ply);
            pos->ply--;
        }

        // Update the progress bar
        const int bar_width = 30;
        const int progress = total_moves == 0
            ? bar_width
            : static_cast<int>((static_cast<double>(moves_processed) / total_moves) * bar_width);
        cout << "\r" << CYAN << "Progress: [" << RESET;
        for (int i = 0; i < bar_width; ++i) {
            if (i < progress)
                cout << "=";
            else if (i == progress)
                cout << ">";
            else
                cout << " ";
        }
        const size_t percent = total_moves == 0 ? 100 : (moves_processed * 100 / total_moves);
        cout << CYAN << "] " << RESET << percent << "%" << std::flush;
    }
    cout << "\n";

    auto duration = std::chrono::high_resolution_clock::now() - start;
    const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    const auto duration_ms = duration_us / 1000;
    const double elapsed_ms = duration_us / 1000.0;
    const double nodes_per_sec = elapsed_ms > 0.0 ? (leaf_nodes * 1000.0) / elapsed_ms : 0.0;

    cout << YELLOW << "\n===== Perft Results =====" << RESET << "\n";
    cout << "Depth:          " << depth << "\n";
    cout << "Total Nodes:    " << leaf_nodes << "\n";
    cout << "Total Time:     " << duration_ms << " ms\n";
    cout << std::fixed << std::setprecision(2);
    cout << "Nodes per Sec:  " << nodes_per_sec << "\n";
    cout << YELLOW << "=========================" << RESET << "\n\n";

    return leaf_nodes;
}

void perft_run_unit_tests() {
    thrawn::Position p;
    std::uint64_t output_nodes = 0;

    struct Test {
        const char* fen;
        int depth;
        std::uint64_t expected_nodes;
    };

    // Define your test cases (make sure these FEN strings and expected values are correct)
    Test tests[] = {
        { start_position, 6, 119060324 },
        { position_2,     5, 193690690 },
        { position_3,     7, 178633661 },
        { position_4,     5, 15833292  },
        { position_5,     5, 89941194  },
        { position_6,     5, 164075551 }
    };

    int total_tests = sizeof(tests) / sizeof(Test);
    int pass_count = 0;
    std::uint64_t total_nodes_accumulated = 0;
    std::uint64_t total_time_ms_accumulated = 0;

    cout << MAGENTA << "\n========================================" << RESET << "\n";
    cout << MAGENTA << "         PERFT UNIT TESTS             " << RESET << "\n";
    cout << MAGENTA << "========================================" << RESET << "\n\n";

    for (int i = 0; i < total_tests; i++) {
        cout << BLUE << "----------------------------------------" << RESET << "\n";
        cout << "Test " << (i + 1) << ":\n";
        cout << "FEN:             " << tests[i].fen << "\n";
        cout << "Expected Nodes:  " << tests[i].expected_nodes << "\n";

        parse_fen(&p, tests[i].fen);

        auto test_start = std::chrono::high_resolution_clock::now();
        output_nodes = perft_test(&p, tests[i].depth);
        auto test_duration = std::chrono::high_resolution_clock::now() - test_start;
        auto test_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(test_duration).count();

        if (output_nodes == tests[i].expected_nodes) {
            cout << GREEN << "Result: Passed" << RESET << "\n";
            pass_count++;
        } else {
            cout << RED << "Result: Failed" << RESET << "\n";
            cout << "Output Nodes: " << output_nodes << "\n";
        }
        cout << "Test Time:     " << test_duration_ms << " ms\n";
        cout << BLUE << "----------------------------------------" << RESET << "\n\n";

        total_nodes_accumulated += output_nodes;
        total_time_ms_accumulated += test_duration_ms;
    }

    double overall_avg_nps = (total_nodes_accumulated * 1000.0) / total_time_ms_accumulated;
    string performance_rating;
    if (overall_avg_nps >= 200e6)
        performance_rating = "Elite";
    else if (overall_avg_nps >= 100e6)
        performance_rating = "Blazingly FAST";
    else if (overall_avg_nps >= 50e6)
        performance_rating = "ABOVE Excellent";
    else if (overall_avg_nps >= 20e6)
        performance_rating = "Excellent";
    else if (overall_avg_nps >= 10e6)
        performance_rating = "Good";
    else if (overall_avg_nps >= 5e6)
        performance_rating = "Average";
    else
        performance_rating = "Below Average";

    cout << MAGENTA << "========================================" << RESET << "\n";
    cout << MAGENTA << "        PERFT UNIT TEST SUMMARY         " << RESET << "\n";
    cout << MAGENTA << "========================================" << RESET << "\n";
    cout << "Passed:                 " << GREEN << pass_count << RESET << "/" << total_tests << "\n";
    cout << "Total Nodes:            " << total_nodes_accumulated << "\n";
    cout << "Total Time:             " << total_time_ms_accumulated << " ms\n";
    cout << std::fixed << std::setprecision(2);
    cout << "Average Nodes per Sec:  " << overall_avg_nps << "\n";
    cout << "Performance Rating:     " << CYAN << performance_rating << RESET << "\n";
    cout << MAGENTA << "========================================" << RESET << "\n\n";
}
