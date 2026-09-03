#ifdef _WIN32
#include <windows.h>
#endif
#include "uci.h"
#include "move_generator.h"
#include "move_helpers.h"
#include "bitboard_helpers.h"
#include "transposition_table.h"
#include "bitboard.h"
#include "fen.h"
#include "perft.h"
#include "search.h"
#include "evaluation.h"
#include "misc.h"
#include "globals.h"
#include "constants.h"
#include "nnue.h"
#include "timeman.h"
#include "threading.h"
#include <stdlib.h>
#include <vector>
#include <cstring>
#include <cctype>
#include <string>
#include <chrono>
#include <sstream>
#include <atomic>
#include <unistd.h>
#include <stdio.h>
#include <algorithm>
#include <cstdint>
#ifndef _WIN32
#include <termios.h>
#include <sys/ioctl.h>
#endif

using namespace std;

/*
TIMING CONTROL AND UCI
UCI PROTOCOL CODE REFERENCES TO VICE CHESS ENGINE BY RICHARD ALBERT
*/

// exit from engine flag
std::atomic<int> quit{0};

// The clock lives in `timeMan` (timeman.h), derived once per `go`.

// variable to flag when the time is up
std::atomic<int> stopped{0};

// Number of threads use for search
int numThreads = 1;

// Requested table size. The table itself is allocated lazily (see
// ensure_tt_allocated) so a process that is about to be told a different Hash
// size never touches the default one.
static int tt_size_mb = TT_DEFAULT_MB;

static void ensure_tt_allocated() {
    if (!tt->isAllocated())
        tt->initTable(tt_size_mb);
}

// While `bench` drives searches internally, stdin must be left alone:
// read_input() consumes whole buffered chunks and would swallow the commands
// queued behind the bench.
static bool bench_running = false;

static std::string trim_option_value(const char* value) {
    if (value == nullptr) {
        return {};
    }

    std::string text(value);
    std::size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }

    std::size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }

    return text.substr(start, end - start);
}

static bool parse_uci_setoption(const char* command,
                                std::string& name,
                                std::string& value) {
    const std::string text = trim_option_value(command);
    const std::string prefix = "setoption name ";
    if (text.rfind(prefix, 0) != 0) {
        return false;
    }

    const std::string rest = text.substr(prefix.size());
    const std::string value_marker = " value ";
    const std::size_t value_pos = rest.find(value_marker);

    if (value_pos == std::string::npos) {
        name = trim_option_value(rest.c_str());
        value.clear();
    } else {
        name = trim_option_value(rest.substr(0, value_pos).c_str());
        value = trim_option_value(rest.substr(value_pos + value_marker.size()).c_str());
    }

    return !name.empty();
}

static bool option_name_equals(const std::string& lhs, const char* rhs) {
    std::size_t rhs_len = std::strlen(rhs);
    if (lhs.size() != rhs_len) {
        return false;
    }

    for (std::size_t i = 0; i < lhs.size(); i++) {
        if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
            std::tolower(static_cast<unsigned char>(rhs[i]))) {
            return false;
        }
    }

    return true;
}

static void uci_report_raw_nnue(const char* command) {
    if (!nnue_loaded()) {
        std::cout << "info string NNUE not loaded\n";
        return;
    }

    const std::string fen = trim_option_value(command + 7);
    if (fen.empty()) {
        std::cout << "info string usage: nnuefen <fen>\n";
        return;
    }

    thrawn::Position scratch;
    parse_fen(&scratch, fen.c_str());

    std::cout << "info string nnue score_stm " << nnue_evaluate_raw(&scratch) << "\n";
}

static void uci_report_engine_nnue(const char* command) {
    if (!nnue_loaded()) {
        std::cout << "info string NNUE not loaded\n";
        return;
    }

    const std::string fen = trim_option_value(command + 13);
    if (fen.empty()) {
        std::cout << "info string usage: nnuefenengine <fen>\n";
        return;
    }

    thrawn::Position scratch;
    parse_fen(&scratch, fen.c_str());

    std::cout << "info string engine nnue score " << evaluate(&scratch) << "\n";
}

static std::string uci_move_to_string(int move) {
    std::string text;
    text += square_to_coordinates[get_move_source(move)];
    text += square_to_coordinates[get_move_target(move)];

    const int promoted_piece = get_promoted_piece(move);
    if (promoted_piece != 0) {
        text += static_cast<char>(std::tolower(ascii_pieces[promoted_piece]));
    }
    return text;
}

static bool uci_nnue_verify_recursive(thrawn::Position* pos,
                                      int depth,
                                      int max_nodes,
                                      int& visited_nodes,
                                      float& max_parity_error_cp,
                                      std::string& error) {
    if (!nnue_verify_position(pos, &error)) {
        return false;
    }
    float parity_error_cp = 0.0f;
    if (!nnue_measure_evaluation_parity(pos, &parity_error_cp, &error)) {
        return false;
    }
    max_parity_error_cp = std::max(max_parity_error_cp, parity_error_cp);

    if (depth <= 0 || visited_nodes >= max_nodes || pos->ply >= MAX_DEPTH - 1) {
        return true;
    }

    ++visited_nodes;

    const bool in_check = is_square_under_attack(
        pos,
        (pos->colour_to_move == white)
            ? get_lsb_index(pos->piece_bitboards[K])
            : get_lsb_index(pos->piece_bitboards[k]),
        pos->colour_to_move ^ 1
    );

    if (!in_check && !noMajorsOrMinorsPieces(pos)) {
        pos->ply++;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;
        make_null_move(pos, pos->ply);

        if (!uci_nnue_verify_recursive(pos, depth - 1, max_nodes, visited_nodes, max_parity_error_cp, error)) {
            error = "after null move: " + error;
            unmake_null_move(pos, pos->ply);
            pos->ply--;
            pos->repetition_index--;
            return false;
        }

        unmake_null_move(pos, pos->ply);
        pos->ply--;
        pos->repetition_index--;
    }

    MoveList moves;
    generate_moves(pos, all_moves, moves);
    for (int move : moves) {
        if (visited_nodes >= max_nodes) {
            break;
        }

        pos->ply++;
        pos->repetition_index++;
        pos->repetition_table[pos->repetition_index] = pos->zobristKey;

        if (!make_move_on_board(pos, move, all_moves, pos->ply)) {
            pos->ply--;
            pos->repetition_index--;
            continue;
        }

        if (!uci_nnue_verify_recursive(pos, depth - 1, max_nodes, visited_nodes, max_parity_error_cp, error)) {
            error = "after move " + uci_move_to_string(move) + ": " + error;
            unmake_move(pos, pos->ply);
            pos->ply--;
            pos->repetition_index--;
            return false;
        }

        unmake_move(pos, pos->ply);
        pos->ply--;
        pos->repetition_index--;
    }

    return true;
}

static void uci_verify_nnue_search(const char* command, thrawn::Position* pos) {
    if (!nnue_loaded()) {
        std::cout << "info string NNUE not loaded\n";
        return;
    }

    int depth = 3;
    const std::string args = trim_option_value(command + 10);
    if (!args.empty()) {
        depth = std::max(0, std::atoi(args.c_str()));
    }

    constexpr int kMaxVerifyNodes = 5000;
    int visited_nodes = 0;
    float max_parity_error_cp = 0.0f;
    std::string error;
    const bool ok = uci_nnue_verify_recursive(pos,
                                              depth,
                                              kMaxVerifyNodes,
                                              visited_nodes,
                                              max_parity_error_cp,
                                              error);

    if (ok) {
        std::cout << "info string NNUE verify ok depth " << depth
                  << " nodes " << visited_nodes
                  << " parity_cp " << max_parity_error_cp << "\n";
    } else {
        std::cout << "info string NNUE verify failed depth " << depth
                  << " nodes " << visited_nodes
                  << " reason " << error << "\n";
    }
}

/*
TIME CONTROL
*/
std::int64_t get_time_ms() {
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int input_waiting() {
#ifdef _WIN32
    static int init = 0, pipe;
    static HANDLE inh;
    DWORD dw;

    if (!init) {
        init = 1;
        inh = GetStdHandle(STD_INPUT_HANDLE);
        pipe = !GetConsoleMode(inh, &dw);
        if (!pipe) {
            SetConsoleMode(inh, dw & ~(ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT));
            FlushConsoleInputBuffer(inh);
        }
    }

    if (pipe) {
        if (!PeekNamedPipe(inh, NULL, 0, NULL, &dw, NULL)) return 1;
        return dw;
    } else {
        GetNumberOfConsoleInputEvents(inh, &dw);
        return dw <= 1 ? 0 : dw;
    }
#else
    int bytesAvailable;
    ioctl(fileno(stdin), FIONREAD, &bytesAvailable);
    return bytesAvailable;
#endif
}

void read_input() {
    // Bytes to read holder
    int bytes;

    // GUI/user input
    char input[256] = "", *endc;

    // "Listen" to STDIN
    if (input_waiting()) {
        // Loop to read bytes from STDIN
        do {
            bytes = read(fileno(stdin), input, sizeof(input) - 1);
        }

        // Until bytes are available
        while (bytes < 0);

        if (bytes <= 0) {
            return;
        }

        input[bytes] = '\0';

        // Searches for the first occurrence of '\n'
        endc = strchr(input, '\n');

        // If a newline character is found, set its value at the pointer to 0
        if (endc)
            *endc = 0;

        // If input is available, stop only for explicit UCI stop commands.
        if (strlen(input) > 0) {
            // Match UCI "quit" command
            if (!strncmp(input, "quit", 4)) {
                // Tell the engine to terminate execution
                quit.store(1, std::memory_order_relaxed);
                stopped.store(1, std::memory_order_relaxed);
            }

            // Match UCI "stop" command
            else if (!strncmp(input, "stop", 4)) {
                // Tell the engine to terminate execution
                stopped.store(1, std::memory_order_relaxed);
            }
        }
    }
}

// a bridge function to interact between search and GUI input
void communicate() {
    // The hard bound. Depth 1 is exempt, so we return a searched move rather
    // than a fallback one; `panic` bounds that exemption at the flag.
    if (timeMan.hard_expired() &&
        (main_completed_depth.load(std::memory_order_relaxed) > 0 ||
         timeMan.panic_expired())) {
        stopped.store(1, std::memory_order_relaxed);
    }

    // read GUI input
	if (!bench_running)
		read_input();
}

/*
UCI PROTOCOL
*/

int uci_parse_move(thrawn::Position* pos, const char *move_str)
{
    MoveList moves;
    generate_moves(pos, all_moves, moves);
    
    int source = (move_str[0] - 'a') + (8-(move_str[1]- '0')) * 8;
    int target = (move_str[2] - 'a') + (8-(move_str[3]- '0')) * 8;
    int promoted_piece = 0;

    for (int move : moves)
    {
        if (source == get_move_source(move) && target == get_move_target(move))
        {
            promoted_piece = get_promoted_piece(move);

            if (promoted_piece)
            {
                if (move_str[4] == promoted_pieces.at(promoted_piece))
                    return move;
                continue;
            }

            // legal move
            return move;
        }
    }

    // illegal move ie. puts king in check
    return 0;
}

void uci_parse_position(thrawn::Position* pos, const char *command) {
    // Create a non-const pointer for manipulation
    const char *non_const_command = command;

    non_const_command += 9; // shift index to skip 'position' in command

    const char *curr_ch = non_const_command;

    // parse 'startpos' command
    
    if (strncmp(non_const_command, "startpos", 8) == 0)
        parse_fen(pos, start_position);

    // parse 'fen' command
    else 
    {
        curr_ch = strstr(non_const_command, "fen");

        // no 'fen' command found
        if (curr_ch == nullptr)
            parse_fen(pos, start_position);

        else {
            // shift index to next token
            curr_ch += 4;
            parse_fen(pos, curr_ch);
        }
    }

    // parse moves after position is set up
    curr_ch = strstr(non_const_command, "moves");
    
    // if moves are found
    if (curr_ch != nullptr) {

        curr_ch += 6;

        while (*curr_ch) {
            int move = uci_parse_move(pos, curr_ch);

            if (move == 0)
                break;

            pos->repetition_index++;
            pos->repetition_table[pos->repetition_index] = pos->zobristKey;

            make_root_move(pos, move, all_moves);

            // Move index to the end of the current move
            while (*curr_ch && *curr_ch != ' ')
                curr_ch++;

            // go to next move
            curr_ch++;
        }

    }

    // for debug
    // print_board(colour_to_move);

}

void uci_parse_go(thrawn::Position* pos, const char* command)
{
    ensure_tt_allocated();
    reset_time_control();

    int depth = -1;
    int uci_time = -1;   // wtime/btime for the side to move
    int inc = 0;         // winc/binc for the side to move
    int movestogo = 0;   // 0 means sudden death / Fischer
    int movetime = -1;

    // Infinite search
    if (strstr(command, "infinite") != nullptr) {}

    // Match UCI "binc" command
    if (strstr(command, "binc") != nullptr && pos->colour_to_move == 1) {
        // Parse black time increment
        inc = atoi(strstr(command, "binc") + 5);
    }

    // Match UCI "winc" command
    if (strstr(command, "winc") != nullptr && pos->colour_to_move == 0) {
        // Parse white time increment
        inc = atoi(strstr(command, "winc") + 5);
    }

    // Match UCI "wtime" command
    if (strstr(command, "wtime") != nullptr && pos->colour_to_move == 0) {
        // Parse white time limit
        uci_time = atoi(strstr(command, "wtime") + 6);
    }

    // Match UCI "btime" command
    if (strstr(command, "btime") != nullptr && pos->colour_to_move == 1) {
        // Parse black time limit
        uci_time = atoi(strstr(command, "btime") + 6);
    }

    // Match UCI "movestogo" command
    if (strstr(command, "movestogo") != nullptr) {
        // Parse number of moves to go
        movestogo = atoi(strstr(command, "movestogo") + 10);
    }

    // Match UCI "movetime" command
    if (strstr(command, "movetime") != nullptr) {
        // Parse amount of time allowed to spend to make a move
        movetime = atoi(strstr(command, "movetime") + 9);
    }

    // Match UCI "depth" command
    if (strstr(command, "depth") != nullptr) {
        // Parse search depth
        depth = atoi(strstr(command, "depth") + 6);
    }

    // Only the one-legal-move case reads this. Stamps the clock's start point.
    const int rootMoveCount = count_legal_moves(pos);
    timeMan.init(uci_time, inc, movestogo, movetime, rootMoveCount);

    // Clamped either way: the search stacks are MAX_DEPTH deep, so `go depth
    // 100` would index past them.
    depth = depth == -1 ? MAX_DEPTH : std::clamp(depth, 1, MAX_DEPTH);

    // Print debug info
    std::cout << "info string time " << uci_time << " start " << timeMan.start
              << " soft " << timeMan.soft << " hard " << timeMan.hard
              << " depth " << depth
              << " timeset " << (timeMan.timeset ? 1 : 0) << std::endl;

    std::cout << "info depth 0 nodes 0 time 0 score cp 0 pv none"<<endl;
    search_position_threaded(pos, depth, numThreads);  
}

// Fixed position set for `bench`. Node counts are reproducible for a given
// depth, so a search-neutral refactor must leave them unchanged, and the nps
// figure is a stable speed measurement.
static const char* BENCH_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 1",
    "4rrk1/pp1n1pp1/3bp2p/q2p4/2PP4/1P1BPN2/P4PPP/R2Q1RK1 w - - 0 1",
    "r1bqkb1r/pp3ppp/2n1pn2/2pp4/3P1B2/2P1PN2/PP1N1PPP/R2QKB1R w KQkq - 0 1",
    "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 1",
    "3r1rk1/p3qppp/1pn1pn2/2bp4/8/1PN1PN2/PBQP1PPP/3R1RK1 w - - 0 1",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1",
    "8/8/8/8/8/8/6k1/4K2R w K - 0 1",
    "8/3k4/8/8/8/8/4P3/4K3 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
    "8/pp3k2/2p2p2/3p4/3P4/2P2P2/PP3K2/8 w - - 0 1",
};

static void uci_bench(thrawn::Position* pos, int depth, int threads, int hashMb)
{
    const int savedThreads = numThreads;
    const int benchThreads = std::clamp(threads, 1, MAX_THREADS);
    numThreads = benchThreads;
    tt->initTable(hashMb);

    bench_running = true;
    reset_time_control();   // fixed depth only: no clock, no move-time budget

    const std::size_t count = sizeof(BENCH_FENS) / sizeof(BENCH_FENS[0]);
    std::uint64_t nodes = 0;
    const std::int64_t begin = get_time_ms();

    for (std::size_t i = 0; i < count; i++) {
        // Clear between positions so each node count depends only on the
        // position and the depth, not on what the previous search left behind.
        tt->reset();
        parse_fen(pos, BENCH_FENS[i]);
        nnue_refresh_root(pos);
        stopped.store(0, std::memory_order_relaxed);
        // No clock: search_position_threaded() restarts it for the info lines.

        std::cout << "info string bench position " << (i + 1) << "/" << count << "\n";
        search_position_threaded(pos, depth, numThreads);
        nodes += total_nodes.load(std::memory_order_relaxed);
    }

    const std::int64_t elapsed = std::max<std::int64_t>(1, get_time_ms() - begin);

    bench_running = false;
    reset_time_control();
    numThreads = savedThreads;

    std::cout << "\n===========================\n"
              << "Depth           : " << depth << "\n"
              << "Threads         : " << benchThreads << "\n"
              << "Hash            : " << hashMb << " MB\n"
              << "Total time (ms) : " << elapsed << "\n"
              << "Nodes searched  : " << nodes << "\n"
              << "Nodes/second    : " << (nodes * 1000 / static_cast<std::uint64_t>(elapsed))
              << "\n\n"
              << nodes << " nodes " << (nodes * 1000 / static_cast<std::uint64_t>(elapsed))
              << " nps\n";
    std::cout.flush();
}

void uci_loop(thrawn::Position* pos)
{
    // just make it big enough
    #define INPUT_BUFFER 20000
    
    // reset STDIN & STDOUT buffers
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);
    
    // define user / GUI input buffer
    char input[INPUT_BUFFER];

    while (true)
    {
        // reset user /GUI input
        memset(input, 0, sizeof(input));
        
        // make sure output reaches the GUI
        fflush(stdout);
        
        // get user / GUI input
        if (!fgets(input, INPUT_BUFFER, stdin))
            break;
        
        // make sure input is available
        if (input[0] == '\n')
            continue;
        
        // parse UCI "isready" command
        if (strncmp(input, "isready", 7) == 0)
        {
            ensure_tt_allocated();
            std::cout << "readyok\n";
            continue;
        }
        
        // parse UCI "position" command
        else if (strncmp(input, "position", 8) == 0)
        {
            uci_parse_position(pos, input);
        }

        // parse UCI "ucinewgame" command
        else if (strncmp(input, "ucinewgame", 10) == 0)
        {
            ensure_tt_allocated();
            tt->reset();
            // No previous move to draw a score trend from.
            previous_root_score = TM_SCORE_UNKNOWN;
            uci_parse_position(pos, "position startpos");
        }
        // parse UCI "go" command
        else if (strncmp(input, "go", 2) == 0) {
            uci_parse_go(pos, input);
            if (quit.load(std::memory_order_relaxed) == 1)
                break;
        }
        
        // `bench [depth] [threads] [hash_mb]` - fixed-position speed/node test.
        else if (strncmp(input, "bench", 5) == 0)
        {
            int benchDepth = 13, benchThreads = 1, benchHash = 16;
            std::istringstream args(input + 5);
            int value = 0;
            if (args >> value && value > 0) benchDepth = value;
            if (args >> value && value > 0) benchThreads = value;
            if (args >> value && value > 0) benchHash = value;
            benchHash = std::clamp(benchHash, TT_MIN_MB, TT_MAX_MB);
            uci_bench(pos, benchDepth, benchThreads, benchHash);
            tt->initTable(tt_size_mb);
        }

        // parse UCI "quit" command
        else if (strncmp(input, "quit", 4) == 0)
            break;
        
        // parse UCI "uci" command
        else if (strncmp(input, "uci", 3) == 0)
        {
            // print engine info
            cout << "id name Thrawn"<< version << "\n";
            cout << "id author Feiyu Lin\n";
            cout << "option name Hash type spin default " << TT_DEFAULT_MB
                 << " min " << TT_MIN_MB << " max " << TT_MAX_MB << "\n";
            cout << "option name Threads type spin default 1 min 1 max "
                 << MAX_THREADS << "\n";
            cout << "option name Move Overhead type spin default "
                 << TM_MOVE_OVERHEAD_DEFAULT << " min " << TM_MOVE_OVERHEAD_MIN
                 << " max " << TM_MOVE_OVERHEAD_MAX << "\n";
            cout << "option name EvalFile type string default thrawn-nn-2.nnue" << "\n";
            cout << "uciok\n";
        }
        
        else if (!strncmp(input, "setoption name ", 15)) {
            std::string optionName;
            std::string optionValue;
            if (!parse_uci_setoption(input, optionName, optionValue)) {
                std::cout << "info string Ignored malformed setoption command\n";
                continue;
            }

            if (option_name_equals(optionName, "Hash")) {
                tt_size_mb = std::clamp(
                    optionValue.empty() ? tt_size_mb : std::atoi(optionValue.c_str()),
                    TT_MIN_MB, TT_MAX_MB);

                std::cout << "info string Set hash table size to " << tt_size_mb << "MB\n";
                tt->initTable(tt_size_mb);
            }
            else if (option_name_equals(optionName, "Threads")) {
                int t = optionValue.empty() ? numThreads : std::atoi(optionValue.c_str());
                numThreads = std::clamp(t, 1, MAX_THREADS);
                std::cout << "info string Set threads = " << numThreads << std::endl;
            }
            else if (option_name_equals(optionName, "Move Overhead")) {
                move_overhead = std::clamp(
                    optionValue.empty() ? move_overhead : std::atoi(optionValue.c_str()),
                    TM_MOVE_OVERHEAD_MIN, TM_MOVE_OVERHEAD_MAX);

                std::cout << "info string Set move overhead to " << move_overhead
                          << "ms\n";
            }
            else if (option_name_equals(optionName, "EvalFile")) {
                std::string path = optionValue;
                if (path.empty()) {
                    path = "thrawn-nn-2.nnue";
                }
                nnue_init(path.c_str());
                nnue_refresh_root(pos);
            }
            else {
                std::cout << "info string Unknown option " << optionName << std::endl;
            }
        }

        else if (strncmp(input, "nnuefenengine", 13) == 0)
        {
            uci_report_engine_nnue(input);
        }

        else if (strncmp(input, "nnuefen", 7) == 0)
        {
            uci_report_raw_nnue(input);
        }

        else if (strncmp(input, "nnueverify", 10) == 0)
        {
            uci_verify_nnue_search(input, pos);
        }

        else if (strncmp(input, "perft", 5) == 0)
        {
            perft_run_unit_tests();
        }
    }
}

void reset_time_control()
{
    quit.store(0, std::memory_order_relaxed);
    timeMan.clear();
    stopped.store(0, std::memory_order_relaxed);
}
