#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstdint>
#include <array>
#include <string>
#include <unordered_map>

using namespace std;

// ----------------------------------------
// Some constants used in the search
// ----------------------------------------
#define INFINITY 50000
#define mateVal 49000
#define mateScore 48000
#define MAX_DEPTH 64

constexpr int KILLER_MOVES   = 2;
constexpr int HISTORY_SIZE   = 12;
constexpr int MAX_THREADS    = 16;
constexpr int NODE_COUNTER_BATCH = 1024;

constexpr std::array<int, 6> PIECE_VALUES = {100, 320, 330, 500, 900, 20000};

constexpr int SEARCH_ASPIRATION_WINDOW_DEPTH = 4;
constexpr int SEARCH_ASPIRATION_WINDOW_SIZE = 24;
constexpr int SEARCH_ASPIRATION_THREAD_DELTA = 2;
constexpr int SEARCH_ASPIRATION_THREAD_CYCLE = 4;
constexpr int SEARCH_CHECK_EXTENSION = 1;

constexpr int SEARCH_HISTORY_MAX = 16384;
constexpr int SEARCH_HISTORY_SCORE_CAP = 6000;
constexpr int SEARCH_HISTORY_BONUS_DEPTH_SQUARED = 16;
constexpr int SEARCH_HISTORY_BONUS_DEPTH_LINEAR = 64;
constexpr int SEARCH_CONTINUATION_HISTORY_NUMERATOR = 3;
constexpr int SEARCH_CONTINUATION_HISTORY_DENOMINATOR = 4;
constexpr int SEARCH_CORRECTION_HISTORY_SIZE = 16384;
constexpr int SEARCH_CORRECTION_HISTORY_MAX = 8192;
constexpr int SEARCH_CORRECTION_HISTORY_GRAIN = 16;
constexpr int SEARCH_CORRECTION_HISTORY_WEIGHT_SCALE = 32;

constexpr int SEARCH_COUNTER_MOVE_SCORE = 7000;
constexpr int SEARCH_COUNTER_MOVE_HISTORY_DIVISOR = 64;
constexpr int SEARCH_COUNTER_MOVE_HISTORY_CAP = 250;

constexpr int SEARCH_QUEEN_PROMOTION_SCORE = 10499;
constexpr int SEARCH_KILLER_MOVE_SCORE_1 = 9000;
constexpr int SEARCH_KILLER_MOVE_SCORE_2 = 8000;

constexpr int SEARCH_REVERSE_FUTILITY_MAX_DEPTH = 2;
constexpr int SEARCH_REVERSE_FUTILITY_MARGIN_1 = 160;
constexpr int SEARCH_REVERSE_FUTILITY_MARGIN_2 = 300;
constexpr int SEARCH_REVERSE_FUTILITY_DEPTH_FACTOR = 110;

constexpr int SEARCH_RAZOR_MAX_DEPTH = 2;
constexpr int SEARCH_RAZOR_MARGIN_1 = 250;
constexpr int SEARCH_RAZOR_MARGIN_2 = 450;
constexpr int SEARCH_RAZOR_MARGIN_DEPTH_N = 600;

constexpr int SEARCH_NULL_MOVE_MIN_DEPTH = 4;
constexpr int SEARCH_NULL_MOVE_BASE_REDUCTION = 2;
constexpr int SEARCH_NULL_MOVE_DEPTH_DIVISOR = 6;
constexpr int SEARCH_NULL_MOVE_EVAL_DIVISOR = 400;
constexpr int SEARCH_NULL_MOVE_EVAL_BONUS_MAX = 1;
constexpr int SEARCH_NULL_MOVE_VERIFICATION_DEPTH = 8;

constexpr int SEARCH_FUTILITY_MAX_DEPTH = 3;
constexpr int SEARCH_FUTILITY_MARGIN_1 = 120;
constexpr int SEARCH_FUTILITY_MARGIN_2 = 220;
constexpr int SEARCH_FUTILITY_MARGIN_3 = 360;

constexpr int SEARCH_LATE_MOVE_PRUNING_MAX_DEPTH = 3;
constexpr int SEARCH_LATE_MOVE_PRUNING_DEPTH_1 = 8;
constexpr int SEARCH_LATE_MOVE_PRUNING_DEPTH_2 = 12;
constexpr int SEARCH_LATE_MOVE_PRUNING_DEPTH_3 = 24;

constexpr int SEARCH_LMR_FULL_DEPTH_MOVES = 3;
constexpr int SEARCH_LMR_REDUCTION_DEPTH_LIMIT = 3;
constexpr int SEARCH_LMR_BASE_REDUCTION = 0;
constexpr int SEARCH_LMR_NON_PV_DEPTH = 5;
constexpr int SEARCH_LMR_MOVE_DEPTH_1 = 6;
constexpr int SEARCH_LMR_MOVE_NUMBER_1 = 8;
constexpr int SEARCH_LMR_MOVE_DEPTH_2 = 8;
constexpr int SEARCH_LMR_MOVE_NUMBER_2 = 16;
constexpr int SEARCH_LMR_GOOD_HISTORY_DIVISOR = 4;
constexpr int SEARCH_LMR_BAD_HISTORY_DIVISOR = 4;

constexpr int SEARCH_SINGULAR_EXTENSION = 1;
constexpr int SEARCH_SINGULAR_EXTENSION_MIN_DEPTH = 6;
constexpr int SEARCH_SINGULAR_EXTENSION_DEPTH_MARGIN = 2;
constexpr int SEARCH_SINGULAR_EXTENSION_BASE_MARGIN = 32;
constexpr int SEARCH_SINGULAR_EXTENSION_DEPTH_FACTOR = 8;

constexpr int SEARCH_PROBCUT_MIN_DEPTH = 5;
constexpr int SEARCH_PROBCUT_REDUCTION = 3;
constexpr int SEARCH_PROBCUT_MARGIN = 160;
constexpr int SEARCH_PROBCUT_SEE_MARGIN = 0;

constexpr int SEARCH_QSEARCH_DELTA_MARGIN = 200;
constexpr int SEARCH_SEE_PRUNE_MAX_DEPTH = 4;
constexpr int SEARCH_SEE_PRUNE_DEPTH_MARGIN = 200;

constexpr int SEARCH_LMR_LOG_DIVISOR = 5;
constexpr int SEARCH_TACTICAL_CAPTURE_BASE_SCORE = 10000;
constexpr int SEARCH_TACTICAL_VICTIM_MULTIPLIER = 8;
constexpr int SEARCH_TACTICAL_ATTACKER_DIVISOR = 8;
constexpr int SEARCH_TACTICAL_CAPTURE_HISTORY_DIVISOR = 4;
constexpr int SEARCH_TACTICAL_QUEEN_PROMOTION_BONUS = 800;
constexpr int SEARCH_TACTICAL_UNDERPROMOTION_OFFSET = 1000;
constexpr int SEARCH_BAD_CAPTURE_BASE_SCORE = -5000;
constexpr int SEARCH_BAD_CAPTURE_VICTIM_DIVISOR = 2;
constexpr int SEARCH_BAD_CAPTURE_HISTORY_DIVISOR = 8;
constexpr int SEARCH_BAD_CAPTURE_SEE_FLOOR = -2000;

constexpr int SEARCH_SMP_VOTE_SCORE_OFFSET = 14;

enum{
    white,
    black,
    both
};

// encode pieces
// white ->-> black
enum {P, N, B, R, Q, K, p, n, b, r, q, k};

enum {rook, bishop};

enum {all_moves, only_captures, only_quiets, only_checks};

enum {wks=1, wqs=2, bks=4, bqs=8};

enum {
    a8, b8, c8, d8, e8, f8, g8, h8,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a1, b1, c1, d1, e1, f1, g1, h1, null_sq
};


constexpr int BOARD_SIZE = 64;


// CONSTANTS (only declarations)

// ASCII pieces
extern const array<char, 12>ascii_pieces;
extern const array<string, 12> unicode_pieces;

extern const unordered_map<char, int> char_pieces;

extern const unordered_map<int, char> promoted_pieces;



extern const uint64_t not_a_file;
extern const uint64_t not_h_file;
extern const uint64_t not_hg_file;
extern const uint64_t not_ab_file;
extern const std::array<const char*, 64> square_to_coordinates;

extern const std::array<int, 64> bishop_relevant_bits;
extern const std::array<int, 64> rook_relevant_bits;

extern const std::array<int, 64> update_castling_right_values;

extern std::array<uint64_t, 64> rook_magic_nums;
extern std::array<uint64_t, 64> bishop_magic_nums;

// move ordering
// [attacker][victim]
extern const std::array<std::array<int, 12>, 12> mvv_lva;

// FEN position test cases
extern const char* empty_board;
extern const char* start_position;
extern const char* position_2;
extern const char* position_3;
extern const char* position_4;
extern const char* position_5;
extern const char* position_6;

#endif
