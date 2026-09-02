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
constexpr int SEARCH_INFINITY = 50000;
#define mateVal 49000
#define mateScore 48000
#define MAX_DEPTH 64

constexpr int KILLER_MOVES   = 2;
constexpr int HISTORY_SIZE   = 12;
constexpr int MAX_THREADS    = 16;

// UCI "Hash" option bounds, in MB. Declared in the `uci` handshake, used to
// clamp `setoption`, and used for the startup allocation, so all three agree.
constexpr int TT_DEFAULT_MB  = 256;
constexpr int TT_MIN_MB      = 1;
constexpr int TT_MAX_MB      = 16384;
constexpr int NODE_COUNTER_BATCH = 1024;

constexpr std::array<int, 6> PIECE_VALUES = {100, 320, 330, 500, 900, 20000};

constexpr int SEARCH_ASPIRATION_WINDOW_DEPTH = 4;
constexpr int SEARCH_ASPIRATION_WINDOW_SIZE = 12;
constexpr int SEARCH_ASPIRATION_THREAD_DELTA = 2;
constexpr int SEARCH_ASPIRATION_THREAD_CYCLE = 4;
constexpr int SEARCH_CHECK_EXTENSION = 1;

// History bonus/malus are linear in depth and capped well below the table
// limit, so a single deep update cannot saturate an entry.
constexpr int SEARCH_HISTORY_MAX = 16384;
constexpr int SEARCH_HISTORY_BONUS_DEPTH = 180;
constexpr int SEARCH_HISTORY_BONUS_BIAS = 16;
constexpr int SEARCH_HISTORY_BONUS_MAX = 1500;
constexpr int SEARCH_HISTORY_MALUS_DEPTH = 260;
constexpr int SEARCH_HISTORY_MALUS_BIAS = -30;
constexpr int SEARCH_HISTORY_MALUS_MAX = 1900;
constexpr int SEARCH_CONTINUATION_HISTORY_NUMERATOR = 3;
constexpr int SEARCH_CONTINUATION_HISTORY_DENOMINATOR = 4;
// Continuation history looks 1 and 2 plies back; the older ply carries less.
constexpr int SEARCH_CONTINUATION_HISTORY_PLIES = 2;
constexpr int SEARCH_CONTINUATION_HISTORY_DECAY_NUMERATOR = 3;
constexpr int SEARCH_CONTINUATION_HISTORY_DECAY_DENOMINATOR = 4;

// Correction history is keyed on pawn structure, so entries generalize across
// positions that share it. Applied as weight * entry / 512.
constexpr int SEARCH_CORRECTION_HISTORY_SIZE = 16384;
constexpr int SEARCH_CORRECTION_HISTORY_MAX = 1024;
constexpr int SEARCH_CORRECTION_HISTORY_WEIGHT = 64;
constexpr int SEARCH_CORRECTION_HISTORY_DEPTH_DIVISOR = 8;

constexpr int SEARCH_QUEEN_PROMOTION_SCORE = 10499;

// Reverse futility: margin scales with depth, one depth step cheaper when the
// eval trend is with us. No lower depth bound beyond the floor.
constexpr int SEARCH_REVERSE_FUTILITY_MAX_DEPTH = 11;
constexpr int SEARCH_REVERSE_FUTILITY_DEPTH_MUL = 87;
constexpr int SEARCH_REVERSE_FUTILITY_MIN = 22;

// Razoring: linear in depth and applied at every depth, since the margin
// outruns any real eval gap long before the depth matters.
constexpr int SEARCH_RAZOR_DEPTH_FACTOR = 352;

constexpr int SEARCH_NULL_MOVE_MIN_DEPTH = 3;
constexpr int SEARCH_NULL_MOVE_BASE_REDUCTION = 4;
constexpr int SEARCH_NULL_MOVE_DEPTH_DIVISOR = 3;
constexpr int SEARCH_NULL_MOVE_EVAL_DIVISOR = 120;
constexpr int SEARCH_NULL_MOVE_EVAL_BONUS_MAX = 4;
constexpr int SEARCH_NULL_MOVE_VERIFICATION_DEPTH = 16;

// Move-loop futility, keyed on the LMR-reduced depth rather than node depth.
constexpr int SEARCH_FUTILITY_MAX_DEPTH = 10;
constexpr int SEARCH_FUTILITY_BASE_MARGIN = 159;
constexpr int SEARCH_FUTILITY_DEPTH_FACTOR = 153;

// Late move pruning: (BASE + depth^2) / (2 - improving) over all moves seen.
// The quadratic bounds its own reach, so no separate depth limit.
constexpr int SEARCH_LATE_MOVE_PRUNING_BASE = 3;

constexpr int SEARCH_HISTORY_PRUNING_MAX_DEPTH = 6;
constexpr int SEARCH_HISTORY_PRUNING_DEPTH_MARGIN = 5900;

// LMR starts at the second move (fourth at the root) from depth 2 up.
constexpr int SEARCH_LMR_FIRST_MOVE = 1;
constexpr int SEARCH_LMR_ROOT_EXTRA_MOVES = 2;
constexpr int SEARCH_LMR_MIN_DEPTH = 2;
constexpr int SEARCH_LMR_CUT_NODE_REDUCTION = 2;
constexpr int SEARCH_LMR_QUIET_HISTORY_DIVISOR = 10900;
constexpr int SEARCH_LMR_CAPTURE_HISTORY_DIVISOR = 4096;
constexpr int SEARCH_LMR_HISTORY_CAP = 5;

constexpr int SEARCH_SINGULAR_EXTENSION_MIN_DEPTH = 6;
constexpr int SEARCH_SINGULAR_EXTENSION_DEPTH_MARGIN = 3;
constexpr int SEARCH_SINGULAR_BETA_DEPTH_MARGIN = 1;
constexpr int SEARCH_SINGULAR_DOUBLE_MARGIN = 14;
constexpr int SEARCH_SINGULAR_NEGATIVE_EXTENSION = 2;

constexpr int SEARCH_PROBCUT_MIN_DEPTH = 5;
constexpr int SEARCH_PROBCUT_REDUCTION = 3;
constexpr int SEARCH_PROBCUT_MARGIN = 190;
constexpr int SEARCH_PROBCUT_IMPROVING_MARGIN = 40;
constexpr int SEARCH_PROBCUT_SEE_MARGIN = 0;

// Internal iterative reductions: with no TT move to guide ordering at a deep
// node, reduce one ply so the cheaper search can seed the TT with a move.
constexpr int SEARCH_IIR_MIN_DEPTH = 4;
constexpr int SEARCH_IIR_REDUCTION = 1;

constexpr int SEARCH_QSEARCH_DELTA_MARGIN = 156;
constexpr int SEARCH_QSEARCH_SEE_MARGIN = -32;

constexpr int SEARCH_SEE_PRUNE_MAX_DEPTH = 10;
constexpr int SEARCH_SEE_PRUNE_CAPTURE_MARGIN = 96;
constexpr int SEARCH_SEE_PRUNE_QUIET_MARGIN = 21;

// lmr_table[d][m] = int(SEARCH_LMR_TABLE_BASE + log(d) * log(m) / SEARCH_LMR_TABLE_DIVISOR)
constexpr double SEARCH_LMR_TABLE_BASE = 0.99;
constexpr double SEARCH_LMR_TABLE_DIVISOR = 3.14;
constexpr int SEARCH_LMR_TABLE_MOVES = 64;

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

// ----------------------------------------
// Time management
// ----------------------------------------
// `soft` gates whether another iterative deepening iteration is started; `hard`
// aborts the search wherever it is. Values are Berserk's (src/uci.c,
// src/search.c). See notes/time-management-2026-09.md.

// UCI "Move Overhead" option bounds, in ms. 50 is what keeps a reserve behind
// each move once the clock settles into its low-time equilibrium.
constexpr int TM_MOVE_OVERHEAD_DEFAULT = 50;
constexpr int TM_MOVE_OVERHEAD_MIN     = 0;
constexpr int TM_MOVE_OVERHEAD_MAX     = 5000;

constexpr int TM_SAFETY_MS      = 10;
constexpr int TM_SINGLE_MOVE_MS = 250;

// Sudden death / Fischer. The horizon assumes the game runs this many more
// moves, so their increment counts as time we can already spend against.
constexpr int    TM_FISCHER_HORIZON = 50;
constexpr double TM_SD_SOFT_CAP     = 0.4193;
constexpr double TM_SD_SOFT_SCALE   = 0.0575;
constexpr double TM_SD_HARD_CAP     = 0.9221;
constexpr double TM_SD_HARD_MULT    = 5.928;

// Cyclic. The horizon divisor front-loads the allocation, since every later
// move re-derives its own share from what is left.
constexpr double TM_CYC_SOFT_CAP    = 0.9;
constexpr double TM_CYC_SOFT_SCALE  = 0.9;
constexpr double TM_CYC_HORIZON_DIV = 2.5;
constexpr double TM_CYC_HARD_CAP    = 0.8;
constexpr double TM_CYC_HARD_MULT   = 5.5;

constexpr int TM_MIN_DEPTH         = 5;
constexpr int TM_SCORE_TREND_PLIES = 3;

// Soft-bound scaler (a): best-move stability.
constexpr double TM_STABILITY_BASE = 1.3110;
constexpr double TM_STABILITY_STEP = 0.0533;
constexpr int    TM_STABILITY_MAX  = 10;

// (b) Score trend. Coefficients are Berserk's 0.0262 / 0.0261 divided by 1.85,
// Thrawn's cp inflation versus the WDL-normalised scale the references print -
// see the eval-scale calibration in notes/search-rewrite-2026-08.md.
constexpr double TM_SCORE_BASE        = 0.1127;
constexpr double TM_SCORE_RECENT_COEF = 0.0142;
constexpr double TM_SCORE_PREV_COEF   = 0.0141;
constexpr double TM_SCORE_MIN         = 0.5028;
constexpr double TM_SCORE_MAX         = 1.6561;

// (c) Node effort: the share of the tree spent on the best root move.
constexpr double TM_NODE_BASE     = 0.4499;
constexpr double TM_NODE_COEF     = 2.2669;
constexpr double TM_NODE_MIN      = 0.5630;
constexpr double TM_NODE_DECISIVE = 0.5;

constexpr int TM_MATE_STABILITY = 3;

enum{
    white,
    black,
    both
};

// encode pieces
// white ->-> black
enum {P, N, B, R, Q, K, p, n, b, r, q, k};

enum {rook, bishop};

enum {all_moves, only_captures, only_quiets};

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
