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
