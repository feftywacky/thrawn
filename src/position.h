#ifndef POSITION_H
#define POSITION_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "constants.h"

namespace thrawn {

// Latest trainer export: halfkp_v1, version 7.
constexpr int NNUE_INPUT_FEATURES = 40960;
constexpr int NNUE_FACTOR_FEATURES = 640;
constexpr int NNUE_MAX_ACTIVE_FEATURES = 30;
constexpr int NNUE_ACCUMULATOR_SIZE = 1024;
constexpr int NNUE_L1_SIZE = 256;
constexpr int NNUE_L2_SIZE = 64;
constexpr int NNUE_HIDDEN_SIZE = NNUE_L1_SIZE; // legacy alias
constexpr int NNUE_OUTPUT_BUCKETS = 1;         // legacy alias
constexpr int NNUE_FT_SIZE = NNUE_ACCUMULATOR_SIZE;
constexpr int NNUE_MAX_PIECES = 32;
constexpr int NNUE_SIMD_ALIGNMENT = 64;

static_assert(NNUE_ACCUMULATOR_SIZE % 8 == 0, "NNUE accumulator must fit NEON lanes");
static_assert(NNUE_ACCUMULATOR_SIZE % 16 == 0, "NNUE accumulator must fit AVX2 lanes");

struct alignas(NNUE_SIMD_ALIGNMENT) NnueState {
    alignas(NNUE_SIMD_ALIGNMENT) std::array<int16_t, NNUE_ACCUMULATOR_SIZE> white_acc{};
    alignas(NNUE_SIMD_ALIGNMENT) std::array<int16_t, NNUE_ACCUMULATOR_SIZE> black_acc{};
    std::array<uint8_t, NNUE_MAX_PIECES> piece_list{};
    std::array<uint8_t, NNUE_MAX_PIECES> square_list{};
    std::array<int8_t, BOARD_SIZE> index_by_square{};
    // -1 means local; otherwise this perspective reads from an ancestor ply until first write.
    int8_t white_acc_source_ply = -1;
    int8_t black_acc_source_ply = -1;
    uint8_t piece_count = 0;
    int8_t white_king_sq = -1;
    int8_t black_king_sq = -1;
    bool valid = false;

    NnueState() {
        index_by_square.fill(-1);
    }
};

class NnueStack {
public:
    NnueStack();
    NnueStack(const NnueStack& other);
    NnueStack& operator=(const NnueStack& other);
    NnueStack(NnueStack&& other) noexcept = default;
    NnueStack& operator=(NnueStack&& other) noexcept = default;

    NnueState& operator[](std::size_t index);
    const NnueState& operator[](std::size_t index) const;

    void copy_up_to(const NnueStack& other, int ply);

private:
    std::unique_ptr<std::array<NnueState, MAX_DEPTH + 1>> states;
};

// Holds the irreversible state and capture delta needed to unmake one move.
struct UndoData {
    int move = 0;
    int captured_piece = -1;
    int captured_square = null_sq;
    int castle_rights = 0;
    int enpassant = null_sq;
    int fifty_move = 0;
    uint64_t zobristKey = 0ULL;
};

class Position {
public:
    //============= CURRENT BOARD STATE =============//

    std::array<uint64_t, 12> piece_bitboards;
    std::array<uint64_t, 3>  occupancies;
    int colour_to_move;
    int enpassant;
    int castle_rights;
    uint64_t zobristKey;
    int fifty_move;

    std::array<uint64_t, 1028> repetition_table; 
    int repetition_index;

    int ply;
    NnueStack nnue_stack;

    //============= ATTACK AND HASHING TABLES =============//

    static std::array<std::array<uint64_t, 64>, 2> pawn_attacks;
    static std::array<uint64_t, 64> knight_attacks;
    static std::array<uint64_t, 64> king_attacks;

    static std::array<uint64_t, 64> bishop_masks;
    static std::array<std::array<uint64_t, 512>, 64> bishop_attacks;
    static std::array<uint64_t, 64> rook_masks;
    static std::array<std::array<uint64_t, 4096>, 64> rook_attacks;

    static uint64_t piece_hashkey[12][64];
    static uint64_t enpassant_hashkey[64];
    static uint64_t castling_hashkey[16];
    static uint64_t colour_to_move_hashkey;

    //============= UNDO STACK =============//
    UndoData undo_stack[MAX_DEPTH + 1];

    //============= CONSTRUCTORS & METHODS =============//
    Position();
    Position(const Position& other);
    Position& operator=(const Position& other);
};

} // namespace thrawn

#endif
