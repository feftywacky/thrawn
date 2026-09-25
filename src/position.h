#ifndef POSITION_H
#define POSITION_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "constants.h"

namespace thrawn {

// Latest trainer export: HalfKAv2_hm (see kCurrentVersion in nnue.cpp).
constexpr int NNUE_INPUT_FEATURES = 22528;
constexpr int NNUE_PS_NB = 11 * 64;
constexpr int NNUE_ACCUMULATOR_SIZE = 1024;
constexpr int NNUE_FT_SIZE = NNUE_ACCUMULATOR_SIZE;
constexpr int NNUE_HIDDEN_SIZE = 31;
constexpr int NNUE_FORWARD_SIZE = 1;
constexpr int NNUE_FC0_OUTPUT_SIZE = NNUE_HIDDEN_SIZE + NNUE_FORWARD_SIZE;
constexpr int NNUE_FC1_INPUT_SIZE = NNUE_HIDDEN_SIZE * 2;
constexpr int NNUE_FC1_OUTPUT_SIZE = 32;
constexpr int NNUE_SIMD_ALIGNMENT = 64;

static_assert(NNUE_ACCUMULATOR_SIZE % 8 == 0, "NNUE accumulator must fit NEON lanes");
static_assert(NNUE_ACCUMULATOR_SIZE % 16 == 0, "NNUE accumulator must fit AVX2 lanes");
static_assert(NNUE_INPUT_FEATURES == 32 * NNUE_PS_NB, "HalfKAv2_hm feature count mismatch");

// One piece change made by the move into a ply (piece removed from or added to
// a square). Castling is the largest move: king and rook each removed + added.
struct NnueDirtyPiece {
    int8_t piece = 0;
    int8_t square = 0;
    bool add = false;
};
constexpr int NNUE_MAX_DIRTY_PIECES = 4;

// Per-ply NNUE state. Accumulators are updated lazily: make_move only records
// the move's piece changes in `dirty`, and an accumulator is materialized (by
// replaying dirty pieces forward from the nearest computed ancestor, or by a
// cached refresh after a king move) only when a node is actually evaluated.
struct alignas(NNUE_SIMD_ALIGNMENT) NnueState {
    alignas(NNUE_SIMD_ALIGNMENT) std::array<int16_t, NNUE_ACCUMULATOR_SIZE> white_acc{};
    alignas(NNUE_SIMD_ALIGNMENT) std::array<int16_t, NNUE_ACCUMULATOR_SIZE> black_acc{};
    // Per perspective: the ply whose buffer holds this ply's accumulator - this
    // ply itself once computed, an ancestor when nothing changed for that
    // perspective (null move) - or -1 while it has not been computed yet.
    int8_t white_acc_ply = -1;
    int8_t black_acc_ply = -1;
    uint8_t dirty_count = 0;
    std::array<NnueDirtyPiece, NNUE_MAX_DIRTY_PIECES> dirty{};
    bool valid = false;
};

// Accumulator refresh cache. One entry per perspective per king
// square: HalfKAv2_hm features depend only on (perspective, king bucket, king
// file half), and every king square maps to a distinct (bucket, half) pair, so
// the king square is an exact key. Each entry holds an accumulator together with
// the piece bitboards it was built from; a refresh diffs those against the live
// board and applies only the add/sub deltas. `generation` ties the entry to the
// network it was built with (0 = never built), so a (re)load invalidates it.
struct alignas(NNUE_SIMD_ALIGNMENT) NnueRefreshEntry {
    alignas(NNUE_SIMD_ALIGNMENT) std::array<int16_t, NNUE_ACCUMULATOR_SIZE> acc;
    std::array<uint64_t, 12> bitboards;
    uint32_t generation;
};

struct NnueRefreshCache {
    std::array<std::array<NnueRefreshEntry, BOARD_SIZE>, 2> entries;  // [white_perspective][king_sq]
};

class NnueStack {
public:
    NnueStack();
    // Copies carry the accumulator stack but start with a cold refresh cache:
    // the cache is a pure speed-up (entries are exact for their bitboards), so a
    // fresh one is always correct, and each search thread owns its own.
    NnueStack(const NnueStack& other);
    NnueStack& operator=(const NnueStack& other);
    NnueStack(NnueStack&& other) noexcept = default;
    NnueStack& operator=(NnueStack&& other) noexcept = default;

    NnueState& operator[](std::size_t index);
    const NnueState& operator[](std::size_t index) const;

    void copy_up_to(const NnueStack& other, int ply);

    NnueRefreshCache& refresh_cache() { return *cache; }

private:
    std::unique_ptr<std::array<NnueState, MAX_DEPTH + 1>> states;
    std::unique_ptr<NnueRefreshCache> cache;
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

    // Piece-on-square lookup (mailbox) kept in sync with piece_bitboards.
    // Holds the piece index (P..k) on each square, or -1 when empty. Lets the
    // make-move hot path resolve a captured piece with one load instead of a
    // branchy linear scan over the 12 piece bitboards. Placed next to the board
    // state so the make/unmake working set stays cache-line local.
    std::array<int8_t, 64> mailbox;

    std::array<uint64_t, 1028> repetition_table; 
    int repetition_index;

    int ply;
    NnueStack nnue_stack;

    //============= ATTACK AND HASHING TABLES =============//

    static std::array<std::array<uint64_t, 64>, 2> pawn_attacks;
    static std::array<uint64_t, 64> knight_attacks;
    static std::array<uint64_t, 64> king_attacks;

    static std::array<uint64_t, 64> bishop_masks;
    static std::array<uint64_t, 64> rook_masks;

#if defined(USE_PEXT)
    // PEXT path: each square owns exactly 2^relevant_bits slots in a flat table
    // addressed via *_attack_offset[square]; index = pext(occupancy, mask).
    // Shrinks rook 2MB->~800KB and bishop 256KB->~41KB (better L2/L3 residency
    // on x86, where this matters more than on M4's 4MB L2).
    static constexpr int BISHOP_TABLE_SIZE = 5248;
    static constexpr int ROOK_TABLE_SIZE = 102400;
    static std::array<int, 64> bishop_attack_offset;
    static std::array<int, 64> rook_attack_offset;
    static std::array<uint64_t, BISHOP_TABLE_SIZE> bishop_attacks;
    static std::array<uint64_t, ROOK_TABLE_SIZE> rook_attacks;
#else
    static std::array<std::array<uint64_t, 512>, 64> bishop_attacks;
    static std::array<std::array<uint64_t, 4096>, 64> rook_attacks;
#endif

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
