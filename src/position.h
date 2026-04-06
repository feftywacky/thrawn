#ifndef POSITION_H
#define POSITION_H

#include <array>
#include <vector>
#include <cstdint>
#include "constants.h"

namespace thrawn {

constexpr int NNUE_FT_SIZE = 256;
constexpr int NNUE_MAX_PIECES = 32;

struct NnueState {
    alignas(64) std::array<int16_t, NNUE_FT_SIZE> white_acc{};
    alignas(64) std::array<int16_t, NNUE_FT_SIZE> black_acc{};
    std::array<uint8_t, NNUE_MAX_PIECES> piece_list{};
    std::array<uint8_t, NNUE_MAX_PIECES> square_list{};
    std::array<int8_t, BOARD_SIZE> index_by_square{};
    uint8_t piece_count = 0;
    bool valid = false;

    NnueState() {
        index_by_square.fill(-1);
    }
};

// Holds all data needed to restore a position
struct UndoData {
    int move;             // the packed move itself: source, target, etc.
    int captured_piece;   // which piece got captured (if any)
    int castle_rights;    // old castle rights before move
    int enpassant;        // old en-passant square
    int fifty_move;       // old halfmove clock
    uint64_t zobristKey;  // old zobrist key (optional but convenient)
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
    std::array<NnueState, MAX_DEPTH + 1> nnue_stack;

    //============= ATTACK AND HASHING TABLES =============//

    std::array<std::array<uint64_t, 64>, 2> pawn_attacks;
    std::array<uint64_t, 64> knight_attacks;
    std::array<uint64_t, 64> king_attacks;

    std::array<uint64_t, 64> bishop_masks;
    std::array<std::array<uint64_t, 512>, 64> bishop_attacks;
    std::array<uint64_t, 64> rook_masks;
    std::array<std::array<uint64_t, 4096>, 64> rook_attacks;

    uint64_t piece_hashkey[12][64];
    uint64_t enpassant_hashkey[64];
    uint64_t castling_hashkey[16];
    uint64_t colour_to_move_hashkey;

    //============= UNDO STACK =============//
    UndoData undo_stack[MAX_DEPTH];

    //============= CONSTRUCTORS & METHODS =============//
    Position();
};

// copying and restoring for move take backs
#define copyBoard(pos) \
    array<uint64_t, 12> piece_bitboards_copy; \
    array<uint64_t, 3> occupancies_copy; \
    int colour_to_move_copy; \
    int enpassant_copy; \
    int castle_rights_copy; \
    uint64_t zobristKey_copy; \
    int fifty_move_copy; \
    piece_bitboards_copy = pos->piece_bitboards; \
    occupancies_copy = pos->occupancies; \
    colour_to_move_copy = pos->colour_to_move; \
    enpassant_copy = pos->enpassant; \
    castle_rights_copy = pos->castle_rights; \
    zobristKey_copy = pos->zobristKey; \
    fifty_move_copy = pos->fifty_move; \

// Restore board state
#define restoreBoard(pos) \
    pos->piece_bitboards = piece_bitboards_copy; \
    pos->occupancies = occupancies_copy; \
    pos->colour_to_move = colour_to_move_copy; \
    pos->enpassant = enpassant_copy; \
    pos->castle_rights = castle_rights_copy; \
    pos->zobristKey = zobristKey_copy; \
    pos->fifty_move = fifty_move_copy; \

} // namespace thrawn

#endif
