#include "position.h"
#include "zobrist_hashing.h"
#include "bitboard.h"

#include <algorithm>
#include <mutex>

namespace {

std::once_flag position_tables_once;

void init_position_tables(thrawn::Position* pos) {
    init_leaping_attacks(pos);
    init_sliding_attacks(pos, bishop);
    init_sliding_attacks(pos, rook);
    init_hashkeys(pos);
}

} // namespace

namespace thrawn {

std::array<std::array<uint64_t, 64>, 2> Position::pawn_attacks{};
std::array<uint64_t, 64> Position::knight_attacks{};
std::array<uint64_t, 64> Position::king_attacks{};

std::array<uint64_t, 64> Position::bishop_masks{};
std::array<std::array<uint64_t, 512>, 64> Position::bishop_attacks{};
std::array<uint64_t, 64> Position::rook_masks{};
std::array<std::array<uint64_t, 4096>, 64> Position::rook_attacks{};

uint64_t Position::piece_hashkey[12][64]{};
uint64_t Position::enpassant_hashkey[64]{};
uint64_t Position::castling_hashkey[16]{};
uint64_t Position::colour_to_move_hashkey = 0ULL;

NnueStack::NnueStack()
    : states(std::make_unique<std::array<NnueState, MAX_DEPTH + 1>>()) {}

NnueStack::NnueStack(const NnueStack& other)
    : NnueStack() {
    *states = *other.states;
}

NnueStack& NnueStack::operator=(const NnueStack& other) {
    if (this != &other) {
        *states = *other.states;
    }
    return *this;
}

NnueState& NnueStack::operator[](std::size_t index) {
    return (*states)[index];
}

const NnueState& NnueStack::operator[](std::size_t index) const {
    return (*states)[index];
}

void NnueStack::copy_up_to(const NnueStack& other, int ply) {
    const int limit = std::clamp(ply, 0, MAX_DEPTH);
    for (int i = 0; i <= limit; ++i) {
        (*states)[i] = other[static_cast<std::size_t>(i)];
    }
    for (int i = limit + 1; i <= MAX_DEPTH; ++i) {
        (*states)[i].valid = false;
        (*states)[i].white_acc_source_ply = -1;
        (*states)[i].black_acc_source_ply = -1;
    }
}

// Default constructor
Position::Position()
    : piece_bitboards{},
      occupancies{},
      colour_to_move(white),
      enpassant(null_sq),
      castle_rights(0),
      zobristKey(0ULL),
      fifty_move(0),
      repetition_table{},
      repetition_index(0),
      ply(0),
      nnue_stack(),
      undo_stack{} {
    std::call_once(position_tables_once, init_position_tables, this);
}

Position::Position(const Position& other)
    : piece_bitboards(other.piece_bitboards),
      occupancies(other.occupancies),
      colour_to_move(other.colour_to_move),
      enpassant(other.enpassant),
      castle_rights(other.castle_rights),
      zobristKey(other.zobristKey),
      fifty_move(other.fifty_move),
      repetition_table(other.repetition_table),
      repetition_index(other.repetition_index),
      ply(other.ply),
      nnue_stack(),
      undo_stack{} {
    for (int i = 0; i <= MAX_DEPTH; ++i) {
        undo_stack[i] = other.undo_stack[i];
    }
    nnue_stack.copy_up_to(other.nnue_stack, other.ply);
    std::call_once(position_tables_once, init_position_tables, this);
}

Position& Position::operator=(const Position& other) {
    if (this == &other) {
        return *this;
    }

    piece_bitboards = other.piece_bitboards;
    occupancies = other.occupancies;
    colour_to_move = other.colour_to_move;
    enpassant = other.enpassant;
    castle_rights = other.castle_rights;
    zobristKey = other.zobristKey;
    fifty_move = other.fifty_move;
    repetition_table = other.repetition_table;
    repetition_index = other.repetition_index;
    ply = other.ply;
    for (int i = 0; i <= MAX_DEPTH; ++i) {
        undo_stack[i] = other.undo_stack[i];
    }
    nnue_stack.copy_up_to(other.nnue_stack, other.ply);
    std::call_once(position_tables_once, init_position_tables, this);
    return *this;
}

} // namespace thrawn
