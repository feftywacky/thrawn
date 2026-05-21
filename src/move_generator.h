#ifndef MOVE_GENERATOR_H
#define MOVE_GENERATOR_H

#include <array>
#include <cstdint>

#include "position.h"

constexpr int MAX_GENERATED_MOVES = 256;

using MoveConsumer = void (*)(int move, void* context);

struct MoveList {
    std::array<int, MAX_GENERATED_MOVES> moves{};
    int count = 0;
    MoveConsumer consumer = nullptr;
    void* consumerContext = nullptr;

    MoveList() = default;
    MoveList(MoveConsumer consumer, void* context)
        : consumer(consumer), consumerContext(context) {}

    void clear() { count = 0; }

    void push_back(int move) {
        if (consumer) {
            consumer(move, consumerContext);
            ++count;
            return;
        }
        if (count < MAX_GENERATED_MOVES)
            moves[count++] = move;
    }
    int size() const { return count; }
    bool empty() const { return count == 0; }
    int* begin() { return moves.data(); }
    int* end() { return moves.data() + count; }
    const int* begin() const { return moves.data(); }
    const int* end() const { return moves.data() + count; }
    int& operator[](int index) { return moves[index]; }
    const int& operator[](int index) const { return moves[index]; }
};

void generate_moves(thrawn::Position* pos, int move_type, MoveList& moves);

int make_move(thrawn::Position* pos, int move, int move_type, int ply);
int make_move_on_board(thrawn::Position* pos, int move, int move_type, int ply);
int make_move_for_perft(thrawn::Position* pos, int move, int ply);
int make_root_move(thrawn::Position* pos, int move, int move_type);
void unmake_move(thrawn::Position* pos, int ply);
void make_null_move(thrawn::Position* pos, int ply);
void unmake_null_move(thrawn::Position* pos, int ply);

#endif
