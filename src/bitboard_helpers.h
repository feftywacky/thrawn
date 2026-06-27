#ifndef BITBOARD_HELPERS_H
#define BITBOARD_HELPERS_H

#include <bitset>
#include <cstdint>
#include <vector>
#include "constants.h"
#include "zobrist_hashing.h"
#include "search.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__BMI2__)
#include <immintrin.h>
#endif

using namespace std;

// Parallel bit extract. On x86 with BMI2 this is the single-cycle (Intel) /
// few-cycle (Zen3+) PEXT instruction used for pext-bitboard slider indexing.
// The portable software fallback produces identical results, so correctness can
// be validated on non-BMI2 hardware (e.g. Apple Silicon); only the speed
// differs. NOTE: PEXT is microcoded and slow on pre-Zen3 AMD — gate USE_PEXT to
// Intel Haswell+ / AMD Zen3+ targets.
inline uint64_t pext_u64(uint64_t val, uint64_t mask) {
#if defined(__BMI2__)
    return _pext_u64(val, mask);
#else
    uint64_t res = 0;
    uint64_t out_bit = 1;
    while (mask) {
        const uint64_t lsb = mask & (0ULL - mask);
        if (val & lsb)
            res |= out_bit;
        out_bit <<= 1;
        mask &= mask - 1;
    }
    return res;
#endif
}

// class Bitboard;

extern unsigned int random_state;

inline uint64_t square_bb(int bit) {
    return 1ULL << bit;
}

inline void set_bit(uint64_t& bitboard, int bit) {
    bitboard |= square_bb(bit);
}

inline void clear_bit(uint64_t& bitboard, int bit) {
    bitboard &= ~square_bb(bit);
}

inline void pop_bit(uint64_t& bitboard, int bit) {
    bitboard &= ~square_bb(bit);
}

inline uint64_t get_bit(uint64_t bitboard, int bit) {
    return bitboard & square_bb(bit);
}

inline bool is_bit_set(uint64_t bitboard, int bit) {
    return (bitboard & square_bb(bit)) != 0;
}

inline void toggle_bit(uint64_t& bitboard, int bit) {
    bitboard ^= square_bb(bit);
}

vector<int> get_squares_from_bb(uint16_t bitboard);  

// BIT MANIPULATION
inline int count_bits(uint64_t bitboard) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bitboard);
#elif defined(_MSC_VER) && defined(_M_X64)
    return static_cast<int>(__popcnt64(bitboard));
#elif defined(_MSC_VER)
    return static_cast<int>(__popcnt(static_cast<unsigned int>(bitboard))) +
           static_cast<int>(__popcnt(static_cast<unsigned int>(bitboard >> 32)));
#else
    int count = 0;
    while (bitboard) {
        bitboard &= bitboard - 1;
        ++count;
    }
    return count;
#endif
}

inline int get_lsb_index(uint64_t bitboard) {
    if (!bitboard) {
        return -1;
    }

#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bitboard);
#elif defined(_MSC_VER) && defined(_M_X64)
    unsigned long index = 0;
    _BitScanForward64(&index, bitboard);
    return static_cast<int>(index);
#elif defined(_MSC_VER)
    unsigned long index = 0;
    const unsigned long lo = static_cast<unsigned long>(bitboard);
    if (lo != 0) {
        _BitScanForward(&index, lo);
        return static_cast<int>(index);
    }
    _BitScanForward(&index, static_cast<unsigned long>(bitboard >> 32));
    return static_cast<int>(index + 32);
#else
    return count_bits((bitboard & -bitboard) - 1);
#endif
}

inline int pop_lsb(uint64_t& bitboard) {
    const int square = get_lsb_index(bitboard);
    bitboard &= bitboard - 1;
    return square;
}

// XOR SHIFT RANDOM NUMBER GEN ALGORITHM
// Generate 32-bit pseudo legal numbers
uint32_t get_random_U32();

// Generate 64-bit pseudo legal numbers
uint64_t get_random_U64();

uint64_t gen_magic_num();

void print_bitboard(uint64_t bitboard);

void print_board(thrawn::Position* pos, int side);

void print_bits(uint64_t num);

#endif
