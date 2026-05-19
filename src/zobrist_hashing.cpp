#include "zobrist_hashing.h"
#include "bitboard.h"
#include "bitboard_helpers.h"
#include <iostream>

//uint64_t piece_hashkey[12][64]; // [piece][square]
//uint64_t enpassant_hashkey[64]; // [enpassant square]
//uint64_t castling_hashkey[16]; // [castle rights]
//uint64_t colour_to_move_hashkey;
//
//uint64_t zobristKey;

void init_hashkeys(thrawn::Position* pos)
{
    random_state = 1804289383;
    // piece hashkey
    for (int piece = P; piece<=k; piece++)
    {
        for (int sq = 0; sq<64; sq++)
        {
            pos->piece_hashkey[piece][sq] = get_random_U64();
        }
    }

    // enpassant hashkey
    for (int sq=0;sq<64;sq++)
    {
        pos->enpassant_hashkey[sq] = get_random_U64();
    }

    // castling hashkey
    for (int i=0;i<16;i++)
    {
        pos->castling_hashkey[i] = get_random_U64();
    }

    pos->colour_to_move_hashkey = get_random_U64();
}

uint64_t gen_hashkey(thrawn::Position* pos)
{
    uint64_t hashkey = 0ULL;
    uint64_t bitboard;

    for (int piece=P;piece<=k;piece++)
    {
        bitboard = pos->piece_bitboards[piece];
        while(bitboard)
        {
            int sq = pop_lsb(bitboard);
            hashkey ^= pos->piece_hashkey[piece][sq];

        }
    }

    if (pos->enpassant!=null_sq)
        hashkey ^= pos->enpassant_hashkey[pos->enpassant];
    
    hashkey ^= pos->castling_hashkey[pos->castle_rights];

    if (pos->colour_to_move == black)
        hashkey ^= pos->colour_to_move_hashkey;
    
    return hashkey;
}
