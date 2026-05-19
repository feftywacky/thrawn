#include "bitboard.h"
#include "evaluation.h"
#include <iostream>
#include <vector>

using namespace std;

// GET OCCUPANCY BITBOARDS
uint64_t get_white_occupancy(thrawn::Position* pos)
{
    uint64_t res = 0ULL;
    for(int i=0;i<6;i++)
        res |= pos->piece_bitboards[i];
    return res;
}

uint64_t get_black_occupancy(thrawn::Position* pos)
{
    uint64_t res = 0ULL;
    for(int i=6;i<12;i++)
        res |= pos->piece_bitboards[i];
    return res;
}

uint64_t get_both_occupancy(thrawn::Position* pos)
{
    uint64_t res = 0ULL;
    res |= get_white_occupancy(pos);
    res |= get_black_occupancy(pos);
    return res;
}


// PRE-COMPUTE PIECE ATTACK BITBOARDS

// pawns
uint64_t get_pawn_attacks(int side, const int& square)
{
    uint64_t attacks = 0ULL;
    uint64_t bitboard = 0ULL;

    set_bit(bitboard, square);

    if (side == white) 
    {
        if ((bitboard >> 7) & not_a_file) 
            attacks |= (bitboard >> 7);
        if ((bitboard >> 9) & not_h_file) 
            attacks |= (bitboard >> 9);
    }
    else 
    {
        if ((bitboard << 7) & not_h_file) 
            attacks |= (bitboard << 7);
        if ((bitboard << 9) & not_a_file) 
            attacks |= (bitboard << 9);
    }

    return attacks;
}

// knights
uint64_t get_knight_attacks(const int& square)
{
    uint64_t attacks = 0ULL;
    uint64_t bitboard = 0ULL;

    set_bit(bitboard, square);

    if ((bitboard >> 17) & not_h_file) 
        attacks |= (bitboard >> 17);
    if ((bitboard >> 15) & not_a_file) 
        attacks |= (bitboard >> 15);
    if ((bitboard >> 10) & not_hg_file) 
        attacks |= (bitboard >> 10);
    if ((bitboard >> 6) & not_ab_file) 
        attacks |= (bitboard >> 6);
    if ((bitboard << 17) & not_a_file) 
        attacks |= (bitboard << 17);
    if ((bitboard << 15) & not_h_file) 
        attacks |= (bitboard << 15);
    if ((bitboard << 10) & not_ab_file) 
        attacks |= (bitboard << 10);
    if ((bitboard << 6) & not_hg_file) 
        attacks |= (bitboard << 6);
    
    return attacks;
}

uint64_t get_king_attacks(const int& square)
{
    uint64_t attacks = 0ULL;
    uint64_t bitboard = 0ULL;
    set_bit(bitboard, square);

    if (bitboard>>8)
        attacks |= (bitboard>>8);
    if ((bitboard>>9) & not_h_file)
        attacks |= (bitboard>>9);
    if ((bitboard>>7) & not_a_file)
        attacks |= (bitboard>>7);
    if ((bitboard >> 1) & not_h_file) 
        attacks |= (bitboard >> 1);

    if (bitboard<<8)
        attacks |= (bitboard<<8);
    if ((bitboard<<9) & not_a_file)
        attacks |= (bitboard<<9);
    if ((bitboard<<7) & not_h_file)
        attacks |= (bitboard<<7);
    if ((bitboard << 1) & not_a_file) 
        attacks |= (bitboard << 1);

    return attacks;
}

// does not include squares on the edge of the board
uint64_t get_bishop_mask(const int& square)
{
    uint64_t attacks = 0ULL;

    int row; int col;
    int curr_row = square/8; 
    int curr_col = square%8;

    // mask relevant bishop occupancy bits

    // bottom right diagonal 
    for (row = curr_row+1, col = curr_col+1; row<=6 && col<=6; row++,col++)
        attacks |= (1ULL << row*8 + col);
    
    // bottom left diagonal 
    for (row = curr_row+1, col = curr_col-1; row<=6 && col>=1; row++, col--)
        attacks |= (1ULL << row*8 + col);

    // top right diagonal
    for (row = curr_row-1, col = curr_col+1; row>=1 && col<=6; row--, col++)
        attacks |= (1ULL << row*8 + col);
    
    // top left diagonal
    for (row = curr_row-1, col = curr_col-1; row>=1 && col>=1; row--, col--)
        attacks |= (1ULL << row*8 + col);

    return attacks;    
}

// does not distinguish the colour of the blocker
uint64_t bishop_attack_runtime_gen(int square, const uint64_t& blockers)
{
    uint64_t attacks = 0ULL;

    int row; int col;
    int curr_row = square/8; 
    int curr_col = square%8;

    // generate bishop attacks

    // bottom right diagonal 
    for (row = curr_row+1, col = curr_col+1; row<=7 && col<=7; row++,col++)
    {
        attacks |= (1ULL << row*8 + col);
        if ( (1ULL << row*8 + col) & blockers )
            break;
    }
    
    // bottom left diagonal 
    for (row = curr_row+1, col = curr_col-1; row<=7 && col>=0; row++, col--)
    {
        attacks |= (1ULL << row*8 + col);
         if ( (1ULL << row*8 + col) & blockers )
            break;
    }

    // top right diagonal
    for (row = curr_row-1, col = curr_col+1; row>=0 && col<=7; row--, col++)
    {
        attacks |= (1ULL << row*8 + col);
         if ( (1ULL << row*8 + col) & blockers )
            break;
    }

    // top left diagonal
    for (row = curr_row-1, col = curr_col-1; row>=0 && col>=0; row--, col--)
    {
        attacks |= (1ULL << row*8 + col);
         if ( (1ULL << row*8 + col) & blockers )
            break;
    }

    return attacks; 
}

uint64_t get_rook_mask(const int& square)
{
    uint64_t attacks = 0ULL;

    int row; int col;
    int curr_row = square/8; 
    int curr_col = square%8;

    // up
    for (row = curr_row-1; row>=1; row--)
        attacks |= (1ULL << row*8 + curr_col);

    // down
    for (row = curr_row+1; row<=6; row++)
        attacks |= (1ULL << row*8 + curr_col);

    // left
    for (col = curr_col-1; col >=1; col--)
        attacks |= (1ULL << curr_row*8 + col);

    // right
    for (col = curr_col+1; col<=6; col++)
        attacks |= (1ULL << curr_row*8 + col);

    return attacks;
}

// does not distinguish the colour of the blocker
uint64_t rook_attack_runtime_gen(int square, uint64_t& blockers)
{
    uint64_t attacks = 0ULL;

    int row; int col;
    int curr_row = square/8; 
    int curr_col = square%8;

    // up
    for (row = curr_row-1; row>=0; row--)
    {
        attacks |= (1ULL << row*8 + curr_col);
        if ( (1ULL << row*8 + curr_col) & blockers )
            break;
    }

    // down
    for (row = curr_row+1; row<=7; row++)
    {
        attacks |= (1ULL << row*8 + curr_col);
        if ( (1ULL << row*8 + curr_col) & blockers )
            break;
    }

    // left
    for (col = curr_col-1; col >=0; col--)
    {
        attacks |= (1ULL << curr_row*8 + col);
        if ( (1ULL << curr_row*8 + col) & blockers)
            break;
    }

    // right
    for (col = curr_col+1; col<=7; col++)
    {
        attacks |= (1ULL << curr_row*8 + col);
        if ( (1ULL << curr_row*8 + col) & blockers)
            break;
    }

    return attacks;
}

uint64_t set_occupancy(const int& index, const int& bits_in_mask, uint64_t attack_mask)
{
    uint64_t occupancy = 0ULL;

    for (int i=0;i<bits_in_mask;i++)
    {
        int square = pop_lsb(attack_mask);

        // make sure occupancy is on board
        if (index & (1 << i))
            // populate occupancy map
            occupancy |= (1ULL << square);
    }
    return occupancy;
}

// MAGIC BITBOARD
uint64_t find_magic_num(const int& square, int relevant_bits, int bishop)
{
    array<uint64_t, 4096> occupancies;
    array<uint64_t, 4096> attacks;
    array<uint64_t, 4096> used_attacks;
    
    uint64_t attack_mask = bishop ? get_bishop_mask(square) : get_rook_mask(square);

    // init occupancy indices
    int occupancy_index = 1 << relevant_bits;

    for (int i=0;i<occupancy_index;i++)
    {
        occupancies[i] = set_occupancy(i, relevant_bits, attack_mask);
        attacks[i] = bishop ? bishop_attack_runtime_gen(square, occupancies[i]) : rook_attack_runtime_gen(square, occupancies[i]);
    }
    
    // testing magic numbers
    for (int random_count = 0; random_count < 100000000; random_count++) 
    {
         uint64_t magic_num = gen_magic_num();

        // skip inappropriate magic numbers
        if (count_bits((attack_mask * magic_num) & 0xFF00000000000000ULL) < 6) continue;

        // init used attacks
        used_attacks.fill(0ULL);

        // init index & fail flag
        int index, fail;

        // test magic index loop
        for (index = 0, fail = 0; !fail && index < occupancy_index; index++) 
        {
            // init magic index
            int magic_index = (int)((occupancies[index] * magic_num) >> (64 - relevant_bits));

            // if magic index works
            if (used_attacks[magic_index] == 0ULL)
                // init used attacks
                used_attacks[magic_index] = attacks[index];

            else if (used_attacks[magic_index] != attacks[index])
                // magic index doesn't work
                fail = 1;
        }

        // if magic number works
        if (!fail)
            return magic_num;
    }

    // if magic number doesn't work
    std::cout << "Magic number fails!" << std::endl;
    return 0ULL;
}


void init_magic_nums()
{
    for (int square = 0; square < 64; square++)
        // init rook magic numbers
        rook_magic_nums[square] = find_magic_num(square, rook_relevant_bits[square], rook);

    for (int square = 0; square < 64; square++)
        // init bishop magic numbers
        bishop_magic_nums[square] = find_magic_num(square, bishop_relevant_bits[square], bishop);
}

// is <square> under attacked by <side> pieces
bool is_square_under_attack(thrawn::Position* pos, int square, int side)
{
    // Attacked by white pawns
    if ((side == white) && (pos->pawn_attacks[black][square] & pos->piece_bitboards[P]))
        return true;

    // Attacked by black pawns
    if ((side == black) && (pos->pawn_attacks[white][square] & pos->piece_bitboards[p]))
        return true;

    if (pos->knight_attacks[square] & ((side == white) ? pos->piece_bitboards[N] : pos->piece_bitboards[n]))
        return true;

    if (get_bishop_attacks(pos, square, pos->occupancies[both]) & ((side == white) ? pos->piece_bitboards[B] : pos->piece_bitboards[b]))
        return true;

    if (get_rook_attacks(pos, square, pos->occupancies[both]) & ((side == white) ? pos->piece_bitboards[R] : pos->piece_bitboards[r]))
        return true;

    if (get_queen_attacks(pos, square, pos->occupancies[both]) & ((side == white) ? pos->piece_bitboards[Q] : pos->piece_bitboards[q]))
        return true;

    if (pos->king_attacks[square] & ((side == white) ? pos->piece_bitboards[K] : pos->piece_bitboards[k]))
        return true;

    return false;
}

bool noMajorsOrMinorsPieces(thrawn::Position* pos)
{
    return !(count_bits(pos->occupancies[both]) - count_bits(pos->piece_bitboards[P]) - count_bits(pos->piece_bitboards[p]) - 2);
}

void init_sliding_attacks(thrawn::Position* pos, int isBishop)
{
    for (int square = 0;square<64;square++)
    {
        if (isBishop)
            pos->bishop_masks[square] = get_bishop_mask(square);
        else
            pos->rook_masks[square] = get_rook_mask(square);

        uint64_t curr_attack_mask = isBishop ? pos->bishop_masks[square] : pos->rook_masks[square];

        int relevant_bits_count = count_bits(curr_attack_mask);

        int occupancy_index = 1 << relevant_bits_count;

        for (int i=0;i<occupancy_index;i++)
        {
            if (isBishop)
            {
                uint64_t occupancy = set_occupancy(i, relevant_bits_count, curr_attack_mask);
                int magic_index = (occupancy * bishop_magic_nums[square]) >> (64-bishop_relevant_bits[square]);
                pos->bishop_attacks[square][magic_index] = bishop_attack_runtime_gen(square, occupancy);
            }
            else
            {

                uint64_t occupancy = set_occupancy(i, relevant_bits_count, curr_attack_mask);
                int magic_index = (occupancy * rook_magic_nums[square]) >> (64-rook_relevant_bits[square]);
                pos->rook_attacks[square][magic_index] = rook_attack_runtime_gen(square, occupancy);
            }
        }
    }
}

void init_leaping_attacks(thrawn::Position* pos)
{
    for (int square = 0; square < BOARD_SIZE; square++) 
    {
        pos->pawn_attacks[white][square] = get_pawn_attacks(white, square);
        pos->pawn_attacks[black][square] = get_pawn_attacks(black, square);
        pos->knight_attacks[square] = get_knight_attacks(square);
        pos->king_attacks[square] = get_king_attacks(square);
    }
}
