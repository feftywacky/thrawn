#include <vector>
#include <bitset>
#include <iostream>
#include "constants.h"
#include "bitboard.h"
#include "zobrist_hashing.h"

using namespace std;

// GLOBAL VARIABLES
unsigned int random_state = 1804289383;

vector<int> get_squares_from_bb(uint16_t bitboard)
{
    vector<int> squares;
    bitset<64> bits(bitboard);

    for (int i=0;i<64;i++)
    {
        if (bits[i]==1)
            squares.push_back(i);
    }
    return squares;
}

// RANDOM NUMBER GEN

// Generate 32-bit pseudo legal numbers
uint32_t get_random_U32()
{
    // Get current state
    uint32_t number = random_state;

    // XOR shift algorithm
    number ^= number << 13;
    number ^= number >> 17;
    number ^= number << 5;

    // Update random number state
    random_state = number;

    return number;
}

// Generate 64-bit pseudo legal numbers
uint64_t get_random_U64()
{
    uint64_t num1, num2, num3, num4;

    num1 = (uint64_t)(get_random_U32()) & 0xFFFF;
    num2 = (uint64_t)(get_random_U32()) & 0xFFFF;
    num3 = (uint64_t)(get_random_U32()) & 0xFFFF;
    num4 = (uint64_t)(get_random_U32()) & 0xFFFF;

    return num1 | (num2 << 16) | (num3 << 32) | (num4 << 48);
}


uint64_t gen_magic_num()
{
    return get_random_U64() & get_random_U64() & get_random_U64();
}

void print_bitboard(uint64_t bitboard) 
{
    std::cout << std::endl;

    // Loop over board ranks
    for (int rank = 0; rank < 8; rank++) {
        // Loop over board files
        for (int file = 0; file < 8; file++) {
            // Convert file & rank into square index
            int square = rank * 8 + file;

            // Print ranks
            if (!file)
                std::cout << "  " << 8 - rank << " ";

            // Print bit state (either 1 or 0)
            std::cout << " " << (bitboard & (1ULL << square) ? 1 : 0);
        }

        // Print a new line every rank
        std::cout << std::endl;
    }

    // Print board files
    std::cout << "\n     a b c d e f g h\n\n";

    // Print bitboard as an unsigned decimal number
    std::cout << "     Bitboard as unsigned decimal: " << bitboard << std::endl;
}

void print_board(thrawn::Position* pos, int side) 
{
    // print offset
    std::cout << "\n";


    for (int rank = 0; rank < 8; rank++) {

        // print ranks
        std::cout << "  " << 8 - rank << " ";

        for (int file = 0; file < 8; file++) {

            int square = rank * 8 + file;

            // define piece variable
            int piece = -1;

            // loop over all piece bitboards
            for (int bb_piece = P; bb_piece <= k; bb_piece++) {
                if (get_bit(pos->piece_bitboards[bb_piece], square))
                    piece = bb_piece;
            }

            std::cout << " " << ((piece == -1) ? '.' : ascii_pieces[piece]);

        }

        // print new line every rank
        std::cout << "\n";
    }

    // print board files
    std::cout << "\n     a b c d e f g h\n\n";

    // print side to move
    std::cout << "     Side:     " << (!side ? "white" : "black") << "\n";

    // print enpassant square
    std::cout << "     Enpassant:   " << ((pos->enpassant != null_sq) ? square_to_coordinates[pos->enpassant] : "no") << "\n";

    // print castling rights
    std::cout << "     Castling:  " << ((pos->castle_rights & wks) ? 'K' : '-') << ((pos->castle_rights & wqs) ? 'Q' : '-')
              << ((pos->castle_rights & bks) ? 'k' : '-') << ((pos->castle_rights & bqs) ? 'q' : '-') << "\n";
    
    std::cout << "     ZobristKey:   " << std::hex <<pos->zobristKey << "\n\n";
    std::cout<<std::dec;
}

void print_bits(uint64_t num)
{
    int num_bits = 64;

    for (int i = 0; i < num_bits; i++) {
        uint64_t bit = (num >> i) & 1;
        std::cout << bit;

        if (i % 8 == 7 && i != num_bits - 1) {
            std::cout << ' ';
        }
    }
    std::cout << std::endl;
}

