#include "fen.h"
#include "bitboard.h"
#include "bitboard_helpers.h"
#include "constants.h"
#include "nnue.h"
#include "zobrist_hashing.h"
#include "search.h"
#include "position.h"
#include <algorithm>
#include <cstdlib>
#include <string>

using namespace std;

class Bitboard;

void parse_fen(thrawn::Position* pos, const char* fen)
{

    // reset piece bitboards and occupancies
    for (int i = 0; i < pos->piece_bitboards.size(); i++)
        pos->piece_bitboards[i] = 0ULL;

    for (int i = 0; i < pos->occupancies.size(); i++)
        pos->occupancies[i] = 0ULL;
    
    // reset gameState variables
    pos->colour_to_move = white;
    pos->enpassant = null_sq;
    pos->castle_rights = 0;
    pos->repetition_index = 0;
    pos->fifty_move = 0;
    pos->ply = 0;
    std::fill(std::begin(pos->repetition_table), std::end(pos->repetition_table), 0);
    

    // loop to parse pieces and empty squares from fen
    for (int r=0;r<8;r++)
    {
        for (int c=0;c<8;c++)
        {
            int square = r*8+c;

            // check if current fen char is a letter and init its corresponding piece
            if ( (*fen >= 'a' && *fen<='z') || (*fen>='A' && *fen<='Z') )
            {
                int piece = char_pieces.at(*fen);
                set_bit(pos->piece_bitboards[piece], square);
                *fen++;
            }

            // check for numbers in fen which represents # of empty squares
            if (*fen>='0' && *fen<='9')
            {
                int empty_squares = *fen-'0';
                int piece = -1;

                // loops over all the pieces (white pawn -> black king)
                for (int i=P;i<=k;i++)
                {
                    if (get_bit(pos->piece_bitboards[i], square))
                        piece = i;
                }
                
                // going back one if no piece is on the square
                if (piece==-1)
                    c--;
                c+=empty_squares;
                fen++;
            }

            // goes to next rank
            if (*fen == '/')
                fen++;
        }
    }

    // go to parse colour_to_move value in fen string
    fen++;

    // parsing colour_to_move
    *fen == 'w' ? (pos->colour_to_move = white) : (pos->colour_to_move = black);

    // go to parse castling rights value in fen string
    fen += 2;

    // parsing castle rights
    while(*fen != ' ')
    {
        if (*fen == 'K')
            pos->castle_rights |= wks;
        else if (*fen == 'Q')
            pos->castle_rights |= wqs;
        else if (*fen == 'k')
            pos->castle_rights |= bks;
        else if (*fen == 'q')
            pos->castle_rights |= bqs;
        else if (*fen == '-')
            break;
        
        fen++;
    }

    // go to parse enpassant square value in fen string
    fen++;
    if (*fen == ' ')
        fen++;
    
    // parsing enpassant square
    if (*fen != '-')
    {
        int col = *fen - 'a';
        // to get to row
        *fen++;
        int row = 8- (*fen -'0');

        pos->enpassant = row*8+col;

    }
    else
        pos->enpassant = null_sq;

    while (*fen && *fen != ' ')
        fen++;

    if (*fen == ' ')
    {
        fen++;
        if (*fen >= '0' && *fen <= '9')
            pos->fifty_move = std::max(0, std::atoi(fen));
    }
    
    // setting white, black, and both occupancies
    pos->occupancies[0] = get_white_occupancy(pos);
    pos->occupancies[1] = get_black_occupancy(pos);
    pos->occupancies[2] = get_both_occupancy(pos);

    // rebuild the piece-on-square mailbox from the freshly parsed bitboards
    pos->mailbox.fill(-1);
    for (int piece = P; piece <= k; piece++)
    {
        uint64_t bb = pos->piece_bitboards[piece];
        while (bb)
            pos->mailbox[pop_lsb(bb)] = static_cast<int8_t>(piece);
    }
    
    // init hashkeys
    pos->zobristKey = gen_hashkey(pos);
    nnue_refresh_root(pos);
}
