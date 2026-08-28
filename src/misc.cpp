#include "misc.h"
#include "bitboard.h"
#include "transposition_table.h"
#include "evaluation.h"
#include "nnue.h"
#include "globals.h"
#include "constants.h"

const std::string version = " v3.1";


void init_all()
{
    // init_magic_nums(); // used to help generate magic bitboards
    
    // init_leaping_attacks(pos);
    // init_sliding_attacks(pos, bishop);
    // init_sliding_attacks(pos, rook);

    // init hashkeys
    // init_hashkeys();

    // The transposition table is deliberately NOT allocated here. A GUI or
    // tournament manager sends `setoption name Hash` right after `uci`, so an
    // eager allocation of the 256 MB default would be touched and thrown away
    // moments later - painful when a match runs dozens of engine processes at
    // once. uci.cpp allocates on demand instead.

    init_eval_masks();

    nnue_init("thrawn-nn-2.nnue");
}
