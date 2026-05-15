#include "misc.h"
#include "bitboard.h"
#include "transposition_table.h"
#include "evaluation.h"
#include "nnue.h"
#include "globals.h"

const std::string version = " v2.2";


void init_all()
{
    // init_magic_nums(); // used to help generate magic bitboards
    
    // init_leaping_attacks(pos);
    // init_sliding_attacks(pos, bishop);
    // init_sliding_attacks(pos, rook);

    // init hashkeys
    // init_hashkeys();

    tt->initTable(128); // default of 128 MB

    init_eval_masks();

    nnue_init("model_v4.nnue");
}
