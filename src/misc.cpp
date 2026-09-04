#include "misc.h"
#include "bitboard.h"
#include "transposition_table.h"
#include "evaluation.h"
#include "nnue.h"
#include "globals.h"
#include "constants.h"

const std::string version = " v3.2";


void init_all()
{
    // init_magic_nums(); // kept: regenerates the magic bitboards in constants.cpp
    // Attack tables and Zobrist keys are initialised lazily by Position.

    init_eval_masks();

    // Loads the network embedded in the binary; `setoption name EvalFile` can
    // override it with one on disk.
    nnue_init(nullptr);
}
