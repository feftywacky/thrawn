#ifndef ZOBRIST_HASHING_H
#define ZOBRIST_HASHING_H

#include <cstdint>
#include "position.h"

void init_hashkeys(thrawn::Position* pos);
uint64_t gen_hashkey(thrawn::Position* pos);


#endif 