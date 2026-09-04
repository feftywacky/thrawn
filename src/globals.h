#ifndef GLOBALS_H
#define GLOBALS_H

#include "transposition_table.h"
#include "threading.h"
#include "position.h"
#include "constants.h"

extern TranspositionTable* tt;
extern thrawn::Position* pos;
extern ThreadData threadDatas[MAX_THREADS];

#endif // GLOBALS_H