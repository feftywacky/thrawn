#ifndef PERFT_H
#define PERFT_H

#include <cstdint>

#include "position.h"

std::uint64_t perft_search(thrawn::Position* pos, int depth);

std::uint64_t perft_test(thrawn::Position* pos, int depth);

void perft_run_unit_tests();

#endif
