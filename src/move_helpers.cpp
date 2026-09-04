#include "move_helpers.h"

#include "constants.h"

#include <iostream>

void print_move(const int& move)
{
    if (get_promoted_piece(move))
    {
    std::cout << square_to_coordinates[get_move_source(move)]
              << square_to_coordinates[get_move_target(move)]
              << promoted_pieces.at(get_promoted_piece(move));
    }
    else
    {
        std::cout << square_to_coordinates[get_move_source(move)]
              << square_to_coordinates[get_move_target(move)];
    }
}

