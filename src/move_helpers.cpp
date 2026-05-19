#include <iostream>
#include <vector>
#include "constants.h"
#include "move_helpers.h"

/* 
 * METHODS BELOW ENCODE AND DECODED MOVE NUMBER 
*/

// PRINTING 


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

void print_move_list(const std::vector<int>& move_list) 
{
    if (move_list.size()==0)
        std::cout<<"NO MOVES AVAILABLE"<<std::endl;

    for (int move : move_list) {
        std::cout << square_to_coordinates[get_move_source(move)]
                  << square_to_coordinates[get_move_target(move)]
                  << promoted_pieces.at(get_promoted_piece(move))
                  << " piece: " << ascii_pieces[get_move_piece(move)]
                  << " " << (get_is_capture_move(move) ? 1 : 0)
                  << " " << (get_is_double_pawn_move(move) ? 1 : 0)
                  << " " << (get_is_move_enpassant(move) ? 1 : 0)
                  << " " << (get_is_move_castling(move) ? 1 : 0)
                  << "" << std::endl;
    }
}
