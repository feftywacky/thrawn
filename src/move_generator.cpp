#include <iostream>
#include <vector>
#include "move_generator.h"
#include "bitboard.h"
#include "constants.h"
#include "bitboard_helpers.h"
#include "move_helpers.h"
#include "nnue.h"
#include "zobrist_hashing.h"
#include "search.h"
#include "position.h"

using namespace std;


vector<int> generate_moves(thrawn::Position* pos)
{
    vector<int> moves;

    // the squares where the pieces started from, and where it will go
    int source;
    int target;

    // bitboard of the current piece, and its attacks
    uint64_t curr; 
    uint64_t attacks;

    // loop over all piece types for both black and white
    for (int piece = P; piece<=k; piece++)
    {
        uint64_t curr = pos->piece_bitboards[piece];

        // to distinguish betweem white and black specific moves
        // includes pawns and castling
        if (pos->colour_to_move == white)
        {
            // generate pawn moves
            if (piece==P)
            {
                // double pawn moves, pawn promotion, enpassant
                parse_white_pawn_moves(pos,curr, moves);
            }

            // castling
            if (piece == K)
            {
                parse_white_castle_moves(pos,moves);
            }
        }

        // for black pieces
        else
        {
            if (piece==p)
            {
                parse_black_pawn_moves(pos,curr, moves);
            }

            // castling
            if (piece == k)
            {
               parse_black_castle_moves(pos,moves);
            }
        }

        // generate for the rest of the pieces that are not colour specific
        // knight
        if ( (pos->colour_to_move == white) ? piece == N : piece == n )
        {
            parse_knight_moves(pos,curr, piece, moves);
        }

        // bishop
        if ( (pos->colour_to_move == white) ? piece == B : piece == b )
        {
            parse_bishop_moves(pos,curr, piece, moves);
        }
        
        // rook
        if ( (pos->colour_to_move == white) ? piece == R : piece == r )
        {
           parse_rook_moves(pos,curr, piece, moves);
        }

        // queen
        if ( (pos->colour_to_move == white) ? piece == Q : piece == q )
        {
            parse_queen_moves(pos,curr, piece, moves);
        }

        // king
        if ( (pos->colour_to_move == white) ? piece == K : piece == k )
        {
            parse_king_moves(pos,curr, piece, moves);
        }

    } // end of looping through all pieces

    return moves;


}

void parse_white_pawn_moves(thrawn::Position* pos, uint64_t& curr, vector<int>& moves)
{
    while (curr)
    {
        int source = get_lsb_index(curr);
        int target = source - 8; // go up one square

        if (target>=a8 && !get_bit(pos->occupancies[both], target))
        {
            // pawn promotion by going up one square (NOT TAKING A PIECE)
            if (source>=a7 && source<=h7)
            {
                moves.push_back(parse_move(source, target, P, Q, 0, 0, 0, 0));
                moves.push_back(parse_move(source, target, P, R, 0, 0, 0, 0));
                moves.push_back(parse_move(source, target, P, N, 0, 0, 0, 0));
                moves.push_back(parse_move(source, target, P, B, 0, 0, 0, 0));
            }

            // one square and two square pawn moves
            else
            {
                // one square
                moves.push_back(parse_move(source, target, P, 0, 0, 0, 0, 0));

                // two square
                if (source>=a2 && source<=h2 && !get_bit(pos->occupancies[both], target-8))
                {
                    moves.push_back(parse_move(source, target-8, P, 0, 0, 1, 0, 0));
                }
            }
        }

        uint64_t attacks = pos->pawn_attacks[pos->colour_to_move][source]  & pos->occupancies[black];

        while (attacks) // while attacks squares are present on the board
        {   
            target = get_lsb_index(attacks);

            if (source>=a7 && source<=h7) // pawn promotions by capturing a piece
            {
                moves.push_back(parse_move(source, target, P, Q, 1, 0, 0, 0));
                moves.push_back(parse_move(source, target, P, R, 1, 0, 0, 0));
                moves.push_back(parse_move(source, target, P, N, 1, 0, 0, 0));
                moves.push_back(parse_move(source, target, P, B, 1, 0, 0, 0));
            }

            // diagonal pawn capture
            else
            {
                moves.push_back(parse_move(source, target, P, 0, 1, 0, 0, 0));
            }

            pop_bit(attacks, target);
        }

        // enpassant
        if (pos->enpassant!=null_sq)
        {
            
            uint64_t enpassant_attacks = pos->pawn_attacks[pos->colour_to_move][source] & (1ULL << pos->enpassant);
            if (enpassant_attacks)
            {
                int enpassant_target = get_lsb_index(enpassant_attacks);
                moves.push_back(parse_move(source, enpassant_target, P, 0, 1, 0, 1, 0));
            }
        }

        // remove ls1b for looping through all the bits
        pop_bit(curr, source);
    }
}

void parse_white_castle_moves(thrawn::Position* pos, vector<int>& moves)
{
    if (pos->castle_rights & wks)
    {
        if (!get_bit(pos->occupancies[both], f1) && !get_bit(pos->occupancies[both], g1))
        {
            // make sure can't castle through check
            // if (!is_square_under_attack(e8, white) && !is_square_under_attack(f1, black) && !is_square_under_attack(g1, black))
            if (!is_square_under_attack(pos,e1, black) && !is_square_under_attack(pos,f1, black))
                moves.push_back(parse_move(e1, g1, K, 0, 0, 0, 0, 1));
        }
    }
    if (pos->castle_rights & wqs)
    {
        if (!get_bit(pos->occupancies[both], b1) && !get_bit(pos->occupancies[both], c1) && !get_bit(pos->occupancies[both], d1))
        {
            // make sure can't castle through check
            // if (!is_square_under_attack(e1, black) && !is_square_under_attack(c1, black) && !is_square_under_attack(d1, black))
            if (!is_square_under_attack(pos,e1, black) && !is_square_under_attack(pos,d1, black))
                moves.push_back(parse_move(e1, c1, K, 0, 0, 0, 0, 1));
        }
    }
}

void parse_black_pawn_moves(thrawn::Position* pos, uint64_t& curr, vector<int>& moves)
{
    while(curr) // while white pawns are present on the board
    {
        int source = get_lsb_index(curr);
        int target = source + 8; // go down one square

        if (target<=h1 && !get_bit(pos->occupancies[both], target))
        {
            // pawn promotion by going down one square (NOT TAKING A PIECE)
            if (source>=a2 && source<=h2)
            {
                moves.push_back(parse_move(source, target, p, q, 0, 0, 0, 0));
                moves.push_back(parse_move(source, target, p, r, 0, 0, 0, 0));
                moves.push_back(parse_move(source, target, p, n, 0, 0, 0, 0));
                moves.push_back(parse_move(source, target, p, b, 0, 0, 0, 0));
            }

            // one square and two square pawn moves
            else
            {
                // one square
                moves.push_back(parse_move(source, target, p, 0, 0, 0, 0, 0));

                // two square
                if (source>=a7 && source<=h7 && !get_bit(pos->occupancies[both], target+8))
                    moves.push_back(parse_move(source, target+8, p, 0, 0, 1, 0, 0));
            }
        }

        uint64_t attacks = pos->pawn_attacks[pos->colour_to_move][source] & pos->occupancies[white];

        while (attacks) // while attacks squares are present on the board
        {   
            target = get_lsb_index(attacks);

            if (source>=a2 && source<=h2) // pawn promotion by capturing piece
            {
                moves.push_back(parse_move(source, target, p, q, 1, 0, 0, 0));
                moves.push_back(parse_move(source, target, p, r, 1, 0, 0, 0));
                moves.push_back(parse_move(source, target, p, n, 1, 0, 0, 0));
                moves.push_back(parse_move(source, target, p, b, 1, 0, 0, 0));
            }

            // diagonal pawn capture
            else
            {
                moves.push_back(parse_move(source, target, p, 0, 1, 0, 0, 0));
            }

            pop_bit(attacks, target);
        }

        // enpassant
        if (pos->enpassant!=null_sq)
        {   
            uint64_t enpassant_attacks = pos->pawn_attacks[pos->colour_to_move][source] & (1ULL << pos->enpassant);
            if (enpassant_attacks)
            {
                int enpassant_target = get_lsb_index(enpassant_attacks);
                moves.push_back(parse_move(source, enpassant_target, p, 0, 1, 0, 1, 0));
            }
        }

        // remove ls1b for looping through all the bits
        pop_bit(curr, source);
    }
}

void parse_black_castle_moves(thrawn::Position* pos, vector<int>& moves)
{
    if (pos->castle_rights & bks)
    {
        if (!get_bit(pos->occupancies[both], f8) && !get_bit(pos->occupancies[both], g8))
        {
            // pruend by make_move() for g8
            // if (!is_square_under_attack(e8, white) && !is_square_under_attack(f8, white) && !is_square_under_attack(g8, white))
            if (!is_square_under_attack(pos,e8, white) && !is_square_under_attack(pos,f8, white))
                moves.push_back(parse_move(e8, g8, k, 0, 0, 0, 0, 1));
        }
    }
    if (pos->castle_rights & bqs)
    {
        if (!get_bit(pos->occupancies[both], b8) && !get_bit(pos->occupancies[both], c8) && !get_bit(pos->occupancies[both], d8))
        {
            // pruend by make_move() 
            // if (!is_square_under_attack(e8, white) && !is_square_under_attack(c8, white) && !is_square_under_attack(d8, white))
            if (!is_square_under_attack(pos, e8, white) && !is_square_under_attack(pos,d8, white))
                moves.push_back(parse_move(e8, c8, k, 0, 0, 0, 0, 1));
        }
    }
}

void parse_knight_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves)
{
    while (curr)
    {
        int source = get_lsb_index(curr);

        uint64_t attacks = pos->knight_attacks[source] & ( (pos->colour_to_move==white) ? ~pos->occupancies[white] : ~pos->occupancies[black]);

        while (attacks)
        {
            int target = get_lsb_index(attacks);
            
            // non-capture move
            if ( !get_bit( (pos->colour_to_move==white) ? pos->occupancies[black] : pos->occupancies[white], target ) )
            {
                moves.push_back(parse_move(source, target, piece, 0, 0, 0, 0, 0));
            }

            else
            {
                moves.push_back(parse_move(source, target, piece, 0, 1, 0, 0, 0));
            }

            pop_bit(attacks, target);
        }
        pop_bit(curr, source);
    }
}

void parse_bishop_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves)
{
    while(curr)
    {
        int source = get_lsb_index(curr);

        uint64_t attacks = get_bishop_attacks(pos, source, pos->occupancies[both]) & ( (pos->colour_to_move==white) ? ~pos->occupancies[white] : ~pos->occupancies[black]);
        while (attacks)
        {
            
            int target = get_lsb_index(attacks);
            
            // non-capture move
            if ( !get_bit( (pos->colour_to_move==white) ? pos->occupancies[black] : pos->occupancies[white], target ) )
            {
                moves.push_back(parse_move(source, target, piece, 0, 0, 0, 0, 0));
            }

            else
            {
                moves.push_back(parse_move(source, target, piece, 0, 1, 0, 0, 0));
            }

            pop_bit(attacks, target);
            
        }
        pop_bit(curr, source);
    } 
}

void parse_rook_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves)
{
    while(curr)
    {
        int source = get_lsb_index(curr);

        uint64_t attacks = get_rook_attacks(pos, source, pos->occupancies[both]) & ( (pos->colour_to_move==white) ? ~pos->occupancies[white] : ~pos->occupancies[black]);

        while (attacks)
        {
            int target = get_lsb_index(attacks);
            
            // non-capture move
            if ( !get_bit( (pos->colour_to_move==white) ? pos->occupancies[black] : pos->occupancies[white], target ) )
            {
                moves.push_back(parse_move(source, target, piece, 0, 0, 0, 0, 0));
            }

            else
            {
                moves.push_back(parse_move(source, target, piece, 0, 1, 0, 0, 0));
            }

            pop_bit(attacks, target);
        }
        pop_bit(curr, source);
    }
}

void parse_queen_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves)
{
    while (curr)
    {
        int source = get_lsb_index(curr);

        uint64_t attacks = get_queen_attacks(pos, source, pos->occupancies[both]) & ( (pos->colour_to_move==white) ? ~pos->occupancies[white] : ~pos->occupancies[black]);

        while (attacks)
        {
            int target = get_lsb_index(attacks);
            
            // non-capture move
            if ( !get_bit( (pos->colour_to_move==white) ? pos->occupancies[black] : pos->occupancies[white], target ) )
            {
                moves.push_back(parse_move(source, target, piece, 0, 0, 0, 0, 0));
            }

            else
            {
                moves.push_back(parse_move(source, target, piece, 0, 1, 0, 0, 0));
            }

            pop_bit(attacks, target);
        }
        pop_bit(curr, source);
    }
}

void parse_king_moves(thrawn::Position* pos, uint64_t& curr, const int& piece, vector<int>& moves)
{
    while (curr)
    {
        int source = get_lsb_index(curr);

        uint64_t attacks = pos->king_attacks[source] & ( (pos->colour_to_move==white) ? ~pos->occupancies[white] : ~pos->occupancies[black]);

        while (attacks)
        {
            int target = get_lsb_index(attacks);
            
            // non-capture move
            if ( !get_bit( (pos->colour_to_move==white) ? pos->occupancies[black] : pos->occupancies[white], target ) )
            {
                moves.push_back(parse_move(source, target, piece, 0, 0, 0, 0, 0));
            }

            else
            {
                moves.push_back(parse_move(source, target, piece, 0, 1, 0, 0, 0));
            }

            pop_bit(attacks, target);
        }
        pop_bit(curr, source);
    }
}

int make_move(thrawn::Position* pos, int move, int move_type, int ply)
{
    if (move_type == all_moves)
    {
        copyBoard(pos);
        // if(ply!=-1)
        // {
        //     pos->undo_stack[ply].move           = move;
        //     pos->undo_stack[ply].captured_piece = -1;
        //     pos->undo_stack[ply].castle_rights  = pos->castle_rights;
        //     pos->undo_stack[ply].enpassant      = pos->enpassant;
        //     pos->undo_stack[ply].fifty_move     = pos->fifty_move;
        //     pos->undo_stack[ply].zobristKey     = pos->zobristKey;
        // }

        // move parsing
        int source = get_move_source(move);
        int target = get_move_target(move);
        int piece = get_move_piece(move);
        int promoted_piece = get_promoted_piece(move);
        int is_capture_move = get_is_capture_move(move);
        int double_pawn_move = get_is_double_pawn_move(move);
        int enpassant_move = get_is_move_enpassant(move);
        int castling = get_is_move_castling(move);
        const int nnue_ply = (ply >= 0) ? ply : pos->ply;
        const bool use_nnue = nnue_loaded();

        if (use_nnue)
            nnue_copy_parent_to_child(pos, nnue_ply);

        // move piece
        pop_bit(pos->piece_bitboards[piece], source);
        set_bit(pos->piece_bitboards[piece], target);
        pos->zobristKey ^= pos->piece_hashkey[piece][source]; // update hash to exclude source
        pos->zobristKey ^= pos->piece_hashkey[piece][target]; // update hash to include target
        if (use_nnue)
        {
            nnue_remove_piece(pos, nnue_ply, piece, source);
        }
        pos->fifty_move++;

        // if pawn moved reset fifty-move rule
        if (piece == P || piece == p)
            pos->fifty_move = 0;

        // if capture move, remove the piece being captured from its corresponding bitboard
        // ie. if white pawn captures black kngiht, remove black knight from black knight bitboard
        if (is_capture_move)
        {
            // if captured a piece reset fifty-move rule
            pos->fifty_move = 0;
            int start_piece;
            int end_piece;
            
            (pos->colour_to_move==white) ? start_piece = p : start_piece = P;
            (pos->colour_to_move==white) ? end_piece = k : end_piece = K;

            for(int i=start_piece; i<=end_piece;i++)
            {
                if (get_bit(pos->piece_bitboards[i], target))
                {
                    if (ply != -1)
                        pos->undo_stack[ply].captured_piece = i;

                    pop_bit(pos->piece_bitboards[i], target);
                    if (use_nnue)
                        nnue_remove_piece(pos, nnue_ply, i, target);
                    
                    // update hashkey to exclude captured piece
                    pos->zobristKey ^= pos->piece_hashkey[i][target];
                    break;
                }
            }
        }

        // handle pawn promotions
        if (promoted_piece)
        {
            // erase the pawn from the target square
            // pop_bit(piece_bitboards[(colour_to_move == white) ? P : p], target);
            if (pos->colour_to_move == white)
            {
                pop_bit(pos->piece_bitboards[P], target);
                pos->zobristKey ^= pos->piece_hashkey[P][target];
                if (use_nnue)
                    nnue_add_piece(pos, nnue_ply, promoted_piece, target);
            }
            else
            {
                pop_bit(pos->piece_bitboards[p], target);
                pos->zobristKey ^= pos->piece_hashkey[p][target];
                if (use_nnue)
                    nnue_add_piece(pos, nnue_ply, promoted_piece, target);
            }
            
            set_bit(pos->piece_bitboards[promoted_piece], target);
            pos->zobristKey ^= pos->piece_hashkey[promoted_piece][target];
        }

        if (use_nnue && !promoted_piece)
            nnue_add_piece(pos, nnue_ply, piece, target);

        // handle enpassant capture
        if (enpassant_move)
        {
            if (pos->colour_to_move==white)
            {
                pop_bit(pos->piece_bitboards[p], target + 8);
                pos->zobristKey ^= pos->piece_hashkey[p][target + 8];
                if (use_nnue)
                    nnue_remove_piece(pos, nnue_ply, p, target + 8);
            }
            else
            {
                pop_bit(pos->piece_bitboards[P], target - 8);
                pos->zobristKey ^= pos->piece_hashkey[P][target- 8];
                if (use_nnue)
                    nnue_remove_piece(pos, nnue_ply, P, target - 8);
            }
        }

        if (pos->enpassant!=null_sq)
        {
            pos->zobristKey ^= pos->enpassant_hashkey[pos->enpassant];
        }
        pos->enpassant = null_sq;

        // set enpassant square when pawn double moves
        if (double_pawn_move)
        {
            // (colour_to_move==white) ? enpassant = target + 8 : enpassant = target - 8;

            if (pos->colour_to_move == white)
            {
                pos->enpassant = target + 8;
                pos->zobristKey ^= pos->enpassant_hashkey[target+8];
            }
            else
            {
                pos->enpassant = target - 8;
                pos->zobristKey ^= pos->enpassant_hashkey[target-8];
            }
        }

        // handle castling
        if (castling)
        {
            if (target == g1)
            {
                pop_bit(pos->piece_bitboards[R], h1);
                set_bit(pos->piece_bitboards[R], f1);
                if (use_nnue)
                {
                    nnue_remove_piece(pos, nnue_ply, R, h1);
                    nnue_add_piece(pos, nnue_ply, R, f1);
                }

                pos->zobristKey ^= pos->piece_hashkey[R][h1];  // remove rook from h1 from hash key
                pos->zobristKey ^= pos->piece_hashkey[R][f1];  // put rook on f1 into a hash key
            }
            else if (target == c1)
            {
                pop_bit(pos->piece_bitboards[R], a1);
                set_bit(pos->piece_bitboards[R], d1);
                if (use_nnue)
                {
                    nnue_remove_piece(pos, nnue_ply, R, a1);
                    nnue_add_piece(pos, nnue_ply, R, d1);
                }

                pos->zobristKey ^= pos->piece_hashkey[R][a1];  // remove rook from a1 from hash key
                pos->zobristKey ^= pos->piece_hashkey[R][d1];  // put rook on d1 into a hash key
            }
            else if (target == g8)
            {
                pop_bit(pos->piece_bitboards[r], h8);
                set_bit(pos->piece_bitboards[r], f8);
                if (use_nnue)
                {
                    nnue_remove_piece(pos, nnue_ply, r, h8);
                    nnue_add_piece(pos, nnue_ply, r, f8);
                }

                pos->zobristKey ^= pos->piece_hashkey[r][h8];  // remove rook from h8 from hash key
                pos->zobristKey ^= pos->piece_hashkey[r][f8];  // put rook on f8 into a hash key
            }
            else if (target == c8)
            {
                pop_bit(pos->piece_bitboards[r], a8);
                set_bit(pos->piece_bitboards[r], d8);
                if (use_nnue)
                {
                    nnue_remove_piece(pos, nnue_ply, r, a8);
                    nnue_add_piece(pos, nnue_ply, r, d8);
                }

                pos->zobristKey ^= pos->piece_hashkey[r][a8];  // remove rook from a8 from hash key
                pos->zobristKey ^= pos->piece_hashkey[r][d8];  // put rook on d8 into a hash key
            }         
        }
        pos->zobristKey ^= pos->castling_hashkey[pos->castle_rights]; // remove castling right hash

        // updating castling rights after every move
        pos->castle_rights &= update_castling_right_values[source];
        pos->castle_rights &= update_castling_right_values[target];

        pos->zobristKey ^= pos->castling_hashkey[pos->castle_rights]; // update castling right hash

        // update colour occupancies
        pos->occupancies[white] = get_white_occupancy(pos);
        pos->occupancies[black] = get_black_occupancy(pos);
        pos->occupancies[both]  = get_both_occupancy(pos);

        // change sides
        pos->colour_to_move ^= 1;

        pos->zobristKey ^= pos->colour_to_move_hashkey;
        
        // uint64_t curr_hash = gen_hashkey(); // new hashkey after move made
        // if (curr_hash != zobristKey)
        // {
        //     cout<<"make_move()"<<"\n";
        //     cout<<"move: ";
        //     print_move(move);
        //     cout<<"\n";
        //     print_board(colour_to_move);
        //     cout<<"correct hashkey: "<<std::hex<<curr_hash<<"\n";
        //     cin.get();
        // }

        // handle illegal moves. if move causes king to check, restore previous position and return illegal move
        if (is_square_under_attack(pos,(pos->colour_to_move==white) ? get_lsb_index(pos->piece_bitboards[k]) : get_lsb_index(pos->piece_bitboards[K]), pos->colour_to_move))
        {   
            restoreBoard(pos);
            return 0;
        }
        else
        {
            nnue_debug_check(pos);
            return 1;
        }
    }
    
    else if (move_type == only_captures)
    {
        if (get_is_capture_move(move)) {
            return make_move(pos, move, all_moves, ply);
        }
        else    
            return 0;
    }

    return 0;
}

int make_root_move(thrawn::Position* pos, int move, int move_type)
{
    pos->ply = 1;
    if (!make_move(pos, move, move_type, pos->ply))
    {
        pos->ply = 0;
        return 0;
    }

    pos->nnue_stack[0] = pos->nnue_stack[1];
    pos->ply = 0;
    nnue_debug_check(pos);
    return 1;
}

// void unmake_move(thrawn::Position* pos, int ply)
// {
//     thrawn::UndoData& ud = pos->undo_stack[ply];
//     int move = ud.move;

//     int source         = get_move_source(move);
//     int target         = get_move_target(move);
//     int piece          = get_move_piece(move);
//     int promoted_piece = get_promoted_piece(move);
//     int is_capture     = get_is_capture_move(move);
//     int enpassant_move = get_is_move_enpassant(move);
//     int castling_move  = get_is_move_castling(move);

//     pos->colour_to_move ^= 1;

//     // Restore old Zobrist, castle rights, enpassant, halfmove from undo stack
//     pos->zobristKey    = ud.zobristKey;
//     pos->castle_rights = ud.castle_rights;
//     pos->enpassant     = ud.enpassant;
//     pos->fifty_move    = ud.fifty_move;

//     if (promoted_piece)
//     {
//         // Remove the new piece from target
//         pop_bit(pos->piece_bitboards[promoted_piece], target);
//         // Put the original pawn back on source
//         set_bit(pos->piece_bitboards[piece], source);
//     }
//     else
//     {
//         // No promotion: occupant on target was 'piece', so remove it
//         pop_bit(pos->piece_bitboards[piece], target);
//         // Return it to source
//         set_bit(pos->piece_bitboards[piece], source);
//     }

//     // If there was a capture, restore the captured piece
//     if (is_capture)
//     {
//         int captured = ud.captured_piece; // e.g. black knight, black rook, etc.
//         if (captured != -1) {
//             set_bit(pos->piece_bitboards[captured], target);
//         }
//     }

//     // If enpassant capture, restore the captured pawn
//     if (enpassant_move) {
//         // figure out the square of the captured pawn
//         if (pos->colour_to_move == white) {
//             // black pawn was captured
//             set_bit(pos->piece_bitboards[p], target + 8);
//         } else {
//             // white pawn was captured
//             set_bit(pos->piece_bitboards[P], target - 8);
//         }
//     }

//     // If castling, move the rook back
//     if (castling_move) {
//         if (target == g1) {
//             // White O-O
//             pop_bit(pos->piece_bitboards[R], f1);
//             set_bit(pos->piece_bitboards[R], h1);
//         }
//         else if (target == c1) {
//             // White O-O-O
//             pop_bit(pos->piece_bitboards[R], d1);
//             set_bit(pos->piece_bitboards[R], a1);
//         }
//         else if (target == g8) {
//             // Black O-O
//             pop_bit(pos->piece_bitboards[r], f8);
//             set_bit(pos->piece_bitboards[r], h8);
//         }
//         else if (target == c8) {
//             // Black O-O-O
//             pop_bit(pos->piece_bitboards[r], d8);
//             set_bit(pos->piece_bitboards[r], a8);
//         }
//     }

//     pos->occupancies[white] = get_white_occupancy(pos);
//     pos->occupancies[black] = get_black_occupancy(pos);
//     pos->occupancies[both]  = pos->occupancies[white] | pos->occupancies[black];
// }
