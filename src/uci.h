#ifndef UCI_H
#define UCI_H

#include <atomic>
#include <cstdint>
#include <string>
#include "position.h"

using namespace std;

extern std::atomic<int> quit;
extern int movestogo;
extern int movetime;
extern int uci_time;
extern int inc;
extern std::int64_t starttime;
extern std::int64_t stoptime;
extern int timeset;
extern std::atomic<int> stopped;
extern int numThreads;

// UCI PROTOCOL
int uci_parse_move(thrawn::Position* pos, const char* move_str);

void uci_parse_position(thrawn::Position* pos, const char* command);

void uci_parse_go(thrawn::Position* pos, const char* command);

void uci_loop(thrawn::Position* pos);


// TIME CONTROL 
int input_waiting();

void read_input();

std::int64_t get_time_ms();

void communicate();

void reset_time_control();

#endif
