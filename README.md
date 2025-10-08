# Thrawn

Thrawn is a free UCI-compliant chess engine. I made this to learn about C++ and performance optmizations.

Thrawn does not include a graphical user interface (GUI) that is required to display a chessboard and to make it easy to input moves. These GUIs are developed independently from Thrawn and are available online. Some recommended GUIs include:
- CuteChess: https://github.com/cutechess/cutechess
- Arena GUI: http://www.playwitharena.de/
- Scid vs PC: https://scidvspc.sourceforge.net/

## Rating/ELO
The rating of Thrawn is evaluated by [CCRL](https://computerchess.org.uk/ccrl/404/) <br>
- Thrawn v2.2 **~3000 ELO** (SF NNUE)
- Thrawn v2.0: **~2800 ELO** (SF NNUE)
- [Thrawn v1.1](https://computerchess.org.uk/ccrl/404/cgi/compare_engines.cgi?class=None&only_best_in_class=on&num_best_in_class=1&e=Thrawn+1.1+64-bit&print=Rating+list&profile_step=50&profile_numbers=1&print=Results+table&print=LOS+table&table_size=100&ct_from_elo=0&ct_to_elo=10000&match_length=30&cross_tables_for_best_versions_only=1&sort_tables=by+rating&diag=0&reference_list=None&recalibrate=no): **~1900 ELO**

## Compiling Thrawn
Version v2.1 and later: supports x64 and ARM chips. Can compile for linux, macos, or windows.

Older verions: only supports x64 and windows compilation.

Ensure you have a gcc compiler version 7.3 or later.

### Building from source

**Compiling Thrawn v2.0 and newer:**
```bash
git clone https://github.com/feftywacky/Thrawn.git
cd Thrawn
cd src

# release build
make # or mingw32-make

# debug build (only available for v3.0 or newer)
make BUILD=debug # or mingw32-make BUILD=debug
```

To clean the build:
```bash
make clean # or mingw32-make clean
```

**Compiling Thrawn v1.1 and older:**
```bash
git clone https://github.com/feftywacky/Thrawn.git
cd Thrawn
cd src
g++ -std=c++17 -Ofast -flto -o Thrawn *.cpp
```

## Testing:
Open source CLI tool to run matches between chess engines: https://github.com/Disservin/fastchess

## Evaluation:
- Thrawn v2.0 and newer uses NNUE
- Thrawn v1.1 uses handcrafted evaluation

## What's Next
- [ ] More move ordering heuristics (counter moves, history moves etc.)
- [ ] Train neural network

## Features
- NNUE for evaluation
- Bitboard data structure
- Multithreaded search (Lazy SMP)
- Transposition table
- Time allocation/Control
- UCI protocol

## References
### General
- https://www.chessprogramming.org/Main_Page
- https://github.com/bluefeversoft/vice
- https://github.com/official-stockfish/Stockfish
- https://github.com/mkd/gargantua 
### Search Algorithms
- https://web.archive.org/web/20071006042845/http://www.brucemo.com/compchess/programming/index.htm
### NNUE
- https://hxim.github.io/Stockfish-Evaluation-Guide/ <br>
- nnue probe framework: https://github.com/dshawul/nnue-probe
- nnue dataset https://tests.stockfishchess.org/nns
