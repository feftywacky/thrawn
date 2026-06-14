# Thrawn

Thrawn is a free UCI-compliant chess engine.

## Rating
The rating of Thrawn is evaluated by [CCRL](https://computerchess.org.uk/ccrl/404/) <br>
- [Thrawn v3.0](https://computerchess.org.uk/404/cgi/compare_engines.cgi?family=Thrawn&print=Rating+list&print=Results+table&print=LOS+table&print=Ponder+hit+table&print=Eval+difference+table&print=Comopp+gamenum+table&print=Overlap+table&print=Score+with+common+opponents) **~2900 ELO** ([thrawn nnue ](https://github.com/feftywacky/thrawn-nnue/releases/tag/thrawn-nn-1) - rating not fully tested yet)
- Thrawn v2.2 **~3000 ELO** (using stockfish nnue)
- Thrawn v2.0: **~2800 ELO** (using stockfish nnue)
- [Thrawn v1.1](https://computerchess.org.uk/ccrl/404/cgi/compare_engines.cgi?class=None&only_best_in_class=on&num_best_in_class=1&e=Thrawn+1.1+64-bit&print=Rating+list&profile_step=50&profile_numbers=1&print=Results+table&print=LOS+table&table_size=100&ct_from_elo=0&ct_to_elo=10000&match_length=30&cross_tables_for_best_versions_only=1&sort_tables=by+rating&diag=0&reference_list=None&recalibrate=no): **~1900 ELO**

## Getting Started
Download from latest release or build from source.

```bash
git clone https://github.com/feftywacky/Thrawn.git
cd Thrawn
cd src
make
```

To clean the build:
```bash
make clean
```

## Testing:
Open source CLI tool to run matches between chess engines: https://github.com/Disservin/fastchess
