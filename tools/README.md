# Search Constant Tuning

This directory contains the local SPSA tuning harness for Thrawn's UCI-exposed
search constants. It uses Fastchess to play paired self-play games, then nudges
the selected `Search*` UCI options toward the side that scored better.

## Files

- `tools/spsa_tune.py`: SPSA tuning script.
- `tools/search_spsa_params.json`: tunable parameter list and current SPSA center values.
- `tools/fastchess`: macOS/Linux Fastchess binary.
- `tools/fastchess.exe`: Windows Fastchess binary.
- `tools/UHO_Lichess_4852_v1.epd`: opening book.

## 1. Build The Engine

From the repo root:

```bash
make -C src BUILD=release
```

On Windows with MinGW:

```powershell
mingw32-make -C src BUILD=release
```

The engine path is explicit. Use `--engine ./src/Thrawn` on macOS/Linux and
`--engine ./src/Thrawn.exe` on Windows, or point `--engine` at any other UCI
engine binary you want to tune.

## 2. Create Or Refresh The Parameter File

Run this after adding/removing UCI tuning options:

```bash
python3 tools/spsa_tune.py \
  --engine ./src/Thrawn \
  --write-template \
  --params tools/search_spsa_params.json
```

Windows:

```powershell
python tools\spsa_tune.py `
  --engine .\src\Thrawn.exe `
  --write-template `
  --params tools\search_spsa_params.json
```

Review `tools/search_spsa_params.json` before a long run:

- `enabled`: whether SPSA tunes this parameter.
- `value`: current center value.
- `c`: plus/minus perturbation size.
- `r`: learning rate scale.
- `min` / `max`: hard UCI clamps.

The committed template starts with a conservative first pass: futility,
razoring, null move pruning, late move pruning, LMR, and qsearch delta.

## 3. Run A Pilot Tune

Use this for a quick first signal:

```bash
python3 tools/spsa_tune.py \
  --engine ./src/Thrawn \
  --book tools/UHO_Lichess_4852_v1.epd \
  --book-format epd \
  --book-order random \
  --tc 10+0.1 \
  --rounds 8 \
  --iterations 200 \
  --concurrency 4 \
  --threads 1 \
  --hash 16
```

Windows:

```powershell
python tools\spsa_tune.py `
  --engine .\src\Thrawn.exe `
  --book tools\UHO_Lichess_4852_v1.epd `
  --book-format epd `
  --book-order random `
  --tc 10+0.1 `
  --rounds 8 `
  --iterations 200 `
  --concurrency 4 `
  --threads 1 `
  --hash 16
```

Use `--book-order random` for real tuning. `sequential` is useful only for
smoke tests because each SPSA iteration starts a fresh Fastchess process.

## 4. Resume Or Continue A Tune

The default state file is `tune/search_spsa_state.json`. Re-run the same command
with more iterations and the tuner resumes from that state:

```bash
python3 tools/spsa_tune.py \
  --engine ./src/Thrawn \
  --book tools/UHO_Lichess_4852_v1.epd \
  --book-format epd \
  --book-order random \
  --tc 10+0.1 \
  --rounds 8 \
  --iterations 500 \
  --concurrency 4 \
  --threads 1 \
  --hash 16
```

Delete `tune/search_spsa_state.json` to restart from `tools/search_spsa_params.json`.

## 5. Inspect Current Candidate Values

Print the rounded candidate constants from the current state without running
games:

```bash
python3 tools/spsa_tune.py \
  --engine ./src/Thrawn \
  --params tools/search_spsa_params.json \
  --state tune/search_spsa_state.json \
  --iterations 0
```

The output is a list of `setoption` lines. Those are candidates, not proof.

## 6. Validate Before Baking Constants In

Run a separate default-vs-tuned Fastchess match using the printed `setoption`
values on the tuned side. Only bake the constants into `src/search_params.*`
after the tuned engine wins a separate validation match with enough games for
your confidence target.

## Useful Flags

- `--fastchess PATH`: override the Fastchess binary. Defaults to `tools/fastchess` or `tools/fastchess.exe`.
- `--state PATH`: use a different resumable state file.
- `--pgn-dir PATH`: write per-iteration PGNs. Use `--pgn-dir ''` to disable PGNs.
- `--fastchess-workdir PATH`: where Fastchess writes its autosave config.
- `--verbose-fastchess`: print each Fastchess command and full output.
- `--no-progress`: disable the progress bar and print plain summaries.

## Iteration Counts

With `--rounds 8 --repeat`, each iteration plays 16 games.

- `200` iterations: pilot run, about 3200 games.
- `500-1000` iterations: better candidate search.
- Separate validation match: required before committing constants.
