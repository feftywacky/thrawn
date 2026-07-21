# Thrawn

Thrawn is a high performance uci-compliant chess engine written in C++.

## Rating
Thrawn's ratings are tracked by [CCRL](https://computerchess.org.uk/ccrl/404/).

| Version | Rating | Eval |
| --- | --- | --- |
| [v3.1 in testing](https://computerchess.org.uk/404/cgi/compare_engines.cgi?family=Thrawn&print=Rating+list&print=Results+table&print=LOS+table&print=Ponder+hit+table&print=Eval+difference+table&print=Comopp+gamenum+table&print=Overlap+table&print=Score+with+common+opponents) | ~3400 ELO | [thrawn-nn-2](https://github.com/feftywacky/thrawn-nnue/releases/tag/thrawn-nn-2) |
| [v3.0](https://computerchess.org.uk/404/cgi/compare_engines.cgi?family=Thrawn&print=Rating+list&print=Results+table&print=LOS+table&print=Ponder+hit+table&print=Eval+difference+table&print=Comopp+gamenum+table&print=Overlap+table&print=Score+with+common+opponents) | ~2941 ELO | [thrawn-nn-1](https://github.com/feftywacky/thrawn-nnue/releases/tag/thrawn-nn-1) |
| v2.2 | ~2900 ELO | Stockfish NNUE |
| v2.0 | ~2800 ELO | Stockfish NNUE |
| [v1.1](https://computerchess.org.uk/ccrl/404/cgi/compare_engines.cgi?class=None&only_best_in_class=on&num_best_in_class=1&e=Thrawn+1.1+64-bit&print=Rating+list&profile_step=50&profile_numbers=1&print=Results+table&print=LOS+table&table_size=100&ct_from_elo=0&ct_to_elo=10000&match_length=30&cross_tables_for_best_versions_only=1&sort_tables=by+rating&diag=0&reference_list=None&recalibrate=no) | ~1896 ELO | HCE |

## Getting Started
Download from latest release or build from source. Released binaries have the
NNUE network compiled in, so they run standalone — no `.nnue` file needs to sit
next to the executable.

```bash
git clone https://github.com/feftywacky/Thrawn.git
cd Thrawn
make                    # native build for the current machine
build/thrawn-v3.1-*     # run the produced binary
```

All binaries are written to a `build/` directory at the repo root and named
`thrawn-v<version>-<os>-<simd>[.exe]`.

To clean intermediates (objects), or wipe the whole `build/` directory:
```bash
make clean       # remove .o / .d intermediates
make distclean   # also remove build/
```

### Embedded NNUE network

The default network is compiled directly into the binary (`nn/thrawn-nn-2.nnue`
is embedded via `.incbin` at `src/nnue_embedded_data.S`), so building requires
that file to be present:

```bash
make NNUE_FILE=/path/to/other.nnue   # embed a different network instead
```

A different net can still be swapped in at runtime without rebuilding, via the
UCI `EvalFile` option — it's tried first, and the engine falls back to the
embedded network if it fails to load:

```
setoption name EvalFile value /path/to/other.nnue
```

## Builds

Pick the binary that matches your OS and CPU:

| Binary | Platform | CPU requirement |
| --- | --- | --- |
| `thrawn-v3.1-macos-arm-neon` | macOS (Apple Silicon) | M1 or newer |
| `thrawn-v3.1-windows-x64-avx2.exe` | Windows x64 | AVX2 (Haswell / Zen1 or newer) |
| `thrawn-v3.1-windows-x64-avx512.exe` | Windows x64 | AVX-512 (Intel Ice Lake / AMD Zen4 or newer) |
| `thrawn-v3.1-linux-x64-avx2` | Linux x64 | AVX2 (Haswell / Zen1 or newer) |
| `thrawn-v3.1-linux-x64-avx512` | Linux x64 | AVX-512 (Intel Ice Lake / AMD Zen4 or newer) |

If unsure, use the **avx2** build — it runs on virtually all x86-64 CPUs from the
last decade. The Windows executables are statically linked (libstdc++ / libgcc /
winpthread are baked in), so they have no external GCC DLL dependencies.

### Building a specific target

```bash
# from the repo root
make TARGET=macos   ARCH=native                                          # Apple Silicon (NEON)
make TARGET=windows ARCH=x86-64-avx2   CXX=x86_64-w64-mingw32-g++         # Windows AVX2
make TARGET=windows ARCH=x86-64-avx512 CXX=x86_64-w64-mingw32-g++         # Windows AVX-512
make TARGET=linux   ARCH=x86-64-avx2   CXX=x86_64-unknown-linux-gnu-g++   # Linux AVX2
make TARGET=linux   ARCH=x86-64-avx512 CXX=x86_64-unknown-linux-gnu-g++   # Linux AVX-512
```

### Building all five at once

`make release` cross-builds every target into `build/`. It expects an Apple
Silicon macOS host with the cross toolchains installed via Homebrew, and
`nn/thrawn-nn-2.nnue` present (see [Embedded NNUE network](#embedded-nnue-network)):

```bash
brew install mingw-w64                                          # Windows cross compiler
brew install messense/macos-cross-toolchains/x86_64-unknown-linux-gnu   # Linux cross compiler
make release                                                    # from the repo root
```

## Testing:
Open source CLI tool to run matches between chess engines: https://github.com/Disservin/fastchess
