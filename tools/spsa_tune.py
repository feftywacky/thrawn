#!/usr/bin/env python3
"""Local SPSA tuner for Thrawn search constants using fastchess."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IS_WINDOWS = sys.platform.startswith("win")
EXECUTABLE_SUFFIX = ".exe" if IS_WINDOWS else ""


SCORE_RE = re.compile(
    r"Score of .*?:\s*(?P<wins>\d+)\s*-\s*(?P<losses>\d+)\s*-\s*(?P<draws>\d+)"
)
GAMES_RE = re.compile(
    r"Games:\s*(?P<games>\d+),\s*Wins:\s*(?P<wins>\d+),\s*"
    r"Losses:\s*(?P<losses>\d+),\s*Draws:\s*(?P<draws>\d+)"
)
OPTION_RE = re.compile(
    r"^option name (?P<name>.+?) type spin default (?P<default>-?\d+) "
    r"min (?P<min>-?\d+) max (?P<max>-?\d+)"
)


DEFAULT_ENABLED = {
    "SearchReverseFutilityMargin1",
    "SearchReverseFutilityMargin2",
    "SearchRazorMargin1",
    "SearchRazorMargin2",
    "SearchNullMoveMinDepth",
    "SearchNullMoveBaseReduction",
    "SearchNullMoveDepthDivisor",
    "SearchNullMoveEvalDivisor",
    "SearchNullMoveEvalBonusMax",
    "SearchNullMoveVerificationDepth",
    "SearchFutilityMargin1",
    "SearchFutilityMargin2",
    "SearchFutilityMargin3",
    "SearchLateMovePruningDepth1",
    "SearchLateMovePruningDepth2",
    "SearchLateMovePruningDepth3",
    "SearchLmrFullDepthMoves",
    "SearchLmrReductionDepthLimit",
    "SearchLmrBaseReduction",
    "SearchLmrNonPvDepth",
    "SearchLmrMoveDepth1",
    "SearchLmrMoveNumber1",
    "SearchLmrMoveDepth2",
    "SearchLmrMoveNumber2",
    "SearchLmrGoodHistoryDivisor",
    "SearchLmrBadHistoryDivisor",
    "SearchQsearchDeltaMargin",
}

PERTURBATION_OVERRIDES = {
    "SearchAspirationWindowDepth": 1.0,
    "SearchAspirationWindowSize": 4.0,
    "SearchAspirationThreadDelta": 1.0,
    "SearchAspirationThreadCycle": 1.0,
    "SearchCheckExtension": 1.0,
    "SearchHistoryMax": 1024.0,
    "SearchHistoryScoreCap": 500.0,
    "SearchHistoryBonusDepthSquared": 1.0,
    "SearchHistoryBonusDepthLinear": 2.0,
    "SearchCounterMoveScore": 500.0,
    "SearchCounterMoveHistoryDivisor": 8.0,
    "SearchCounterMoveHistoryCap": 25.0,
    "SearchTtMoveScore": 1000.0,
    "SearchPvMoveScore": 1000.0,
    "SearchQueenPromotionScore": 500.0,
    "SearchKillerMoveScore1": 500.0,
    "SearchKillerMoveScore2": 500.0,
    "SearchReverseFutilityMaxDepth": 1.0,
    "SearchReverseFutilityMargin1": 20.0,
    "SearchReverseFutilityMargin2": 30.0,
    "SearchReverseFutilityDepthFactor": 15.0,
    "SearchRazorMaxDepth": 1.0,
    "SearchRazorMargin1": 25.0,
    "SearchRazorMargin2": 40.0,
    "SearchRazorMarginDepthN": 50.0,
    "SearchNullMoveMinDepth": 1.0,
    "SearchNullMoveBaseReduction": 1.0,
    "SearchNullMoveDepthDivisor": 1.0,
    "SearchNullMoveEvalDivisor": 40.0,
    "SearchNullMoveEvalBonusMax": 1.0,
    "SearchNullMoveVerificationDepth": 1.0,
    "SearchFutilityMaxDepth": 1.0,
    "SearchFutilityMargin1": 15.0,
    "SearchFutilityMargin2": 25.0,
    "SearchFutilityMargin3": 35.0,
    "SearchLateMovePruningMaxDepth": 1.0,
    "SearchLateMovePruningDepth1": 2.0,
    "SearchLateMovePruningDepth2": 3.0,
    "SearchLateMovePruningDepth3": 4.0,
    "SearchLmrFullDepthMoves": 1.0,
    "SearchLmrReductionDepthLimit": 1.0,
    "SearchLmrBaseReduction": 1.0,
    "SearchLmrNonPvDepth": 1.0,
    "SearchLmrMoveDepth1": 1.0,
    "SearchLmrMoveNumber1": 2.0,
    "SearchLmrMoveDepth2": 1.0,
    "SearchLmrMoveNumber2": 4.0,
    "SearchLmrGoodHistoryDivisor": 1.0,
    "SearchLmrBadHistoryDivisor": 1.0,
    "SearchQsearchDeltaMargin": 25.0,
    "SearchSmpVoteScoreOffset": 2.0,
}


def default_fastchess_executable() -> str:
    return str(Path("tools") / f"fastchess{EXECUTABLE_SUFFIX}")


@dataclass
class TuneParam:
    name: str
    value: float
    min_value: int
    max_value: int
    c: float
    r: float
    enabled: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TuneParam":
        return cls(
            name=str(data["name"]),
            value=float(data.get("value", data.get("default", 0))),
            min_value=int(data["min"]),
            max_value=int(data["max"]),
            c=float(data.get("c", 1.0)),
            r=float(data.get("r", 0.002)),
            enabled=bool(data.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "min": self.min_value,
            "max": self.max_value,
            "c": self.c,
            "r": self.r,
            "enabled": self.enabled,
        }

    def rounded_value(self) -> int:
        return clamp_int(round(self.value), self.min_value, self.max_value)


def clamp_int(value: int | float, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))


def clamp_float(value: float, min_value: int, max_value: int) -> float:
    return max(float(min_value), min(float(max_value), value))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds == math.inf:
        return "--:--"

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    def __init__(self, total: int, width: int, enabled: bool) -> None:
        self.total = max(0, total)
        self.width = max(10, width)
        self.enabled = enabled and self.total > 0
        self.start_time = time.monotonic()
        self.current = 0
        self.last_message = ""
        self.last_line_len = 0

    def _line(self, current: int, label: str) -> str:
        fraction = current / self.total if self.total else 1.0
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self.start_time
        eta = None
        if current > 0 and current < self.total:
            eta = elapsed / current * (self.total - current)

        return (
            f"\rSPSA [{bar}] {current}/{self.total} "
            f"{fraction * 100:5.1f}% elapsed {format_duration(elapsed)} "
            f"eta {format_duration(eta)} {label}"
        )

    def render(self, current: int, label: str = "") -> None:
        if not self.enabled:
            return

        self.current = max(0, min(self.total, current))
        self.last_message = label
        line = self._line(self.current, label)
        padding = max(0, self.last_line_len - len(line))
        sys.stderr.write(line + " " * padding)
        sys.stderr.flush()
        self.last_line_len = len(line)

    def advance(self, label: str) -> None:
        self.render(self.current + 1, label)

    def clear(self) -> None:
        if not self.enabled or self.last_line_len == 0:
            return
        sys.stderr.write("\r" + " " * self.last_line_len + "\r")
        sys.stderr.flush()
        self.last_line_len = 0

    def finish(self) -> None:
        if not self.enabled:
            return
        if self.current < self.total:
            self.render(self.total, self.last_message)
        sys.stderr.write("\n")
        sys.stderr.flush()


def command_path(path_text: str) -> str:
    path = Path(path_text).expanduser()
    if path.exists() or path.parent != Path("."):
        return str(path.resolve())
    return path_text


def is_path_like(command: str) -> bool:
    path = Path(command)
    return path.is_absolute() or path.parent != Path(".")


def validate_command(command: str, label: str) -> bool:
    if is_path_like(command) and not Path(command).exists():
        print(f"{label} not found: {command}", file=sys.stderr)
        return False
    return True


def optional_file_path(path_text: str | None) -> str | None:
    if not path_text:
        return None
    return str(Path(path_text).expanduser().resolve())


def configure_paths(args: argparse.Namespace) -> None:
    args.fastchess_cmd = command_path(args.fastchess)
    args.engine_cmd = command_path(args.engine)

    if args.engine_dir:
        args.engine_dir_resolved = str(Path(args.engine_dir).expanduser().resolve())
    else:
        args.engine_dir_resolved = str(Path.cwd().resolve())

    args.book_path = optional_file_path(args.book)
    args.eval_file_path = optional_file_path(args.eval_file)
    args.pgn_dir_path = Path(args.pgn_dir).expanduser().resolve() if args.pgn_dir else None

    if args.fastchess_workdir:
        args.fastchess_workdir_path = Path(args.fastchess_workdir).expanduser().resolve()
    elif args.state:
        args.fastchess_workdir_path = Path(args.state).expanduser().resolve().parent / "fastchess-work"
    else:
        args.fastchess_workdir_path = Path.cwd().resolve() / "tune" / "fastchess-work"

    args.fastchess_workdir_path.mkdir(parents=True, exist_ok=True)


def discover_search_options(engine: str, engine_dir: str | None) -> list[dict[str, Any]]:
    process = subprocess.Popen(
        [engine],
        cwd=engine_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    process.stdin.write("uci\n")
    process.stdin.flush()

    options: list[dict[str, Any]] = []
    for line in process.stdout:
        text = line.strip()
        match = OPTION_RE.match(text)
        if match and match.group("name").startswith("Search"):
            name = match.group("name")
            default = int(match.group("default"))
            min_value = int(match.group("min"))
            max_value = int(match.group("max"))
            span_c = max(1.0, (max_value - min_value) / 20.0)
            options.append(
                {
                    "name": name,
                    "value": default,
                    "min": min_value,
                    "max": max_value,
                    "c": PERTURBATION_OVERRIDES.get(name, span_c),
                    "r": 0.002,
                    "enabled": name in DEFAULT_ENABLED,
                }
            )
        if text == "uciok":
            break

    process.stdin.write("quit\n")
    process.stdin.flush()
    process.communicate(timeout=5)
    return options


def load_params(params_path: Path, state_path: Path | None) -> tuple[int, list[TuneParam], list[dict[str, Any]]]:
    source_path = state_path if state_path is not None and state_path.exists() else params_path
    data = read_json(source_path)
    params = [TuneParam.from_dict(item) for item in data["parameters"]]
    return int(data.get("iteration", 0)), params, list(data.get("history", []))


def save_state(path: Path, iteration: int, params: list[TuneParam], history: list[dict[str, Any]]) -> None:
    write_json(
        path,
        {
            "iteration": iteration,
            "parameters": [param.to_dict() for param in params],
            "history": history[-200:],
        },
    )


def value_map(params: list[TuneParam], deltas: dict[str, int], sign: int, iteration: int, args: argparse.Namespace) -> dict[str, int]:
    values: dict[str, int] = {}
    k = iteration + 1
    for param in params:
        if not param.enabled:
            values[param.name] = param.rounded_value()
            continue

        c_k = param.c / math.pow(k, args.gamma)
        raw = param.value + sign * c_k * deltas[param.name]
        values[param.name] = clamp_int(round(raw), param.min_value, param.max_value)

    return values


def engine_args(args: argparse.Namespace, name: str, values: dict[str, int]) -> list[str]:
    opts = ["-engine", f"cmd={args.engine_cmd}", f"name={name}"]
    if args.engine_dir_resolved:
        opts.append(f"dir={args.engine_dir_resolved}")
    opts.append(f"option.Hash={args.hash}")
    opts.append(f"option.Threads={args.threads}")
    if args.eval_file_path:
        opts.append(f"option.EvalFile={args.eval_file_path}")
    for param_name, value in sorted(values.items()):
        opts.append(f"option.{param_name}={value}")
    return opts


def fastchess_command(args: argparse.Namespace, plus_values: dict[str, int], minus_values: dict[str, int], iteration: int) -> list[str]:
    command = [args.fastchess_cmd]
    command.extend(engine_args(args, "plus", plus_values))
    command.extend(engine_args(args, "minus", minus_values))
    command.extend(["-each", f"tc={args.tc}", "proto=uci"])
    command.extend(["-rounds", str(args.rounds), "-repeat"])
    command.extend(["-concurrency", str(args.concurrency)])
    command.extend(["-report", "penta=false"])

    if args.book_path:
        command.extend([
            "-openings",
            f"file={args.book_path}",
            f"format={args.book_format}",
            f"order={args.book_order}",
        ])

    if args.pgn_dir_path:
        pgn_dir = args.pgn_dir_path
        pgn_dir.mkdir(parents=True, exist_ok=True)
        command.extend([
            "-pgnout",
            f"file={pgn_dir / f'spsa_{iteration + 1:05d}.pgn'}",
            "notation=uci",
            "append=false",
        ])

    return command


def parse_fastchess_score(output: str) -> tuple[int, int, int] | None:
    result: tuple[int, int, int] | None = None
    for line in output.splitlines():
        match = SCORE_RE.search(line)
        if match:
            result = (
                int(match.group("wins")),
                int(match.group("losses")),
                int(match.group("draws")),
            )
            continue

        match = GAMES_RE.search(line)
        if match:
            result = (
                int(match.group("wins")),
                int(match.group("losses")),
                int(match.group("draws")),
            )
    return result


def parse_pgn_score(path: Path) -> tuple[int, int, int] | None:
    if not path.exists():
        return None

    wins = losses = draws = 0
    white = black = result = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("[White "):
            white = line.split('"', 2)[1]
        elif line.startswith("[Black "):
            black = line.split('"', 2)[1]
        elif line.startswith("[Result "):
            result = line.split('"', 2)[1]
        elif not line and white and black and result:
            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0" and white == "plus") or (result == "0-1" and black == "plus"):
                wins += 1
            elif result in {"1-0", "0-1"}:
                losses += 1
            white = black = result = None

    if white and black and result:
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0" and white == "plus") or (result == "0-1" and black == "plus"):
            wins += 1
        elif result in {"1-0", "0-1"}:
            losses += 1

    games = wins + losses + draws
    return (wins, losses, draws) if games else None


def run_match(args: argparse.Namespace, plus_values: dict[str, int], minus_values: dict[str, int], iteration: int) -> tuple[int, int, int]:
    command = fastchess_command(args, plus_values, minus_values, iteration)
    if args.verbose_fastchess:
        print(" ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=args.fastchess_workdir_path,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if args.verbose_fastchess:
        print(output, flush=True)

    if completed.returncode != 0:
        if not args.verbose_fastchess and output:
            print(output, file=sys.stderr, flush=True)
        raise RuntimeError(f"fastchess failed with exit code {completed.returncode}")

    score = parse_fastchess_score(output)
    if score is not None:
        return score

    if args.pgn_dir_path:
        pgn_path = args.pgn_dir_path / f"spsa_{iteration + 1:05d}.pgn"
        score = parse_pgn_score(pgn_path)
        if score is not None:
            return score

    raise RuntimeError("could not parse fastchess score")


def update_params(params: list[TuneParam], deltas: dict[str, int], wins: int, losses: int, iteration: int, args: argparse.Namespace) -> None:
    signal = wins - losses
    k = iteration + 1 + args.stability_offset

    for param in params:
        if not param.enabled:
            continue

        c_k = param.c / math.pow(iteration + 1, args.gamma)
        r_k = param.r / math.pow(k, args.alpha)
        param.value += r_k * c_k * signal * deltas[param.name]
        param.value = clamp_float(param.value, param.min_value, param.max_value)


def print_final(params: list[TuneParam]) -> None:
    print("\nFinal rounded enabled parameters:")
    for param in params:
        if param.enabled:
            print(f"setoption name {param.name} value {param.rounded_value()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--engine", required=True, help="Path to the engine binary to tune.")
    parser.add_argument("--engine-dir", help="Working directory for the engine.")
    parser.add_argument("--fastchess", default=default_fastchess_executable(), help="Path to fastchess.")
    parser.add_argument("--fastchess-workdir", help="Working directory for fastchess autosave files.")
    parser.add_argument("--params", default="tools/search_spsa_params.json", help="Parameter JSON path.")
    parser.add_argument("--state", default="tune/search_spsa_state.json", help="Resumable state JSON path.")
    parser.add_argument("--write-template", action="store_true", help="Discover UCI Search* options and write --params, then exit.")
    parser.add_argument("--iterations", type=int, default=100, help="SPSA iterations to run.")
    parser.add_argument("--rounds", type=int, default=8, help="Fastchess rounds per iteration.")
    parser.add_argument("--tc", default="10+0.1", help="Fastchess time control.")
    parser.add_argument("--book", help="Opening book path.")
    parser.add_argument("--book-format", choices=["epd", "pgn"], default="epd")
    parser.add_argument("--book-order", choices=["random", "sequential"], default="random")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1, help="Engine Threads option.")
    parser.add_argument("--hash", type=int, default=16, help="Engine Hash option in MB.")
    parser.add_argument("--eval-file", help="Engine EvalFile option.")
    parser.add_argument("--pgn-dir", default="tune/pgn", help="Directory for per-iteration PGNs.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the SPSA progress bar.")
    parser.add_argument("--progress-width", type=int, default=32, help="Width of the progress bar.")
    parser.add_argument("--verbose-fastchess", action="store_true", help="Print each fastchess command and full fastchess output.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.602, help="SPSA learning-rate decay exponent.")
    parser.add_argument("--gamma", type=float, default=0.101, help="SPSA perturbation decay exponent.")
    parser.add_argument("--stability-offset", type=float, default=10.0, help="SPSA learning-rate stability offset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_paths(args)
    if not validate_command(args.engine_cmd, "Engine"):
        return 1
    if not validate_command(args.fastchess_cmd, "Fastchess"):
        return 1
    params_path = Path(args.params)
    state_path = Path(args.state) if args.state else None

    if args.write_template:
        options = discover_search_options(args.engine_cmd, args.engine_dir_resolved)
        if not options:
            print("No Search* spin options discovered. Build the engine first?", file=sys.stderr)
            return 1
        write_json(params_path, {"iteration": 0, "parameters": options, "history": []})
        print(f"Wrote {len(options)} parameters to {params_path}")
        return 0

    if not params_path.exists():
        print(f"Parameter file not found: {params_path}", file=sys.stderr)
        print("Run with --write-template first.", file=sys.stderr)
        return 1

    start_iteration, params, history = load_params(params_path, state_path)
    enabled = [param for param in params if param.enabled]
    if not enabled:
        print("No enabled parameters in parameter file.", file=sys.stderr)
        return 1

    progress = ProgressBar(
        total=args.iterations,
        width=args.progress_width,
        enabled=not args.no_progress,
    )
    progress.render(0, f"starting at iteration {start_iteration + 1}")

    completed_normally = False
    try:
        for run_index, iteration in enumerate(
            range(start_iteration, start_iteration + args.iterations),
            start=1,
        ):
            progress.render(run_index - 1, f"running iteration {iteration + 1}")
            iteration_rng = random.Random(args.seed + iteration * 1_000_003)
            deltas = {param.name: iteration_rng.choice([-1, 1]) for param in enabled}
            plus_values = value_map(params, deltas, 1, iteration, args)
            minus_values = value_map(params, deltas, -1, iteration, args)

            wins, losses, draws = run_match(args, plus_values, minus_values, iteration)
            update_params(params, deltas, wins, losses, iteration, args)

            entry = {
                "iteration": iteration + 1,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "signal": wins - losses,
                "enabled": {param.name: param.rounded_value() for param in enabled},
            }
            history.append(entry)

            if state_path is not None:
                save_state(state_path, iteration + 1, params, history)

            label = f"iter {iteration + 1}: +{wins} -{losses} ={draws} signal {wins - losses:+d}"
            progress.advance(label)
            if args.no_progress or args.verbose_fastchess:
                print(label, flush=True)
        completed_normally = True
    except KeyboardInterrupt:
        progress.clear()
        print("Interrupted; latest completed iteration is saved in the state file.", file=sys.stderr)
        return 130
    finally:
        if completed_normally:
            progress.finish()
        else:
            progress.clear()

    print_final(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
