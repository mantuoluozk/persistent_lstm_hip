from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ShapeCase:
    hidden: int
    batch: int
    seq: int


def _parse_cases(raw: str) -> list[ShapeCase]:
    cases: list[ShapeCase] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        parts = piece.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError("MIOPEN_ADAPTIVE_SWEEP must look like 128:512,256:512,512:128 or hidden:batch:seq")
        hidden = int(parts[0])
        batch = int(parts[1])
        seq = int(parts[2]) if len(parts) == 3 else int(os.environ.get("MIOPEN_ADAPTIVE_SEQ", "1000"))
        cases.append(ShapeCase(hidden=hidden, batch=batch, seq=seq))
    if not cases:
        raise ValueError("MIOPEN_ADAPTIVE_SWEEP produced no cases")
    return cases


def _extract(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def main() -> None:
    raw_cases = os.environ.get("MIOPEN_ADAPTIVE_SWEEP", "128:512,256:512,512:128")
    cases = _parse_cases(raw_cases)
    repeats = int(os.environ.get("MIOPEN_ADAPTIVE_SWEEP_REPEATS", "1"))
    if repeats <= 0:
        raise ValueError("MIOPEN_ADAPTIVE_SWEEP_REPEATS must be positive")
    base_env = os.environ.copy()
    base_env.setdefault("MIOPEN_ADAPTIVE_LSTM_DEBUG", "1")

    rows: list[tuple[ShapeCase, float, float, str, str, str]] = []
    for case in cases:
        elapsed_values: list[float] = []
        throughput_values: list[float] = []
        max_abs = ""
        mean_abs = ""
        kernel = ""
        for repeat_idx in range(repeats):
            env = base_env.copy()
            env["MIOPEN_ADAPTIVE_HIDDEN"] = str(case.hidden)
            env["MIOPEN_ADAPTIVE_BATCH"] = str(case.batch)
            env["MIOPEN_ADAPTIVE_SEQ"] = str(case.seq)

            repeat_label = f" repeat={repeat_idx + 1}/{repeats}" if repeats > 1 else ""
            print(
                f"\n=== hidden={case.hidden} batch={case.batch} seq={case.seq}{repeat_label} ===",
                flush=True,
            )
            result = subprocess.run(
                [sys.executable, str(ROOT / "run_adaptive_lstm.py")],
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.returncode != 0:
                raise SystemExit(result.returncode)

            output = result.stdout
            elapsed = float(_extract(r"elapsed=([0-9.]+)s", output, "nan"))
            throughput = float(_extract(r"throughput=([0-9.]+)", output, "nan"))
            elapsed_values.append(elapsed)
            throughput_values.append(throughput)
            max_abs = _extract(r"max_abs=([0-9.eE+-]+)", output, max_abs)
            mean_abs = _extract(r"mean_abs=([0-9.eE+-]+)", output, mean_abs)
            kernel = _extract(r"kernel=([^,\s]+)", output, kernel)
        rows.append((case, median(elapsed_values), median(throughput_values), max_abs, mean_abs, kernel))

    print("\nsummary:")
    print("hidden,batch,seq,elapsed_s,throughput,max_abs,mean_abs,kernel")
    for case, elapsed, throughput, max_abs, mean_abs, kernel in rows:
        print(
            f"{case.hidden},{case.batch},{case.seq},{elapsed:.6f},{throughput:.3f},"
            f"{max_abs},{mean_abs},{kernel}"
        )


if __name__ == "__main__":
    main()
