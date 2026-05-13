from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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
    cases = _parse_cases(os.environ.get("MIOPEN_ADAPTIVE_SWEEP", "128:512,256:512,512:128"))
    base_env = os.environ.copy()
    base_env["MIOPEN_ADAPTIVE_LSTM_DEBUG"] = "1"
    base_env["MIOPEN_ADAPTIVE_LSTM_PROFILE"] = "1"
    base_env.setdefault("MIOPEN_ADAPTIVE_LSTM_PROFILE_SKIP", "11")
    base_env.setdefault("MIOPEN_ADAPTIVE_ITERS", "20")

    rows: list[tuple[ShapeCase, str, float, float, float, float, float]] = []
    for case in cases:
        env = base_env.copy()
        env["MIOPEN_ADAPTIVE_HIDDEN"] = str(case.hidden)
        env["MIOPEN_ADAPTIVE_BATCH"] = str(case.batch)
        env["MIOPEN_ADAPTIVE_SEQ"] = str(case.seq)

        print(f"\n=== profile hidden={case.hidden} batch={case.batch} seq={case.seq} ===", flush=True)
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
        kernel = _extract(r"kernel=([^,\s]+)", output)
        elapsed = float(_extract(r"elapsed=([0-9.]+)s", output, "nan"))
        layer_rows = [
            (int(layer), float(proj), float(recur))
            for layer, proj, recur in re.findall(
                r"profile: layer=([0-9]+), input_proj_ms=([0-9.]+), recurrent_ms=([0-9.]+)",
                output,
            )
        ]
        linear_ms = float(_extract(r"profile: linear_ms=([0-9.]+)", output, "0"))
        total_proj = sum(proj for _, proj, _ in layer_rows)
        total_recur = sum(recur for _, _, recur in layer_rows)
        rows.append((case, kernel, elapsed, total_proj, total_recur, linear_ms, total_proj + total_recur + linear_ms))

    print("\nprofile_summary:")
    print("hidden,batch,seq,kernel,elapsed_s,input_proj_ms,recurrent_ms,linear_ms,profiled_total_ms")
    for case, kernel, elapsed, total_proj, total_recur, linear_ms, profiled_total in rows:
        print(
            f"{case.hidden},{case.batch},{case.seq},{kernel},{elapsed:.6f},"
            f"{total_proj:.3f},{total_recur:.3f},{linear_ms:.3f},{profiled_total:.3f}"
        )


if __name__ == "__main__":
    main()
