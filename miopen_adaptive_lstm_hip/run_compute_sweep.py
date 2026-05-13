from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ResultRow:
    mode: str
    hidden: int
    batch: int
    seq: int
    elapsed: str
    throughput: str
    max_abs: str
    mean_abs: str
    kernel: str


def _extract(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def main() -> None:
    raw_modes = os.environ.get(
        "MIOPEN_ADAPTIVE_COMPUTE_SWEEP",
        "fp32;auto_fast;auto_balanced;auto_aggressive;fp16",
    )
    modes = [mode.strip() for mode in raw_modes.split(";") if mode.strip()]
    if not modes:
        raise ValueError("MIOPEN_ADAPTIVE_COMPUTE_SWEEP produced no modes")

    base_env = os.environ.copy()
    base_env.setdefault("MIOPEN_ADAPTIVE_LSTM_DEBUG", "1")

    rows: list[ResultRow] = []
    for mode in modes:
        env = base_env.copy()
        env["MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE"] = mode
        print(f"\n######## recurrent_compute={mode} ########", flush=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_shape_sweep.py")],
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

        for line in result.stdout.splitlines():
            if not re.match(r"^[0-9]+,[0-9]+,[0-9]+,", line):
                continue
            hidden, batch, seq, elapsed, throughput, max_abs, mean_abs, kernel = line.split(",", 7)
            rows.append(
                ResultRow(
                    mode=mode,
                    hidden=int(hidden),
                    batch=int(batch),
                    seq=int(seq),
                    elapsed=elapsed,
                    throughput=throughput,
                    max_abs=max_abs,
                    mean_abs=mean_abs,
                    kernel=kernel,
                )
            )

    print("\ncompute_summary:")
    print("mode,hidden,batch,seq,elapsed_s,throughput,max_abs,mean_abs,kernel")
    for row in rows:
        print(
            f"{row.mode},{row.hidden},{row.batch},{row.seq},{row.elapsed},"
            f"{row.throughput},{row.max_abs},{row.mean_abs},{row.kernel}"
        )


if __name__ == "__main__":
    main()
