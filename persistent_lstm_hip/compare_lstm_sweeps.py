from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path


@dataclass(frozen=True)
class SweepRow:
    hidden: int
    batch: int
    seq: int
    elapsed_s: float
    throughput: float
    kernel: str


def _read_text(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _extract_csv_block(text: str) -> str:
    lines = text.splitlines()
    header_idx = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith("hidden,batch,seq,elapsed_s,throughput"):
            header_idx = idx
    if header_idx < 0:
        raise ValueError("could not find sweep summary CSV header")

    block: list[str] = []
    for line in lines[header_idx:]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("hidden,") or stripped[0].isdigit():
            block.append(stripped)
    return "\n".join(block)


def _parse_rows(text: str) -> dict[tuple[int, int, int], SweepRow]:
    block = _extract_csv_block(text)
    rows: dict[tuple[int, int, int], SweepRow] = {}
    reader = csv.DictReader(StringIO(block))
    for raw in reader:
        hidden = int(raw["hidden"])
        batch = int(raw["batch"])
        seq = int(raw["seq"])
        rows[(hidden, batch, seq)] = SweepRow(
            hidden=hidden,
            batch=batch,
            seq=seq,
            elapsed_s=float(raw["elapsed_s"]),
            throughput=float(raw["throughput"]),
            kernel=raw.get("kernel", ""),
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare native and adaptive LSTM sweep summaries.")
    parser.add_argument("--native", required=True, help="Path to native sweep output, or '-' for stdin")
    parser.add_argument("--adaptive", required=True, help="Path to adaptive sweep output, or '-' for stdin")
    args = parser.parse_args()

    native = _parse_rows(_read_text(args.native))
    adaptive = _parse_rows(_read_text(args.adaptive))
    common = sorted(set(native) & set(adaptive))
    if not common:
        raise SystemExit("no common hidden,batch,seq rows found")

    print("hidden,batch,seq,native_elapsed,adaptive_elapsed,elapsed_speedup,native_tput,adaptive_tput,tput_speedup,kernel")
    for key in common:
        n = native[key]
        a = adaptive[key]
        elapsed_speedup = n.elapsed_s / a.elapsed_s
        tput_speedup = a.throughput / n.throughput
        print(
            f"{a.hidden},{a.batch},{a.seq},"
            f"{n.elapsed_s:.6f},{a.elapsed_s:.6f},{elapsed_speedup:.3f},"
            f"{n.throughput:.3f},{a.throughput:.3f},{tput_speedup:.3f},{a.kernel}"
        )


if __name__ == "__main__":
    main()
