from __future__ import annotations

import os
import time
from dataclasses import dataclass
from statistics import median

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ShapeCase:
    hidden: int
    batch: int
    seq: int


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.linear(self.dropout(out[:, -1, :]))


def _parse_cases(raw: str) -> list[ShapeCase]:
    cases: list[ShapeCase] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        parts = piece.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError("NATIVE_LSTM_SWEEP must look like 128:512,256:512,512:128 or hidden:batch:seq")
        hidden = int(parts[0])
        batch = int(parts[1])
        seq = int(parts[2]) if len(parts) == 3 else int(os.environ.get("MIOPEN_ADAPTIVE_SEQ", "1000"))
        cases.append(ShapeCase(hidden=hidden, batch=batch, seq=seq))
    if not cases:
        raise ValueError("NATIVE_LSTM_SWEEP produced no cases")
    return cases


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def main() -> None:
    torch.backends.cudnn.benchmark = True

    device = os.environ.get("NATIVE_LSTM_DEVICE", "cuda:0")
    input_size = _env_int("MIOPEN_ADAPTIVE_INPUT", 5)
    output_size = _env_int("MIOPEN_ADAPTIVE_OUTPUT", 24)
    num_layers = _env_int("MIOPEN_ADAPTIVE_LAYERS", 4)
    iterations = _env_int("MIOPEN_ADAPTIVE_ITERS", 100)
    warmup = _env_int("NATIVE_LSTM_WARMUP", 10)
    repeats = _env_int("NATIVE_LSTM_SWEEP_REPEATS", _env_int("MIOPEN_ADAPTIVE_SWEEP_REPEATS", 1))
    if repeats <= 0:
        raise ValueError("NATIVE_LSTM_SWEEP_REPEATS must be positive")
    dropout = float(os.environ.get("NATIVE_LSTM_DROPOUT", "0.2"))
    use_eval = os.environ.get("NATIVE_LSTM_EVAL", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    raw_cases = os.environ.get(
        "NATIVE_LSTM_SWEEP",
        os.environ.get("MIOPEN_ADAPTIVE_SWEEP", "128:512,256:512,512:128"),
    )
    cases = _parse_cases(raw_cases)

    rows: list[tuple[ShapeCase, float, float]] = []
    for case in cases:
        elapsed_values: list[float] = []
        throughput_values: list[float] = []
        for repeat_idx in range(repeats):
            repeat_label = f" repeat={repeat_idx + 1}/{repeats}" if repeats > 1 else ""
            print(
                f"\n=== native hidden={case.hidden} batch={case.batch} seq={case.seq}{repeat_label} ===",
                flush=True,
            )
            model = LSTMRegressor(input_size, case.hidden, output_size, num_layers, dropout=dropout).to(device).half()
            if use_eval:
                model.eval()
            x = torch.ones((case.batch, case.seq, input_size), device=device, dtype=torch.float16)

            with torch.inference_mode():
                for _ in range(warmup):
                    _ = model(x)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(iterations):
                    _ = model(x)
                torch.cuda.synchronize()

            elapsed = time.time() - start
            throughput = iterations / elapsed * case.batch
            print(f"backend: native_pytorch_lstm")
            print(
                f"native_lstm debug: batch={case.batch}, seq_len={case.seq}, input_size={input_size}, "
                f"hidden_size={case.hidden}, num_layers={num_layers}, eval={use_eval}"
            )
            print(f"elapsed={elapsed:.6f}s")
            print(f"throughput={throughput:.3f} samples/s")
            elapsed_values.append(elapsed)
            throughput_values.append(throughput)
        rows.append((case, median(elapsed_values), median(throughput_values)))

    print("\nsummary:")
    print("hidden,batch,seq,elapsed_s,throughput,kernel")
    for case, elapsed, throughput in rows:
        print(f"{case.hidden},{case.batch},{case.seq},{elapsed:.6f},{throughput:.3f},native_pytorch_lstm")


if __name__ == "__main__":
    main()
