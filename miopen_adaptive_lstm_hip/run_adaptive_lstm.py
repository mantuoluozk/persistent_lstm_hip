from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from miopen_adaptive_lstm_hip import convert_regressor_module


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.linear(self.dropout(out[:, -1, :]))


def main() -> None:
    device = "cuda:0"
    batch_size = int(os.environ.get("MIOPEN_ADAPTIVE_BATCH", "512"))
    seq_len = int(os.environ.get("MIOPEN_ADAPTIVE_SEQ", "1000"))
    input_size = int(os.environ.get("MIOPEN_ADAPTIVE_INPUT", "5"))
    hidden_size = int(os.environ.get("MIOPEN_ADAPTIVE_HIDDEN", "128"))
    output_size = int(os.environ.get("MIOPEN_ADAPTIVE_OUTPUT", "24"))
    num_layers = int(os.environ.get("MIOPEN_ADAPTIVE_LAYERS", "4"))
    iterations = int(os.environ.get("MIOPEN_ADAPTIVE_ITERS", "100"))

    native = LSTMRegressor(input_size, hidden_size, output_size, num_layers).to(device).half().eval()
    model = convert_regressor_module(native).to(device).half().eval()
    x = torch.ones((batch_size, seq_len, input_size), device=device, dtype=torch.float16)

    print(f"backend: {getattr(model, 'backend_name', 'native_pytorch')}")
    with torch.inference_mode():
        native_out = native(x)
        model_out = model(x)
        torch.cuda.synchronize()
        diff = (model_out.float() - native_out.float()).abs()
        print(f"accuracy_vs_native_lstm: max_abs={diff.max().item():.6g}, mean_abs={diff.mean().item():.6g}")

        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model(x)
        torch.cuda.synchronize()

    elapsed = time.time() - start
    print(f"elapsed={elapsed:.6f}s")
    print(f"throughput={iterations / elapsed * batch_size:.3f} samples/s")


if __name__ == "__main__":
    main()

