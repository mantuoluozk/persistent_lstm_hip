"""Compare accuracy: system MIOpen vs R&D optimized MIOpen package.

Usage:
    python test_rd_accuracy.py               # system MIOpen only
    LD_LIBRARY_PATH=/path/to/pkg python test_rd_accuracy.py  # with R&D package
"""
import os
import time
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.linear(last_output)


def test_size(hidden_size, seq_len=1000, batch=512):
    torch.manual_seed(42)
    model = LSTMRegressor(5, hidden_size, 24, 4).to("cuda:0").half().eval()
    # Random input to avoid "all-ones" overfitting
    x = torch.randn((batch, seq_len, 5), device="cuda:0", dtype=torch.float16)

    with torch.inference_mode():
        out = model(x)
        # Warmup
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

    return out.float(), elapsed


def main():
    has_rd = "/data1/zk/miopen-package" in os.environ.get("LD_LIBRARY_PATH", "")
    label = "R&D" if has_rd else "Baseline"

    for hs in [128, 256, 512]:
        out, elapsed = test_size(hs)
        print(f"H{hs:>3}  {label:>8}  elapsed={elapsed:.3f}s  "
              f"out_sum={out.sum().item():.3f}  out_mean={out.mean().item():.6f}")


if __name__ == "__main__":
    main()
