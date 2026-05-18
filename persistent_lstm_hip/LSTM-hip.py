import os
import time

import torch
import torch.nn as nn

from persistent_lstm_hip import convert_regressor_module


torch.backends.cudnn.benchmark = True


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.linear(last_output)


def maybe_convert_backend(model: nn.Module) -> nn.Module:
    if os.environ.get("USE_PERSISTENT_LSTM_HIP", "1") != "1":
        return model
    return convert_regressor_module(model)


def report_accuracy_gap(native_model: nn.Module, candidate_model: nn.Module, x: torch.Tensor) -> None:
    if os.environ.get("PERSISTENT_LSTM_HIP_ACCURACY", "1") != "1":
        return

    with torch.inference_mode():
        native_out = native_model(x)
        candidate_out = candidate_model(x)
    torch.cuda.synchronize()

    diff = (candidate_out.float() - native_out.float()).abs()
    denom = native_out.float().abs().clamp_min(1.0e-6)
    rel = diff / denom
    print(
        "accuracy_vs_native_lstm: "
        f"max_abs={diff.max().item():.6g}, "
        f"mean_abs={diff.mean().item():.6g}, "
        f"max_rel={rel.max().item():.6g}"
    )


def main() -> None:
    native_model = LSTMRegressor(
        input_dim=5,
        hidden_dim=128,
        output_dim=24,
        n_layers=4,
    ).to("cuda:0").half().eval()

    model = maybe_convert_backend(native_model).to("cuda:0").half().eval()

    seq_length = 1000
    batch_size = 512
    input_size = 5
    iterations = 100

    x = torch.ones((batch_size, seq_length, input_size), device="cuda:0", dtype=torch.float16)

    backend_name = getattr(model, "backend_name", "native_pytorch")
    print(f"backend: {backend_name}")
    report_accuracy_gap(native_model, model, x)

    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(iterations):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print(elapsed_time)
    print(f"吞吐量(含batchsize): {iterations / elapsed_time * batch_size}")


if __name__ == "__main__":
    main()
