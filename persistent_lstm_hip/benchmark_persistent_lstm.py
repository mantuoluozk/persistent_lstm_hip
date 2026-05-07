import time

import torch

from persistent_lstm_hip.model import (
    PersistentLSTMRegressor,
    StandardLSTMRegressor,
)


def benchmark() -> None:
    device = "cuda:0"
    seq_length = 1000
    batch_size = 512
    input_size = 5
    hidden_size = 128
    output_size = 24
    n_layers = 4
    iterations = 100

    native_model = StandardLSTMRegressor(
        input_dim=input_size,
        hidden_dim=hidden_size,
        output_dim=output_size,
        n_layers=n_layers,
    ).to(device).half()
    native_model.eval()

    persistent_model = PersistentLSTMRegressor.from_native_module(native_model).to(device).half()
    persistent_model.eval()

    x = torch.randn((batch_size, seq_length, input_size), device=device, dtype=torch.float16)

    with torch.no_grad():
        native_out = native_model(x)
        persistent_out = persistent_model(x)
        diff = torch.max(torch.abs(native_out - persistent_out))
        print("当前对比口径: 4 层 LSTM + Linear，自定义 HIP op 骨架")
        print(f"FP16 max diff: {diff.item():.6f}")
        print(f"Backend: {persistent_model.backend_name}")

    for _ in range(10):
        _ = native_model(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = native_model(x)
    torch.cuda.synchronize()
    native_time = time.time() - start

    for _ in range(10):
        _ = persistent_model(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = persistent_model(x)
    torch.cuda.synchronize()
    persistent_time = time.time() - start

    print(f"Native PyTorch time: {native_time:.4f} s")
    print(f"Persistent skeleton time: {persistent_time:.4f} s")
    print(f"Speedup: {native_time / persistent_time:.2f}x")


if __name__ == "__main__":
    benchmark()
