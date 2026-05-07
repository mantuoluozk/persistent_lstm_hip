import math
import os
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def maybe_compile(model: nn.Module, example_input: Optional[torch.Tensor] = None) -> nn.Module:
    if os.environ.get("LSTM_USE_COMPILE", "0") != "1":
        return model

    if not hasattr(torch, "compile"):
        return model

    try:
        compiled = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
        )
    except TypeError:
        try:
            compiled = torch.compile(model, mode="reduce-overhead")
        except Exception:
            return model
    except Exception:
        return model

    if example_input is None:
        return compiled

    try:
        with torch.no_grad():
            _ = compiled(example_input)
        return compiled
    except Exception:
        return model


class ManualLSTMLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def step(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = F.linear(x_t, self.weight_ih, self.bias_ih)
        gates = gates + F.linear(h, self.weight_hh, self.bias_hh)

        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        c = f_gate * c + i_gate * g_gate
        h = o_gate * torch.tanh(c)
        return h, c


class LastStateOnlyLSTMRegressor(nn.Module):
    """
    面向当前这个回归任务的优化版：
    1. 仍然保持 4 层 LSTM 语义
    2. 只保留每层当前时刻状态，不再物化前 3 层整段序列输出
    3. 最终只输出顶层最后一个时间步，贴合原始模型真实需求

    这和 Triton persistent 的目标一致：
    尽量把时间维上的状态留在计算图内部，而不是频繁写回全局显存。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_p = dropout

        layers: List[ManualLSTMLayer] = []
        for layer_idx in range(n_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.append(ManualLSTMLayer(layer_input_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if state is None:
            h_list = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
                for _ in range(self.n_layers)
            ]
            c_list = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
                for _ in range(self.n_layers)
            ]
        else:
            h_all, c_all = state
            h_list = [h_all[layer_idx] for layer_idx in range(self.n_layers)]
            c_list = [c_all[layer_idx] for layer_idx in range(self.n_layers)]

        top_h = h_list[-1]

        for t in range(seq_len):
            layer_input = x[:, t, :]

            for layer_idx, layer in enumerate(self.layers):
                h_new, c_new = layer.step(layer_input, h_list[layer_idx], c_list[layer_idx])
                h_list[layer_idx] = h_new
                c_list[layer_idx] = c_new

                if layer_idx < self.n_layers - 1:
                    if self.training and self.dropout_p > 0.0:
                        layer_input = self.dropout(h_new)
                    else:
                        layer_input = h_new
                else:
                    top_h = h_new

        if self.training and self.dropout_p > 0.0:
            top_h = self.dropout(top_h)

        return self.linear(top_h)

    @classmethod
    def from_native_module(cls, native_module: nn.Module) -> "LastStateOnlyLSTMRegressor":
        lstm = native_module.lstm
        dropout_module = getattr(native_module, "dropout", None)
        dropout = dropout_module.p if dropout_module is not None else getattr(lstm, "dropout", 0.0)

        optimized = cls(
            input_dim=lstm.input_size,
            hidden_dim=lstm.hidden_size,
            output_dim=native_module.linear.out_features,
            n_layers=lstm.num_layers,
            dropout=dropout,
        )
        optimized.load_from_native_module(native_module)
        return optimized

    def load_from_native_module(self, native_module: nn.Module) -> None:
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                layer.weight_ih.copy_(getattr(native_module.lstm, f"weight_ih_l{layer_idx}"))
                layer.weight_hh.copy_(getattr(native_module.lstm, f"weight_hh_l{layer_idx}"))
                layer.bias_ih.copy_(getattr(native_module.lstm, f"bias_ih_l{layer_idx}"))
                layer.bias_hh.copy_(getattr(native_module.lstm, f"bias_hh_l{layer_idx}"))

            self.linear.weight.copy_(native_module.linear.weight)
            self.linear.bias.copy_(native_module.linear.bias)


class NativeLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, dropout: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        if self.training:
            last_output = self.dropout(last_output)
        return self.linear(last_output)


def benchmark_demo() -> None:
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)

    device = "cuda:0"
    seq_length = 1000
    batch_size = 512
    input_size = 5
    hidden_size = 128
    output_size = 24
    n_layers = 4
    iterations = 100

    native_model = NativeLSTMRegressor(
        input_dim=input_size,
        hidden_dim=hidden_size,
        output_dim=output_size,
        n_layers=n_layers,
        dropout=0.2,
    ).to(device).half()

    native_model.eval()
    input_tensor = torch.randn((batch_size, seq_length, input_size), device=device, dtype=torch.float16)
    optimized_model = LastStateOnlyLSTMRegressor.from_native_module(native_model).to(device).half()
    optimized_model.eval()
    optimized_model = maybe_compile(optimized_model, example_input=input_tensor)

    with torch.no_grad():
        native_out = native_model(input_tensor)
        optimized_out = optimized_model(input_tensor)
        diff = torch.max(torch.abs(native_out - optimized_out))
        print(f"FP16 max diff: {diff.item():.6f}")

    for _ in range(10):
        _ = native_model(input_tensor)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = native_model(input_tensor)
    torch.cuda.synchronize()
    native_time = time.time() - start_time

    for _ in range(10):
        _ = optimized_model(input_tensor)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = optimized_model(input_tensor)
    torch.cuda.synchronize()
    optimized_time = time.time() - start_time

    print(f"Native PyTorch time: {native_time:.4f} s")
    print(f"Last-state optimized time: {optimized_time:.4f} s")
    print(f"Speedup: {native_time / optimized_time:.2f}x")


if __name__ == "__main__":
    benchmark_demo()
