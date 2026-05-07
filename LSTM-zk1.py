import math
import os
import time
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lstm_scan_impl(
    projected_inputs: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对单层 LSTM 做时间维扫描。

    这里假设 input-to-hidden 已经被提前投影成 [B, T, 4H]，
    从而把每个时间步的一次大 GEMM 收敛为整段序列的一次大 GEMM。
    """
    outputs: List[torch.Tensor] = []

    for x_t in projected_inputs.unbind(dim=1):
        recurrent = torch.matmul(h, weight_hh.t()) + bias_hh
        gates = x_t + recurrent

        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        c = f_gate * c + i_gate * g_gate
        h = o_gate * torch.tanh(c)
        outputs.append(h)

    return torch.stack(outputs, dim=1), h, c


try:
    lstm_scan: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ] = torch.jit.script(_lstm_scan_impl)
except Exception:
    lstm_scan = _lstm_scan_impl


class FusedLSTMLayer(nn.Module):
    """
    更适合 BW10/ROCm 的单层 LSTM：
    1. 先做整段序列的 input projection
    2. 时间步内只保留 recurrent matmul + 门控更新
    3. 让逐元素逻辑尽量落到统一编译图
    """

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

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)

        if state is None:
            h = torch.zeros(
                batch_size,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )
            c = torch.zeros(
                batch_size,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            h, c = state

        # 整段输入先做一次大投影，减少时间步上的 GEMM 调度次数。
        projected_inputs = F.linear(x, self.weight_ih, self.bias_ih).contiguous()
        outputs, h, c = lstm_scan(projected_inputs, self.weight_hh, self.bias_hh, h, c)
        return outputs, (h, c)


class BW10OptimizedLSTMRegressor(nn.Module):
    """
    面向海光 BW10 / ROCm 的 LSTM 回归模型。

    与标准 nn.LSTM 相比，这个版本的核心变化是：
    1. 每层先对整段序列做 input-to-hidden 大 GEMM
    2. 时间维上只保留 recurrent 路径
    3. 逐元素门控逻辑通过 JIT / compile 更容易融合

    注意：
    这不是在 Python 层完全复刻 NVIDIA cuDNN 的 Persistent RNN Kernel，
    但在 AMD/ROCm 上通常比直接走 nn.LSTM 更容易减少 kernel 碎片化。
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

        layers = []
        for layer_idx in range(n_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.append(FusedLSTMLayer(layer_input_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward_features(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        layer_input = x
        final_h: List[torch.Tensor] = []
        final_c: List[torch.Tensor] = []

        if state is None:
            states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.n_layers
        else:
            h_all, c_all = state
            states = [(h_all[layer_idx], c_all[layer_idx]) for layer_idx in range(self.n_layers)]

        for layer_idx, layer in enumerate(self.layers):
            layer_output, (h, c) = layer(layer_input, states[layer_idx])
            final_h.append(h)
            final_c.append(c)

            if layer_idx < self.n_layers - 1 and self.dropout_p > 0.0:
                layer_input = self.dropout(layer_output)
            else:
                layer_input = layer_output

        stacked_h = torch.stack(final_h, dim=0)
        stacked_c = torch.stack(final_c, dim=0)
        return layer_input, (stacked_h, stacked_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.forward_features(x)
        last_output = sequence_output[:, -1, :]
        last_output = self.dropout(last_output)
        return self.linear(last_output)

    @classmethod
    def from_standard_module(
        cls,
        standard_module: nn.Module,
    ) -> "BW10OptimizedLSTMRegressor":
        """
        从原始结构为 `module.lstm + module.linear + module.dropout`
        的标准模型复制权重，方便直接替换现有实现。
        """
        lstm = standard_module.lstm
        dropout_module = getattr(standard_module, "dropout", None)
        dropout = dropout_module.p if dropout_module is not None else getattr(lstm, "dropout", 0.0)

        optimized = cls(
            input_dim=lstm.input_size,
            hidden_dim=lstm.hidden_size,
            output_dim=standard_module.linear.out_features,
            n_layers=lstm.num_layers,
            dropout=dropout,
        )
        optimized.load_from_standard_module(standard_module)
        return optimized

    def load_from_standard_module(self, standard_module: nn.Module) -> None:
        with torch.no_grad():
            for layer_idx, fused_layer in enumerate(self.layers):
                fused_layer.weight_ih.copy_(getattr(standard_module.lstm, f"weight_ih_l{layer_idx}"))
                fused_layer.weight_hh.copy_(getattr(standard_module.lstm, f"weight_hh_l{layer_idx}"))
                fused_layer.bias_ih.copy_(getattr(standard_module.lstm, f"bias_ih_l{layer_idx}"))
                fused_layer.bias_hh.copy_(getattr(standard_module.lstm, f"bias_hh_l{layer_idx}"))

            self.linear.weight.copy_(standard_module.linear.weight)
            self.linear.bias.copy_(standard_module.linear.bias)


def maybe_compile(model: nn.Module, example_input: Optional[torch.Tensor] = None) -> nn.Module:
    """
    针对固定 shape 推理场景，尝试启用 torch.compile。
    在 ROCm 上这一步常常能进一步减少细碎 kernel。
    """
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


class StandardLSTMRegressor(nn.Module):
    """
    保留一个标准版本，方便你做权重迁移或 A/B 对照。
    """

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
        last_output = self.dropout(lstm_out[:, -1, :])
        return self.linear(last_output)


def build_bw10_model(
    device: str = "cuda:0",
    use_compile: bool = True,
) -> nn.Module:
    model = BW10OptimizedLSTMRegressor(
        input_dim=5,
        hidden_dim=128,
        output_dim=24,
        n_layers=4,
        dropout=0.2,
    ).to(device).half()

    model.eval()
    return model


def benchmark_demo() -> None:
    """
    保留与你原脚本一致的基准形式，方便你到服务器上直接跑。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("需要在 CUDA/ROCm 设备上运行该脚本。")

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = "cuda:0"
    seq_length = 1000
    batch_size = 512
    input_size = 5
    iterations = 100
    use_compile = True

    model = build_bw10_model(device=device, use_compile=True)
    x = torch.ones((batch_size, seq_length, input_size), device=device, dtype=torch.float16)
    if use_compile:
        model = maybe_compile(model, example_input=x)

    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(iterations):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print(elapsed_time)
    print(f"吞吐量(按 batchsize): {iterations / elapsed_time * batch_size}")


def convert_standard_checkpoint_example() -> nn.Module:
    """
    如果你已经有标准 nn.LSTM 训练好的权重，可以参考这个函数迁移。
    """
    baseline = StandardLSTMRegressor(
        input_dim=5,
        hidden_dim=128,
        output_dim=24,
        n_layers=4,
        dropout=0.2,
    )
    optimized = BW10OptimizedLSTMRegressor.from_standard_module(baseline)
    return optimized


if __name__ == "__main__":
    benchmark_demo()
