from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .packing import PackedLSTMWeights, PackedRegressorWeights


def reference_lstm_forward(
    x: torch.Tensor,
    packed: PackedLSTMWeights,
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if not packed.batch_first:
        raise ValueError("当前 reference 实现只支持 batch_first=True。")

    batch_size, seq_len, _ = x.shape
    hidden_size = packed.hidden_size

    if state is None:
        h_list: List[torch.Tensor] = [
            torch.zeros((batch_size, hidden_size), device=x.device, dtype=x.dtype)
            for _ in range(packed.num_layers)
        ]
        c_list: List[torch.Tensor] = [
            torch.zeros((batch_size, hidden_size), device=x.device, dtype=x.dtype)
            for _ in range(packed.num_layers)
        ]
    else:
        h_all, c_all = state
        h_list = [h_all[layer_idx] for layer_idx in range(packed.num_layers)]
        c_list = [c_all[layer_idx] for layer_idx in range(packed.num_layers)]

    layer_input = x
    for layer_idx, layer in enumerate(packed.layers):
        outputs = []
        h = h_list[layer_idx]
        c = c_list[layer_idx]

        for t in range(seq_len):
            x_t = layer_input[:, t, :]
            gates = (
                F.linear(x_t, layer.weight_ih, layer.bias_ih) +
                F.linear(h, layer.weight_hh, layer.bias_hh)
            )
            i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            g_gate = torch.tanh(g_gate)
            o_gate = torch.sigmoid(o_gate)
            c = f_gate * c + i_gate * g_gate
            h = o_gate * torch.tanh(c)
            outputs.append(h)

        h_list[layer_idx] = h
        c_list[layer_idx] = c
        layer_input = torch.stack(outputs, dim=1)

    h_n = torch.stack(h_list, dim=0)
    c_n = torch.stack(c_list, dim=0)
    return layer_input, (h_n, c_n)


def reference_regressor_forward(
    x: torch.Tensor,
    packed: PackedRegressorWeights,
) -> torch.Tensor:
    out, _ = reference_lstm_forward(x, packed.lstm)
    last = out[:, -1, :]
    return F.linear(last, packed.linear_weight, packed.linear_bias)
