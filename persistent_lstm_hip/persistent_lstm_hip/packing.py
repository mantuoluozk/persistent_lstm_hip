from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class PackedLSTMLayerWeights:
    weight_ih: torch.Tensor
    weight_hh: torch.Tensor
    bias_ih: torch.Tensor
    bias_hh: torch.Tensor

    @property
    def bias(self) -> torch.Tensor:
        return (self.bias_ih + self.bias_hh).contiguous()


@dataclass
class PackedLSTMWeights:
    input_size: int
    hidden_size: int
    num_layers: int
    batch_first: bool
    layers: List[PackedLSTMLayerWeights]


@dataclass
class PackedRegressorWeights:
    lstm: PackedLSTMWeights
    linear_weight: torch.Tensor
    linear_bias: torch.Tensor


def pack_lstm_module(lstm: nn.LSTM) -> PackedLSTMWeights:
    if not lstm.batch_first:
        raise ValueError("当前骨架只支持 batch_first=True。")
    if lstm.bidirectional:
        raise ValueError("当前骨架只支持单向 LSTM。")
    if lstm.proj_size != 0:
        raise ValueError("当前骨架暂不支持 proj_size。")

    layers: List[PackedLSTMLayerWeights] = []
    for layer_idx in range(lstm.num_layers):
        layers.append(
            PackedLSTMLayerWeights(
                weight_ih=getattr(lstm, f"weight_ih_l{layer_idx}").contiguous(),
                weight_hh=getattr(lstm, f"weight_hh_l{layer_idx}").contiguous(),
                bias_ih=getattr(lstm, f"bias_ih_l{layer_idx}").contiguous(),
                bias_hh=getattr(lstm, f"bias_hh_l{layer_idx}").contiguous(),
            )
        )

    return PackedLSTMWeights(
        input_size=lstm.input_size,
        hidden_size=lstm.hidden_size,
        num_layers=lstm.num_layers,
        batch_first=lstm.batch_first,
        layers=layers,
    )


def pack_regressor_module(native_module: nn.Module) -> PackedRegressorWeights:
    lstm = native_module.lstm
    linear = native_module.linear
    return PackedRegressorWeights(
        lstm=pack_lstm_module(lstm),
        linear_weight=linear.weight.contiguous(),
        linear_bias=linear.bias.contiguous(),
    )
