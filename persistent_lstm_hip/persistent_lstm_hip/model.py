from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .extension import load_extension
from .packing import (
    PackedLSTMLayerWeights,
    PackedLSTMWeights,
    PackedRegressorWeights,
    pack_lstm_module,
    pack_regressor_module,
)
from .reference import reference_lstm_forward, reference_regressor_forward


def _pack_weight_kpairs(weight: torch.Tensor) -> torch.Tensor:
    transposed = weight.transpose(0, 1).contiguous()
    k_size, out_size = transposed.shape
    if k_size % 2 != 0:
        pad = torch.zeros((1, out_size), device=transposed.device, dtype=transposed.dtype)
        transposed = torch.cat((transposed, pad), dim=0)
        k_size += 1
    return transposed.view(k_size // 2, 2, out_size).permute(0, 2, 1).contiguous()


class StandardLSTMRegressor(nn.Module):
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
        out, _ = self.lstm(x)
        return self.linear(self.dropout(out[:, -1, :]))


class PersistentLSTM(nn.Module):
    """
    一个尽量贴近 nn.LSTM 调用方式的包装层。

    当前支持：
    - batch_first=True
    - bidirectional=False
    - proj_size=0

    当前后端策略：
    - 命中特化条件时，后续可走 HIP specialized kernel
    - 其他结构自动回退到通用 reference 路径
    """

    def __init__(self, packed: PackedLSTMWeights):
        super().__init__()
        self.input_size = packed.input_size
        self.hidden_size = packed.hidden_size
        self.num_layers = packed.num_layers
        self.batch_first = packed.batch_first
        self.bidirectional = False
        self.proj_size = 0

        for layer_idx, layer in enumerate(packed.layers):
            setattr(self, f"weight_ih_l{layer_idx}", nn.Parameter(layer.weight_ih))
            setattr(self, f"weight_hh_l{layer_idx}", nn.Parameter(layer.weight_hh))
            setattr(self, f"bias_ih_l{layer_idx}", nn.Parameter(layer.bias_ih))
            setattr(self, f"bias_hh_l{layer_idx}", nn.Parameter(layer.bias_hh))

    @classmethod
    def from_native_module(cls, lstm: nn.LSTM) -> "PersistentLSTM":
        return cls(pack_lstm_module(lstm))

    @property
    def backend_name(self) -> str:
        if self.can_use_specialized_hip():
            return "hip_specialized_candidate"
        return "python_reference_generic"

    def can_use_specialized_hip(self) -> bool:
        ext = load_extension()
        if ext is None:
            return False
        return (
            self.input_size == 5 and
            self.hidden_size == 128 and
            self.num_layers == 4 and
            self.batch_first
        )

    def _packed(self) -> PackedLSTMWeights:
        return PackedLSTMWeights(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            layers=[
                PackedLSTMLayerWeights(
                    weight_ih=getattr(self, f"weight_ih_l{layer_idx}"),
                    weight_hh=getattr(self, f"weight_hh_l{layer_idx}"),
                    bias_ih=getattr(self, f"bias_ih_l{layer_idx}"),
                    bias_hh=getattr(self, f"bias_hh_l{layer_idx}"),
                )
                for layer_idx in range(self.num_layers)
            ],
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return reference_lstm_forward(x, self._packed(), state=hx)


class PersistentLSTMRegressor(nn.Module):
    def __init__(self, packed: PackedRegressorWeights):
        super().__init__()
        self.lstm = PersistentLSTM(packed.lstm)
        self.linear = nn.Linear(packed.linear_weight.shape[1], packed.linear_weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(packed.linear_weight)
            self.linear.bias.copy_(packed.linear_bias)
        self._specialized_cache_key: tuple[object, ...] | None = None
        self._hip_specialized_args: tuple[torch.Tensor, ...] | None = None

    @property
    def backend_name(self) -> str:
        if self._can_use_specialized_regressor_hip():
            return "hip_specialized_4layer_regressor"
        return "python_reference_generic"

    @classmethod
    def from_native_module(cls, native_module: nn.Module) -> "PersistentLSTMRegressor":
        return cls(pack_regressor_module(native_module))

    def _packed(self) -> PackedRegressorWeights:
        return PackedRegressorWeights(
            lstm=self.lstm._packed(),
            linear_weight=self.linear.weight,
            linear_bias=self.linear.bias,
        )

    def _can_use_specialized_regressor_hip(self) -> bool:
        ext = load_extension()
        if ext is None:
            return False
        return (
            self.lstm.input_size == 5 and
            self.lstm.hidden_size == 128 and
            self.lstm.num_layers == 4 and
            self.linear.out_features == 24
        )

    def _current_specialized_cache_key(self) -> tuple[object, ...]:
        params = (
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            self.lstm.weight_ih_l2,
            self.lstm.weight_hh_l2,
            self.lstm.bias_ih_l2,
            self.lstm.bias_hh_l2,
            self.lstm.weight_ih_l3,
            self.lstm.weight_hh_l3,
            self.lstm.bias_ih_l3,
            self.lstm.bias_hh_l3,
            self.linear.weight,
            self.linear.bias,
        )
        key_parts: list[object] = []
        for tensor in params:
            key_parts.extend(
                (
                    id(tensor),
                    tensor.device.type,
                    tensor.device.index,
                    str(tensor.dtype),
                    tensor._version,
                )
            )
        return tuple(key_parts)

    def _ensure_specialized_cache(self) -> tuple[torch.Tensor, ...]:
        cache_key = self._current_specialized_cache_key()
        if self._specialized_cache_key != cache_key or self._hip_specialized_args is None:
            self._hip_specialized_args = (
                _pack_weight_kpairs(self.lstm.weight_ih_l0),
                _pack_weight_kpairs(self.lstm.weight_hh_l0),
                (self.lstm.bias_ih_l0 + self.lstm.bias_hh_l0).contiguous(),
                _pack_weight_kpairs(self.lstm.weight_ih_l1),
                _pack_weight_kpairs(self.lstm.weight_hh_l1),
                (self.lstm.bias_ih_l1 + self.lstm.bias_hh_l1).contiguous(),
                _pack_weight_kpairs(self.lstm.weight_ih_l2),
                _pack_weight_kpairs(self.lstm.weight_hh_l2),
                (self.lstm.bias_ih_l2 + self.lstm.bias_hh_l2).contiguous(),
                _pack_weight_kpairs(self.lstm.weight_ih_l3),
                _pack_weight_kpairs(self.lstm.weight_hh_l3),
                (self.lstm.bias_ih_l3 + self.lstm.bias_hh_l3).contiguous(),
                self.linear.weight.contiguous(),
                self.linear.bias.contiguous(),
            )
            self._specialized_cache_key = cache_key
        return self._hip_specialized_args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = load_extension()
        if self._can_use_specialized_regressor_hip():
            return ext.persistent_lstm4_forward_interleaved(x, *self._ensure_specialized_cache())

        if ext is not None and hasattr(ext, "persistent_lstm_generic_forward"):
            return ext.persistent_lstm_generic_forward(x)

        return reference_regressor_forward(x, self._packed())


PersistentLSTM4LayerRegressor = PersistentLSTMRegressor
