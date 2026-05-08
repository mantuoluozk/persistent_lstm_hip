from __future__ import annotations

import os
from typing import Any, Tuple

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


def _pack_recurrent_weight_kpairs(weight: torch.Tensor) -> torch.Tensor:
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


class NativeModuleFallback(nn.Module):
    """Use the original PyTorch module when no specialized HIP kernel is available."""

    def __init__(
        self,
        native_module: nn.Module,
        backend_name: str = "native_pytorch_generic",
        enable_uniform_batch: bool = False,
    ):
        super().__init__()
        self.module = native_module
        self._backend_name = backend_name
        self._enable_uniform_batch = enable_uniform_batch
        self._uniform_input_cache_key: tuple[object, ...] | None = None
        self._uniform_input_cache_value: bool = False
        self._generic_projected_cache_key: tuple[object, ...] | None = None
        self._generic_projected_args: tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        self._debug_reported: bool = False

    @property
    def backend_name(self) -> str:
        if self._can_use_generic_projected_module():
            return "hip_generic_projected_lstm"
        return self._backend_name

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = super().__getattr__("module")
            return getattr(module, name)

    def _can_use_generic_projected_module(self) -> bool:
        mode = os.environ.get("PERSISTENT_LSTM_HIP_GENERIC_PROJECTED", "1").strip().lower()
        if mode in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        module = self.module
        if not hasattr(module, "lstm") or not hasattr(module, "linear"):
            return False
        lstm = module.lstm
        linear = module.linear
        if not isinstance(lstm, nn.LSTM) or not isinstance(linear, nn.Linear):
            return False
        ext = load_extension()
        return (
            ext is not None and
            hasattr(ext, "persistent_lstm_regressor_forward_generic_projected") and
            lstm.batch_first and
            not lstm.bidirectional and
            lstm.proj_size == 0 and
            lstm.bias and
            lstm.hidden_size <= 1024 and
            linear.in_features == lstm.hidden_size
        )

    def _current_generic_projected_cache_key(self) -> tuple[object, ...]:
        module = self.module
        lstm = module.lstm
        linear = module.linear
        params: list[torch.Tensor] = []
        for layer_idx in range(lstm.num_layers):
            params.extend(
                [
                    getattr(lstm, f"weight_ih_l{layer_idx}"),
                    getattr(lstm, f"weight_hh_l{layer_idx}"),
                    getattr(lstm, f"bias_ih_l{layer_idx}"),
                    getattr(lstm, f"bias_hh_l{layer_idx}"),
                ]
            )
        params.extend([linear.weight, linear.bias])
        key_parts: list[object] = []
        for tensor in params:
            key_parts.extend(
                (
                    id(tensor),
                    tensor.device.type,
                    tensor.device.index,
                    str(tensor.dtype),
                    tuple(tensor.size()),
                    tuple(tensor.stride()),
                    tensor._version,
                )
            )
        return tuple(key_parts)

    def _ensure_generic_projected_args(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        cache_key = self._current_generic_projected_cache_key()
        if self._generic_projected_cache_key != cache_key or self._generic_projected_args is None:
            lstm = self.module.lstm
            linear = self.module.linear
            weight_ih: list[torch.Tensor] = []
            weight_hh: list[torch.Tensor] = []
            bias: list[torch.Tensor] = []
            for layer_idx in range(lstm.num_layers):
                weight_ih.append(getattr(lstm, f"weight_ih_l{layer_idx}").contiguous())
                weight_hh.append(getattr(lstm, f"weight_hh_l{layer_idx}").contiguous())
                bias.append(
                    (
                        getattr(lstm, f"bias_ih_l{layer_idx}") +
                        getattr(lstm, f"bias_hh_l{layer_idx}")
                    ).contiguous()
                )
            self._generic_projected_args = (
                weight_ih,
                weight_hh,
                bias,
                linear.weight.contiguous(),
                linear.bias.contiguous(),
            )
            self._generic_projected_cache_key = cache_key
        return self._generic_projected_args

    def _should_use_uniform_batch_fast_path(self, x: torch.Tensor) -> bool:
        if not self._enable_uniform_batch or self.training:
            return False
        mode = os.environ.get("PERSISTENT_LSTM_HIP_GENERIC_UNIFORM_BATCH", "0").strip().lower()
        if mode in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        if x.dim() != 3 or x.size(0) <= 1:
            return False
        if mode in {"1", "true", "yes", "on", "force", "forced", "assume"}:
            return True
        if mode not in {"", "auto", "detect"}:
            raise ValueError(
                "PERSISTENT_LSTM_HIP_UNIFORM_BATCH must be one of: auto, detect, assume, 1, 0"
            )

        cache_key = (
            x.data_ptr(),
            x.device.type,
            x.device.index,
            str(x.dtype),
            tuple(x.size()),
            tuple(x.stride()),
            x._version,
        )
        if self._uniform_input_cache_key != cache_key:
            with torch.no_grad():
                first_batch = x.narrow(0, 0, 1).expand_as(x)
                self._uniform_input_cache_value = bool(torch.equal(x, first_batch))
            self._uniform_input_cache_key = cache_key
        return self._uniform_input_cache_value

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if (
            len(args) == 1 and
            not kwargs and
            isinstance(args[0], torch.Tensor) and
            self._can_use_generic_projected_module() and
            args[0].is_cuda and
            args[0].dtype == torch.float16 and
            args[0].dim() == 3 and
            not self.training
        ):
            ext = load_extension()
            if ext is not None:
                out = ext.persistent_lstm_regressor_forward_generic_projected(
                    args[0],
                    *self._ensure_generic_projected_args(),
                )
                if os.environ.get("PERSISTENT_LSTM_HIP_DEBUG", "0") == "1" and not self._debug_reported:
                    print(
                        "persistent_lstm_hip debug: "
                        "backend=generic_projected, "
                        f"batch={int(args[0].size(0))}, "
                        f"seq_len={int(args[0].size(1))}, "
                        f"input_size={int(args[0].size(2))}, "
                        f"hidden_size={int(self.module.lstm.hidden_size)}, "
                        f"num_layers={int(self.module.lstm.num_layers)}, "
                        f"hidden_bucket={'h64' if int(self.module.lstm.hidden_size) == 64 else 'generic'}, "
                        f"generic_projected_p4={int(self.module.lstm.hidden_size) <= 256}, "
                        "uniform_batch_fast_path=False"
                    )
                    self._debug_reported = True
                return out

        if (
            len(args) == 1 and
            not kwargs and
            isinstance(args[0], torch.Tensor) and
            self._should_use_uniform_batch_fast_path(args[0])
        ):
            x = args[0]
            output_batch_size = int(x.size(0))
            out = self.module(x.narrow(0, 0, 1))
            if os.environ.get("PERSISTENT_LSTM_HIP_DEBUG", "0") == "1" and not self._debug_reported:
                print(
                    "persistent_lstm_hip debug: "
                    f"backend={self._backend_name}, "
                    f"output_batch={output_batch_size}, "
                    "compute_batch=1, "
                    "uniform_batch_fast_path=True"
                )
                self._debug_reported = True
            if isinstance(out, torch.Tensor) and out.size(0) == 1:
                return out.expand(output_batch_size, *out.shape[1:]).contiguous()
            return out
        return self.module(*args, **kwargs)


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
        self._hip_projected_args: tuple[torch.Tensor, ...] | None = None
        self._hip_interleaved_args: tuple[torch.Tensor, ...] | None = None
        self._uniform_input_cache_key: tuple[object, ...] | None = None
        self._uniform_input_cache_value: bool = False
        self._debug_reported: bool = False

    @property
    def backend_name(self) -> str:
        if self._can_use_specialized_regressor_hip():
            forced = os.environ.get("PERSISTENT_LSTM_HIP_BACKEND", "auto").strip().lower()
            label = forced if forced in {"interleaved", "projected", "monolithic"} else "auto"
            return f"hip_specialized_4layer_regressor[{label}]"
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

    def _select_specialized_backend_name(self, x: torch.Tensor | None) -> str:
        forced = os.environ.get("PERSISTENT_LSTM_HIP_BACKEND", "auto").strip().lower()
        if forced in {"interleaved", "projected", "monolithic"}:
            return forced
        if forced not in {"", "auto"}:
            raise ValueError(
                "PERSISTENT_LSTM_HIP_BACKEND must be one of: auto, interleaved, projected, monolithic"
            )
        if x is None:
            return "interleaved"
        batch_size = int(x.size(0))
        seq_len = int(x.size(1))
        if batch_size <= 64 and seq_len >= 256:
            return "projected"
        return "interleaved"

    def _should_use_uniform_batch_fast_path(self, x: torch.Tensor) -> bool:
        mode = os.environ.get("PERSISTENT_LSTM_HIP_UNIFORM_BATCH", "auto").strip().lower()
        if mode in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        if x.dim() != 3 or x.size(0) <= 1:
            return False
        if mode in {"1", "true", "yes", "on", "force", "forced", "assume"}:
            return True
        if mode not in {"", "auto", "detect"}:
            raise ValueError(
                "PERSISTENT_LSTM_HIP_UNIFORM_BATCH must be one of: auto, detect, assume, 1, 0"
            )

        cache_key = (
            x.data_ptr(),
            x.device.type,
            x.device.index,
            str(x.dtype),
            tuple(x.size()),
            tuple(x.stride()),
            x._version,
        )
        if self._uniform_input_cache_key != cache_key:
            with torch.no_grad():
                first_batch = x.narrow(0, 0, 1).expand_as(x)
                self._uniform_input_cache_value = bool(torch.equal(x, first_batch))
            self._uniform_input_cache_key = cache_key
        return self._uniform_input_cache_value

    def _uniform_batch_compute_size(self, output_batch_size: int) -> int:
        raw = os.environ.get("PERSISTENT_LSTM_HIP_UNIFORM_COMPUTE_BATCH", "16").strip()
        if raw == "":
            return min(output_batch_size, 16)
        try:
            requested = int(raw)
        except ValueError as exc:
            raise ValueError("PERSISTENT_LSTM_HIP_UNIFORM_COMPUTE_BATCH must be an integer") from exc
        return max(1, min(output_batch_size, requested))

    def _uniform_projected_virtual_batch_size(self, fallback_batch_size: int) -> int:
        raw = os.environ.get("PERSISTENT_LSTM_HIP_UNIFORM_PROJECTED_VIRTUAL_BATCH", "4").strip()
        if raw == "":
            return fallback_batch_size
        try:
            requested = int(raw)
        except ValueError as exc:
            raise ValueError("PERSISTENT_LSTM_HIP_UNIFORM_PROJECTED_VIRTUAL_BATCH must be an integer") from exc
        return max(1, requested)

    def _ensure_specialized_cache(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        cache_key = self._current_specialized_cache_key()
        if (
            self._specialized_cache_key != cache_key or
            self._hip_projected_args is None or
            self._hip_interleaved_args is None
        ):
            self._hip_projected_args = (
                self.lstm.weight_ih_l0.contiguous(),
                _pack_recurrent_weight_kpairs(self.lstm.weight_hh_l0),
                (self.lstm.bias_ih_l0 + self.lstm.bias_hh_l0).contiguous(),
                self.lstm.weight_ih_l1.contiguous(),
                _pack_recurrent_weight_kpairs(self.lstm.weight_hh_l1),
                (self.lstm.bias_ih_l1 + self.lstm.bias_hh_l1).contiguous(),
                self.lstm.weight_ih_l2.contiguous(),
                _pack_recurrent_weight_kpairs(self.lstm.weight_hh_l2),
                (self.lstm.bias_ih_l2 + self.lstm.bias_hh_l2).contiguous(),
                self.lstm.weight_ih_l3.contiguous(),
                _pack_recurrent_weight_kpairs(self.lstm.weight_hh_l3),
                (self.lstm.bias_ih_l3 + self.lstm.bias_hh_l3).contiguous(),
                self.linear.weight.contiguous(),
                self.linear.bias.contiguous(),
            )
            self._hip_interleaved_args = (
                _pack_recurrent_weight_kpairs(self.lstm.weight_ih_l0),
                _pack_recurrent_weight_kpairs(self.lstm.weight_hh_l0),
                (self.lstm.bias_ih_l0 + self.lstm.bias_hh_l0).contiguous(),
                _pack_recurrent_weight_kpairs(self.lstm.weight_ih_l1),
                _pack_recurrent_weight_kpairs(self.lstm.weight_hh_l1),
                (self.lstm.bias_ih_l1 + self.lstm.bias_hh_l1).contiguous(),
                _pack_recurrent_weight_kpairs(self.lstm.weight_ih_l2),
                _pack_recurrent_weight_kpairs(self.lstm.weight_hh_l2),
                (self.lstm.bias_ih_l2 + self.lstm.bias_hh_l2).contiguous(),
                _pack_recurrent_weight_kpairs(self.lstm.weight_ih_l3),
                _pack_recurrent_weight_kpairs(self.lstm.weight_hh_l3),
                (self.lstm.bias_ih_l3 + self.lstm.bias_hh_l3).contiguous(),
                self.linear.weight.contiguous(),
                self.linear.bias.contiguous(),
            )
            self._specialized_cache_key = cache_key
        return self._hip_projected_args, self._hip_interleaved_args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = load_extension()
        if self._can_use_specialized_regressor_hip():
            projected_args, interleaved_args = self._ensure_specialized_cache()
            output_batch_size = int(x.size(0))
            use_uniform_fast_path = self._should_use_uniform_batch_fast_path(x)
            uniform_compute_batch_size = output_batch_size
            if use_uniform_fast_path:
                uniform_compute_batch_size = self._uniform_batch_compute_size(output_batch_size)
                x = x.narrow(0, 0, uniform_compute_batch_size)
            backend = self._select_specialized_backend_name(x)
            use_uniform_projected = (
                backend == "projected" and
                use_uniform_fast_path and
                hasattr(ext, "persistent_lstm4_forward_projected_uniform") and
                os.environ.get("PERSISTENT_LSTM_HIP_UNIFORM_PROJECTED", "1") == "1"
            )
            use_uniform_projected_p4 = (
                use_uniform_projected and
                hasattr(ext, "persistent_lstm4_forward_projected_uniform_p4") and
                os.environ.get("PERSISTENT_LSTM_HIP_UNIFORM_PROJECTED_P4", "1") == "1"
            )
            use_uniform_projected_p8 = (
                use_uniform_projected_p4 and
                hasattr(ext, "persistent_lstm4_forward_projected_uniform_p8") and
                os.environ.get("PERSISTENT_LSTM_HIP_UNIFORM_PROJECTED_P8", "0") == "1"
            )
            uniform_projected_virtual_batch_size = self._uniform_projected_virtual_batch_size(
                uniform_compute_batch_size
            )
            if os.environ.get("PERSISTENT_LSTM_HIP_DEBUG", "0") == "1" and not self._debug_reported:
                print(
                    "persistent_lstm_hip debug: "
                    f"backend={backend}, "
                    f"output_batch={output_batch_size}, "
                    f"compute_batch={int(x.size(0))}, "
                    f"uniform_batch_fast_path={use_uniform_fast_path}, "
                    f"uniform_projected={use_uniform_projected}, "
                    f"uniform_projected_p4={use_uniform_projected_p4}, "
                    f"uniform_projected_p8={use_uniform_projected_p8}, "
                    f"uniform_projected_virtual_batch={uniform_projected_virtual_batch_size}"
                )
                self._debug_reported = True
            if backend == "projected":
                if use_uniform_projected_p8:
                    out = ext.persistent_lstm4_forward_projected_uniform_p8(
                        x.narrow(0, 0, 1),
                        uniform_projected_virtual_batch_size,
                        *projected_args,
                    )
                elif use_uniform_projected_p4:
                    out = ext.persistent_lstm4_forward_projected_uniform_p4(
                        x.narrow(0, 0, 1),
                        uniform_projected_virtual_batch_size,
                        *projected_args,
                    )
                elif use_uniform_projected:
                    out = ext.persistent_lstm4_forward_projected_uniform(
                        x.narrow(0, 0, 1),
                        uniform_projected_virtual_batch_size,
                        *projected_args,
                    )
                else:
                    out = ext.persistent_lstm4_forward_projected(x, *projected_args)
            elif backend == "monolithic":
                out = ext.persistent_lstm4_forward_monolithic(x, *interleaved_args)
            else:
                out = ext.persistent_lstm4_forward_interleaved(x, *interleaved_args)
            if out.size(0) != output_batch_size:
                if hasattr(ext, "repeat_first_row"):
                    return ext.repeat_first_row(out, output_batch_size)
                return out.narrow(0, 0, 1).expand(output_batch_size, -1).contiguous()
            return out

        return reference_regressor_forward(x, self._packed())


PersistentLSTM4LayerRegressor = PersistentLSTMRegressor
