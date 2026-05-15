from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from .descriptors import HardwareDescriptor, RNNDescriptor, RuntimeMode, SeqTensorDescriptor
from .extension import load_extension
from .modular import build_forward_plan


@dataclass(frozen=True)
class AdaptiveLayerArgs:
    weight_ih: torch.Tensor
    weight_ih_t: torch.Tensor
    weight_hh: torch.Tensor
    weight_hh_t: torch.Tensor
    bias: torch.Tensor
    packed_weight_hh: torch.Tensor | None = None


@dataclass
class H128GemmScanWorkspace:
    key: tuple[object, ...]
    gate: torch.Tensor
    h_state: torch.Tensor
    c_state: torch.Tensor
    recur: torch.Tensor
    seq_buffers: tuple[torch.Tensor, ...]
    last_out: torch.Tensor


@dataclass
class H128SeqMajorAccumWorkspace:
    key: tuple[object, ...]
    gate: torch.Tensor
    h_state: torch.Tensor
    c_state: torch.Tensor
    seq_buffers: tuple[torch.Tensor, ...]
    last_out: torch.Tensor


@dataclass
class GenericGemmScanWorkspace:
    key: tuple[object, ...]
    gate: torch.Tensor
    h_state: torch.Tensor
    c_state: torch.Tensor
    recur: torch.Tensor
    seq_buffers: tuple[torch.Tensor, ...]
    last_out: torch.Tensor


@dataclass
class LinearWorkspace:
    key: tuple[object, ...]
    out: torch.Tensor


def _device_cu_count(x: torch.Tensor) -> int:
    if not x.is_cuda:
        return 120
    props = torch.cuda.get_device_properties(x.device)
    return int(getattr(props, "multi_processor_count", 120))


def _wavefront_width() -> int:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_WAVEFRONT", "64").strip()
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_WAVEFRONT must be an integer") from exc


def _recurrent_partitions(hidden_size: int) -> int:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_PARTITIONS", "auto").strip().lower()
    if raw in {"", "auto"}:
        return 8 if hidden_size >= 256 else 4
    try:
        partitions = int(raw)
    except ValueError as exc:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_PARTITIONS must be auto, 2, 4, or 8") from exc
    if partitions not in {2, 4, 8}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_PARTITIONS must be auto, 2, 4, or 8")
    return partitions


def _use_h128_cached_path(ext, hidden_size: int) -> bool:
    mode = os.environ.get("MIOPEN_ADAPTIVE_LSTM_H128_CACHED", "1").strip().lower()
    if mode in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if mode not in {"", "1", "true", "yes", "on", "auto"}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_H128_CACHED must be 1, 0, or auto")
    return hidden_size == 128 and hasattr(ext, "adaptive_lstm_h128_cached_update_forward")


def _recurrent_backend(ext, hidden_size: int) -> str:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND", "gemm_scan").strip().lower()
    if raw in {"", "auto", "best"}:
        raw = "gemm_scan"
    if raw not in {"seqmajor_accum", "gemm_scan", "cached", "partitioned", "scalar", "persistent_mmac", "profile_variants"}:
        raise ValueError(
            "MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND must be auto, seqmajor_accum, gemm_scan, cached, partitioned, scalar, persistent_mmac, or profile_variants"
        )
    if (
        raw == "seqmajor_accum"
        and not (hidden_size == 128 and hasattr(ext, "adaptive_lstm_h128_seqmajor_accum_update_forward"))
    ):
        return "gemm_scan"
    if raw == "cached" and not _use_h128_cached_path(ext, hidden_size):
        return "partitioned"
    if raw == "persistent_mmac":
        packed_fn = f"adaptive_lstm_h{hidden_size}_mmac_packed_variant_forward_workspace"
        h128_fn = "adaptive_lstm_h128_persistent_mmac_update_forward_workspace"
        if not (hidden_size == 128 and hasattr(ext, h128_fn)) and not hasattr(ext, packed_fn):
            return "gemm_scan"
    if (
        raw == "profile_variants"
        and not (hidden_size == 128 and hasattr(ext, "adaptive_lstm_h128_mmac_profile_variant_forward_workspace"))
    ):
        return "gemm_scan"
    return raw


def _gemm_scan_read_block(default_read_block: int, hidden_size: int) -> int:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_GEMM_SCAN_READ_BLOCK", "auto").strip().lower()
    if raw in {"", "auto"}:
        # Keep the measured-safe default. On K100_AI, forcing vectorized
        # read blocks for H256/H512 reduced pointwise work but hurt overall
        # recurrent scan time, likely from lower memory coalescing/occupancy in
        # the tiny per-timestep update kernel.
        return int(default_read_block)
    try:
        read_block = int(raw)
    except ValueError as exc:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_GEMM_SCAN_READ_BLOCK must be auto, 1, 2, or 4") from exc
    if read_block not in {1, 2, 4}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_GEMM_SCAN_READ_BLOCK must be auto, 1, 2, or 4")
    return read_block


def _valid_gemm_scan_read_block(default_read_block: int, hidden_size: int) -> int:
    read_block = _gemm_scan_read_block(default_read_block, hidden_size)
    if hidden_size % read_block == 0:
        return read_block
    return 1


def _use_gemm_scan_workspace(ext) -> bool:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_GEMM_SCAN_WORKSPACE", "1").strip().lower()
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if raw not in {"", "1", "true", "yes", "on", "auto"}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_GEMM_SCAN_WORKSPACE must be 1, 0, or auto")
    return hasattr(ext, "adaptive_lstm_h128_gemm_scan_update_forward_workspace") or hasattr(
        ext, "adaptive_lstm_gemm_scan_update_forward_workspace"
    )


def _use_gate_workspace() -> bool:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_GATE_WORKSPACE", "1").strip().lower()
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if raw not in {"", "1", "true", "yes", "on", "auto"}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_GATE_WORKSPACE must be 1, 0, or auto")
    return True


def _use_direct_blas() -> bool:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS", "1").strip().lower()
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if raw not in {"", "1", "true", "yes", "on", "auto"}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS must be 1, 0, or auto")
    return True


def _recurrent_compute_mode() -> str:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE", "fp32").strip().lower()
    if raw in {"", "fp32", "32", "float"}:
        return "fp32"
    if raw in {"fp16", "16", "half"}:
        return "fp16"
    if raw == "fp16_first":
        return "fp16_first"
    if raw == "fp16_except_last":
        return "fp16_except_last"
    if raw in {"auto_fast", "fast", "auto_balanced", "balanced", "auto_aggressive", "aggressive"}:
        if raw in {"auto_balanced", "balanced"}:
            return "auto_balanced"
        if raw in {"auto_aggressive", "aggressive"}:
            return "auto_aggressive"
        return "auto_fast"
    if raw.startswith("fp16_layers:"):
        return raw
    raise ValueError(
        "MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE must be fp32, fp16, fp16_first, fp16_except_last, "
        "auto_fast, auto_balanced, auto_aggressive, or fp16_layers:<ids>"
    )


def _recurrent_compute_for_layer(layer_idx: int, num_layers: int, hidden_size: int) -> int:
    mode = _recurrent_compute_mode()
    if mode == "fp16":
        return 1
    if mode == "auto_fast":
        return 1 if hidden_size >= 256 and layer_idx == 0 else 0
    if mode == "auto_balanced":
        return 1 if hidden_size >= 256 and layer_idx in {0, 1} else 0
    if mode == "auto_aggressive":
        return 1 if hidden_size >= 256 and layer_idx != num_layers - 1 else 0
    if mode == "fp16_first":
        return 1 if layer_idx == 0 else 0
    if mode == "fp16_except_last":
        return 1 if layer_idx != num_layers - 1 else 0
    if mode.startswith("fp16_layers:"):
        raw_layers = mode.split(":", 1)[1]
        if raw_layers.strip() == "":
            return 0
        try:
            fp16_layers = {int(piece.strip()) for piece in raw_layers.split(",") if piece.strip()}
        except ValueError as exc:
            raise ValueError("fp16_layers must contain comma-separated integer layer ids") from exc
        return 1 if layer_idx in fp16_layers else 0
    return 0


def _use_input_gemm_workspace(ext) -> bool:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_INPUT_GEMM", "1").strip().lower()
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if raw not in {"", "1", "true", "yes", "on", "auto"}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_INPUT_GEMM must be 1, 0, or auto")
    return hasattr(ext, "adaptive_lstm_input_gemm_forward_workspace")


def _use_fixed_hidden_scan() -> bool:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_FIXED_HIDDEN_SCAN", "1").strip().lower()
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if raw not in {"", "1", "true", "yes", "on", "auto"}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_FIXED_HIDDEN_SCAN must be 1, 0, or auto")
    return True


def _profile_variant() -> int:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_PROFILE_VARIANT", "3").strip().lower()
    if raw in {"0", "a"}:
        return 0
    if raw in {"1", "b"}:
        return 1
    if raw in {"2", "c"}:
        return 2
    if raw in {"3", "d", "full"}:
        return 3
    try:
        v = int(raw)
        if 0 <= v <= 3:
            return v
    except ValueError:
        pass
    raise ValueError("MIOPEN_ADAPTIVE_LSTM_PROFILE_VARIANT must be 0, 1, 2, 3, a, b, c, d, or full")




def _use_packed_mmac(ext, hidden_size: int) -> bool:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_MMAC_PACKED", "1").strip().lower()
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if raw not in {"", "1", "true", "yes", "on", "auto"}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_MMAC_PACKED must be 1, 0, or auto")
    fn_name = f"adaptive_lstm_h{hidden_size}_mmac_packed_variant_forward_workspace"
    return hasattr(ext, fn_name)


def _pack_mmac_weight(weight_hh: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Pack native [4H, H] recurrent weight for HCU MMAC B-load.

    Packed layout: [htile][ktile][krow=16][ngroup=4][gate=4][frag=4]
    Total size = H*H*4 half values.
    """
    kH = hidden_size
    assert weight_hh.shape == (4 * kH, kH), f"Expected [{4*kH},{kH}], got {list(weight_hh.shape)}"
    kMmacK = 16
    n_tiles = kH // kMmacK
    device = weight_hh.device
    dtype = weight_hh.dtype
    wh = weight_hh.detach().contiguous().cpu()
    packed = torch.empty((kH * kH * 4,), dtype=dtype)
    idx = 0
    for ht in range(n_tiles):
        h0 = ht * kMmacK
        for kt in range(n_tiles):
            k0 = kt * kMmacK
            for krow in range(kMmacK):
                src_k = k0 + krow
                for ng in range(4):
                    n0 = h0 + ng * 4
                    for gate in range(4):
                        src_row = gate * kH + n0
                        packed[idx:idx + 4] = wh[src_row:src_row + 4, src_k]
                        idx += 4
    return packed.to(device=device, dtype=dtype, non_blocking=False).contiguous()


def _get_packed_mmac_fn(ext, hidden_size: int):
    fn_name = f"adaptive_lstm_h{hidden_size}_mmac_packed_variant_forward_workspace"
    return getattr(ext, fn_name, None)


def _use_gate_accum_scan() -> bool:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_GATE_ACCUM", "0").strip().lower()
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if raw not in {"", "1", "true", "yes", "on", "auto"}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_GATE_ACCUM must be 1, 0, or auto")
    return True


def _h128_cached_batch_tile(ext, hidden_size: int, batch_size: int) -> int:
    raw = os.environ.get("MIOPEN_ADAPTIVE_LSTM_H128_BATCH_TILE", "4").strip().lower()
    if raw in {"", "auto"}:
        requested = 4
    else:
        try:
            requested = int(raw)
        except ValueError as exc:
            raise ValueError("MIOPEN_ADAPTIVE_LSTM_H128_BATCH_TILE must be auto, 2, 4, or 8") from exc
    if requested not in {2, 4, 8}:
        raise ValueError("MIOPEN_ADAPTIVE_LSTM_H128_BATCH_TILE must be auto, 2, 4, or 8")
    if (
        hidden_size == 128
        and requested == 8
        and batch_size >= 8
        and hasattr(ext, "adaptive_lstm_h128_cached_b8_update_forward")
    ):
        return 8
    if (
        hidden_size == 128
        and requested >= 4
        and batch_size >= 4
        and hasattr(ext, "adaptive_lstm_h128_cached_b4_update_forward")
    ):
        return 4
    return 2


class NativeModuleFallback(nn.Module):
    def __init__(self, module: nn.Module, backend_name: str = "native_pytorch"):
        super().__init__()
        self.module = module
        self._backend_name = backend_name

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class AdaptiveLSTMRegressor(nn.Module):
    """Experimental MIOpen-inspired backend for `nn.LSTM + nn.Linear`.

    This is inference-oriented and intentionally separate from the existing
    production `persistent_lstm_hip` path.
    """

    def __init__(self, native_module: nn.Module):
        super().__init__()
        self.native_module = copy.deepcopy(native_module)
        self.lstm = self.native_module.lstm
        self.linear = self.native_module.linear
        self._cache_key: tuple[object, ...] | None = None
        self._layer_args: tuple[AdaptiveLayerArgs, ...] | None = None
        self._linear_weight_t: torch.Tensor | None = None
        self._linear_bias: torch.Tensor | None = None
        self._debug_reported = False
        self._forward_count = 0
        self._forward_plan_cache_key: tuple[object, ...] | None = None
        self._forward_plan = None
        self._h128_gemm_scan_workspace: H128GemmScanWorkspace | None = None
        self._h128_seqmajor_workspace: H128SeqMajorAccumWorkspace | None = None
        self._generic_gemm_scan_workspace: GenericGemmScanWorkspace | None = None
        self._linear_workspace: LinearWorkspace | None = None

    @property
    def backend_name(self) -> str:
        return "miopen_adaptive_lstm_hip"

    def _cache_key_current(self) -> tuple[object, ...]:
        tensors: list[torch.Tensor] = []
        for layer_idx in range(self.lstm.num_layers):
            tensors.extend(
                [
                    getattr(self.lstm, f"weight_ih_l{layer_idx}"),
                    getattr(self.lstm, f"weight_hh_l{layer_idx}"),
                    getattr(self.lstm, f"bias_ih_l{layer_idx}"),
                    getattr(self.lstm, f"bias_hh_l{layer_idx}"),
                ]
            )
        tensors.extend([self.linear.weight, self.linear.bias])
        key: list[object] = []
        for tensor in tensors:
            key.extend(
                (
                    id(tensor),
                    tensor.device.type,
                    tensor.device.index,
                    str(tensor.dtype),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor._version,
                )
            )
        return tuple(key)

    def _ensure_layer_args(self) -> tuple[AdaptiveLayerArgs, ...]:
        key = self._cache_key_current()
        if self._cache_key != key or self._layer_args is None:
            args: list[AdaptiveLayerArgs] = []
            for layer_idx in range(self.lstm.num_layers):
                weight_hh = getattr(self.lstm, f"weight_hh_l{layer_idx}").contiguous()
                args.append(
                    AdaptiveLayerArgs(
                        weight_ih=getattr(self.lstm, f"weight_ih_l{layer_idx}").contiguous(),
                        weight_ih_t=getattr(self.lstm, f"weight_ih_l{layer_idx}").transpose(0, 1).contiguous(),
                        weight_hh=weight_hh,
                        weight_hh_t=getattr(self.lstm, f"weight_hh_l{layer_idx}").transpose(0, 1).contiguous(),
                        bias=(
                            getattr(self.lstm, f"bias_ih_l{layer_idx}")
                            + getattr(self.lstm, f"bias_hh_l{layer_idx}")
                        ).contiguous(),
                        packed_weight_hh=(
                            _pack_mmac_weight(weight_hh, int(self.lstm.hidden_size))
                            if weight_hh.dtype == torch.float16 and weight_hh.is_cuda
                            else None
                        ),
                    )
                )
            self._layer_args = tuple(args)
            self._linear_weight_t = self.linear.weight.transpose(0, 1).contiguous()
            self._linear_bias = self.linear.bias.contiguous()
            self._cache_key = key
        return self._layer_args

    def _build_plan(self, x: torch.Tensor):
        key = (
            int(x.size(0)),
            int(x.size(1)),
            int(x.size(2)),
            int(self.lstm.hidden_size),
            int(self.lstm.num_layers),
            _device_cu_count(x),
            _wavefront_width(),
        )
        if self._forward_plan_cache_key == key and self._forward_plan is not None:
            return self._forward_plan

        self._forward_plan = build_forward_plan(
            rnn_desc=RNNDescriptor(
                hidden_size=int(self.lstm.hidden_size),
                num_layers=int(self.lstm.num_layers),
                algo_mode="rounded_dynamic",
                dropout=0.0,
                bias=bool(self.lstm.bias),
            ),
            x_desc=SeqTensorDescriptor(
                batch_size=int(x.size(0)),
                seq_len=int(x.size(1)),
                input_size=int(x.size(2)),
                batch_first=bool(self.lstm.batch_first),
            ),
            hardware=HardwareDescriptor(
                max_compute_units=_device_cu_count(x),
                wavefront_width=_wavefront_width(),
            ),
            runtime=RuntimeMode(
                fwd_mode="inference",
                allow_inference_adaptation=True,
            ),
        )
        self._forward_plan_cache_key = key
        return self._forward_plan

    def _ensure_h128_gemm_scan_workspace(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> H128GemmScanWorkspace:
        key = (
            int(batch_size),
            int(seq_len),
            128,
            int(self.lstm.num_layers),
            device.type,
            device.index,
            str(dtype),
        )
        if self._h128_gemm_scan_workspace is not None and self._h128_gemm_scan_workspace.key == key:
            return self._h128_gemm_scan_workspace

        seq_buffer_count = 2 if int(self.lstm.num_layers) > 1 else 1
        workspace = H128GemmScanWorkspace(
            key=key,
            gate=torch.empty((batch_size * seq_len, 512), device=device, dtype=dtype),
            h_state=torch.empty((batch_size, 128), device=device, dtype=dtype),
            c_state=torch.empty((batch_size, 128), device=device, dtype=torch.float32),
            recur=torch.empty((batch_size, 512), device=device, dtype=dtype),
            seq_buffers=tuple(
                torch.empty((batch_size, seq_len, 128), device=device, dtype=dtype)
                for _ in range(seq_buffer_count)
            ),
            last_out=torch.empty((batch_size, 128), device=device, dtype=dtype),
        )
        self._h128_gemm_scan_workspace = workspace
        return workspace

    def _ensure_h128_seqmajor_workspace(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> H128SeqMajorAccumWorkspace:
        key = (
            int(batch_size),
            int(seq_len),
            128,
            int(self.lstm.num_layers),
            device.type,
            device.index,
            str(dtype),
        )
        if self._h128_seqmajor_workspace is not None and self._h128_seqmajor_workspace.key == key:
            return self._h128_seqmajor_workspace

        seq_buffer_count = 2 if int(self.lstm.num_layers) > 1 else 1
        workspace = H128SeqMajorAccumWorkspace(
            key=key,
            gate=torch.empty((seq_len * batch_size, 512), device=device, dtype=dtype),
            h_state=torch.empty((batch_size, 128), device=device, dtype=dtype),
            c_state=torch.empty((batch_size, 128), device=device, dtype=torch.float32),
            seq_buffers=tuple(
                torch.empty((seq_len, batch_size, 128), device=device, dtype=dtype)
                for _ in range(seq_buffer_count)
            ),
            last_out=torch.empty((batch_size, 128), device=device, dtype=dtype),
        )
        self._h128_seqmajor_workspace = workspace
        return workspace

    def _ensure_generic_gemm_scan_workspace(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> GenericGemmScanWorkspace:
        key = (
            int(batch_size),
            int(seq_len),
            int(hidden_size),
            int(self.lstm.num_layers),
            device.type,
            device.index,
            str(dtype),
        )
        if self._generic_gemm_scan_workspace is not None and self._generic_gemm_scan_workspace.key == key:
            return self._generic_gemm_scan_workspace

        seq_buffer_count = 2 if int(self.lstm.num_layers) > 1 else 1
        workspace = GenericGemmScanWorkspace(
            key=key,
            gate=torch.empty((batch_size * seq_len, 4 * hidden_size), device=device, dtype=dtype),
            h_state=torch.empty((batch_size, hidden_size), device=device, dtype=dtype),
            c_state=torch.empty((batch_size, hidden_size), device=device, dtype=torch.float32),
            recur=torch.empty((batch_size, 4 * hidden_size), device=device, dtype=dtype),
            seq_buffers=tuple(
                torch.empty((batch_size, seq_len, hidden_size), device=device, dtype=dtype)
                for _ in range(seq_buffer_count)
            ),
            last_out=torch.empty((batch_size, hidden_size), device=device, dtype=dtype),
        )
        self._generic_gemm_scan_workspace = workspace
        return workspace

    def _ensure_linear_workspace(
        self,
        batch_size: int,
        output_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LinearWorkspace:
        key = (int(batch_size), int(output_size), device.type, device.index, str(dtype))
        if self._linear_workspace is not None and self._linear_workspace.key == key:
            return self._linear_workspace
        workspace = LinearWorkspace(
            key=key,
            out=torch.empty((batch_size, output_size), device=device, dtype=dtype),
        )
        self._linear_workspace = workspace
        return workspace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        ext = load_extension()
        if ext is None:
            return self.native_module(x)
        if self.training or not x.is_cuda or x.dtype != torch.float16 or x.dim() != 3:
            return self.native_module(x)

        with torch.inference_mode():
            self._forward_count += 1
            profile_skip = int(os.environ.get("MIOPEN_ADAPTIVE_LSTM_PROFILE_SKIP", "0"))
            if profile_skip < 0:
                raise ValueError("MIOPEN_ADAPTIVE_LSTM_PROFILE_SKIP must be non-negative")
            should_report = self._forward_count > profile_skip
            plan = self._build_plan(x)
            if not plan.use_dynamic_algo:
                return self.native_module(x)

            layer_input = x.contiguous()
            layer_input_is_seqmajor = False
            layer_args = self._ensure_layer_args()
            linear_weight_t = self._linear_weight_t
            linear_bias = self._linear_bias
            if linear_weight_t is None or linear_bias is None:
                linear_weight_t = self.linear.weight.transpose(0, 1).contiguous()
                linear_bias = self.linear.bias.contiguous()
            actual_kernel_name = plan.recurrent_kernel.name
            actual_pipeline_name = plan.recurrent_kernel.pipeline
            actual_cached_batch_tile = 0
            actual_gemm_scan_read_block = 0
            actual_gemm_scan_workspace = False
            actual_gate_workspace = False
            actual_seqmajor_accum = False
            actual_direct_blas = False
            actual_input_gemm = False
            profile_enabled = os.environ.get("MIOPEN_ADAPTIVE_LSTM_PROFILE", "0") == "1" and should_report
            profile_rows: list[tuple[int, float, float]] = []
            linear_ms = 0.0
            for layer_idx, args in enumerate(layer_args):
                if layer_input_is_seqmajor:
                    seq_len, batch_size, input_size = layer_input.shape
                else:
                    batch_size, seq_len, input_size = layer_input.shape
                hidden_size = int(self.lstm.hidden_size)
                partitions = _recurrent_partitions(hidden_size)
                backend = _recurrent_backend(ext, hidden_size)
                is_seqmajor_accum = (
                    backend == "seqmajor_accum"
                    and hidden_size == 128
                    and hasattr(ext, "adaptive_lstm_h128_seqmajor_accum_update_forward")
                )
                use_h128_gemm_scan_workspace = (
                    backend in {"gemm_scan", "persistent_mmac", "profile_variants"}
                    and hidden_size == 128
                    and (
                        hasattr(ext, "adaptive_lstm_h128_gemm_scan_update_forward")
                        or hasattr(ext, "adaptive_lstm_h128_persistent_mmac_update_forward_workspace")
                    )
                    and _use_gemm_scan_workspace(ext)
                )
                use_generic_gemm_scan_workspace = (
                    backend == "gemm_scan"
                    and hidden_size != 128
                    and hasattr(ext, "adaptive_lstm_gemm_scan_update_forward_workspace")
                    and _use_gemm_scan_workspace(ext)
                )
                workspace = (
                    self._ensure_h128_gemm_scan_workspace(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        device=layer_input.device,
                        dtype=layer_input.dtype,
                    )
                    if use_h128_gemm_scan_workspace
                    else None
                )
                generic_workspace = (
                    self._ensure_generic_gemm_scan_workspace(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        hidden_size=hidden_size,
                        device=layer_input.device,
                        dtype=layer_input.dtype,
                    )
                    if use_generic_gemm_scan_workspace
                    else None
                )
                if profile_enabled:
                    proj_start = torch.cuda.Event(enable_timing=True)
                    proj_end = torch.cuda.Event(enable_timing=True)
                    recur_start = torch.cuda.Event(enable_timing=True)
                    recur_end = torch.cuda.Event(enable_timing=True)
                    proj_start.record()
                if is_seqmajor_accum:
                    seq_workspace = self._ensure_h128_seqmajor_workspace(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        device=layer_input.device,
                        dtype=layer_input.dtype,
                    )
                    if layer_input_is_seqmajor:
                        input_2d = layer_input.view(seq_len * batch_size, input_size)
                    else:
                        input_2d = layer_input.transpose(0, 1).contiguous().view(seq_len * batch_size, input_size)
                    torch.mm(input_2d, args.weight_ih_t, out=seq_workspace.gate)
                    seq_workspace.gate.add_(args.bias)
                    gate = seq_workspace.gate.view(seq_len, batch_size, 4 * hidden_size)
                    actual_gate_workspace = True
                    actual_seqmajor_accum = True
                elif workspace is not None and _use_gate_workspace():
                    input_2d = layer_input.view(batch_size * seq_len, input_size)
                    if _use_input_gemm_workspace(ext):
                        ext.adaptive_lstm_input_gemm_forward_workspace(input_2d, args.weight_ih_t, workspace.gate)
                        actual_input_gemm = True
                    else:
                        torch.mm(input_2d, args.weight_ih_t, out=workspace.gate)
                    gate = workspace.gate.view(batch_size, seq_len, 4 * hidden_size)
                    actual_gate_workspace = True
                elif generic_workspace is not None and _use_gate_workspace():
                    input_2d = layer_input.view(batch_size * seq_len, input_size)
                    if _use_input_gemm_workspace(ext):
                        ext.adaptive_lstm_input_gemm_forward_workspace(
                            input_2d, args.weight_ih_t, generic_workspace.gate
                        )
                        actual_input_gemm = True
                    else:
                        torch.mm(input_2d, args.weight_ih_t, out=generic_workspace.gate)
                    gate = generic_workspace.gate.view(batch_size, seq_len, 4 * hidden_size)
                    actual_gate_workspace = True
                else:
                    input_2d = layer_input.view(batch_size * seq_len, input_size)
                    gate = torch.matmul(input_2d, args.weight_ih_t)
                    gate = gate.view(batch_size, seq_len, 4 * hidden_size).contiguous()
                if profile_enabled:
                    proj_end.record()
                is_last = layer_idx == len(layer_args) - 1
                if profile_enabled:
                    recur_start.record()
                if is_seqmajor_accum:
                    actual_gemm_scan_read_block = _valid_gemm_scan_read_block(
                        plan.hidden_launch.read_block, hidden_size
                    )
                    actual_direct_blas = _use_direct_blas()
                    out_workspace = seq_workspace.last_out if is_last else seq_workspace.seq_buffers[layer_idx & 1]
                    layer_input = ext.adaptive_lstm_h128_seqmajor_accum_update_forward(
                        gate,
                        args.weight_hh_t,
                        seq_workspace.h_state,
                        seq_workspace.c_state,
                        out_workspace,
                        not is_last,
                        actual_gemm_scan_read_block,
                    )
                    layer_input_is_seqmajor = not is_last
                    actual_kernel_name = "h128_seqmajor_accum"
                    actual_pipeline_name = "seqmajor_accum"
                elif backend == "persistent_mmac" and (
                    (hidden_size == 128 and hasattr(ext, "adaptive_lstm_h128_persistent_mmac_update_forward_workspace"))
                    or hasattr(ext, f"adaptive_lstm_h{hidden_size}_mmac_packed_variant_forward_workspace")
                ):
                    actual_gemm_scan_read_block = _valid_gemm_scan_read_block(
                        plan.hidden_launch.read_block, hidden_size
                    )
                    use_packed_mmac = _use_packed_mmac(ext, hidden_size) and args.packed_weight_hh is not None
                    packed_fn = _get_packed_mmac_fn(ext, hidden_size) if use_packed_mmac else None
                    prefix = f"h{hidden_size}"
                    if workspace is not None:
                        out_workspace = workspace.last_out if is_last else workspace.seq_buffers[layer_idx & 1]
                        if use_packed_mmac and packed_fn:
                            layer_input = packed_fn(
                                gate,
                                args.packed_weight_hh,
                                args.bias,
                                workspace.h_state,
                                workspace.c_state,
                                out_workspace,
                                workspace.recur,
                                not is_last,
                                actual_gemm_scan_read_block,
                                3,
                            )
                            actual_kernel_name = f"{prefix}_persistent_mmac_packed"
                            actual_pipeline_name = "persistent_mmac_packed"
                        elif hidden_size == 128:
                            layer_input = ext.adaptive_lstm_h128_persistent_mmac_update_forward_workspace(
                                gate,
                                args.weight_hh,
                                args.bias,
                                workspace.h_state,
                                workspace.c_state,
                                out_workspace,
                                not is_last,
                                actual_gemm_scan_read_block,
                                0,
                            )
                            actual_kernel_name = f"{prefix}_persistent_mmac"
                            actual_pipeline_name = "persistent_mmac"
                        actual_gemm_scan_workspace = True
                    else:
                        _h_state = torch.zeros((batch_size, hidden_size), device=gate.device, dtype=gate.dtype)
                        _c_state = torch.zeros((batch_size, hidden_size), device=gate.device, dtype=torch.float32)
                        _out = torch.empty(
                            (batch_size, seq_len, hidden_size) if not is_last else (batch_size, hidden_size),
                            device=gate.device, dtype=gate.dtype)
                        if use_packed_mmac and packed_fn:
                            _pout = torch.empty((batch_size, 4 * hidden_size), device=gate.device, dtype=gate.dtype)
                            layer_input = packed_fn(
                                gate,
                                args.packed_weight_hh,
                                args.bias,
                                _h_state,
                                _c_state,
                                _out,
                                _pout,
                                not is_last,
                                actual_gemm_scan_read_block,
                                3,
                            )
                            actual_kernel_name = f"{prefix}_persistent_mmac_packed"
                            actual_pipeline_name = "persistent_mmac_packed"
                        elif hidden_size == 128:
                            layer_input = ext.adaptive_lstm_h128_persistent_mmac_update_forward_workspace(
                                gate,
                                args.weight_hh,
                                args.bias,
                                _h_state,
                                _c_state,
                                _out,
                                not is_last,
                                actual_gemm_scan_read_block,
                                0,
                            )
                            actual_kernel_name = f"{prefix}_persistent_mmac"
                            actual_pipeline_name = "persistent_mmac"
                    layer_input_is_seqmajor = False
                elif (
                    backend == "profile_variants"
                    and hidden_size == 128
                    and hasattr(ext, "adaptive_lstm_h128_mmac_profile_variant_forward_workspace")
                ):
                    variant = _profile_variant()
                    variant_names = {0: "mmac_only", 1: "mmac_bias", 2: "mmac_act", 3: "mmac_full"}
                    actual_gemm_scan_read_block = _valid_gemm_scan_read_block(
                        plan.hidden_launch.read_block, hidden_size
                    )
                    if workspace is not None:
                        out_workspace = workspace.last_out if is_last else workspace.seq_buffers[layer_idx & 1]
                        layer_input = ext.adaptive_lstm_h128_mmac_profile_variant_forward_workspace(
                            gate,
                            args.weight_hh,
                            args.bias,
                            workspace.h_state,
                            workspace.c_state,
                            out_workspace,
                            workspace.recur,
                            not is_last,
                            actual_gemm_scan_read_block,
                            variant,
                        )
                        actual_gemm_scan_workspace = True
                    else:
                        _h_state = torch.zeros((batch_size, hidden_size), device=gate.device, dtype=gate.dtype)
                        _c_state = torch.zeros((batch_size, hidden_size), device=gate.device, dtype=torch.float32)
                        _out = torch.empty(
                            (batch_size, seq_len, hidden_size) if not is_last else (batch_size, hidden_size),
                            device=gate.device, dtype=gate.dtype)
                        _pout = torch.empty((batch_size, 512), device=gate.device, dtype=gate.dtype)
                        layer_input = ext.adaptive_lstm_h128_mmac_profile_variant_forward_workspace(
                            gate,
                            args.weight_hh,
                            args.bias,
                            _h_state,
                            _c_state,
                            _out,
                            _pout,
                            not is_last,
                            actual_gemm_scan_read_block,
                            variant,
                        )
                    actual_kernel_name = f"h128_profile_v{variant}_{variant_names.get(variant, '?')}"
                    actual_pipeline_name = "profile_variants"
                    layer_input_is_seqmajor = False
                elif backend == "gemm_scan" and hidden_size == 128 and hasattr(ext, "adaptive_lstm_h128_gemm_scan_update_forward"):
                    actual_gemm_scan_read_block = _valid_gemm_scan_read_block(
                        plan.hidden_launch.read_block, hidden_size
                    )
                    actual_direct_blas = _use_direct_blas()
                    if workspace is not None:
                        out_workspace = workspace.last_out if is_last else workspace.seq_buffers[layer_idx & 1]
                        layer_input = ext.adaptive_lstm_h128_gemm_scan_update_forward_workspace(
                            gate,
                            args.weight_hh_t,
                            args.bias,
                            workspace.h_state,
                            workspace.c_state,
                            workspace.recur,
                            out_workspace,
                            not is_last,
                            actual_gemm_scan_read_block,
                        )
                        actual_gemm_scan_workspace = True
                    else:
                        layer_input = ext.adaptive_lstm_h128_gemm_scan_update_forward(
                            gate,
                            args.weight_hh_t,
                            args.bias,
                            not is_last,
                            actual_gemm_scan_read_block,
                        )
                    actual_kernel_name = "h128_gemm_scan"
                    actual_pipeline_name = "gemm_scan"
                    layer_input_is_seqmajor = False
                elif backend == "gemm_scan":
                    if generic_workspace is not None:
                        out_workspace = generic_workspace.last_out if is_last else generic_workspace.seq_buffers[layer_idx & 1]
                        actual_gemm_scan_read_block = _valid_gemm_scan_read_block(
                            plan.hidden_launch.read_block, hidden_size
                        )
                        actual_direct_blas = _use_direct_blas()
                        use_gate_accum_scan = _use_gate_accum_scan()
                        use_fixed_hidden_scan = _use_fixed_hidden_scan()
                        if (
                            use_gate_accum_scan
                            and hidden_size == 256
                            and hasattr(ext, "adaptive_lstm_h256_gate_accum_update_forward_workspace")
                        ):
                            layer_input = ext.adaptive_lstm_h256_gate_accum_update_forward_workspace(
                                gate,
                                args.weight_hh_t,
                                args.bias,
                                generic_workspace.h_state,
                                generic_workspace.c_state,
                                out_workspace,
                                not is_last,
                                actual_gemm_scan_read_block,
                            )
                            actual_kernel_name = "h256_gate_accum"
                        elif (
                            use_gate_accum_scan
                            and hidden_size == 512
                            and hasattr(ext, "adaptive_lstm_h512_gate_accum_update_forward_workspace")
                        ):
                            layer_input = ext.adaptive_lstm_h512_gate_accum_update_forward_workspace(
                                gate,
                                args.weight_hh_t,
                                args.bias,
                                generic_workspace.h_state,
                                generic_workspace.c_state,
                                out_workspace,
                                not is_last,
                                actual_gemm_scan_read_block,
                            )
                            actual_kernel_name = "h512_gate_accum"
                        elif (
                            use_fixed_hidden_scan
                            and hidden_size == 256
                            and hasattr(ext, "adaptive_lstm_h256_gemm_scan_update_forward_workspace")
                        ):
                            layer_input = ext.adaptive_lstm_h256_gemm_scan_update_forward_workspace(
                                gate,
                                args.weight_hh_t,
                                args.bias,
                                generic_workspace.h_state,
                                generic_workspace.c_state,
                                generic_workspace.recur,
                                out_workspace,
                                not is_last,
                                actual_gemm_scan_read_block,
                                _recurrent_compute_for_layer(layer_idx, len(layer_args), hidden_size),
                            )
                            actual_kernel_name = "h256_gemm_scan"
                        elif (
                            use_fixed_hidden_scan
                            and hidden_size == 512
                            and hasattr(ext, "adaptive_lstm_h512_gemm_scan_update_forward_workspace")
                        ):
                            layer_input = ext.adaptive_lstm_h512_gemm_scan_update_forward_workspace(
                                gate,
                                args.weight_hh_t,
                                args.bias,
                                generic_workspace.h_state,
                                generic_workspace.c_state,
                                generic_workspace.recur,
                                out_workspace,
                                not is_last,
                                actual_gemm_scan_read_block,
                                _recurrent_compute_for_layer(layer_idx, len(layer_args), hidden_size),
                            )
                            actual_kernel_name = "h512_gemm_scan"
                        else:
                            layer_input = ext.adaptive_lstm_gemm_scan_update_forward_workspace(
                                gate,
                                args.weight_hh_t,
                                args.bias,
                                generic_workspace.h_state,
                                generic_workspace.c_state,
                                generic_workspace.recur,
                                out_workspace,
                                not is_last,
                                actual_gemm_scan_read_block,
                            )
                            actual_kernel_name = "generic_gemm_scan"
                        actual_pipeline_name = "gemm_scan"
                        actual_gemm_scan_workspace = True
                        layer_input_is_seqmajor = False
                    else:
                        layer_input = self._forward_generic_gemm_scan_layer(
                            gate,
                            args.weight_hh_t,
                            args.bias,
                            not is_last,
                        )
                        actual_kernel_name = "generic_gemm_scan_python"
                        actual_pipeline_name = "gemm_scan_python"
                        layer_input_is_seqmajor = False
                elif backend == "cached" and _use_h128_cached_path(ext, hidden_size):
                    cached_batch_tile = _h128_cached_batch_tile(ext, hidden_size, batch_size)
                    actual_cached_batch_tile = cached_batch_tile
                    if cached_batch_tile == 8:
                        actual_kernel_name = "h128_cached_b8"
                        actual_pipeline_name = "cached_b8"
                        layer_input = ext.adaptive_lstm_h128_cached_b8_update_forward(
                            gate,
                            args.weight_hh,
                            args.bias,
                            not is_last,
                        )
                        layer_input_is_seqmajor = False
                    elif cached_batch_tile == 4:
                        actual_kernel_name = "h128_cached_b4"
                        actual_pipeline_name = "cached_b4"
                        layer_input = ext.adaptive_lstm_h128_cached_b4_update_forward(
                            gate,
                            args.weight_hh,
                            args.bias,
                            not is_last,
                        )
                        layer_input_is_seqmajor = False
                    else:
                        actual_kernel_name = "h128_cached_b2"
                        actual_pipeline_name = "cached_b2"
                        layer_input = ext.adaptive_lstm_h128_cached_update_forward(
                            gate,
                            args.weight_hh,
                            args.bias,
                            not is_last,
                        )
                        layer_input_is_seqmajor = False
                elif backend == "partitioned" and hasattr(ext, "adaptive_lstm_hidden_update_partitioned_forward"):
                    actual_kernel_name = "partitioned_hidden_update"
                    actual_pipeline_name = "partitioned"
                    layer_input = ext.adaptive_lstm_hidden_update_partitioned_forward(
                        gate,
                        args.weight_hh,
                        args.bias,
                        not is_last,
                        partitions,
                    )
                    layer_input_is_seqmajor = False
                else:
                    actual_kernel_name = "scalar_hidden_update"
                    actual_pipeline_name = "scalar"
                    layer_input = ext.adaptive_lstm_hidden_update_forward(
                        gate,
                        args.weight_hh,
                        args.bias,
                        not is_last,
                        int(plan.hidden_launch.read_block),
                        int(plan.hidden_launch.items_per_group),
                    )
                    layer_input_is_seqmajor = False
                if profile_enabled:
                    recur_end.record()
                    torch.cuda.synchronize()
                    profile_rows.append(
                        (
                            layer_idx,
                            float(proj_start.elapsed_time(proj_end)),
                            float(recur_start.elapsed_time(recur_end)),
                        )
                    )

            if profile_enabled:
                linear_start = torch.cuda.Event(enable_timing=True)
                linear_end = torch.cuda.Event(enable_timing=True)
                linear_start.record()
            linear_workspace = self._ensure_linear_workspace(
                batch_size=int(layer_input.size(0)),
                output_size=int(linear_weight_t.size(1)),
                device=layer_input.device,
                dtype=layer_input.dtype,
            )
            torch.mm(layer_input, linear_weight_t, out=linear_workspace.out)
            linear_workspace.out.add_(linear_bias)
            out = linear_workspace.out
            if profile_enabled:
                linear_end.record()
                torch.cuda.synchronize()
                linear_ms = float(linear_start.elapsed_time(linear_end))

            if (
                os.environ.get("MIOPEN_ADAPTIVE_LSTM_DEBUG", "0") == "1"
                and not self._debug_reported
                and should_report
            ):
                print(
                    "miopen_adaptive_lstm_hip debug: "
                    f"batch={int(x.size(0))}, "
                    f"seq_len={int(x.size(1))}, "
                    f"input_size={int(x.size(2))}, "
                    f"hidden_size={int(self.lstm.hidden_size)}, "
                    f"num_layers={int(self.lstm.num_layers)}, "
                    f"read_block={plan.hidden_launch.read_block}, "
                    f"items_per_group={plan.hidden_launch.items_per_group}, "
                    f"partitions={_recurrent_partitions(int(self.lstm.hidden_size))}, "
                    f"h128_cached={_use_h128_cached_path(ext, int(self.lstm.hidden_size))}, "
                    f"h128_batch_tile={actual_cached_batch_tile}, "
                    f"gemm_scan_read_block={actual_gemm_scan_read_block}, "
                    f"gemm_scan_workspace={actual_gemm_scan_workspace}, "
                    f"gate_workspace={actual_gate_workspace}, "
                    f"seqmajor_accum={actual_seqmajor_accum}, "
                    f"direct_blas={actual_direct_blas}, "
                    f"recurrent_compute={_recurrent_compute_mode()}, "
                    f"input_gemm={actual_input_gemm}, "
                    "cuda_graph=False, "
                    f"kernel={actual_kernel_name}, "
                    f"pipeline={actual_pipeline_name}, "
                    f"recurrent_algo={actual_kernel_name}"
                )
                if profile_rows:
                    for layer_idx, proj_ms, recur_ms in profile_rows:
                        print(
                            "miopen_adaptive_lstm_hip profile: "
                            f"layer={layer_idx}, input_proj_ms={proj_ms:.3f}, recurrent_ms={recur_ms:.3f}"
                        )
                    print(f"miopen_adaptive_lstm_hip profile: linear_ms={linear_ms:.3f}")
                self._debug_reported = True
            return out

    def _forward_generic_gemm_scan_layer(
        self,
        gate: torch.Tensor,
        weight_hh_t: torch.Tensor,
        bias: torch.Tensor,
        write_sequence: bool,
    ) -> torch.Tensor:
        batch_size = int(gate.size(0))
        seq_len = int(gate.size(1))
        hidden_size = int(gate.size(2) // 4)
        h_state = torch.zeros((batch_size, hidden_size), device=gate.device, dtype=gate.dtype)
        c_state = torch.zeros((batch_size, hidden_size), device=gate.device, dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            recur = torch.matmul(h_state, weight_hh_t)
            gates = gate[:, t, :] + recur + bias
            i_gate, f_gate, g_gate, o_gate = gates.float().chunk(4, dim=1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            g_gate = torch.tanh(g_gate)
            o_gate = torch.sigmoid(o_gate)
            c_state = f_gate * c_state + i_gate * g_gate
            h_state = (o_gate * torch.tanh(c_state)).to(gate.dtype)
            if write_sequence:
                outputs.append(h_state)
        if write_sequence:
            return torch.stack(outputs, dim=1).contiguous()
        return h_state


def can_convert_regressor_module(model: nn.Module) -> bool:
    return (
        hasattr(model, "lstm")
        and hasattr(model, "linear")
        and isinstance(model.lstm, nn.LSTM)
        and isinstance(model.linear, nn.Linear)
        and model.lstm.batch_first
        and not model.lstm.bidirectional
        and model.lstm.proj_size == 0
        and model.lstm.bias
        and model.linear.in_features == model.lstm.hidden_size
    )


def convert_regressor_module(model: nn.Module) -> nn.Module:
    if not can_convert_regressor_module(model):
        return NativeModuleFallback(copy.deepcopy(model), "native_pytorch_unsupported")
    if load_extension() is None:
        return NativeModuleFallback(copy.deepcopy(model), "native_pytorch_no_adaptive_ext")
    return AdaptiveLSTMRegressor(model)
