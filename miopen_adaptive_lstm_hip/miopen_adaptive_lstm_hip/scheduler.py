from __future__ import annotations

from .descriptors import AlgoMode, FwdMode, InputMode, RNNMode
from .modular import (
    Pow2Segment,
    hidden_prop_segments,
    lower_bound_pow2,
    masked_pow2_range,
    x_prop_segments,
)
from .pipeline import HiddenUpdateLaunch, choose_hidden_update_launch as _choose_hidden_update_launch
from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicAlgoConfig:
    algo_mode: AlgoMode = "default"
    fwd_mode: FwdMode = "inference"
    direction_count: int = 1
    input_mode: InputMode = "linear"
    rnn_mode: RNNMode = "lstm"
    dropout: float = 0.0
    force_dynamic: bool = False
    allow_inference_adaptation: bool = True


@dataclass(frozen=True)
class AdaptiveLSTMPlan:
    use_dynamic_algo: bool
    recurrent_algo: str
    hidden_launch: HiddenUpdateLaunch
    x_prop_segments: tuple[Pow2Segment, ...]
    hidden_prop_segments: tuple[Pow2Segment, ...]
    notes: tuple[str, ...]


def check_dynamic_algo_selection(config: DynamicAlgoConfig) -> bool:
    """Return whether the MIOpen-style dynamic path is eligible.

    MIOpen's `CheckDynamicAlgoSelection()` rejects inference. This project is
    inference-first, so `allow_inference_adaptation=True` keeps the useful
    selector constraints while allowing our local experimental backend to run
    during inference.
    """

    if config.fwd_mode == "inference" and not config.allow_inference_adaptation:
        return False

    algo_mode_match = (
        config.algo_mode == "rounded_dynamic"
        or config.force_dynamic
        or config.algo_mode == "forced_dynamic"
    )
    rnn_config_match = (
        config.direction_count == 1
        and config.input_mode == "linear"
        and config.rnn_mode == "lstm"
        and config.dropout == 0.0
    )
    return algo_mode_match and rnn_config_match


def choose_hidden_update_launch(
    *,
    max_compute_units: int,
    wavefront_width: int,
    max_batch: int,
    hidden_size: int,
) -> HiddenUpdateLaunch:
    """Mirror MIOpen's hidden-state update launch sizing heuristic."""

    if max_compute_units <= 0:
        raise ValueError("max_compute_units must be positive")
    if wavefront_width <= 0:
        raise ValueError("wavefront_width must be positive")
    if max_batch <= 0:
        raise ValueError("max_batch must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")

    return _choose_hidden_update_launch(
        max_active_threads=max_compute_units * wavefront_width * 32,
        max_batch=max_batch,
        hidden_size=hidden_size,
    )


def select_project_algo(hidden_size: int, use_dynamic_algo: bool) -> str:
    if not use_dynamic_algo:
        return "native_or_existing"
    if hidden_size % 64 == 0 and hidden_size >= 128:
        return "adaptive_tiled_mfma_candidate"
    if hidden_size <= 256:
        return "adaptive_tiled_scalar"
    return "adaptive_gemm_scan"


def build_adaptive_plan(
    *,
    batch_size: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    max_compute_units: int = 120,
    wavefront_width: int = 64,
    config: DynamicAlgoConfig | None = None,
) -> AdaptiveLSTMPlan:
    if batch_size <= 0 or seq_len <= 0 or input_size <= 0 or num_layers <= 0:
        raise ValueError("batch_size, seq_len, input_size, and num_layers must be positive")

    dynamic_config = config or DynamicAlgoConfig(algo_mode="rounded_dynamic")
    use_dynamic = check_dynamic_algo_selection(dynamic_config)
    launch = choose_hidden_update_launch(
        max_compute_units=max_compute_units,
        wavefront_width=wavefront_width,
        max_batch=batch_size,
        hidden_size=hidden_size,
    )
    recurrent_algo = select_project_algo(hidden_size, use_dynamic)

    notes = [
        "input projection remains GEMM-backed",
        "recurrent update keeps the sequence loop inside the kernel",
    ]
    if recurrent_algo == "adaptive_tiled_mfma_candidate":
        notes.append("hidden size is MFMA-friendly; use tiled recurrent prototype first")
    elif recurrent_algo == "adaptive_gemm_scan":
        notes.append("large hidden sizes should stay GEMM-scan until tiled MFMA is ready")

    return AdaptiveLSTMPlan(
        use_dynamic_algo=use_dynamic,
        recurrent_algo=recurrent_algo,
        hidden_launch=launch,
        x_prop_segments=x_prop_segments(seq_len),
        hidden_prop_segments=hidden_prop_segments(seq_len),
        notes=tuple(notes),
    )
