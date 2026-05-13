from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .descriptors import HardwareDescriptor, RNNDescriptor, RuntimeMode, SeqTensorDescriptor
from .pipeline import HiddenUpdateLaunch, RecurrentKernelPlan, choose_hidden_update_launch, choose_recurrent_kernel_plan
from .selector import AlgoSelection, check_dynamic_algo_selection


OperationKind = Literal["prepare", "prop_x", "prop_hidden_y", "hidden_update", "prop_y", "linear"]


@dataclass(frozen=True)
class Pow2Segment:
    offset: int
    size: int


@dataclass(frozen=True)
class ModularStep:
    kind: OperationKind
    layer: int
    segment: Pow2Segment | None
    write_sequence: bool
    note: str


@dataclass(frozen=True)
class AdaptiveForwardPlan:
    selection: AlgoSelection
    rnn_desc: RNNDescriptor
    x_desc: SeqTensorDescriptor
    hardware: HardwareDescriptor
    hidden_launch: HiddenUpdateLaunch
    recurrent_kernel: RecurrentKernelPlan
    x_prop_segments: tuple[Pow2Segment, ...]
    hidden_prop_segments: tuple[Pow2Segment, ...]
    steps: tuple[ModularStep, ...]

    @property
    def use_dynamic_algo(self) -> bool:
        return self.selection.use_dynamic

    @property
    def recurrent_algo(self) -> str:
        return self.recurrent_kernel.name


def lower_bound_pow2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value.bit_length() - 1)


def masked_pow2_range(value: int) -> tuple[int, ...]:
    if value <= 0:
        raise ValueError("value must be positive")
    pieces: list[int] = []
    remaining = value
    step = lower_bound_pow2(value)
    while step:
        if remaining & step:
            pieces.append(step)
            remaining -= step
        step >>= 1
    return tuple(pieces)


def x_prop_segments(total_seq_count: int) -> tuple[Pow2Segment, ...]:
    offset = 0
    segments: list[Pow2Segment] = []
    for step_size in masked_pow2_range(total_seq_count):
        segments.append(Pow2Segment(offset=offset, size=step_size))
        offset += step_size
    return tuple(segments)


def hidden_prop_segments(total_seq_count: int) -> tuple[Pow2Segment, ...]:
    segments: list[Pow2Segment] = []
    seq_it = 0
    step = lower_bound_pow2(total_seq_count)
    while seq_it < total_seq_count and step:
        if (total_seq_count - seq_it) & step:
            segments.append(Pow2Segment(offset=seq_it, size=step))
            seq_it += step
        step >>= 1
    return tuple(segments)


def build_forward_steps(
    *,
    num_layers: int,
    x_segments: tuple[Pow2Segment, ...],
    hidden_segments: tuple[Pow2Segment, ...],
) -> tuple[ModularStep, ...]:
    steps: list[ModularStep] = [ModularStep("prepare", 0, None, True, "prepare reserve/work buffers")]
    for layer in range(num_layers):
        for segment in x_segments:
            steps.append(ModularStep("prop_x", layer, segment, True, "input projection GEMM segment"))
        for segment in hidden_segments:
            steps.append(ModularStep("prop_hidden_y", layer, segment, layer != num_layers - 1, "recurrent propagation segment"))
        steps.append(ModularStep("hidden_update", layer, None, layer != num_layers - 1, "LSTM gate/cell/hidden update"))
    steps.append(ModularStep("linear", num_layers - 1, None, False, "linear head on final hidden state"))
    return tuple(steps)


def build_forward_plan(
    *,
    rnn_desc: RNNDescriptor,
    x_desc: SeqTensorDescriptor,
    hardware: HardwareDescriptor,
    runtime: RuntimeMode,
) -> AdaptiveForwardPlan:
    selection = check_dynamic_algo_selection(rnn_desc, x_desc, runtime)
    hidden_launch = choose_hidden_update_launch(
        max_active_threads=hardware.max_active_threads,
        max_batch=x_desc.max_batch,
        hidden_size=rnn_desc.hidden_size,
    )
    recurrent_kernel = choose_recurrent_kernel_plan(rnn_desc.hidden_size, x_desc.batch_size)
    xs = x_prop_segments(x_desc.total_seq_count)
    hs = hidden_prop_segments(x_desc.total_seq_count)
    return AdaptiveForwardPlan(
        selection=selection,
        rnn_desc=rnn_desc,
        x_desc=x_desc,
        hardware=hardware,
        hidden_launch=hidden_launch,
        recurrent_kernel=recurrent_kernel,
        x_prop_segments=xs,
        hidden_prop_segments=hs,
        steps=build_forward_steps(num_layers=rnn_desc.num_layers, x_segments=xs, hidden_segments=hs),
    )

