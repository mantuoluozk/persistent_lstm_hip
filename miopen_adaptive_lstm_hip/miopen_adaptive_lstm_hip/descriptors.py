from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AlgoMode = Literal["default", "rounded_dynamic", "forced_dynamic", "native"]
FwdMode = Literal["inference", "training"]
InputMode = Literal["linear", "skip"]
RNNMode = Literal["lstm", "gru", "rnn_relu", "rnn_tanh"]


@dataclass(frozen=True)
class RNNDescriptor:
    """Small local equivalent of the MIOpen RNN descriptor fields we use."""

    hidden_size: int
    num_layers: int
    input_mode: InputMode = "linear"
    direction_count: int = 1
    rnn_mode: RNNMode = "lstm"
    algo_mode: AlgoMode = "rounded_dynamic"
    dropout: float = 0.0
    bias: bool = True


@dataclass(frozen=True)
class SeqTensorDescriptor:
    batch_size: int
    seq_len: int
    input_size: int
    batch_first: bool = True

    @property
    def total_seq_count(self) -> int:
        return self.seq_len

    @property
    def max_batch(self) -> int:
        return self.batch_size


@dataclass(frozen=True)
class HardwareDescriptor:
    max_compute_units: int
    wavefront_width: int = 64

    @property
    def max_active_threads(self) -> int:
        return self.max_compute_units * self.wavefront_width * 32


@dataclass(frozen=True)
class RuntimeMode:
    fwd_mode: FwdMode = "inference"
    allow_inference_adaptation: bool = True
    force_dynamic: bool = False

