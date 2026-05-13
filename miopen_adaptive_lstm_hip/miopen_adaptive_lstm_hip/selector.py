from __future__ import annotations

from dataclasses import dataclass

from .descriptors import RNNDescriptor, RuntimeMode, SeqTensorDescriptor


@dataclass(frozen=True)
class AlgoSelection:
    use_dynamic: bool
    algo_name: str
    reason: str


def check_dynamic_algo_selection(
    rnn_desc: RNNDescriptor,
    x_desc: SeqTensorDescriptor,
    runtime: RuntimeMode,
) -> AlgoSelection:
    """MIOpen `CheckDynamicAlgoSelection()` adapted for inference research.

    Official MIOpen rejects inference for the rounded dynamic path. This project
    is inference-first, so `allow_inference_adaptation` lets us keep the same
    structural constraints while enabling the experimental path.
    """

    if x_desc.seq_len <= 0 or x_desc.batch_size <= 0:
        return AlgoSelection(False, "native_or_existing", "empty sequence or batch")

    if runtime.fwd_mode == "inference" and not runtime.allow_inference_adaptation:
        return AlgoSelection(False, "native_or_existing", "MIOpen-compatible inference rejection")

    algo_mode_match = (
        rnn_desc.algo_mode == "rounded_dynamic"
        or runtime.force_dynamic
        or rnn_desc.algo_mode == "forced_dynamic"
    )
    if not algo_mode_match:
        return AlgoSelection(False, "native_or_existing", "algo mode is not rounded dynamic")

    config_match = (
        rnn_desc.direction_count == 1
        and rnn_desc.input_mode == "linear"
        and rnn_desc.rnn_mode == "lstm"
        and rnn_desc.dropout == 0.0
        and rnn_desc.bias
    )
    if not config_match:
        return AlgoSelection(False, "native_or_existing", "RNN config is not dynamic-LSTM compatible")

    return AlgoSelection(True, "rounded_dynamic_adaptive", "matched dynamic LSTM selector")

