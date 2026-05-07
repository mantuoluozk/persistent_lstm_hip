from __future__ import annotations

import copy

import torch.nn as nn

from .model import PersistentLSTM, PersistentLSTMRegressor


def convert_lstm_module(lstm: nn.LSTM) -> PersistentLSTM:
    return PersistentLSTM.from_native_module(lstm)


def convert_regressor_module(model: nn.Module) -> PersistentLSTMRegressor:
    if not hasattr(model, "lstm") or not hasattr(model, "linear"):
        raise ValueError("convert_regressor_module 需要模型至少包含 `.lstm` 和 `.linear`。")
    if not isinstance(model.lstm, nn.LSTM):
        raise ValueError("model.lstm 必须是 nn.LSTM。")
    return PersistentLSTMRegressor.from_native_module(model)


def replace_lstm_inplace(model: nn.Module) -> nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, nn.LSTM):
            setattr(model, name, PersistentLSTM.from_native_module(child))
        else:
            replace_lstm_inplace(child)
    return model


def convert_model_copy(model: nn.Module) -> nn.Module:
    copied = copy.deepcopy(model)
    return replace_lstm_inplace(copied)
