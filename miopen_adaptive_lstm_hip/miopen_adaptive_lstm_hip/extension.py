from __future__ import annotations

import importlib
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def load_extension():
    try:
        return importlib.import_module("miopen_adaptive_lstm_hip._C")
    except Exception as exc:
        if os.environ.get("MIOPEN_ADAPTIVE_LSTM_DEBUG", "0") == "1":
            print(f"miopen_adaptive_lstm_hip extension load failed: {exc}")
        return None
