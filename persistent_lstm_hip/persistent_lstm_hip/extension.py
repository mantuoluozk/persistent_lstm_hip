from __future__ import annotations

import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def load_extension():
    try:
        return importlib.import_module("persistent_lstm_hip_ext")
    except Exception:
        return None

