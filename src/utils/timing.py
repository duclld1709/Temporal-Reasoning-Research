from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.perf_counter()
    state = {"elapsed": 0.0}
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - start
