"""Interface chung cho mọi method.

Mỗi method nhận model + config, expose `predict(sample) -> str` (raw output).
Method mới = thêm file + đăng ký vào REGISTRY.
"""

from __future__ import annotations

from typing import Protocol

from ..data.schema import Sample
from ..models.base import ChatLM


class Method(Protocol):
    name: str

    def predict(self, sample: Sample) -> str:
        ...


# Default generation hyperparams cho evaluation.
DEFAULT_GEN_KWARGS = {
    "duration": {"max_new_tokens": 8, "do_sample": False, "temperature": 0.0},
    "date_arith": {"max_new_tokens": 24, "do_sample": False, "temperature": 0.0},
}


def gen_kwargs_for(task: str) -> dict:
    return dict(DEFAULT_GEN_KWARGS.get(task, {"max_new_tokens": 32, "do_sample": False}))
