"""Registry cho methods. Thêm method mới = đăng ký tại đây."""

from __future__ import annotations

from typing import Any, Callable

from ..data.schema import Sample
from ..models.base import ChatLM
from .few_shot import FewShotMethod, fixed_shots
from .zero_shot import ZeroShotMethod


def build_zero_shot(model: ChatLM, **kwargs: Any) -> ZeroShotMethod:
    return ZeroShotMethod(model=model, enable_thinking=kwargs.get("enable_thinking", False))


def build_few_shot(
    model: ChatLM,
    shots: list[Sample],
    **kwargs: Any,
) -> FewShotMethod:
    return FewShotMethod(
        model=model,
        shot_selector=fixed_shots(shots),
        enable_thinking=kwargs.get("enable_thinking", False),
    )


METHOD_BUILDERS: dict[str, Callable[..., Any]] = {
    "zero_shot": build_zero_shot,
    "few_shot": build_few_shot,
}


def build_method(name: str, model: ChatLM, **kwargs: Any):
    if name not in METHOD_BUILDERS:
        raise KeyError(f"Unknown method {name!r}. Available: {list(METHOD_BUILDERS)}")
    return METHOD_BUILDERS[name](model, **kwargs)
