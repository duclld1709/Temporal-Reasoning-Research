"""Few-shot method — k shot cố định, chọn thủ công từ train set.

Để dễ mở rộng sang Dynamic Few-shot: chỉ cần thay `shot_selector`
(một callable nhận sample test → list[Sample]).
"""

from __future__ import annotations

from typing import Callable, Sequence

from ..data.schema import Sample
from ..models.base import ChatLM
from ..prompts.templates import build_messages
from .base import gen_kwargs_for


ShotSelector = Callable[[Sample], Sequence[Sample]]


def fixed_shots(shots: Sequence[Sample]) -> ShotSelector:
    def _select(_s: Sample) -> Sequence[Sample]:
        return shots
    return _select


class FewShotMethod:
    name = "few_shot"

    def __init__(
        self,
        model: ChatLM,
        shot_selector: ShotSelector,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.shot_selector = shot_selector
        self.enable_thinking = enable_thinking

    def predict(self, sample: Sample) -> str:
        shots = self.shot_selector(sample)
        messages = build_messages(sample, shots=shots)
        kwargs = gen_kwargs_for(sample["task"])
        kwargs["enable_thinking"] = self.enable_thinking
        if self.enable_thinking:
            kwargs["max_new_tokens"] = max(kwargs["max_new_tokens"], 256)
        return self.model.generate(messages, **kwargs)
