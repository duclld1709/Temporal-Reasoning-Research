"""Zero-shot method."""

from __future__ import annotations

from ..data.schema import Sample
from ..models.base import ChatLM
from ..prompts.templates import build_messages
from .base import gen_kwargs_for


class ZeroShotMethod:
    name = "zero_shot"

    def __init__(self, model: ChatLM, enable_thinking: bool = False):
        self.model = model
        self.enable_thinking = enable_thinking

    def predict(self, sample: Sample) -> str:
        messages = build_messages(sample, shots=())
        kwargs = gen_kwargs_for(sample["task"])
        kwargs["enable_thinking"] = self.enable_thinking
        if self.enable_thinking:
            kwargs["max_new_tokens"] = max(kwargs["max_new_tokens"], 256)
        return self.model.generate(messages, **kwargs)
