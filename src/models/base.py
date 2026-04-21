"""Base interface cho mọi LM wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatLM(Protocol):
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        ...
