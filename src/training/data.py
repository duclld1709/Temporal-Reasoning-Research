"""Helpers để chuyển ``Sample`` sang định dạng SFT (chat-format string).

Tận dụng prompt template hiện có (``src/prompts/templates.py``) để training
distribution khớp 100% với inference prompt — adapter học đúng format model
thấy lúc eval.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Sequence

from ..data.schema import Sample
from ..prompts.templates import build_messages, get_template

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


def split_train_val(
    samples: Sequence[Sample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Shuffle (deterministic) rồi cắt val cuối list.

    Trả về (train, val). Không động vào input list gốc.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio phải trong (0, 1), got {val_ratio}")
    items = list(samples)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * val_ratio)))
    train = items[:-n_val]
    val = items[-n_val:]
    return train, val


def _render_one(
    sample: Sample,
    tokenizer: "PreTrainedTokenizerBase",
    enable_thinking: bool = False,
) -> str:
    """Render 1 sample thành chuỗi chat-template gồm system+user+assistant.

    Assistant content = gold của sample (đã chuẩn hoá theo schema).
    """
    msgs = build_messages(sample, shots=())
    chat = [{"role": m.role, "content": m.content} for m in msgs]
    tmpl = get_template(sample["task"], sample["language"])
    chat.append({"role": "assistant", "content": tmpl.render_shot_assistant(sample)})
    text = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return text


def samples_to_chat_dataset(
    samples: Sequence[Sample],
    tokenizer: "PreTrainedTokenizerBase",
    enable_thinking: bool = False,
) -> "Dataset":
    """Convert list[Sample] -> ``datasets.Dataset`` với 1 cột ``text``."""
    from datasets import Dataset

    rows = [{"text": _render_one(s, tokenizer, enable_thinking=enable_thinking)}
            for s in samples]
    return Dataset.from_list(rows)


def resolve_assistant_response_template(
    tokenizer: "PreTrainedTokenizerBase",
) -> str:
    """Trả về chuỗi đánh dấu mở turn assistant trong chat template.

    Dùng cho ``DataCollatorForCompletionOnlyLM`` để mask loss tới user, chỉ tính
    loss trên phần assistant. Resolve động bằng cách so sánh template với và
    không có ``add_generation_prompt`` để tách phần đuôi.
    """
    dummy_user = [{"role": "user", "content": "x"}]
    with_gen = tokenizer.apply_chat_template(
        dummy_user, tokenize=False, add_generation_prompt=True
    )
    without_gen = tokenizer.apply_chat_template(
        dummy_user, tokenize=False, add_generation_prompt=False
    )
    if not with_gen.startswith(without_gen):
        # Fallback: hardcode cho Qwen chat-ml format.
        return "<|im_start|>assistant\n"
    return with_gen[len(without_gen):]
