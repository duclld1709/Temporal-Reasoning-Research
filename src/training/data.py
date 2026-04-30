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

    Dùng để mask loss tới user, chỉ tính loss trên phần assistant. Resolve động
    bằng cách so sánh template với và không có ``add_generation_prompt`` để tách
    phần đuôi.
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


class CompletionOnlyCollator:
    """Collator: pad batch, mask label cho phần prompt, chỉ tính loss trên assistant.

    Thay thế cho ``trl.DataCollatorForCompletionOnlyLM`` (đã được loại bỏ trong
    các phiên bản TRL gần đây). Tìm subsequence ``response_token_ids`` trong
    ``input_ids`` của từng sample, set labels[:end_of_response] = -100. Nếu
    không tìm thấy template (ví dụ sample bị truncate quá ngắn) → mask toàn bộ
    để sample đó không đóng góp loss.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        response_template: str,
        pad_to_multiple_of: int | None = 8,
    ):
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id set.")
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.response_template = response_template
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        if not self.response_token_ids:
            raise ValueError(
                f"Empty tokenization for response_template={response_template!r}"
            )

    def __call__(self, examples):
        import torch

        # Pad input_ids + attention_mask (right-padding).
        batch = self.tokenizer.pad(
            [{"input_ids": ex["input_ids"]} for ex in examples],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()
        # Mask pad tokens.
        labels[attention_mask == 0] = -100
        # Mask prompt tokens (tới hết response_template) cho mỗi sample.
        rti = self.response_token_ids
        n = len(rti)
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            cut = -1
            for start in range(len(ids) - n + 1):
                if ids[start : start + n] == rti:
                    cut = start + n
                    break
            if cut < 0:
                labels[i, :] = -100  # template not found → don't train on this sample
            else:
                labels[i, :cut] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
