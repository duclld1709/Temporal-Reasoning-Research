"""Smoke tests cho src.training.data — không cần GPU/internet.

Mock tokenizer để verify rằng:
- split_train_val: chia đúng size, deterministic theo seed.
- samples_to_chat_dataset: render đủ system+user+assistant cho VI DateArith.
- resolve_assistant_response_template: trả về phần đuôi sau when add_generation_prompt.
"""

from __future__ import annotations

import importlib.util

import pytest

from src.data.schema import Sample
from src.training.data import (
    resolve_assistant_response_template,
    samples_to_chat_dataset,
    split_train_val,
)

_HAS_DATASETS = importlib.util.find_spec("datasets") is not None


class _FakeTokenizer:
    """Tokenizer giả lập chat template kiểu chat-ml (như Qwen)."""

    def apply_chat_template(
        self,
        chat,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        assert tokenize is False
        parts = []
        for msg in chat:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


def _vi_date_sample(idx: int, gold: str = "Tháng 4, 1321") -> Sample:
    return Sample(
        sample_id=f"vlsp_date-{idx}",
        task="date_arith",
        language="vi",
        dataset="vlsp_date",
        context="",
        question=f"Thời gian 1 năm sau tháng {idx % 12 + 1}, 1320?",
        gold=gold,
        meta={"all_answers": [gold]},
    )


def test_split_train_val_sizes_and_disjoint():
    samples = [_vi_date_sample(i) for i in range(100)]
    train, val = split_train_val(samples, val_ratio=0.1, seed=42)
    assert len(train) == 90
    assert len(val) == 10
    train_ids = {s["sample_id"] for s in train}
    val_ids = {s["sample_id"] for s in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {f"vlsp_date-{i}" for i in range(100)}


def test_split_train_val_deterministic():
    samples = [_vi_date_sample(i) for i in range(50)]
    a1, b1 = split_train_val(samples, val_ratio=0.1, seed=7)
    a2, b2 = split_train_val(samples, val_ratio=0.1, seed=7)
    assert [s["sample_id"] for s in a1] == [s["sample_id"] for s in a2]
    assert [s["sample_id"] for s in b1] == [s["sample_id"] for s in b2]


@pytest.mark.skipif(not _HAS_DATASETS, reason="`datasets` package not installed")
def test_samples_to_chat_dataset_renders_system_user_assistant():
    samples = [_vi_date_sample(0, gold="Tháng 4, 1321")]
    ds = samples_to_chat_dataset(samples, _FakeTokenizer())
    assert len(ds) == 1
    text = ds[0]["text"]
    # System (VI DateArith) + user (question) + assistant (gold) đều phải có mặt.
    assert "<|im_start|>system" in text
    assert "Tháng M, YYYY" in text  # phần system prompt VI DateArith
    assert "<|im_start|>user" in text
    assert "Câu hỏi:" in text or "tháng" in text.lower()
    assert "<|im_start|>assistant" in text
    assert "Tháng 4, 1321" in text
    # Không có generation prompt vì add_generation_prompt=False.
    assert not text.endswith("<|im_start|>assistant\n")


def test_resolve_assistant_response_template():
    tmpl = resolve_assistant_response_template(_FakeTokenizer())
    assert tmpl == "<|im_start|>assistant\n"
