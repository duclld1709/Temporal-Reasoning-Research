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
    CompletionOnlyCollator,
    resolve_assistant_response_template,
    samples_to_chat_dataset,
    split_train_val,
)

_HAS_DATASETS = importlib.util.find_spec("datasets") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None


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


class _FakeTokenizerForCollator:
    """Tokenizer giả tối thiểu cho CompletionOnlyCollator: encode + pad."""

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False):
        # Mỗi ký tự thành 1 token id (offset +1 để tránh trùng pad=0).
        return [ord(c) + 1 for c in text]

    def pad(self, encoded_inputs, padding=True, pad_to_multiple_of=None,
            return_tensors=None):
        import torch

        seqs = [ex["input_ids"] for ex in encoded_inputs]
        max_len = max(len(s) for s in seqs)
        if pad_to_multiple_of:
            rem = max_len % pad_to_multiple_of
            if rem != 0:
                max_len += pad_to_multiple_of - rem
        input_ids = torch.full((len(seqs), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_completion_only_collator_masks_prompt():
    tokenizer = _FakeTokenizerForCollator()
    response_template = "RESP"
    collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template=response_template,
        pad_to_multiple_of=None,
    )
    # Sample có "PROMPT_RESP_ANSWER" → mask tới hết "RESP".
    sample = {"input_ids": tokenizer.encode("PROMPT_RESP_ANSWER")}
    batch = collator([sample])
    labels = batch["labels"][0].tolist()
    input_ids = batch["input_ids"][0].tolist()
    # Tất cả tokens trước+gồm "RESP" phải = -100.
    resp_ids = tokenizer.encode(response_template)
    # Tìm vị trí RESP.
    n = len(resp_ids)
    cut = next(i + n for i in range(len(input_ids) - n + 1)
               if input_ids[i : i + n] == resp_ids)
    for j in range(cut):
        assert labels[j] == -100, f"label[{j}] should be masked"
    # Các token sau RESP phải == input_ids (loss được tính).
    for j in range(cut, len(input_ids)):
        if input_ids[j] != tokenizer.pad_token_id:
            assert labels[j] == input_ids[j], f"label[{j}] should equal input_id"


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_completion_only_collator_masks_all_when_template_missing():
    tokenizer = _FakeTokenizerForCollator()
    collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template="MISSING",
        pad_to_multiple_of=None,
    )
    sample = {"input_ids": tokenizer.encode("PROMPT_ONLY_NO_TEMPLATE_HERE")}
    batch = collator([sample])
    labels = batch["labels"][0].tolist()
    # Toàn bộ phải -100 vì template không match.
    assert all(label == -100 for label in labels)
