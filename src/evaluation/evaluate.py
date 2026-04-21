"""Compute metric cho 1 dataset từ list predictions.jsonl rows."""

from __future__ import annotations

from typing import Sequence

from .extractor import extract, normalize_gold
from .metrics import accuracy, binary_f1_yes


def score_records(
    records: Sequence[dict],
    task: str,
    language: str,
) -> dict:
    """Từ records có field gold + extracted, tính metric chuẩn theo task."""
    golds = [r["gold_normalized"] for r in records]
    preds = [r["extracted"] for r in records]
    if task == "duration":
        return binary_f1_yes(golds, preds)
    if task == "date_arith":
        return accuracy(golds, preds)
    raise KeyError(f"Unknown task {task!r}")


def build_record(
    sample: dict,
    raw_output: str,
    elapsed_sec: float,
) -> dict:
    task = sample["task"]
    language = sample["language"]
    extracted = extract(task, language, raw_output)
    gold_norm = normalize_gold(task, language, sample["gold"])
    correct = (extracted == gold_norm) if extracted is not None else False
    rec = {
        "sample_id": sample["sample_id"],
        "task": task,
        "language": language,
        "dataset": sample["dataset"],
        "question": sample["question"],
        "gold_raw": sample["gold"],
        "gold_normalized": gold_norm,
        "raw_output": raw_output,
        "extracted": extracted,
        "correct": correct,
        "elapsed_sec": elapsed_sec,
    }
    # Thêm candidate_answer cho task duration
    if task == "duration":
        rec["candidate_answer"] = sample["meta"].get("candidate_answer", "")
    # Thêm context cho task date_arith (nếu có)
    if task == "date_arith" and "context" in sample:
        rec["context"] = sample["context"]
    return rec
