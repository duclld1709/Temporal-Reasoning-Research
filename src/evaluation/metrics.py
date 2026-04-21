"""Metrics Phase 1.

- F1 Score (binary, positive class = 'yes') cho Duration Reasoning.
- Accuracy (string match sau extract + normalize) cho Date Arithmetic.
- Avg inference time per sample (giây).
"""

from __future__ import annotations

from typing import Iterable, Sequence


def binary_f1_yes(
    gold: Sequence[str],
    pred: Sequence[str | None],
) -> dict[str, float]:
    """F1 cho class 'yes'. pred=None (không parse được) → coi là 'no' (fail).

    Trả về precision, recall, f1, support + đếm parse-fail.
    """
    if len(gold) != len(pred):
        raise ValueError(f"len mismatch: gold={len(gold)} pred={len(pred)}")
    tp = fp = fn = tn = 0
    parse_fail = 0
    for g, p in zip(gold, pred):
        g_pos = (g == "yes")
        if p is None:
            parse_fail += 1
            p_pos = False
        else:
            p_pos = (p == "yes")
        if g_pos and p_pos:
            tp += 1
        elif not g_pos and p_pos:
            fp += 1
        elif g_pos and not p_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": len(gold),
        "parse_fail": parse_fail,
    }


def accuracy(
    gold: Sequence[str],
    pred: Sequence[str | None],
) -> dict[str, float]:
    if len(gold) != len(pred):
        raise ValueError(f"len mismatch: gold={len(gold)} pred={len(pred)}")
    correct = 0
    parse_fail = 0
    for g, p in zip(gold, pred):
        if p is None:
            parse_fail += 1
            continue
        if g == p:
            correct += 1
    n = len(gold)
    return {
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "support": n,
        "parse_fail": parse_fail,
    }


def avg_inference_time(times: Iterable[float]) -> float:
    ts = list(times)
    return sum(ts) / len(ts) if ts else 0.0
