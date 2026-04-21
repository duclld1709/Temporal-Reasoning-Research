"""Loader cho BigBench DateUnderstanding (English, Date Arithmetic).

Format: JSON với examples[i] = {"input": str, "target_scores": {answer: 0|1}}.
Ta chuyển về open-ended bằng cách giữ duy nhất đáp án có score == 1.
File gốc chỉ có 369 examples → Phase 1 mặc định dùng toàn bộ.
"""

from __future__ import annotations

from pathlib import Path

from ..utils.io import read_json
from .schema import Sample

DEFAULT_MAX_SAMPLES: int | None = None  # = dùng hết


def load_bigbench_date(
    path: str | Path,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> list[Sample]:
    data = read_json(path)
    examples = data.get("examples", [])
    samples: list[Sample] = []
    for idx, ex in enumerate(examples):
        if max_samples is not None and idx >= max_samples:
            break
        question = ex["input"]
        scores = ex.get("target_scores", {})
        correct = [ans for ans, s in scores.items() if s == 1]
        if len(correct) != 1:
            raise ValueError(
                f"BigBench example {idx} expected exactly 1 correct answer, got {correct}"
            )
        gold = correct[0]
        samples.append(
            Sample(
                sample_id=f"bigbench-{idx}",
                task="date_arith",
                language="en",
                dataset="bigbench_date",
                context="",
                question=question,
                gold=gold,
                meta={"all_candidates": list(scores.keys())},
            )
        )
    return samples
