"""Loader cho VLSP 2025 ViTempQA DateArith (Vietnamese).

Format: JSONL mỗi dòng {"question", "answer": [str], "context"}.
Gold: answer[0] (chuỗi "Tháng M, YYYY").
"""

from __future__ import annotations

from pathlib import Path

from ..utils.io import read_jsonl
from .schema import Sample

DEFAULT_MAX_SAMPLES = 1500


def load_vlsp_date(
    path: str | Path,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> list[Sample]:
    samples: list[Sample] = []
    for idx, row in enumerate(read_jsonl(path)):
        if max_samples is not None and idx >= max_samples:
            break
        answers = row.get("answer") or []
        if not answers:
            raise ValueError(f"VLSP Date line {idx} has no answer: {row}")
        gold = answers[0]
        samples.append(
            Sample(
                sample_id=f"vlsp_date-{idx}",
                task="date_arith",
                language="vi",
                dataset="vlsp_date",
                context=row.get("context", "") or "",
                question=row["question"],
                gold=gold,
                meta={"all_answers": answers},
            )
        )
    return samples
