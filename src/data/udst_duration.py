"""Loader cho UDST-DurationQA (English, Duration Reasoning).

Format thực tế của test.tsv (4 cột, no header):
    context \t question \t candidate_answer \t label (yes/no)

Mỗi dòng TSV = 1 sample binary. Phase 1: lấy `max_samples` dòng đầu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .schema import Sample

DEFAULT_MAX_SAMPLES = 1500


def load_udst_duration(
    path: str | Path,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"UDST line {idx} expected 4 columns, got {len(parts)}: {parts!r}"
                )
            context, question, candidate, label = parts
            label_norm = label.strip().lower()
            if label_norm not in {"yes", "no"}:
                raise ValueError(f"UDST line {idx} bad label: {label!r}")
            samples.append(
                Sample(
                    sample_id=f"udst-{idx}",
                    task="duration",
                    language="en",
                    dataset="udst_duration",
                    context=context,
                    question=question,
                    gold=label_norm,
                    meta={"candidate_answer": candidate},
                )
            )
    return samples


def iter_udst_duration(
    path: str | Path,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> Iterator[Sample]:
    for s in load_udst_duration(path, max_samples=max_samples):
        yield s
