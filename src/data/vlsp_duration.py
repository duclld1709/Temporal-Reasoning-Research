"""Loader cho VLSP 2025 ViTempQA DurationQA (Vietnamese).

Format gốc: JSONL mỗi dòng có "context","question","options" (4),"labels" (4×yes/no),"qid".
Mỗi question expand thành 4 row binary (option, yes/no).

Phase 1: max_samples áp dụng TRÊN rows đã expand (mặc định 1500 rows).
"""

from __future__ import annotations

from pathlib import Path

from ..utils.io import read_jsonl
from .schema import Sample

DEFAULT_MAX_SAMPLES = 1500


def load_vlsp_duration(
    path: str | Path,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> list[Sample]:
    samples: list[Sample] = []
    for q_idx, row in enumerate(read_jsonl(path)):
        options = row.get("options") or []
        labels = row.get("labels") or []
        if len(options) != 4 or len(labels) != 4:
            raise ValueError(
                f"VLSP Duration line {q_idx} expected 4 options/labels, got "
                f"{len(options)}/{len(labels)}"
            )
        qid = row.get("qid", q_idx)
        context = row.get("context", "") or ""
        question = row["question"]
        for opt_idx, (opt, lab) in enumerate(zip(options, labels)):
            lab_norm = lab.strip().lower()
            if lab_norm not in {"yes", "no"}:
                raise ValueError(
                    f"VLSP Duration qid={qid} opt={opt_idx} bad label: {lab!r}"
                )
            samples.append(
                Sample(
                    sample_id=f"vlsp_duration-{qid}-{opt_idx}",
                    task="duration",
                    language="vi",
                    dataset="vlsp_duration",
                    context=context,
                    question=question,
                    gold=lab_norm,
                    meta={
                        "candidate_answer": opt,
                        "qid": qid,
                        "option_index": opt_idx,
                    },
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples
