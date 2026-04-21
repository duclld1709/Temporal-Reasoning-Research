"""Pool loader cho dynamic few-shot retrieval.

Lấy các sample NẰM SAU `exclude_first_n` (mặc định 1500) của dataset gốc
để dùng làm pool retrieve shots. Không trùng với eval set.

Với `vlsp_duration`: ngoài việc skip first N rows (đã expand), có thể
loại thêm các qid trùng với eval set để đảm bảo không leak câu hỏi gốc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .registry import load_dataset
from .schema import Sample


def load_retrieval_pool(
    dataset_name: str,
    exclude_first_n: int = 1500,
    eval_qids: Iterable | None = None,
    path: str | Path | None = None,
) -> list[Sample]:
    kwargs = {"max_samples": None}
    if path is not None:
        kwargs["path"] = path
    full = load_dataset(dataset_name, **kwargs)
    if exclude_first_n > 0:
        pool = full[exclude_first_n:]
    else:
        pool = list(full)

    if eval_qids is not None:
        qid_set = set(eval_qids)
        pool = [s for s in pool if s.get("meta", {}).get("qid") not in qid_set]

    return pool


def collect_qids(samples: list[Sample]) -> set:
    return {s["meta"]["qid"] for s in samples if "qid" in s.get("meta", {})}
