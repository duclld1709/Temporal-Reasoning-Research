"""Registry thống nhất cho 4 dataset của Phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .bigbench_date import load_bigbench_date
from .schema import Sample
from .udst_duration import load_udst_duration
from .vlsp_date import load_vlsp_date
from .vlsp_duration import load_vlsp_duration

DATASET_LOADERS: dict[str, Callable[..., list[Sample]]] = {
    "udst_duration": load_udst_duration,
    "bigbench_date": load_bigbench_date,
    "vlsp_date": load_vlsp_date,
    "vlsp_duration": load_vlsp_duration,
}

# Default paths (relative to project root) — có thể override qua config.
DEFAULT_PATHS: dict[str, str] = {
    "udst_duration": "Dataset/Raw/English/UDST-DurationQA/data/test.tsv",
    "bigbench_date": "Dataset/Raw/English/BigBench_DateUnderstanding/task.json",
    "vlsp_date": (
        "Dataset/Raw/Vietnamese/VLSP 2025 ViTempQA (DateArith + DurationQA) Task/"
        "TrainingDataset/date_train_dataset/date_training_dataset.txt"
    ),
    "vlsp_duration": (
        "Dataset/Raw/Vietnamese/VLSP 2025 ViTempQA (DateArith + DurationQA) Task/"
        "TrainingDataset/durationQA_train_dataset/duration_training_dataset.txt"
    ),
}

# Default max_samples cho Phase 1.
DEFAULT_MAX_SAMPLES: dict[str, int | None] = {
    "udst_duration": 1500,
    "bigbench_date": None,  # dùng toàn bộ 369
    "vlsp_date": 1500,
    "vlsp_duration": 1500,  # 1500 rows SAU expand
}


def load_dataset(
    name: str,
    path: str | Path | None = None,
    max_samples: int | None = ...,  # sentinel để phân biệt None (dùng hết) vs chưa truyền
) -> list[Sample]:
    if name not in DATASET_LOADERS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list(DATASET_LOADERS)}")
    loader = DATASET_LOADERS[name]
    if path is None:
        path = DEFAULT_PATHS[name]
    if max_samples is ...:
        max_samples = DEFAULT_MAX_SAMPLES[name]
    return loader(path, max_samples=max_samples)
