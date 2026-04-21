"""Script preprocess: đọc raw datasets, chuẩn hoá về schema Sample, dump JSONL.

Chạy từ project root:
    python -m src.data.preprocess

Output: Dataset/Preprocessed/<dataset_name>.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.io import write_jsonl
from .registry import DATASET_LOADERS, DEFAULT_MAX_SAMPLES, DEFAULT_PATHS, load_dataset

PREPROCESSED_DIR = Path("Dataset/Preprocessed")


def preprocess_all(output_dir: Path = PREPROCESSED_DIR) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in DATASET_LOADERS:
        samples = load_dataset(name)
        out_path = output_dir / f"{name}.jsonl"
        write_jsonl(out_path, samples)
        counts[name] = len(samples)
        print(
            f"[{name}] {len(samples)} samples -> {out_path} "
            f"(max_samples={DEFAULT_MAX_SAMPLES[name]}, src={DEFAULT_PATHS[name]})"
        )
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(PREPROCESSED_DIR))
    args = ap.parse_args()
    preprocess_all(Path(args.output_dir))


if __name__ == "__main__":
    main()
