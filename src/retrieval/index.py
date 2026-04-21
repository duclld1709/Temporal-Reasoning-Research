"""FAISS index builder + cache cho retrieval pool.

- build_index: encode pool dưới mode 'passage' rồi add vào IndexFlatIP.
- save / load: cache ra disk (tránh re-encode mỗi Colab session).
- cache key dựa trên (encoder_name, dataset, exclude_first_n, text_scope, pool_size).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..data.schema import Sample
from .encoder import E5Encoder


def default_text_fn(sample: Sample) -> str:
    ctx = sample.get("context") or ""
    q = sample.get("question") or ""
    return f"{ctx}\n{q}".strip()


def question_only_text_fn(sample: Sample) -> str:
    return sample.get("question") or ""


TEXT_FN_REGISTRY: dict[str, Callable[[Sample], str]] = {
    "context_question": default_text_fn,
    "question": question_only_text_fn,
}


def resolve_text_fn(scope: str) -> Callable[[Sample], str]:
    if scope not in TEXT_FN_REGISTRY:
        raise KeyError(f"Unknown text_scope {scope!r}. Available: {list(TEXT_FN_REGISTRY)}")
    return TEXT_FN_REGISTRY[scope]


@dataclass
class BuiltIndex:
    index: object  # faiss.Index
    pool: list[Sample]
    dim: int


def _cache_key(
    encoder_name: str,
    dataset_name: str,
    exclude_first_n: int,
    text_scope: str,
    pool_size: int,
) -> str:
    raw = f"{encoder_name}|{dataset_name}|{exclude_first_n}|{text_scope}|{pool_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _cache_paths(cache_dir: Path, dataset_name: str, key: str) -> tuple[Path, Path, Path]:
    stem = f"{dataset_name}-{key}"
    return (
        cache_dir / f"{stem}.faiss",
        cache_dir / f"{stem}.samples.jsonl",
        cache_dir / f"{stem}.meta.json",
    )


def build_index(
    pool: list[Sample],
    encoder: E5Encoder,
    text_fn: Callable[[Sample], str],
    show_progress: bool = True,
) -> BuiltIndex:
    import faiss

    texts = [text_fn(s) for s in pool]
    vecs = encoder.encode(texts, mode="passage", show_progress=show_progress)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return BuiltIndex(index=index, pool=list(pool), dim=vecs.shape[1])


def save_index(
    built: BuiltIndex,
    cache_dir: str | Path,
    dataset_name: str,
    encoder_name: str,
    exclude_first_n: int,
    text_scope: str,
) -> Path:
    import faiss

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(encoder_name, dataset_name, exclude_first_n, text_scope, len(built.pool))
    faiss_path, samples_path, meta_path = _cache_paths(cache_dir, dataset_name, key)

    faiss.write_index(built.index, str(faiss_path))
    with open(samples_path, "w", encoding="utf-8") as f:
        for s in built.pool:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    meta = {
        "encoder_name": encoder_name,
        "dataset_name": dataset_name,
        "exclude_first_n": exclude_first_n,
        "text_scope": text_scope,
        "pool_size": len(built.pool),
        "dim": built.dim,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return faiss_path


def load_index(
    cache_dir: str | Path,
    dataset_name: str,
    encoder_name: str,
    exclude_first_n: int,
    text_scope: str,
    expected_pool_size: int,
) -> BuiltIndex | None:
    import faiss

    cache_dir = Path(cache_dir)
    key = _cache_key(encoder_name, dataset_name, exclude_first_n, text_scope, expected_pool_size)
    faiss_path, samples_path, meta_path = _cache_paths(cache_dir, dataset_name, key)
    if not (faiss_path.exists() and samples_path.exists() and meta_path.exists()):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("pool_size") != expected_pool_size:
        return None

    index = faiss.read_index(str(faiss_path))
    pool: list[Sample] = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            pool.append(json.loads(line))
    return BuiltIndex(index=index, pool=pool, dim=meta.get("dim", index.d))


def build_or_load_index(
    pool: list[Sample],
    encoder: E5Encoder,
    text_fn: Callable[[Sample], str],
    dataset_name: str,
    exclude_first_n: int,
    text_scope: str,
    cache_dir: str | Path | None,
    show_progress: bool = True,
) -> BuiltIndex:
    if cache_dir is not None:
        cached = load_index(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            encoder_name=encoder.model_name,
            exclude_first_n=exclude_first_n,
            text_scope=text_scope,
            expected_pool_size=len(pool),
        )
        if cached is not None:
            print(f"[retrieval] loaded cached index: {dataset_name} pool_size={len(cached.pool)}")
            return cached

    print(f"[retrieval] building index: {dataset_name} pool_size={len(pool)} encoder={encoder.model_name}")
    built = build_index(pool, encoder, text_fn, show_progress=show_progress)
    if cache_dir is not None:
        path = save_index(
            built,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            encoder_name=encoder.model_name,
            exclude_first_n=exclude_first_n,
            text_scope=text_scope,
        )
        print(f"[retrieval] cached index -> {path}")
    return built
