"""Entry-point runner: load config YAML, chạy method trên dataset, lưu output.

Usage:
    python -m src.runner --config configs/<name>.yaml

Output:
    <output_dir>/<method>/<dataset>/predictions.jsonl
    <output_dir>/<method>/<dataset>/metrics.json
    append 1 row vào <output_dir>/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .data.pools import collect_qids, load_retrieval_pool
from .data.registry import load_dataset
from .evaluation.evaluate import build_record, score_records
from .evaluation.metrics import avg_inference_time
from .methods.registry import build_method
from .models.qwen import QwenChatLM, QwenConfig
from .prompts.shot_pools import get_shots
from .utils.io import write_json, write_jsonl
from .utils.seed import set_seed
from .utils.timing import timer


@dataclass
class RunConfig:
    experiment_name: str
    method: str
    dataset: str
    seed: int = 42
    model_name: str = "Qwen/Qwen3-8B"
    dtype: str = "bfloat16"
    enable_thinking: bool = False
    k_shot: int = 0
    max_samples: int | None = ...  # ...= dùng default theo dataset
    dataset_path: str | None = None
    output_dir: str = "outputs"
    progress_every: int = 50
    adapter_path: str | None = None  # PEFT/LoRA adapter dir (optional)
    # Verbose controls
    verbose: bool = False            # bật log per-sample
    verbose_first_n: int = 5         # full log cho N sample đầu (raw + extracted + gold + correct)
    verbose_every: int = 0           # >0: log rút gọn mỗi N sample
    running_score_every: int = 100   # in F1/accuracy tạm thời mỗi N sample
    # Dynamic few-shot retrieval (chỉ dùng khi method == "dynamic_few_shot")
    encoder_name: str = "intfloat/multilingual-e5-base"
    text_scope: str = "context_question"     # context_question | question
    pool_exclude_first_n: int = 1500
    retrieval_cache_dir: str | None = "outputs/retrieval_cache"
    balance_by_label: bool = False
    unique_by_qid: bool = False


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw.get("max_samples", "default") == "default":
        raw["max_samples"] = ...
    return RunConfig(**raw)


def _summary_row(cfg: RunConfig, metrics: dict, avg_time: float, n: int) -> dict:
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "experiment": cfg.experiment_name,
        "method": cfg.method,
        "dataset": cfg.dataset,
        "k_shot": cfg.k_shot,
        "enable_thinking": cfg.enable_thinking,
        "seed": cfg.seed,
        "model": cfg.model_name,
        "adapter_path": cfg.adapter_path or "",
        "num_samples": n,
        "avg_inference_sec": round(avg_time, 4),
    }
    # Luôn xuất đủ 4 cột để summary.csv giữ schema cố định bất kể task
    # (precision/recall = "" với accuracy task) — nếu không, header dựa trên row
    # đầu tiên sẽ thiếu cột khi rows sau có metric khác.
    row["metric"] = ""
    row["score"] = ""
    row["precision"] = ""
    row["recall"] = ""
    if "f1" in metrics:
        row["metric"] = "f1"
        row["score"] = round(metrics["f1"], 4)
        row["precision"] = round(metrics["precision"], 4)
        row["recall"] = round(metrics["recall"], 4)
    elif "accuracy" in metrics:
        row["metric"] = "accuracy"
        row["score"] = round(metrics["accuracy"], 4)
    row["parse_fail"] = metrics.get("parse_fail", 0)
    return row


def _append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header_needed:
            writer.writeheader()
        writer.writerow(row)


def _running_score(records: list[dict], task: str) -> str:
    """Running F1 (duration) hoặc accuracy (date_arith) tạm thời."""
    if not records:
        return "n/a"
    if task == "duration":
        from .evaluation.metrics import binary_f1_yes
        m = binary_f1_yes(
            [r["gold_normalized"] for r in records],
            [r["extracted"] for r in records],
        )
        return (
            f"f1={m['f1']:.3f} p={m['precision']:.3f} r={m['recall']:.3f} "
            f"parse_fail={m['parse_fail']}"
        )
    from .evaluation.metrics import accuracy
    m = accuracy(
        [r["gold_normalized"] for r in records],
        [r["extracted"] for r in records],
    )
    return f"acc={m['accuracy']:.3f} correct={m['correct']}/{m['support']} parse_fail={m['parse_fail']}"


def _log_sample(idx: int, total: int, rec: dict, full: bool) -> None:
    mark = "✓" if rec["correct"] else "✗"
    if full:
        raw_short = rec["raw_output"].replace("\n", "\\n")
        if len(raw_short) > 200:
            raw_short = raw_short[:200] + "…"
        print(
            f"  [{idx+1}/{total}] {mark} "
            f"gold={rec['gold_normalized']!r} "
            f"extracted={rec['extracted']!r} "
            f"elapsed={rec['elapsed_sec']:.2f}s"
        )
        print(f"      Q: {rec['question'][:160]}")
        print(f"      raw: {raw_short}")
    else:
        print(
            f"  [{idx+1}/{total}] {mark} "
            f"gold={rec['gold_normalized']!r} extracted={rec['extracted']!r}"
        )


def _build_dynamic_selector(cfg: RunConfig, eval_samples: list[dict]):
    """Build retrieval pool + FAISS index + DynamicShotSelector.

    Lazy import để các run không dùng dynamic_few_shot không phải cài
    sentence-transformers / faiss.
    """
    from .retrieval.encoder import E5Encoder
    from .retrieval.index import build_or_load_index, resolve_text_fn
    from .retrieval.selector import DynamicShotSelector

    dataset = cfg.dataset
    text_fn = resolve_text_fn(cfg.text_scope)

    eval_qids = collect_qids(eval_samples) if cfg.unique_by_qid else None
    pool = load_retrieval_pool(
        dataset_name=dataset,
        exclude_first_n=cfg.pool_exclude_first_n,
        eval_qids=eval_qids,
    )
    if not pool:
        raise RuntimeError(
            f"Retrieval pool cho {dataset!r} rỗng (exclude_first_n="
            f"{cfg.pool_exclude_first_n}). Kiểm tra kích thước dataset gốc."
        )
    print(f"[runner] retrieval pool size={len(pool)} "
          f"(skip first {cfg.pool_exclude_first_n}, exclude_eval_qids={eval_qids is not None})")

    encoder = E5Encoder(model_name=cfg.encoder_name)
    built = build_or_load_index(
        pool=pool,
        encoder=encoder,
        text_fn=text_fn,
        dataset_name=dataset,
        exclude_first_n=cfg.pool_exclude_first_n,
        text_scope=cfg.text_scope,
        cache_dir=cfg.retrieval_cache_dir,
    )
    selector = DynamicShotSelector(
        built=built,
        encoder=encoder,
        text_fn=text_fn,
        k=cfg.k_shot,
        balance_by_label=cfg.balance_by_label,
        unique_by_qid=cfg.unique_by_qid,
    )
    print(f"[runner] dynamic few-shot k={cfg.k_shot} "
          f"balance_by_label={cfg.balance_by_label} unique_by_qid={cfg.unique_by_qid}")
    return selector


def run(
    cfg: RunConfig,
    *,
    model: QwenChatLM | None = None,
) -> dict:
    """Chạy 1 experiment. Truyền model đã load để tránh reload giữa các cell."""
    set_seed(cfg.seed)
    print(f"[runner] experiment={cfg.experiment_name} method={cfg.method} dataset={cfg.dataset} "
          f"verbose={cfg.verbose}")

    # Load dataset
    kwargs: dict[str, Any] = {}
    if cfg.dataset_path:
        kwargs["path"] = cfg.dataset_path
    if cfg.max_samples is not ...:
        kwargs["max_samples"] = cfg.max_samples
    samples = load_dataset(cfg.dataset, **kwargs)
    print(f"[runner] loaded {len(samples)} samples "
          f"(task={samples[0]['task']}, lang={samples[0]['language']})")

    # Build / reuse model
    if model is None:
        model = QwenChatLM(QwenConfig(
            model_name=cfg.model_name,
            dtype=cfg.dtype,
            adapter_path=cfg.adapter_path,
        ))
        model.load()
    else:
        print(f"[runner] reusing pre-loaded model: {model.config.model_name}"
              + (f" (+adapter={model.config.adapter_path})" if model.config.adapter_path else ""))

    # Build method
    method_kwargs: dict[str, Any] = {"enable_thinking": cfg.enable_thinking}
    dynamic_selector = None
    if cfg.method == "few_shot":
        task = samples[0]["task"]
        language = samples[0]["language"]
        method_kwargs["shots"] = get_shots(task, language, cfg.k_shot)
        print(f"[runner] few-shot k={cfg.k_shot} for ({task},{language})")
    elif cfg.method == "dynamic_few_shot":
        dynamic_selector = _build_dynamic_selector(cfg, samples)
        method_kwargs["shot_selector"] = dynamic_selector
    method = build_method(cfg.method, model, **method_kwargs)

    task = samples[0]["task"]
    records: list[dict] = []
    times: list[float] = []
    n = len(samples)
    for i, sample in enumerate(samples):
        with timer() as t:
            raw = method.predict(sample)
        elapsed = t["elapsed"]
        times.append(elapsed)
        extra: dict | None = None
        if dynamic_selector is not None:
            extra = {
                "retrieved_shot_ids": list(dynamic_selector.last_retrieved_ids),
                "retrieved_scores": list(dynamic_selector.last_scores),
            }
        rec = build_record(sample, raw, elapsed, extra=extra)
        records.append(rec)

        if cfg.verbose:
            in_first_n = i < cfg.verbose_first_n
            periodic = cfg.verbose_every > 0 and ((i + 1) % cfg.verbose_every == 0)
            if in_first_n or periodic:
                _log_sample(i, n, rec, full=in_first_n)

        if cfg.running_score_every > 0 and (i + 1) % cfg.running_score_every == 0:
            print(
                f"  [{i+1}/{n}] running: {_running_score(records, task)} "
                f"avg_time={sum(times)/len(times):.3f}s"
            )
        elif (not cfg.verbose) and (i + 1) % cfg.progress_every == 0:
            print(f"  [{i+1}/{n}] avg={sum(times)/len(times):.3f}s")

    task = samples[0]["task"]
    language = samples[0]["language"]
    metrics = score_records(records, task, language)
    avg_t = avg_inference_time(times)

    out_dir = Path(cfg.output_dir) / cfg.method / cfg.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", records)
    metrics_payload = {
        "experiment": cfg.experiment_name,
        "method": cfg.method,
        "dataset": cfg.dataset,
        "task": task,
        "language": language,
        "k_shot": cfg.k_shot,
        "enable_thinking": cfg.enable_thinking,
        "seed": cfg.seed,
        "model": cfg.model_name,
        "num_samples": len(samples),
        "avg_inference_sec": avg_t,
        "metrics": metrics,
        "config": cfg.__dict__ if cfg.max_samples is not ... else {**cfg.__dict__, "max_samples": "default"},
    }
    write_json(out_dir / "metrics.json", metrics_payload)
    _append_summary(
        Path(cfg.output_dir) / "summary.csv",
        _summary_row(cfg, metrics, avg_t, len(samples)),
    )
    print(f"[runner] done. metrics: {json.dumps(metrics, ensure_ascii=False)} "
          f"avg_time={avg_t:.3f}s -> {out_dir}")
    return metrics_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verbose", action="store_true", help="override cfg.verbose=True")
    ap.add_argument("--verbose-first-n", type=int, default=None)
    ap.add_argument("--verbose-every", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.verbose:
        cfg.verbose = True
    if args.verbose_first_n is not None:
        cfg.verbose_first_n = args.verbose_first_n
    if args.verbose_every is not None:
        cfg.verbose_every = args.verbose_every
    run(cfg)


if __name__ == "__main__":
    main()
