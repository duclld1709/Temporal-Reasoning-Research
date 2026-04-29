"""Supervised fine-tuning (LoRA) cho Qwen3-8B trên VLSP DateArith VI.

Lưu ý:
- LoRA bf16 (không quantize) — cần ~24GB VRAM (Colab Pro A100 ổn).
- Loss chỉ tính trên assistant tokens (dùng ``DataCollatorForCompletionOnlyLM``).
- Sample render qua chính prompt template inference → distribution khớp eval.
- Adapter chỉ save (không merge vào base) để tiết kiệm disk; eval load qua
  ``QwenConfig(adapter_path=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.registry import load_dataset
from ..data.schema import Sample
from ..utils.seed import set_seed
from .data import (
    resolve_assistant_response_template,
    samples_to_chat_dataset,
    split_train_val,
)


@dataclass
class SFTRunConfig:
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True

    # Data
    dataset: str = "vlsp_date"
    dataset_path: str | None = None
    train_pool_start: int = 1500   # rows 0..1499 = eval Phase 1, KHÔNG được train
    train_pool_size: int = 1500    # rows 1500..2999
    val_ratio: float = 0.1
    shuffle_seed: int = 42

    # Output
    output_dir: str = "checkpoints/lora_vlsp_date_bf16"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Training hyperparams
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_seq_length: int = 256
    bf16: bool = True

    # Logging / checkpointing
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "none"

    seed: int = 42


def _load_train_pool(cfg: SFTRunConfig) -> list[Sample]:
    """Load đủ rows để có thể slice train pool, rồi cắt rows [start:start+size]."""
    needed = cfg.train_pool_start + cfg.train_pool_size
    kwargs: dict[str, Any] = {"max_samples": needed}
    if cfg.dataset_path:
        kwargs["path"] = cfg.dataset_path
    samples = load_dataset(cfg.dataset, **kwargs)
    pool = samples[cfg.train_pool_start : cfg.train_pool_start + cfg.train_pool_size]
    if not pool:
        raise RuntimeError(
            f"Train pool rỗng: dataset={cfg.dataset!r} chỉ có {len(samples)} rows, "
            f"không thể slice [{cfg.train_pool_start}:{cfg.train_pool_start + cfg.train_pool_size}]"
        )
    return pool


def train_sft(cfg: SFTRunConfig) -> str:
    """Chạy SFT, save adapter vào ``cfg.output_dir``, trả về đường dẫn đó."""
    set_seed(cfg.seed)

    # Lazy imports — chỉ cần khi train.
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import (  # type: ignore
        DataCollatorForCompletionOnlyLM,
        SFTConfig,
        SFTTrainer,
    )

    # 1. Data
    pool = _load_train_pool(cfg)
    train_samples, val_samples = split_train_val(
        pool, val_ratio=cfg.val_ratio, seed=cfg.shuffle_seed
    )
    print(f"[sft] pool={len(pool)} -> train={len(train_samples)} val={len(val_samples)}")

    # 2. Tokenizer + base model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sft] loading base model {cfg.model_name} dtype={cfg.dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
    )
    model.config.use_cache = False  # cần thiết khi train với gradient checkpointing

    # 3. LoRA config
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Build chat-format datasets
    train_ds = samples_to_chat_dataset(train_samples, tokenizer)
    val_ds = samples_to_chat_dataset(val_samples, tokenizer)

    # 5. Response-only loss masking
    response_template = resolve_assistant_response_template(tokenizer)
    print(f"[sft] response_template (loss-mask boundary) = {response_template!r}")
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 6. SFTConfig (extends TrainingArguments)
    sft_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=cfg.report_to,
        seed=cfg.seed,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 8. Train + save adapter
    trainer.train()
    out = Path(cfg.output_dir)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[sft] adapter saved to {out.resolve()}")
    return str(out)
