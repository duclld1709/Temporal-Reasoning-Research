"""Qwen3-8B wrapper dùng HuggingFace transformers.

Qwen3 hỗ trợ 2 chế độ qua `apply_chat_template(..., enable_thinking=...)`:
- non-thinking (default eval): phản hồi ngắn, đúng format.
- thinking: phát sinh `<think>...</think>` trước đáp án; extractor sẽ strip.

Model được load 1 lần, có thể reuse qua nhiều method/dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ChatMessage

DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"


@dataclass
class QwenConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dtype: str = "bfloat16"  # "float16" | "bfloat16" | "float32"
    device_map: str | None = "auto"
    trust_remote_code: bool = True
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    adapter_path: str | None = None  # PEFT/LoRA adapter directory (optional)


class QwenChatLM:
    def __init__(self, config: QwenConfig | None = None):
        self.config = config or QwenConfig()
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        import torch  # lazy import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            **self.config.load_kwargs,
        )
        if self.config.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                self._model, self.config.adapter_path
            )
        self._model.eval()

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        if self._model is None or self._tokenizer is None:
            self.load()
        import torch

        chat = [{"role": m.role, "content": m.content} for m in messages]
        prompt_text = self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text
