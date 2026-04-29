"""SFT (LoRA) training module cho Phase 2.

Public API:
- ``SFTRunConfig``: dataclass cấu hình train (model, LoRA, hyperparams).
- ``train_sft``: chạy SFT, save adapter, trả về đường dẫn adapter.
- ``samples_to_chat_dataset``, ``split_train_val``: helper xử lý data.
"""

from .data import samples_to_chat_dataset, split_train_val
from .sft import SFTRunConfig, train_sft

__all__ = [
    "SFTRunConfig",
    "train_sft",
    "samples_to_chat_dataset",
    "split_train_val",
]
