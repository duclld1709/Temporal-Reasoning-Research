"""Sentence encoder wrapper — multilingual E5.

E5 convention: prepend "query: " hoặc "passage: " trước khi encode.
Vector trả về đã L2-normalize để cosine = dot product.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

Mode = Literal["query", "passage"]


class E5Encoder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str | None = None,
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _prefix(self, texts: Sequence[str], mode: Mode) -> list[str]:
        tag = "query: " if mode == "query" else "passage: "
        return [tag + t for t in texts]

    def encode(
        self,
        texts: Sequence[str],
        mode: Mode,
        show_progress: bool = False,
    ) -> np.ndarray:
        prefixed = self._prefix(texts, mode)
        vecs = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)
