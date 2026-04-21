"""Retrieval utilities cho dynamic few-shot.

- encoder.E5Encoder: wrapper multilingual-e5 (query / passage prefix).
- index: FAISS IndexFlatIP + cache helpers.
- selector.DynamicShotSelector: ShotSelector retrieve top-k theo cosine.
"""
