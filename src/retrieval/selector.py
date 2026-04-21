"""DynamicShotSelector — implement ShotSelector interface bằng retrieval.

Dùng chung FewShotMethod: shot_selector(test_sample) -> list[Sample].

Cơ chế:
1. Encode test sample dưới mode 'query'.
2. FAISS search top N (over-fetch) để có dư cho filter.
3. (Optional) balance_by_label: chia top theo sample['gold'] lấy k/2 yes + k/2 no.
4. (Optional) unique_by_qid: dedup theo meta['qid'] (VLSP duration — tránh 4 row cùng 1 qid).
5. Trả k shot, sắp theo similarity tăng dần (shot giống nhất đứng gần câu query —
   recency bias của prompt few-shot).

Expose `last_retrieved_ids`, `last_scores` để runner ghi log.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..data.schema import Sample
from .encoder import E5Encoder
from .index import BuiltIndex


class DynamicShotSelector:
    def __init__(
        self,
        built: BuiltIndex,
        encoder: E5Encoder,
        text_fn: Callable[[Sample], str],
        k: int,
        balance_by_label: bool = False,
        unique_by_qid: bool = False,
        overfetch_mult: int = 5,
    ):
        self.built = built
        self.encoder = encoder
        self.text_fn = text_fn
        self.k = k
        self.balance_by_label = balance_by_label
        self.unique_by_qid = unique_by_qid
        self.overfetch_mult = overfetch_mult
        self.last_retrieved_ids: list[str] = []
        self.last_scores: list[float] = []

    def __call__(self, test_sample: Sample) -> Sequence[Sample]:
        pool = self.built.pool
        n_pool = len(pool)
        want = self.k
        fetch = min(n_pool, max(want * self.overfetch_mult, want))

        q_text = self.text_fn(test_sample)
        q_vec = self.encoder.encode([q_text], mode="query")  # (1, dim)
        scores, idxs = self.built.index.search(q_vec, fetch)  # (1, fetch)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        candidates: list[tuple[float, Sample]] = []
        for s, i in zip(scores, idxs):
            if i < 0 or i >= n_pool:
                continue
            candidates.append((float(s), pool[i]))

        if self.unique_by_qid:
            seen = set()
            deduped: list[tuple[float, Sample]] = []
            for score, samp in candidates:
                qid = samp.get("meta", {}).get("qid")
                if qid is None:
                    deduped.append((score, samp))
                    continue
                if qid in seen:
                    continue
                seen.add(qid)
                deduped.append((score, samp))
            candidates = deduped

        if self.balance_by_label:
            chosen = self._balance_pick(candidates, want)
        else:
            chosen = candidates[:want]

        # Sắp tăng dần theo similarity (shot giống nhất ở cuối, gần câu query thật).
        chosen.sort(key=lambda x: x[0])

        self.last_retrieved_ids = [s.get("sample_id", "") for _, s in chosen]
        self.last_scores = [round(sc, 4) for sc, _ in chosen]
        return [s for _, s in chosen]

    @staticmethod
    def _balance_pick(
        candidates: list[tuple[float, Sample]],
        k: int,
    ) -> list[tuple[float, Sample]]:
        yes_q = [c for c in candidates if c[1].get("gold") == "yes"]
        no_q = [c for c in candidates if c[1].get("gold") == "no"]
        half = k // 2
        extra = k - 2 * half  # 0 nếu k chẵn, 1 nếu k lẻ
        # Bên ít hơn sẽ nhận slot lẻ (để đa dạng nếu 1 bên cạn)
        target_yes = half
        target_no = half
        if extra:
            if len(yes_q) < len(no_q):
                target_yes += extra
            else:
                target_no += extra
        picked_yes = yes_q[:target_yes]
        picked_no = no_q[:target_no]
        picked = picked_yes + picked_no
        if len(picked) < k:
            used_ids = {id(s) for _, s in picked}
            for c in candidates:
                if id(c[1]) in used_ids:
                    continue
                picked.append(c)
                if len(picked) >= k:
                    break
        return picked[:k]
