"""Integration tests — chạy loader thật trên raw files (không cần GPU).

Skip nếu file không tồn tại (ví dụ CI chưa có dataset)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.data.bigbench_date import load_bigbench_date
from src.data.registry import DEFAULT_PATHS
from src.data.udst_duration import load_udst_duration
from src.data.vlsp_date import load_vlsp_date
from src.data.vlsp_duration import load_vlsp_duration

ROOT = Path(__file__).resolve().parents[1]


def _p(name: str) -> Path:
    return ROOT / DEFAULT_PATHS[name]


@pytest.mark.skipif(not _p("udst_duration").exists(), reason="UDST raw missing")
def test_udst_loader_1500():
    samples = load_udst_duration(_p("udst_duration"), max_samples=1500)
    assert len(samples) == 1500
    s = samples[0]
    assert s["task"] == "duration" and s["language"] == "en"
    assert s["gold"] in {"yes", "no"}
    assert s["meta"]["candidate_answer"]


@pytest.mark.skipif(not _p("bigbench_date").exists(), reason="BigBench raw missing")
def test_bigbench_loader_full():
    samples = load_bigbench_date(_p("bigbench_date"), max_samples=None)
    assert len(samples) == 369
    s = samples[0]
    assert s["task"] == "date_arith" and s["language"] == "en"
    # gold phải dạng MM/DD/YYYY (có thể 1 chữ số)
    assert "/" in s["gold"] and len(s["gold"].split("/")) == 3


@pytest.mark.skipif(not _p("vlsp_date").exists(), reason="VLSP Date raw missing")
def test_vlsp_date_loader_1500():
    samples = load_vlsp_date(_p("vlsp_date"), max_samples=1500)
    assert len(samples) == 1500
    s = samples[0]
    assert s["task"] == "date_arith" and s["language"] == "vi"
    assert "Tháng" in s["gold"]


@pytest.mark.skipif(not _p("vlsp_duration").exists(), reason="VLSP Duration raw missing")
def test_vlsp_duration_loader_1500_expanded():
    samples = load_vlsp_duration(_p("vlsp_duration"), max_samples=1500)
    assert len(samples) == 1500  # 1500 rows sau expand
    s = samples[0]
    assert s["task"] == "duration" and s["language"] == "vi"
    assert s["gold"] in {"yes", "no"}
    assert s["meta"]["candidate_answer"]
    # qid 375 samples đầu sẽ xuất hiện 4 lần mỗi qid
    qids = [x["meta"]["qid"] for x in samples]
    assert qids[0] == qids[1] == qids[2] == qids[3]
