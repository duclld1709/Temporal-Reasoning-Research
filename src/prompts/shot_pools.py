"""Manual few-shot pools cho Phase 1.

Mỗi pool là danh sách Sample (schema chuẩn) dùng làm shot cho FewShotMethod.
Tác dụng: giữ shot cố định, có thể audit dễ và reproducible.

Shot lấy cảm hứng từ phân phối train set nhưng viết tay để tránh leak test.
"""

from __future__ import annotations

from ..data.schema import Sample

DURATION_EN_SHOTS: list[Sample] = [
    Sample(
        sample_id="shot-dur-en-1",
        task="duration", language="en", dataset="manual",
        context="He boiled the kettle to make a cup of tea.",
        question="How long does it take to boil the kettle?",
        gold="yes",
        meta={"candidate_answer": "3 minutes"},
    ),
    Sample(
        sample_id="shot-dur-en-2",
        task="duration", language="en", dataset="manual",
        context="He boiled the kettle to make a cup of tea.",
        question="How long does it take to boil the kettle?",
        gold="no",
        meta={"candidate_answer": "5 years"},
    ),
    Sample(
        sample_id="shot-dur-en-3",
        task="duration", language="en", dataset="manual",
        context="She is writing her PhD dissertation this year.",
        question="How long does it take to write a PhD dissertation?",
        gold="yes",
        meta={"candidate_answer": "several months"},
    ),
    Sample(
        sample_id="shot-dur-en-4",
        task="duration", language="en", dataset="manual",
        context="She is writing her PhD dissertation this year.",
        question="How long does it take to write a PhD dissertation?",
        gold="no",
        meta={"candidate_answer": "10 seconds"},
    ),
]


DURATION_VI_SHOTS: list[Sample] = [
    Sample(
        sample_id="shot-dur-vi-1",
        task="duration", language="vi", dataset="manual",
        context="Cô ấy đang pha một tách cà phê cho buổi sáng.",
        question="Mất bao lâu để pha một tách cà phê?",
        gold="yes",
        meta={"candidate_answer": "5 phút"},
    ),
    Sample(
        sample_id="shot-dur-vi-2",
        task="duration", language="vi", dataset="manual",
        context="Cô ấy đang pha một tách cà phê cho buổi sáng.",
        question="Mất bao lâu để pha một tách cà phê?",
        gold="no",
        meta={"candidate_answer": "10 năm"},
    ),
    Sample(
        sample_id="shot-dur-vi-3",
        task="duration", language="vi", dataset="manual",
        context="Anh ấy đang xây một ngôi nhà ba tầng cho gia đình.",
        question="Mất bao lâu để xây một ngôi nhà ba tầng?",
        gold="yes",
        meta={"candidate_answer": "6 tháng"},
    ),
    Sample(
        sample_id="shot-dur-vi-4",
        task="duration", language="vi", dataset="manual",
        context="Anh ấy đang xây một ngôi nhà ba tầng cho gia đình.",
        question="Mất bao lâu để xây một ngôi nhà ba tầng?",
        gold="no",
        meta={"candidate_answer": "2 giây"},
    ),
]


DATE_EN_SHOTS: list[Sample] = [
    Sample(
        sample_id="shot-date-en-1",
        task="date_arith", language="en", dataset="manual",
        context="",
        question="Today is January 15, 2020. What is the date 10 days later in MM/DD/YYYY?",
        gold="01/25/2020",
        meta={},
    ),
    Sample(
        sample_id="shot-date-en-2",
        task="date_arith", language="en", dataset="manual",
        context="",
        question="Yesterday was March 3, 2005. What is the date one week ago in MM/DD/YYYY?",
        gold="02/25/2005",
        meta={},
    ),
    Sample(
        sample_id="shot-date-en-3",
        task="date_arith", language="en", dataset="manual",
        context="",
        question="Today is June 30, 1999. What is the date one month from now in MM/DD/YYYY?",
        gold="07/30/1999",
        meta={},
    ),
]


DATE_VI_SHOTS: list[Sample] = [
    Sample(
        sample_id="shot-date-vi-1",
        task="date_arith", language="vi", dataset="manual",
        context="",
        question="Hãy tính thời điểm 5 năm sau tháng 2, 1800",
        gold="Tháng 2, 1805",
        meta={},
    ),
    Sample(
        sample_id="shot-date-vi-2",
        task="date_arith", language="vi", dataset="manual",
        context="",
        question="Thời gian 3 tháng trước tháng 5, 1900 là khi nào?",
        gold="Tháng 2, 1900",
        meta={},
    ),
    Sample(
        sample_id="shot-date-vi-3",
        task="date_arith", language="vi", dataset="manual",
        context="",
        question="Giả sử bạn đang ở tháng 10, 1500, thời gian sau 1 năm 4 tháng, thì là thời điểm nào?",
        gold="Tháng 2, 1502",
        meta={},
    ),
]


SHOT_POOLS: dict[tuple[str, str], list[Sample]] = {
    ("duration", "en"): DURATION_EN_SHOTS,
    ("duration", "vi"): DURATION_VI_SHOTS,
    ("date_arith", "en"): DATE_EN_SHOTS,
    ("date_arith", "vi"): DATE_VI_SHOTS,
}


def get_shots(task: str, language: str, k: int) -> list[Sample]:
    pool = SHOT_POOLS[(task, language)]
    if k > len(pool):
        raise ValueError(
            f"Requested k={k} shots but pool for ({task},{language}) has only {len(pool)}"
        )
    return list(pool[:k])
