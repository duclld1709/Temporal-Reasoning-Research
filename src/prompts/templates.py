"""Prompt templates cho (task × ngôn ngữ).

Mỗi entry gồm:
- system: chuỗi ràng buộc format.
- render(sample) -> user message.
- few_shot_block(example_samples) -> đoạn text chèn trước câu hỏi thật.

Few-shot dùng messages format: mỗi shot là 1 cặp user/assistant bơm trước user cuối.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..data.schema import Sample
from ..models.base import ChatMessage


@dataclass
class PromptTemplate:
    system: str
    render_user: Callable[[Sample], str]
    render_shot_user: Callable[[Sample], str]
    render_shot_assistant: Callable[[Sample], str]


# ------------------------- Duration EN (UDST) -------------------------

_SYS_DURATION_EN = (
    "You judge whether a candidate duration is plausible for the event in the context. "
    "Respond with only the single word 'yes' or 'no'. "
    "No reasoning, no punctuation, no extra tokens."
)


def _user_duration_en(s: Sample) -> str:
    cand = s["meta"].get("candidate_answer", "")
    return (
        f"Context: {s['context']}\n"
        f"Question: {s['question']}\n"
        f"Candidate duration: {cand}\n"
        f"Is this a plausible duration? Answer 'yes' or 'no'."
    )


def _shot_assistant_yes_no(s: Sample) -> str:
    return s["gold"]


DURATION_EN = PromptTemplate(
    system=_SYS_DURATION_EN,
    render_user=_user_duration_en,
    render_shot_user=_user_duration_en,
    render_shot_assistant=_shot_assistant_yes_no,
)


# ------------------------- Duration VI (VLSP) -------------------------

_SYS_DURATION_VI = (
    "Bạn đánh giá một khoảng thời gian có hợp lý cho sự kiện trong ngữ cảnh hay không. "
    "Chỉ trả lời duy nhất 'yes' hoặc 'no'. "
    "Không giải thích, không thêm bất kỳ ký tự nào khác."
)


def _user_duration_vi(s: Sample) -> str:
    cand = s["meta"].get("candidate_answer", "")
    return (
        f"Ngữ cảnh: {s['context']}\n"
        f"Câu hỏi: {s['question']}\n"
        f"Khoảng thời gian đề xuất: {cand}\n"
        f"Khoảng thời gian này có hợp lý không? Trả lời 'yes' hoặc 'no'."
    )


DURATION_VI = PromptTemplate(
    system=_SYS_DURATION_VI,
    render_user=_user_duration_vi,
    render_shot_user=_user_duration_vi,
    render_shot_assistant=_shot_assistant_yes_no,
)


# ------------------------- DateArith EN (BigBench) -------------------------

_SYS_DATE_EN = (
    "You are a date arithmetic solver. "
    "Respond with ONLY the target date in MM/DD/YYYY format. "
    "No reasoning, no words, no punctuation other than the slashes."
)


def _user_date_en(s: Sample) -> str:
    return s["question"]


def _shot_assistant_date_en(s: Sample) -> str:
    return s["gold"]


DATE_EN = PromptTemplate(
    system=_SYS_DATE_EN,
    render_user=_user_date_en,
    render_shot_user=_user_date_en,
    render_shot_assistant=_shot_assistant_date_en,
)


# ------------------------- DateArith VI (VLSP) -------------------------

_SYS_DATE_VI = (
    "Bạn là bộ giải các phép tính thời gian. "
    "Chỉ trả lời duy nhất theo đúng mẫu 'Tháng M, YYYY' (ví dụ: 'Tháng 4, 1321'). "
    "Không giải thích, không thêm ký tự hay văn bản nào khác."
)


def _user_date_vi(s: Sample) -> str:
    ctx = s.get("context") or ""
    if ctx:
        return f"Ngữ cảnh: {ctx}\nCâu hỏi: {s['question']}"
    return s["question"]


def _shot_assistant_date_vi(s: Sample) -> str:
    return s["gold"]


DATE_VI = PromptTemplate(
    system=_SYS_DATE_VI,
    render_user=_user_date_vi,
    render_shot_user=_user_date_vi,
    render_shot_assistant=_shot_assistant_date_vi,
)


# ------------------------- Registry -------------------------

TEMPLATES: dict[tuple[str, str], PromptTemplate] = {
    ("duration", "en"): DURATION_EN,
    ("duration", "vi"): DURATION_VI,
    ("date_arith", "en"): DATE_EN,
    ("date_arith", "vi"): DATE_VI,
}


def get_template(task: str, language: str) -> PromptTemplate:
    key = (task, language)
    if key not in TEMPLATES:
        raise KeyError(f"No prompt template for {key}")
    return TEMPLATES[key]


def build_messages(
    sample: Sample,
    shots: Sequence[Sample] = (),
) -> list[ChatMessage]:
    tmpl = get_template(sample["task"], sample["language"])
    msgs: list[ChatMessage] = [ChatMessage(role="system", content=tmpl.system)]
    for shot in shots:
        msgs.append(ChatMessage(role="user", content=tmpl.render_shot_user(shot)))
        msgs.append(
            ChatMessage(role="assistant", content=tmpl.render_shot_assistant(shot))
        )
    msgs.append(ChatMessage(role="user", content=tmpl.render_user(sample)))
    return msgs
