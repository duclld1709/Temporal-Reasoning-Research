from src.data.schema import Sample
from src.prompts.shot_pools import get_shots
from src.prompts.templates import build_messages


def _make_sample(task, lang, candidate=None, question="q?", context=""):
    return Sample(
        sample_id="x", task=task, language=lang, dataset="manual",
        context=context, question=question,
        gold="yes" if task == "duration" else ("05/01/2021" if lang == "en" else "Tháng 4, 1321"),
        meta={"candidate_answer": candidate} if candidate else {},
    )


def test_build_messages_zero_shot_duration_en():
    s = _make_sample("duration", "en", candidate="5 minutes")
    msgs = build_messages(s, shots=())
    roles = [m.role for m in msgs]
    assert roles == ["system", "user"]
    assert "yes" in msgs[0].content.lower()
    assert "5 minutes" in msgs[1].content


def test_build_messages_few_shot_duration_vi():
    s = _make_sample("duration", "vi", candidate="5 phút", question="Mất bao lâu?")
    shots = get_shots("duration", "vi", k=2)
    msgs = build_messages(s, shots=shots)
    # system + 2*(user,assistant) + user = 6
    assert len(msgs) == 2 + 2 * 2
    assert msgs[-1].role == "user"
    assert msgs[1].role == "user" and msgs[2].role == "assistant"


def test_build_messages_date_en():
    s = _make_sample("date_arith", "en", question="What is 5 days after 04/01/2020 in MM/DD/YYYY?")
    msgs = build_messages(s, shots=get_shots("date_arith", "en", k=1))
    assert "MM/DD/YYYY" in msgs[0].content
    assert msgs[-1].role == "user"


def test_build_messages_date_vi():
    s = _make_sample("date_arith", "vi", question="Thời gian 1 năm sau tháng 1, 2020?")
    msgs = build_messages(s, shots=get_shots("date_arith", "vi", k=1))
    assert "Tháng" in msgs[0].content
