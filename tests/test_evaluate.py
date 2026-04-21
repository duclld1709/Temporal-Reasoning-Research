from src.data.schema import Sample
from src.evaluation.evaluate import build_record, score_records


def test_build_record_duration():
    s = Sample(
        sample_id="id1", task="duration", language="en", dataset="udst_duration",
        context="c", question="q", gold="yes",
        meta={"candidate_answer": "5 minutes"},
    )
    rec = build_record(s, raw_output="Yes.", elapsed_sec=0.1)
    assert rec["extracted"] == "yes"
    assert rec["gold_normalized"] == "yes"
    assert rec["correct"] is True
    assert rec["elapsed_sec"] == 0.1


def test_build_record_date_en_thinking():
    s = Sample(
        sample_id="id2", task="date_arith", language="en", dataset="bigbench_date",
        context="", question="q", gold="05/01/2021", meta={},
    )
    rec = build_record(s, raw_output="<think>calc...</think>05/01/2021", elapsed_sec=0.2)
    assert rec["extracted"] == "05/01/2021"
    assert rec["correct"] is True


def test_score_records_duration_f1():
    recs = [
        {"gold_normalized": "yes", "extracted": "yes"},
        {"gold_normalized": "no", "extracted": "yes"},
        {"gold_normalized": "yes", "extracted": "no"},
        {"gold_normalized": "no", "extracted": "no"},
    ]
    m = score_records(recs, "duration", "en")
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert m["f1"] == 0.5


def test_score_records_date_accuracy():
    recs = [
        {"gold_normalized": "Tháng 4, 1321", "extracted": "Tháng 4, 1321"},
        {"gold_normalized": "Tháng 2, 1078", "extracted": "Tháng 2, 1077"},
        {"gold_normalized": "Tháng 1, 2020", "extracted": None},
    ]
    m = score_records(recs, "date_arith", "vi")
    assert m["correct"] == 1
    assert m["support"] == 3
    assert m["parse_fail"] == 1
