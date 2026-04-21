from src.evaluation.metrics import accuracy, avg_inference_time, binary_f1_yes


def test_binary_f1_perfect():
    m = binary_f1_yes(["yes", "no", "yes"], ["yes", "no", "yes"])
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["parse_fail"] == 0


def test_binary_f1_parse_fail_counts_as_negative():
    m = binary_f1_yes(["yes", "yes"], [None, "yes"])
    # 1 TP, 1 FN (None treated as no)
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["parse_fail"] == 1
    assert 0 < m["f1"] < 1


def test_binary_f1_mix():
    m = binary_f1_yes(["yes", "no", "no", "yes"], ["yes", "yes", "no", "no"])
    # TP=1, FP=1, FN=1, TN=1
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_accuracy():
    m = accuracy(["a", "b", "c"], ["a", "x", "c"])
    assert m["accuracy"] == 2 / 3
    assert m["correct"] == 2
    assert m["parse_fail"] == 0

    m2 = accuracy(["a", "b"], [None, "b"])
    assert m2["parse_fail"] == 1
    assert m2["correct"] == 1


def test_avg_time():
    assert avg_inference_time([1.0, 2.0, 3.0]) == 2.0
    assert avg_inference_time([]) == 0.0
