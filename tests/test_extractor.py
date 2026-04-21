from src.evaluation.extractor import (
    extract_mmddyyyy,
    extract_vi_month_year,
    extract_yes_no,
    normalize_mmddyyyy,
    normalize_vi_month_year,
    strip_thinking,
)


def test_strip_thinking():
    assert strip_thinking("<think>abc</think>yes") == "yes"
    assert strip_thinking("<THINK>x</THINK>\nno") == "no"
    assert strip_thinking("plain") == "plain"


def test_yes_no_basic():
    assert extract_yes_no("yes") == "yes"
    assert extract_yes_no("No.") == "no"
    assert extract_yes_no(" Yes, it is plausible.") == "yes"
    assert extract_yes_no("Không hợp lý") == "no"
    assert extract_yes_no("Có, điều này hợp lý") == "yes"
    assert extract_yes_no("") is None
    assert extract_yes_no("hmm") is None


def test_yes_no_thinking():
    assert extract_yes_no("<think>Consider...</think>yes") == "yes"
    assert extract_yes_no("<think>long</think>\n\nno") == "no"


def test_mmddyyyy():
    assert extract_mmddyyyy("05/01/2021") == "05/01/2021"
    assert extract_mmddyyyy("The answer is 5/1/2021.") == "05/01/2021"
    assert extract_mmddyyyy("<think>calc</think>12/31/1999") == "12/31/1999"
    assert extract_mmddyyyy("no date here") is None
    assert normalize_mmddyyyy("5/1/2021") == "05/01/2021"


def test_vi_month_year():
    assert extract_vi_month_year("Tháng 4, 1321") == "Tháng 4, 1321"
    assert extract_vi_month_year("tháng 12, 1999") == "Tháng 12, 1999"
    assert extract_vi_month_year("Kết quả: Tháng 2, 1078.") == "Tháng 2, 1078"
    assert extract_vi_month_year("không có") is None
    assert normalize_vi_month_year("Tháng 04, 1321") == "Tháng 4, 1321"
