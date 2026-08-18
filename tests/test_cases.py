"""Tests for the evaluation-set loader.

The point of these is that a typo in your own articles file produces a message
naming the file and the row, not a KeyError three frames deep.
"""

import json

import pytest

from news_classifier.categories import Category
from news_classifier.hard_cases import CASE_SET, CaseSetError, load_cases

ARTICLE = (
    "The Senate voted 58-42 on Thursday to advance the immigration bill, "
    "sending it to the House after three weeks of floor debate."
)


def write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return p


def test_bundled_set_is_flagged_synthetic():
    """The shipped articles were written for this project; results must say so."""
    assert CASE_SET.synthetic is True
    assert len(CASE_SET) == 12


def test_loads_json_list(tmp_path):
    path = write(tmp_path, "c.json", [{"text": ARTICLE, "expected": "Politics"}])
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases.cases[0].expected is Category.POLITICS


def test_loads_json_object_with_metadata(tmp_path):
    path = write(tmp_path, "c.json", {
        "name": "My real articles",
        "synthetic": False,
        "cases": [{"text": ARTICLE, "expected": "Politics", "source": "Reuters"}],
    })
    cases = load_cases(path)
    assert cases.name == "My real articles"
    assert cases.synthetic is False
    assert cases.cases[0].source == "Reuters"


def test_your_own_file_is_assumed_real(tmp_path):
    """Only the bundled default defaults to synthetic."""
    path = write(tmp_path, "mine.json", [{"text": ARTICLE, "expected": "Politics"}])
    assert load_cases(path).synthetic is False


def test_loads_csv(tmp_path):
    path = write(
        tmp_path, "c.csv",
        f'text,expected,hard_because\n"{ARTICLE}",Politics,a floor vote\n',
    )
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases.cases[0].hard_because == "a floor vote"


def test_label_casing_is_forgiving(tmp_path):
    path = write(tmp_path, "c.json", [{"text": ARTICLE, "expected": "sports"}])
    assert load_cases(path).cases[0].expected is Category.SPORTS


def test_whitespace_in_articles_is_normalised(tmp_path):
    path = write(tmp_path, "c.json", [
        {"text": "The  Senate\n\n  voted 58-42 on Thursday to advance the bill today.",
         "expected": "Politics"}
    ])
    assert "  " not in load_cases(path).cases[0].text


@pytest.mark.parametrize(
    ("case", "fragment"),
    [
        ({"text": ARTICLE, "expected": "Weather"}, "must be one of"),
        ({"text": "too short", "expected": "Politics"}, "shorter than"),
        ({"expected": "Politics"}, "missing"),
    ],
)
def test_bad_rows_name_the_problem(tmp_path, case, fragment):
    path = write(tmp_path, "c.json", [case])
    with pytest.raises(CaseSetError, match=fragment):
        load_cases(path)


def test_error_names_the_offending_row(tmp_path):
    path = write(tmp_path, "c.json", [
        {"text": ARTICLE, "expected": "Politics"},
        {"text": ARTICLE, "expected": "Nonsense"},
    ])
    with pytest.raises(CaseSetError, match=r"case 1"):
        load_cases(path)


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(CaseSetError, match="No evaluation set at"):
        load_cases(tmp_path / "nope.json")


def test_malformed_json_is_reported_clearly(tmp_path):
    path = write(tmp_path, "c.json", "{not json")
    with pytest.raises(CaseSetError, match="invalid JSON"):
        load_cases(path)


def test_empty_set_is_rejected(tmp_path):
    with pytest.raises(CaseSetError, match="non-empty"):
        load_cases(write(tmp_path, "c.json", []))


def test_env_var_selects_the_set(tmp_path, monkeypatch):
    path = write(tmp_path, "env.json", [{"text": ARTICLE, "expected": "Business"}])
    monkeypatch.setenv("NEWS_CLASSIFIER_CASES", str(path))
    cases = load_cases()
    assert len(cases) == 1
    assert cases.cases[0].expected is Category.BUSINESS


def test_broken_user_file_falls_back_without_crashing(tmp_path, monkeypatch):
    """A typo must not stop the app importing -- but must not pass silently."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"text": ARTICLE, "expected": "Weather"}]), encoding="utf-8")
    monkeypatch.setenv("NEWS_CLASSIFIER_CASES", str(bad))

    import importlib

    from news_classifier import hard_cases

    reloaded = importlib.reload(hard_cases)
    try:
        assert reloaded.CASE_SET_ERROR is not None
        assert "must be one of" in reloaded.CASE_SET_ERROR
        # Fell back to the bundled set rather than leaving the app with nothing.
        assert reloaded.CASE_SET.synthetic is True
        assert len(reloaded.CASE_SET) == 12
    finally:
        monkeypatch.delenv("NEWS_CLASSIFIER_CASES")
        importlib.reload(hard_cases)


def test_healthy_default_records_no_error():
    from news_classifier.hard_cases import CASE_SET_ERROR

    assert CASE_SET_ERROR is None
