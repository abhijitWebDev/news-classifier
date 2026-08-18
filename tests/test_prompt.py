"""Offline tests -- these never call the API.

They guard the parts that silently rot: the few-shot examples staying valid and
balanced, and the prompt prefix keeping the exact shape the API expects.
"""

from collections import Counter

import pytest

from news_classifier import Category, Classification, FEW_SHOT_EXAMPLES
from news_classifier.classifier import SYSTEM_PROMPT, _few_shot_turns


def test_every_category_is_demonstrated():
    """A category with no example is the one the model gets wrong."""
    shown = {result.category for _, result in FEW_SHOT_EXAMPLES}
    assert shown == set(Category)


def test_examples_are_not_lopsided():
    """Skewed example counts bias the model toward the over-represented label."""
    counts = Counter(result.category for _, result in FEW_SHOT_EXAMPLES)
    assert max(counts.values()) - min(counts.values()) <= 2, counts


def test_few_shot_turns_alternate_user_assistant():
    turns = _few_shot_turns()
    assert len(turns) == 2 * len(FEW_SHOT_EXAMPLES)
    for i, turn in enumerate(turns):
        assert turn["role"] == ("user" if i % 2 == 0 else "assistant")


def test_prefix_is_byte_stable_across_calls():
    """OpenAI caches automatically, but only on an identical prefix -- any
    per-call variation in the system prompt or examples silently loses it."""
    assert _few_shot_turns() == _few_shot_turns()


def test_assistant_turns_parse_back_into_the_schema():
    """The demonstrated output must be exactly what the model is asked to emit."""
    for turn in _few_shot_turns()[1::2]:
        Classification.model_validate_json(turn["content"])


def test_system_prompt_names_all_four_categories():
    for category in Category:
        assert category.value in SYSTEM_PROMPT


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_article_is_rejected_before_any_api_call(bad, monkeypatch):
    from news_classifier.classifier import NewsClassifier

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    classifier = NewsClassifier()
    with pytest.raises(ValueError):
        classifier.classify(bad)


def test_missing_credentials_fail_fast(monkeypatch):
    from news_classifier.classifier import NewsClassifier

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No OpenAI credentials"):
        NewsClassifier()


def test_guide_ablation_removes_the_boundary_rules():
    """The ablation must actually change the prompt, or the comparison is fake."""
    from news_classifier.classifier import build_system_prompt

    with_guide, without = build_system_prompt(True), build_system_prompt(False)
    assert len(without) < len(with_guide)
    assert "Rules:" in without          # the task instructions stay
    assert "is Business, not Technology" not in without   # the guide goes


def test_hard_cases_are_balanced_and_annotated():
    from collections import Counter

    from news_classifier.hard_cases import HARD_CASES

    assert len(HARD_CASES) >= 10
    counts = Counter(c.expected for c in HARD_CASES)
    assert set(counts) == set(Category)
    assert max(counts.values()) - min(counts.values()) <= 2, counts
    for case in HARD_CASES:
        assert case.hard_because.strip(), "every case must record its trap"


def test_hard_cases_are_not_reused_from_the_few_shot_examples():
    """Reusing an example would measure recall, not generalisation."""
    from news_classifier.hard_cases import HARD_CASES

    shown = {article.strip() for article, _ in FEW_SHOT_EXAMPLES}
    for case in HARD_CASES:
        assert case.text.strip() not in shown
