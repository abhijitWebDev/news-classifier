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


def _bad_request(message: str):
    """Build a real BadRequestError -- the SDK needs a live response object.

    The OpenAI SDK moved to httpx2; import it via the SDK so this keeps working
    if that changes again.
    """
    import openai
    from openai._base_client import httpx2 as httpx

    response = httpx.Response(
        400, request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    return openai.BadRequestError(message, response=response, body=None)


UNSUPPORTED_REASONING = (
    "Error code: 400 - {'error': {'message': \"Unsupported parameter: "
    "'reasoning.effort' is not supported with this model.\"}}"
)


class _FakeResponses:
    """Stands in for client.responses, recording what each call was sent."""

    def __init__(self, reject_reasoning: bool, error: Exception | None = None):
        self.reject_reasoning = reject_reasoning
        self.error = error
        self.calls: list[bool] = []   # True when 'reasoning' was included

    def parse(self, **kwargs):
        sent_reasoning = "reasoning" in kwargs
        self.calls.append(sent_reasoning)
        if sent_reasoning and self.reject_reasoning:
            raise self.error or _bad_request(UNSUPPORTED_REASONING)
        return type("R", (), {"output_parsed": _RESULT, "status": "completed"})()


class _FakeClient:
    def __init__(self, responses):
        self.responses = responses


_RESULT = Classification(
    rationale="r", category=Category.SPORTS, confidence=0.9
)


def _classifier(responses):
    from news_classifier.classifier import NewsClassifier

    return NewsClassifier(client=_FakeClient(responses))


def test_reasoning_is_dropped_for_models_that_reject_it():
    """gpt-4o-mini and friends 400 on 'reasoning'; the retry must recover."""
    fake = _FakeResponses(reject_reasoning=True)
    classifier = _classifier(fake)

    assert classifier.classify("Liverpool beat Chelsea 3-0 at Anfield.") is _RESULT
    # Tried with reasoning, then retried without.
    assert fake.calls == [True, False]
    assert classifier._supports_reasoning is False


def test_unsupported_reasoning_is_only_probed_once():
    """The retry must not repeat on every call -- that would double the cost."""
    fake = _FakeResponses(reject_reasoning=True)
    classifier = _classifier(fake)

    for _ in range(3):
        classifier.classify("Liverpool beat Chelsea 3-0 at Anfield.")

    # One probe pair, then three plain calls -- not three probe pairs.
    assert fake.calls == [True, False, False, False]


def test_reasoning_is_kept_for_models_that_accept_it():
    fake = _FakeResponses(reject_reasoning=False)
    classifier = _classifier(fake)

    classifier.classify("Liverpool beat Chelsea 3-0 at Anfield.")
    classifier.classify("Liverpool beat Chelsea 3-0 at Anfield.")

    assert fake.calls == [True, True]
    assert classifier._supports_reasoning is True


def test_other_bad_requests_are_not_swallowed():
    """Only the reasoning-unsupported 400 triggers the retry."""
    import openai

    unrelated = _bad_request("Error code: 400 - context_length_exceeded")
    fake = _FakeResponses(reject_reasoning=True, error=unrelated)
    classifier = _classifier(fake)

    with pytest.raises(openai.BadRequestError, match="context_length_exceeded"):
        classifier.classify("Liverpool beat Chelsea 3-0 at Anfield.")
    assert fake.calls == [True]
