"""Classify short news articles into Sports, Politics, Business or Technology."""

from __future__ import annotations

import os

import openai

from .categories import CATEGORY_GUIDE, Classification
from .examples import FEW_SHOT_EXAMPLES

# Terra balances quality against cost for this task. Luna is roughly 10x
# cheaper if you are running volume; Sol is the flagship if the boundary cases
# matter more than the bill. Older non-reasoning models such as gpt-4o-mini
# work too -- the reasoning parameter is dropped automatically for them. Use
# `news-classifier-eval --compare` to decide rather than guessing.
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Reasoning tokens count against max_output_tokens, so a classification-sized
# budget of ~256 would truncate the response before the JSON was emitted.
MAX_OUTPUT_TOKENS = 2048

# Effort for models that accept it. Non-reasoning models (gpt-4o-mini and the
# rest of the 4o family) reject the parameter outright, so it is sent
# optimistically and dropped on the specific 400 -- see NewsClassifier.classify.
REASONING_EFFORT = "low"

_RULES = """\
Rules:
- Choose the single category the article is *about*, not every topic it mentions.
  Decide by the framing of the lede, not by keywords appearing later.
- Every article gets one of the four categories. There is no "Other" -- pick the
  closest fit and lower the confidence instead.
- Confidence reflects how cleanly the article sits inside one category: use 0.95+
  for unambiguous articles, 0.7-0.9 when it straddles a boundary, below 0.7 when
  the article could reasonably be filed either way.
- Keep the rationale to one sentence naming the decisive signal.
"""

_HEADER = (
    "You classify short news articles into exactly one of four categories: "
    "Sports, Politics, Business, Technology.\n"
)


def build_system_prompt(use_guide: bool = True) -> str:
    """Assemble the system prompt.

    `use_guide` exists for ablation. CATEGORY_GUIDE states each category
    boundary in both directions -- it is the part of the prompt that most
    directly competes with the few-shot examples, so turning it off is how you
    find out what the examples are actually contributing.
    """
    parts = [_HEADER]
    if use_guide:
        parts.append("\n" + CATEGORY_GUIDE)
    parts.append("\n" + _RULES)
    return "".join(parts)


# The default prompt, kept as a module constant for callers and tests.
SYSTEM_PROMPT = build_system_prompt()


def _is_unsupported_reasoning(exc: openai.BadRequestError) -> bool:
    """True when the 400 is specifically 'this model has no reasoning param'.

    Matched on the message because the API gives no machine-readable code for
    it. Kept narrow: any other bad request must still surface as an error.
    """
    message = (getattr(exc, "message", None) or str(exc)).lower()
    return "reasoning" in message and (
        "unsupported parameter" in message or "not supported" in message
    )


def describe_api_error(exc: openai.APIError) -> str:
    """Turn an SDK exception into a line that says what to actually do.

    The common failures here are not bugs -- they are an unset key, an empty
    account, or a rate limit -- and each has a different fix.
    """
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
    if isinstance(exc, openai.RateLimitError):
        # OpenAI returns 429 for both throttling and an exhausted quota.
        message = getattr(exc, "message", str(exc))
        if "quota" in message.lower() or "billing" in message.lower():
            return (
                "The OpenAI account is out of quota. Add credit at "
                "https://platform.openai.com/settings/organization/billing"
            )
        return "Rate limited by the OpenAI API. Wait a moment and retry."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach the OpenAI API. Check your network."
    if isinstance(exc, openai.NotFoundError):
        return f"Model {MODEL!r} is not available to this account."
    return f"OpenAI API error: {getattr(exc, 'message', exc)}"


def _few_shot_turns() -> list[dict[str, str]]:
    """Replay the examples as alternating user/assistant turns.

    Demonstrated turns beat described rules for this task: the model sees the
    exact input format, the exact JSON shape, and how the boundary cases were
    resolved. Keeping the system prompt and these turns byte-identical on every
    call also lets OpenAI's automatic prompt caching serve the prefix -- there
    is no cache_control to set, but a prefix that varies gets no discount.
    """
    turns: list[dict[str, str]] = []
    for article, result in FEW_SHOT_EXAMPLES:
        turns.append({"role": "user", "content": article})
        turns.append({"role": "assistant", "content": result.model_dump_json()})
    return turns


class NewsClassifier:
    """Few-shot news category classifier backed by the OpenAI API.

    Build one instance and reuse it -- the system prompt and few-shot examples
    are identical on every call, so they are served from the prompt cache after
    the first request.
    """

    def __init__(
        self,
        client: openai.OpenAI | None = None,
        model: str = MODEL,
        use_few_shot: bool = True,
        use_guide: bool = True,
    ) -> None:
        if client is None and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "No OpenAI credentials found. Set OPENAI_API_KEY in your "
                "environment or in a .env file (see .env.example)."
            )
        self.client = client or openai.OpenAI()
        self.model = model
        # `use_few_shot=False` exists so the evaluation script can measure what
        # the examples actually buy us. Production callers should leave it on.
        self.use_few_shot = use_few_shot
        self.use_guide = use_guide
        # None = not yet known. Set to False the first time the model rejects
        # the reasoning parameter, so the retry happens once per instance
        # rather than on every call.
        self._supports_reasoning: bool | None = None
        self._system = build_system_prompt(use_guide)
        self._prefix = _few_shot_turns() if use_few_shot else []

    def classify(self, article: str) -> Classification:
        """Classify one article. Raises ValueError on empty input."""
        article = article.strip()
        if not article:
            raise ValueError("Article text is empty.")

        request = {
            "model": self.model,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "input": [
                {"role": "system", "content": self._system},
                *self._prefix,
                {"role": "user", "content": article},
            ],
            "text_format": Classification,
        }

        # Low effort keeps a one-label decision cheap and fast; the few-shot
        # examples, not deliberation, are what make it consistent. Models with
        # no reasoning mode (gpt-4o-mini and friends) reject the parameter
        # outright, so send it, and on that one specific 400 drop it and
        # remember. Beats a hardcoded list of reasoning models, which goes
        # stale every release.
        if self._supports_reasoning is False:
            response = self.client.responses.parse(**request)
        else:
            try:
                response = self.client.responses.parse(
                    **request, reasoning={"effort": REASONING_EFFORT}
                )
                self._supports_reasoning = True
            except openai.BadRequestError as exc:
                if not _is_unsupported_reasoning(exc):
                    raise
                self._supports_reasoning = False
                response = self.client.responses.parse(**request)

        parsed = response.output_parsed
        if parsed is None:
            # Happens when the model hit the token ceiling or refused, in which
            # case output_parsed is None rather than an exception being raised.
            raise RuntimeError(
                f"Model returned no parsed result (status={response.status!r})."
            )
        return parsed

    def classify_many(self, articles: list[str]) -> list[Classification]:
        """Classify several articles sequentially, reusing the cached prefix."""
        return [self.classify(a) for a in articles]
