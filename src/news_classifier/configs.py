"""The prompt configurations the evaluation compares.

Two independent switches -- the category guide in the system prompt, and the
few-shot examples -- give a 2x2. Comparing only `full` against `zero_shot`
answers "do examples help a prompt that already explains the boundaries?"; the
`minimal` vs `examples_only` pair answers "do examples teach the boundaries on
their own?". The second question is the one the first eval could not reach.
"""

from typing import NamedTuple


class PromptConfig(NamedTuple):
    key: str
    label: str
    use_guide: bool
    use_few_shot: bool
    description: str
    # "zero-shot" or "few-shot" -- the axis the assignment is about. Shown as a
    # tag in the UI so which rows form a comparable pair is never ambiguous.
    shot_type: str


CONFIGS: list[PromptConfig] = [
    PromptConfig(
        "zero_shot", "Zero-shot", True, False,
        "The full system prompt, with no worked examples. The baseline.",
        "zero-shot",
    ),
    PromptConfig(
        "full", "Few-shot", True, True,
        "The same prompt plus 10 worked examples. What this project ships.",
        "few-shot",
    ),
    PromptConfig(
        "minimal", "Zero-shot, bare prompt", False, False,
        "Category names only -- no boundary rules, no examples.",
        "zero-shot",
    ),
    PromptConfig(
        "examples_only", "Few-shot, bare prompt", False, True,
        "No boundary rules; the examples alone have to teach them.",
        "few-shot",
    ),
]

BY_KEY = {c.key: c for c in CONFIGS}

# The comparison the assignment asks for: identical prompt, examples off vs on.
# Everything else is held constant, so any difference is the examples.
HEADLINE_PAIR = ("zero_shot", "full")

# The same question asked of a bare prompt, where the category guide is not
# already doing the examples' job. Kept for diagnosis, not for the headline.
BARE_PAIR = ("minimal", "examples_only")

PAIRS = [
    (HEADLINE_PAIR, "with the full system prompt"),
    (BARE_PAIR, "with a bare prompt"),
]
