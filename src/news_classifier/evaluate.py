"""Measure classification consistency, with and without the few-shot examples.

Consistency here means two things, measured on a held-out set of deliberately
ambiguous articles:

  agreement -- classify the same article N times; what fraction of runs land on
               the modal label? 1.0 means the classifier never wavers.
  accuracy  -- does the modal label match the label we expect?

Run `news-classifier-eval --compare` to see both configurations side by side.
"""

from __future__ import annotations

import argparse
from collections import Counter

import openai
from dotenv import load_dotenv

from .categories import Category
from .classifier import NewsClassifier, describe_api_error
from .configs import BY_KEY, CONFIGS
from .hard_cases import ENV_VAR
from .hard_cases import CASE_SET, CASE_SET_ERROR, HARD_CASES

# Held out from examples.py on purpose -- these are new boundary cases, so the
# few-shot examples have to generalise rather than be recalled.
EVAL_SET: list[tuple[str, Category]] = [
    (
        "Tesla delivered 495,000 vehicles last quarter, missing Wall Street "
        "estimates of 512,000. The company said price cuts had failed to offset "
        "softening demand in Europe, and its shares fell 6% premarket.",
        Category.BUSINESS,
    ),
    (
        "The Justice Department sued to block the merger of the two largest "
        "grocery chains, arguing in a filing on Monday that the deal would raise "
        "prices for millions of households.",
        Category.POLITICS,
    ),
    (
        "The International Olympic Committee will allow athletes to compete "
        "under a neutral flag at the next Games, reversing a ban imposed two "
        "years ago. Fourteen athletes have already qualified.",
        Category.SPORTS,
    ),
    (
        "A team at CERN has recorded the first direct evidence of a rare kaon "
        "decay predicted by the Standard Model, using a detector upgrade that "
        "improved timing resolution to 30 picoseconds.",
        Category.TECHNOLOGY,
    ),
    (
        "A ransomware group encrypted patient records at 12 hospitals across the "
        "region, forcing staff onto paper charts. Investigators traced the "
        "intrusion to an unpatched VPN appliance.",
        Category.TECHNOLOGY,
    ),
    (
        "The city council approved $700 million in public financing for a new "
        "stadium after a six-hour hearing, with four members voting against and "
        "residents' groups promising a ballot challenge.",
        Category.POLITICS,
    ),
]


def _evaluate(
    classifier: NewsClassifier,
    runs: int,
    cases: list[tuple[str, Category]] | None = None,
) -> tuple[float, float, list[str]]:
    """Return (mean agreement, accuracy, per-article report lines)."""
    cases = cases if cases is not None else EVAL_SET
    agreements: list[float] = []
    correct = 0
    lines: list[str] = []

    for article, expected in cases:
        labels = [classifier.classify(article).category for _ in range(runs)]
        counts = Counter(labels)
        modal, modal_count = counts.most_common(1)[0]
        agreement = modal_count / runs
        agreements.append(agreement)
        hit = modal is expected
        correct += hit

        spread = ", ".join(f"{c.value}x{n}" for c, n in counts.most_common())
        lines.append(
            f"  [{'ok ' if hit else 'MISS'}] agreement {agreement:.0%}  "
            f"expected {expected.value:<10} got {spread}\n"
            f"         {article.strip()[:70]}..."
        )

    return sum(agreements) / len(agreements), correct / len(cases), lines


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="news-classifier-eval",
        description="Measure how consistently the classifier labels ambiguous articles.",
    )
    parser.add_argument(
        "-n", "--runs", type=int, default=3, help="Runs per article (default: 3)."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also evaluate without few-shot examples, to show what they buy.",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Use the hard eval set (lede/body conflicts, loaded vocabulary).",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Run all four prompt configurations (implies --hard).",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.ablate:
        args.hard = True
        runs = [(c.label, c.key, c.use_guide, c.use_few_shot) for c in CONFIGS]
    elif args.compare:
        runs = [("with few-shot", "full", True, True), ("zero-shot", "zero_shot", True, False)]
    else:
        runs = [("with few-shot", "full", True, True)]

    cases = (
        [(c.text, c.expected) for c in HARD_CASES] if args.hard else EVAL_SET
    )
    print(f"{len(cases)} articles x {args.runs} runs")
    if args.hard:
        if CASE_SET_ERROR:
            print(f"error: could not load your evaluation set — {CASE_SET_ERROR}")
            print("Fix the file, or unset $" + ENV_VAR + " to use the bundled set.")
            return 1
        print(f"  set: {CASE_SET.name} ({CASE_SET.path})")
        if CASE_SET.synthetic:
            print("  NOTE: synthetic articles -- written for this project, not real "
                  "reporting.\n        Results are weaker evidence than a real set. "
                  f"Set ${ENV_VAR} to use your own.")

    scores: dict[str, tuple[float, float]] = {}
    for label, key, use_guide, use_few_shot in runs:
        try:
            classifier = NewsClassifier(use_guide=use_guide, use_few_shot=use_few_shot)
        except RuntimeError as exc:
            print(f"error: {exc}")
            return 1
        try:
            agreement, accuracy, lines = _evaluate(classifier, args.runs, cases)
        except openai.APIError as exc:
            print(f"error: {describe_api_error(exc)}")
            return 1
        scores[key] = (agreement, accuracy)
        print(f"\n=== {label} ===")
        print("\n".join(lines))
        print(f"  mean agreement {agreement:.0%}   accuracy {accuracy:.0%}")

    _print_deltas(scores)
    return 0


def _print_deltas(scores: dict[str, tuple[float, float]]) -> None:
    """State the few-shot effect as a number, including when it is zero."""
    pairs = [
        ("minimal", "examples_only", "with no boundary rules in the prompt"),
        ("zero_shot", "full", "on top of the full category guide"),
    ]
    lines = [
        f"  {where}: agreement {(scores[b][0] - scores[a][0]) * 100:+.0f} pts, "
        f"accuracy {(scores[b][1] - scores[a][1]) * 100:+.0f} pts"
        for a, b, where in pairs
        if a in scores and b in scores
    ]
    if lines:
        print("\n=== few-shot effect ===")
        print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
