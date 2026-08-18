"""Command line interface for the news category classifier."""

from __future__ import annotations

import argparse
import json
import sys

import openai
from dotenv import load_dotenv

from .classifier import describe_api_error
from .categories import Classification
from .classifier import NewsClassifier

DEMO_ARTICLES = [
    "India chased down 287 with four balls to spare in Kolkata, Shubman Gill "
    "anchoring the innings with 112 not out to seal the series 2-1.",
    "Voters in the state go to the polls on Tuesday in a special election that "
    "could decide which party controls the chamber for the rest of the term.",
    "Apple said supply constraints on its new headset would ease by the summer, "
    "as it reported services revenue of $24 billion for the quarter.",
    "Engineers have built a photonic chip that performs matrix multiplication "
    "with light instead of electrons, cutting energy use per operation by an "
    "order of magnitude in early benchmarks.",
]


def _format(article: str, result: Classification, as_json: bool) -> str:
    if as_json:
        return json.dumps({"article": article, **result.model_dump(mode="json")})
    return (
        f"{result.category.value:<12} (confidence {result.confidence:.2f})\n"
        f"  why: {result.rationale}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="news-classifier",
        description="Classify a short news article as Sports, Politics, Business or Technology.",
    )
    parser.add_argument(
        "article",
        nargs="?",
        help="Article text. Omit to read from stdin, or use --demo.",
    )
    parser.add_argument("-f", "--file", help="Read the article from a file instead.")
    parser.add_argument(
        "--demo", action="store_true", help="Classify four built-in sample articles."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON Lines.")
    args = parser.parse_args()

    load_dotenv()

    if args.demo:
        articles = DEMO_ARTICLES
    elif args.file:
        with open(args.file, encoding="utf-8") as fh:
            articles = [fh.read()]
    elif args.article:
        articles = [args.article]
    elif not sys.stdin.isatty():
        articles = [sys.stdin.read()]
    else:
        parser.error("No article given. Pass text, --file, --demo, or pipe to stdin.")

    try:
        classifier = NewsClassifier()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for article in articles:
        try:
            result = classifier.classify(article)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except openai.APIError as exc:
            print(f"error: {describe_api_error(exc)}", file=sys.stderr)
            return 1
        if not args.json:
            print(f"\n{article.strip()[:100]}...")
        print(_format(article, result, args.json))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
