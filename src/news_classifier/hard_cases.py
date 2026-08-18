"""The evaluation set: loaded from data, not hardcoded.

The shipped default (`data/hard_cases.json`) is SYNTHETIC -- twelve articles
written for this project to stress the category boundaries. That is stated in
the file, carried on `CaseSet.synthetic`, and surfaced in the UI, because a
result measured on articles written by the same hand as the few-shot examples
is weaker evidence than one measured on real reporting.

To use your own articles, point `NEWS_CLASSIFIER_CASES` at a .json or .csv
file:

    NEWS_CLASSIFIER_CASES=my_articles.csv uv run news-classifier-eval --hard

CSV needs a header row with at least `text` and `expected`; `hard_because` and
`source` are optional. JSON is either a bare list of case objects or an object
with a `cases` key plus optional `name`/`synthetic`/`description`. A set loaded
from your own file defaults to `synthetic: false` -- say so explicitly in the
JSON if it is not.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .categories import Category

DEFAULT_PATH = Path(__file__).parent / "data" / "hard_cases.json"
ENV_VAR = "NEWS_CLASSIFIER_CASES"

# Enough text to be classifiable; below this it is a headline, not an article.
MIN_TEXT_CHARS = 40


class CaseSetError(ValueError):
    """Raised with a message naming the file and row, so a typo is findable."""


@dataclass(frozen=True)
class Case:
    text: str
    expected: Category
    hard_because: str = ""
    source: str | None = None


@dataclass(frozen=True)
class CaseSet:
    cases: list[Case]
    name: str = "Evaluation set"
    # True only when the articles were written for this project. Real articles
    # you supply are assumed real unless the JSON says otherwise.
    synthetic: bool = False
    description: str = ""
    path: Path | None = field(default=None, compare=False)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)


def _parse_case(raw: dict, where: str) -> Case:
    text = str(raw.get("text", "")).strip()
    if len(text) < MIN_TEXT_CHARS:
        raise CaseSetError(
            f"{where}: 'text' is missing or shorter than {MIN_TEXT_CHARS} characters."
        )

    label = str(raw.get("expected", "")).strip()
    try:
        # Accept any casing: "sports", "SPORTS" and "Sports" all work.
        expected = Category(label.title())
    except ValueError:
        valid = ", ".join(c.value for c in Category)
        raise CaseSetError(
            f"{where}: 'expected' is {label!r}; must be one of {valid}."
        ) from None

    return Case(
        text=" ".join(text.split()),
        expected=expected,
        hard_because=str(raw.get("hard_because", "") or "").strip(),
        source=(str(raw["source"]).strip() or None) if raw.get("source") else None,
    )


def load_cases(path: str | Path | None = None) -> CaseSet:
    """Load an evaluation set from JSON or CSV.

    Resolution order: explicit `path`, then $NEWS_CLASSIFIER_CASES, then the
    bundled synthetic default.
    """
    explicit = path is not None or bool(os.environ.get(ENV_VAR))
    resolved = Path(path or os.environ.get(ENV_VAR) or DEFAULT_PATH)

    if not resolved.is_file():
        raise CaseSetError(f"No evaluation set at {resolved}.")

    if resolved.suffix.lower() == ".csv":
        with resolved.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise CaseSetError(f"{resolved}: no rows (is the header row present?).")
        cases = [_parse_case(r, f"{resolved} row {i}") for i, r in enumerate(rows, 2)]
        meta: dict = {}
    else:
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CaseSetError(f"{resolved}: invalid JSON — {exc}") from None
        meta = payload if isinstance(payload, dict) else {}
        raw_cases = meta.get("cases") if isinstance(payload, dict) else payload
        if not isinstance(raw_cases, list) or not raw_cases:
            raise CaseSetError(f"{resolved}: expected a non-empty list of cases.")
        cases = [
            _parse_case(c, f"{resolved} case {i}") for i, c in enumerate(raw_cases)
        ]

    return CaseSet(
        cases=cases,
        name=str(meta.get("name") or resolved.stem.replace("_", " ").title()),
        # Your own file is treated as real unless it declares otherwise.
        synthetic=bool(meta.get("synthetic", False if explicit else True)),
        description=str(meta.get("description") or ""),
        path=resolved,
    )


def _load_at_import() -> tuple[CaseSet, str | None]:
    """Load the configured set, surviving a broken one.

    A typo in a user-supplied file must not stop the server from starting --
    but it must not silently substitute the synthetic set either, or you would
    believe you measured on real articles when you did not. So: fall back to
    the bundled default AND keep the error, which the CLI, the API and the UI
    all report.
    """
    try:
        return load_cases(), None
    except CaseSetError as exc:
        try:
            return load_cases(DEFAULT_PATH), str(exc)
        except CaseSetError:  # pragma: no cover - the bundled file is tested
            return CaseSet(cases=[], name="unavailable"), str(exc)


CASE_SET, CASE_SET_ERROR = _load_at_import()

# Kept as the module-level name the rest of the code already imports.
HARD_CASES: list[Case] = CASE_SET.cases
