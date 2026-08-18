"""FastAPI service wrapping the classifier: a JSON API plus the web UI."""

from __future__ import annotations

import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .categories import Category, Classification
from .classifier import MODEL, NewsClassifier, describe_api_error
from .configs import BY_KEY, CONFIGS, HEADLINE_PAIR, PAIRS
from .hard_cases import CASE_SET, CASE_SET_ERROR, HARD_CASES
from .examples import FEW_SHOT_EXAMPLES

STATIC_DIR = Path(__file__).parent / "static"

# Serverless platforms cap how long a request may run (60s on Vercel by
# default). A sequential batch of Opus calls blows straight through that, so
# batches fan out across a small pool instead. Bounded to stay well inside
# OpenAI rate limits -- override with BATCH_CONCURRENCY.
BATCH_CONCURRENCY = int(os.environ.get("BATCH_CONCURRENCY", "5"))
MAX_BATCH = 25

SAMPLES: list[dict[str, str]] = [
    {
        "label": "Sports",
        "text": "India chased down 287 with four balls to spare in Kolkata, "
        "Shubman Gill anchoring the innings with 112 not out to seal the "
        "series 2-1. The hosts had lost three wickets inside the powerplay.",
    },
    {
        "label": "Politics",
        "text": "Voters in the state go to the polls on Tuesday in a special "
        "election that could decide which party controls the chamber for the "
        "rest of the term. Both campaigns spent a combined $40 million.",
    },
    {
        "label": "Business",
        "text": "Apple said supply constraints on its new headset would ease by "
        "the summer, as it reported services revenue of $24 billion for the "
        "quarter, slightly ahead of analyst expectations.",
    },
    {
        "label": "Technology",
        "text": "Engineers have built a photonic chip that performs matrix "
        "multiplication with light instead of electrons, cutting energy use per "
        "operation by an order of magnitude in early benchmarks.",
    },
    {
        "label": "Tricky",
        "text": "A consortium backed by a sovereign wealth fund agreed to buy a "
        "70% stake in the Premier League club for £1.2 billion, in a deal that "
        "values the club at a record multiple of its annual revenue.",
    },
]


class ClassifyRequest(BaseModel):
    article: str = Field(min_length=1, max_length=20_000)


class ClassifyResponse(BaseModel):
    category: Category
    confidence: float
    rationale: str
    elapsed_ms: int


class CompareRequest(BaseModel):
    """Run one pasted article through several prompt configurations."""

    article: str = Field(min_length=1, max_length=20_000)
    runs: int = Field(default=3, ge=1, le=10)
    config_keys: list[str] = Field(
        default_factory=lambda: list(HEADLINE_PAIR),
        min_length=1,
        max_length=len(CONFIGS),
    )


class FewShotEffect(BaseModel):
    """The measured difference the examples made, for one comparable pair."""

    baseline_key: str
    few_shot_key: str
    context: str
    agreement_delta: float
    confidence_delta: float
    changed_label: bool


class CompareOutcome(BaseModel):
    config_key: str
    label: str
    shot_type: str
    use_guide: bool
    use_few_shot: bool
    modal_category: Category
    agreement: float
    spread: dict[str, int]
    mean_confidence: float
    # Why it chose the label it reports.
    rationale: str
    # When the runs disagreed, the label and reasoning from a dissenting run.
    # Showing only the winner hides the fact that the model wavered and why.
    dissenting_category: Category | None = None
    dissenting_rationale: str | None = None


class CompareResponse(BaseModel):
    outcomes: list[CompareOutcome]
    effects: list[FewShotEffect]
    # True when every configuration settled on the same label. A False here is
    # what makes an article interesting: it means the prompt is deciding the
    # answer, not the article.
    consensus: bool
    categories_seen: list[str]
    elapsed_ms: int


class EvalRunRequest(BaseModel):
    """Evaluate one article across configurations.

    One article per request on purpose: the browser drives the loop, so progress
    appears live and no single request approaches a serverless timeout.
    """

    case_index: int = Field(ge=0)
    runs: int = Field(default=3, ge=1, le=10)
    config_keys: list[str] = Field(min_length=1, max_length=len(CONFIGS))


class ConfigOutcome(BaseModel):
    config_key: str
    modal_category: Category
    agreement: float
    correct: bool
    spread: dict[str, int]
    mean_confidence: float


class EvalRunResponse(BaseModel):
    case_index: int
    expected: Category
    outcomes: list[ConfigOutcome]


class BatchRequest(BaseModel):
    articles: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    results: list[ClassifyResponse]


_classifier: NewsClassifier | None = None
_startup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the classifier once at startup, reusing its cached prompt prefix.

    A missing API key is reported through /api/health and a 503 on classify
    rather than crashing the server -- the UI can then say something useful.
    """
    global _classifier, _startup_error
    load_dotenv()
    try:
        _classifier = NewsClassifier()
    except RuntimeError as exc:
        _startup_error = str(exc)
    yield


app = FastAPI(
    title="News Category Classifier",
    description="Classify short news articles as Sports, Politics, Business or Technology.",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_classifier() -> NewsClassifier:
    if _classifier is None:
        raise HTTPException(status_code=503, detail=_startup_error or "Classifier unavailable.")
    return _classifier


def _classify_one(classifier: NewsClassifier, article: str) -> ClassifyResponse:
    """Run one classification, mapping SDK failures onto sensible HTTP codes."""
    started = time.perf_counter()
    try:
        result: Classification = classifier.classify(article)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except openai.AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=describe_api_error(exc)) from exc
    except openai.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=describe_api_error(exc)) from exc
    except openai.APIConnectionError as exc:
        raise HTTPException(status_code=503, detail=describe_api_error(exc)) from exc
    except openai.APIStatusError as exc:
        raise HTTPException(status_code=502, detail=describe_api_error(exc)) from exc

    return ClassifyResponse(
        category=result.category,
        confidence=result.confidence,
        rationale=result.rationale,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/api/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a single article."""
    return _classify_one(_require_classifier(), request.article)


@app.post("/api/classify/batch", response_model=BatchResponse)
def classify_batch(request: BatchRequest) -> BatchResponse:
    """Classify up to 25 articles, reusing the cached prefix.

    Runs concurrently so a full batch finishes inside a serverless request
    window. `map` preserves input order, and the first exception surfaces when
    the result is read -- so an HTTPException raised by a worker still becomes
    that response.
    """
    classifier = _require_classifier()
    workers = min(BATCH_CONCURRENCY, len(request.articles))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda a: _classify_one(classifier, a), request.articles))
    return BatchResponse(results=results)


# Classifiers are built per configuration and cached: each has its own stable
# prompt prefix, and rebuilding one per request would throw away the prefix.
_eval_classifiers: dict[str, NewsClassifier] = {}


def _eval_classifier(key: str) -> NewsClassifier:
    if key not in BY_KEY:
        raise HTTPException(status_code=400, detail=f"Unknown configuration {key!r}.")
    if key not in _eval_classifiers:
        _require_classifier()  # surfaces missing credentials as a 503
        cfg = BY_KEY[key]
        _eval_classifiers[key] = NewsClassifier(
            use_guide=cfg.use_guide, use_few_shot=cfg.use_few_shot
        )
    return _eval_classifiers[key]


@app.post("/api/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    """Classify one arbitrary article under each configuration, `runs` times.

    This is the ad-hoc counterpart to /api/eval/run: same measurement, but on
    text the caller supplies rather than a fixed case from the hard set.
    """
    started = time.perf_counter()
    article = request.article.strip()
    if not article:
        raise HTTPException(status_code=400, detail="Article text is empty.")

    outcomes: list[CompareOutcome] = []
    for key in request.config_keys:
        classifier = _eval_classifier(key)
        jobs = [article] * request.runs
        workers = min(BATCH_CONCURRENCY, len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda a: _classify_one(classifier, a), jobs))

        counts = Counter(r.category for r in results)
        modal, modal_count = counts.most_common(1)[0]
        cfg = BY_KEY[key]
        outcomes.append(
            CompareOutcome(
                config_key=key,
                label=cfg.label,
                shot_type=cfg.shot_type,
                use_guide=cfg.use_guide,
                use_few_shot=cfg.use_few_shot,
                modal_category=modal,
                agreement=modal_count / request.runs,
                spread={c.value: n for c, n in counts.most_common()},
                mean_confidence=sum(r.confidence for r in results) / len(results),
                # Show a rationale from a run that produced the modal label, so
                # the explanation matches the label displayed beside it.
                rationale=next(r.rationale for r in results if r.category is modal),
                dissenting_category=next(
                    (r.category for r in results if r.category is not modal), None
                ),
                dissenting_rationale=next(
                    (r.rationale for r in results if r.category is not modal), None
                ),
            )
        )

    by_key = {o.config_key: o for o in outcomes}
    effects = [
        FewShotEffect(
            baseline_key=base,
            few_shot_key=shot,
            context=context,
            agreement_delta=by_key[shot].agreement - by_key[base].agreement,
            confidence_delta=by_key[shot].mean_confidence - by_key[base].mean_confidence,
            changed_label=by_key[shot].modal_category is not by_key[base].modal_category,
        )
        for (base, shot), context in PAIRS
        if base in by_key and shot in by_key
    ]

    seen = {o.modal_category for o in outcomes}
    return CompareResponse(
        outcomes=outcomes,
        effects=effects,
        consensus=len(seen) == 1,
        categories_seen=sorted(c.value for c in seen),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


@app.get("/api/eval/config")
def eval_config() -> dict:
    """Everything the evaluation UI needs to render itself."""
    return {
        "configs": [c._asdict() for c in CONFIGS],
        # The UI labels results honestly: a synthetic set is weaker evidence.
        "case_set": {
            "name": CASE_SET.name,
            "synthetic": CASE_SET.synthetic,
            "description": CASE_SET.description,
            "count": len(CASE_SET),
            "path": str(CASE_SET.path) if CASE_SET.path else None,
            "error": CASE_SET_ERROR,
        },
        "cases": [
            {
                "index": i,
                "text": c.text,
                "expected": c.expected.value,
                "hard_because": c.hard_because,
                "source": c.source,
            }
            for i, c in enumerate(HARD_CASES)
        ],
    }


@app.post("/api/eval/run", response_model=EvalRunResponse)
def eval_run(request: EvalRunRequest) -> EvalRunResponse:
    """Classify one article `runs` times under each requested configuration."""
    if request.case_index >= len(HARD_CASES):
        raise HTTPException(status_code=404, detail="No such case.")

    case = HARD_CASES[request.case_index]
    outcomes: list[ConfigOutcome] = []

    for key in request.config_keys:
        classifier = _eval_classifier(key)
        jobs = [case.text] * request.runs
        workers = min(BATCH_CONCURRENCY, len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda a: _classify_one(classifier, a), jobs))

        counts = Counter(r.category for r in results)
        modal, modal_count = counts.most_common(1)[0]
        outcomes.append(
            ConfigOutcome(
                config_key=key,
                modal_category=modal,
                agreement=modal_count / request.runs,
                correct=modal is case.expected,
                spread={c.value: n for c, n in counts.most_common()},
                mean_confidence=sum(r.confidence for r in results) / len(results),
            )
        )

    return EvalRunResponse(
        case_index=request.case_index, expected=case.expected, outcomes=outcomes
    )


@app.get("/eval")
def eval_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "eval.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if _classifier is not None else "no_credentials",
        "detail": _startup_error,
        "model": MODEL,
        "categories": [c.value for c in Category],
        "few_shot_examples": len(FEW_SHOT_EXAMPLES),
    }


@app.get("/api/samples")
def samples() -> list[dict[str, str]]:
    """Sample articles the UI offers as one-click chips."""
    return SAMPLES


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> int:
    """Entry point for `news-classifier-serve`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="news-classifier-serve",
        description="Serve the news classifier web UI and JSON API.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    args = parser.parse_args()

    print(f"\n  News Category Classifier -> http://{args.host}:{args.port}\n")
    uvicorn.run(
        "news_classifier.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0
