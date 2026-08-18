"""API tests. A stub classifier stands in for the model -- no network, no key."""

import pytest
from fastapi.testclient import TestClient

from news_classifier import api
from news_classifier.categories import Category, Classification


class StubClassifier:
    """Records what it was asked and returns a fixed answer."""

    def __init__(self, result=None, error=None):
        self.result = result or Classification(
            rationale="A floor vote on a bill.",
            category=Category.POLITICS,
            confidence=0.99,
        )
        self.error = error
        self.seen: list[str] = []

    def classify(self, article: str) -> Classification:
        self.seen.append(article)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client(monkeypatch):
    """A live app with the model swapped out."""
    stub = StubClassifier()
    monkeypatch.setattr(api, "_classifier", stub)
    monkeypatch.setattr(api, "_startup_error", None)
    with TestClient(api.app) as c:
        # TestClient runs lifespan, which rebuilds _classifier -- reassert the stub.
        monkeypatch.setattr(api, "_classifier", stub)
        c.stub = stub
        yield c


def test_classify_returns_the_full_result(client):
    r = client.post("/api/classify", json={"article": "The Senate voted 58-42..."})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Politics"
    assert body["confidence"] == 0.99
    assert body["rationale"] == "A floor vote on a bill."
    assert isinstance(body["elapsed_ms"], int)


def test_article_reaches_the_classifier_unmodified(client):
    article = "  Liverpool beat Chelsea 3-0 at Anfield.  "
    client.post("/api/classify", json={"article": article})
    assert client.stub.seen == [article]


def test_empty_article_is_rejected_by_validation(client):
    assert client.post("/api/classify", json={"article": ""}).status_code == 422


def test_oversized_article_is_rejected(client):
    r = client.post("/api/classify", json={"article": "x" * 20_001})
    assert r.status_code == 422


def test_batch_classifies_every_article(client):
    r = client.post("/api/classify/batch", json={"articles": ["one", "two", "three"]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3
    assert client.stub.seen == ["one", "two", "three"]


def test_batch_is_capped(client):
    r = client.post("/api/classify/batch", json={"articles": ["a"] * 26})
    assert r.status_code == 422


def test_missing_credentials_surface_as_503(monkeypatch):
    monkeypatch.setattr(api, "_classifier", None)
    monkeypatch.setattr(api, "_startup_error", "No OpenAI credentials found.")
    with TestClient(api.app) as c:
        monkeypatch.setattr(api, "_classifier", None)
        r = c.post("/api/classify", json={"article": "anything"})
    assert r.status_code == 503
    assert "credentials" in r.json()["detail"]


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ValueError("Article text is empty."), 400),
        (RuntimeError("boom"), 500),
    ],
)
def test_classifier_failures_map_to_status_codes(monkeypatch, exc, status):
    monkeypatch.setattr(api, "_classifier", StubClassifier(error=exc))
    monkeypatch.setattr(api, "_startup_error", None)
    with TestClient(api.app, raise_server_exceptions=False) as c:
        monkeypatch.setattr(api, "_classifier", StubClassifier(error=exc))
        r = c.post("/api/classify", json={"article": "text"})
    assert r.status_code == status


def test_health_reports_configuration(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model"] == api.MODEL
    assert body["categories"] == ["Sports", "Politics", "Business", "Technology"]
    assert body["few_shot_examples"] == 10


def test_samples_cover_every_category(client):
    labels = {s["label"] for s in client.get("/api/samples").json()}
    assert {c.value for c in Category} <= labels


def test_index_serves_the_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "News Category Classifier" in r.text


def test_static_assets_are_served(client):
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_batch_preserves_input_order():
    """Concurrency must not reorder results relative to the articles sent."""

    class PerArticleStub:
        def classify(self, article: str) -> Classification:
            # Reverse-ordered sleeps: without order-preserving collection the
            # fastest call would land first.
            import time

            time.sleep((10 - int(article)) * 0.01)
            return Classification(
                rationale=article, category=Category.BUSINESS, confidence=0.5
            )

    import pytest as _pytest  # local import keeps the fixture list unchanged

    monkeypatch = _pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(api, "_classifier", PerArticleStub())
        monkeypatch.setattr(api, "_startup_error", None)
        with TestClient(api.app) as c:
            monkeypatch.setattr(api, "_classifier", PerArticleStub())
            body = c.post(
                "/api/classify/batch", json={"articles": [str(i) for i in range(8)]}
            ).json()
    finally:
        monkeypatch.undo()

    assert [r["rationale"] for r in body["results"]] == [str(i) for i in range(8)]


def test_vercel_entrypoint_exposes_the_app():
    """app.py is what Vercel loads; it must import without the package installed."""
    import app as entrypoint

    assert entrypoint.app is api.app


def test_eval_config_exposes_configs_and_cases(client):
    body = client.get("/api/eval/config").json()
    keys = [c["key"] for c in body["configs"]]
    assert keys == ["zero_shot", "full", "minimal", "examples_only"]
    # Baseline first, few-shot second: the UI reads them as ordered pairs.
    assert [c["shot_type"] for c in body["configs"]] == [
        "zero-shot", "few-shot", "zero-shot", "few-shot",
    ]
    assert len(body["cases"]) >= 10
    case = body["cases"][0]
    assert {"index", "text", "expected", "hard_because"} <= case.keys()


def test_eval_run_aggregates_across_runs(client, monkeypatch):
    """Agreement and spread must reflect what the model actually returned."""
    labels = iter([Category.SPORTS, Category.SPORTS, Category.BUSINESS])

    class Wavering:
        def classify(self, article):
            return Classification(
                rationale="r", category=next(labels), confidence=0.8
            )

    monkeypatch.setitem(api._eval_classifiers, "full", Wavering())
    body = client.post(
        "/api/eval/run", json={"case_index": 0, "runs": 3, "config_keys": ["full"]}
    ).json()

    outcome = body["outcomes"][0]
    assert outcome["modal_category"] == "Sports"
    assert outcome["agreement"] == pytest.approx(2 / 3)
    assert outcome["spread"] == {"Sports": 2, "Business": 1}
    assert outcome["mean_confidence"] == pytest.approx(0.8)


def test_eval_run_rejects_unknown_config(client):
    r = client.post(
        "/api/eval/run", json={"case_index": 0, "runs": 1, "config_keys": ["nope"]}
    )
    assert r.status_code == 400


def test_eval_run_rejects_out_of_range_case(client):
    r = client.post(
        "/api/eval/run", json={"case_index": 9999, "runs": 1, "config_keys": ["full"]}
    )
    assert r.status_code == 404


def test_eval_runs_are_capped(client):
    r = client.post(
        "/api/eval/run", json={"case_index": 0, "runs": 99, "config_keys": ["full"]}
    )
    assert r.status_code == 422


def test_eval_page_and_asset_are_served(client):
    assert "Consistency Lab" in client.get("/eval").text
    assert client.get("/static/eval.js").status_code == 200


def test_compare_reports_consensus_when_configs_agree(client, monkeypatch):
    for key in ["full", "zero_shot", "examples_only", "minimal"]:
        monkeypatch.setitem(api._eval_classifiers, key, StubClassifier())

    body = client.post(
        "/api/compare", json={"article": "The Senate voted 58-42...", "runs": 2}
    ).json()

    assert body["consensus"] is True
    assert body["categories_seen"] == ["Politics"]
    # Defaults to the headline pair only: zero-shot vs few-shot.
    assert [o["config_key"] for o in body["outcomes"]] == ["zero_shot", "full"]
    assert [o["shot_type"] for o in body["outcomes"]] == ["zero-shot", "few-shot"]
    assert body["outcomes"][1]["use_few_shot"] is True
    assert body["outcomes"][0]["use_few_shot"] is False
    assert body["outcomes"][0]["agreement"] == 1.0


def test_compare_flags_disagreement_between_configs(client, monkeypatch):
    """A split verdict is the signal that the prompt, not the article, decides."""
    labels = {
        "full": Category.BUSINESS,
        "zero_shot": Category.BUSINESS,
        "examples_only": Category.TECHNOLOGY,
        "minimal": Category.SPORTS,
    }
    for key, cat in labels.items():
        monkeypatch.setitem(
            api._eval_classifiers,
            key,
            StubClassifier(
                result=Classification(rationale=key, category=cat, confidence=0.7)
            ),
        )

    body = client.post(
        "/api/compare",
        json={
            "article": "text",
            "runs": 1,
            "config_keys": ["zero_shot", "full", "examples_only", "minimal"],
        },
    ).json()

    assert body["consensus"] is False
    assert body["categories_seen"] == ["Business", "Sports", "Technology"]


def test_compare_rationale_matches_the_reported_label(client, monkeypatch):
    """The shown rationale must come from a run that produced the shown label."""
    seq = iter([Category.SPORTS, Category.BUSINESS, Category.BUSINESS])

    class Mixed:
        def classify(self, article):
            cat = next(seq)
            return Classification(rationale=f"because {cat.value}", category=cat, confidence=0.8)

    monkeypatch.setitem(api._eval_classifiers, "full", Mixed())
    body = client.post(
        "/api/compare", json={"article": "t", "runs": 3, "config_keys": ["full"]}
    ).json()

    outcome = body["outcomes"][0]
    assert outcome["modal_category"] == "Business"
    assert outcome["rationale"] == "because Business"


def test_compare_rejects_blank_article(client):
    assert client.post("/api/compare", json={"article": "   "}).status_code == 400
    assert client.post("/api/compare", json={"article": ""}).status_code == 422


def test_compare_reports_the_few_shot_effect(client, monkeypatch):
    """The delta between the pair is the deliverable, so it is computed server-side."""
    monkeypatch.setitem(
        api._eval_classifiers,
        "zero_shot",
        StubClassifier(
            result=Classification(
                rationale="baseline", category=Category.BUSINESS, confidence=0.60
            )
        ),
    )
    monkeypatch.setitem(
        api._eval_classifiers,
        "full",
        StubClassifier(
            result=Classification(
                rationale="few-shot", category=Category.POLITICS, confidence=0.95
            )
        ),
    )

    body = client.post("/api/compare", json={"article": "t", "runs": 2}).json()

    assert len(body["effects"]) == 1
    effect = body["effects"][0]
    assert effect["baseline_key"] == "zero_shot"
    assert effect["few_shot_key"] == "full"
    assert effect["agreement_delta"] == pytest.approx(0.0)   # both perfectly stable
    assert effect["confidence_delta"] == pytest.approx(0.35)
    assert effect["changed_label"] is True


def test_compare_effect_tracks_a_real_consistency_gain(client, monkeypatch):
    """Baseline wavers, few-shot does not -- the delta must be positive."""
    wobble = iter([Category.SPORTS, Category.BUSINESS])

    class Wavering:
        def classify(self, article):
            return Classification(
                rationale="r", category=next(wobble), confidence=0.5
            )

    monkeypatch.setitem(api._eval_classifiers, "zero_shot", Wavering())
    monkeypatch.setitem(
        api._eval_classifiers,
        "full",
        StubClassifier(
            result=Classification(
                rationale="steady", category=Category.SPORTS, confidence=0.9
            )
        ),
    )

    body = client.post("/api/compare", json={"article": "t", "runs": 2}).json()
    assert body["effects"][0]["agreement_delta"] == pytest.approx(0.5)  # 50% -> 100%


def test_compare_classifies_exactly_the_pasted_text(client, monkeypatch):
    """The box drives the comparison -- the bundled article set is never involved."""
    seen: dict[str, list[str]] = {}

    def recorder(key):
        class Recording:
            def classify(self, article):
                seen.setdefault(key, []).append(article)
                return Classification(
                    rationale="r", category=Category.SPORTS, confidence=0.9
                )

        return Recording()

    for key in ("zero_shot", "full"):
        monkeypatch.setitem(api._eval_classifiers, key, recorder(key))

    pasted = (
        "Rishabh Pant became the fastest batter to reach 100 Test sixes "
        "during the first Test against Sri Lanka in Galle."
    )
    client.post("/api/compare", json={"article": pasted, "runs": 3})

    # Every call, under every configuration, saw the pasted text verbatim.
    assert set(seen) == {"zero_shot", "full"}
    for key, articles in seen.items():
        assert articles == [pasted] * 3, key

    # And nothing from the bundled set leaked in.
    bundled = {c.text for c in api.HARD_CASES}
    assert not bundled & {a for calls in seen.values() for a in calls}


def test_compare_strips_surrounding_whitespace_only(client, monkeypatch):
    seen = []

    class Recording:
        def classify(self, article):
            seen.append(article)
            return Classification(rationale="r", category=Category.SPORTS, confidence=0.9)

    monkeypatch.setitem(api._eval_classifiers, "zero_shot", Recording())
    client.post(
        "/api/compare",
        json={"article": "  Pant hit 100 Test sixes in Galle today.  ", "runs": 1,
              "config_keys": ["zero_shot"]},
    )
    assert seen == ["Pant hit 100 Test sixes in Galle today."]


def test_compare_reports_the_dissenting_reason_when_runs_disagree(client, monkeypatch):
    """A wavering prompt must show both sides -- the split is the finding."""
    seq = iter(
        [
            (Category.SPORTS, "a cricket tournament report"),
            (Category.SPORTS, "framed around the semi-finals"),
            (Category.TECHNOLOGY, "explains how the camera array works"),
        ]
    )

    class Wavering:
        def classify(self, article):
            cat, why = next(seq)
            return Classification(rationale=why, category=cat, confidence=0.8)

    monkeypatch.setitem(api._eval_classifiers, "zero_shot", Wavering())
    body = client.post(
        "/api/compare",
        json={"article": "ball tracking article", "runs": 3, "config_keys": ["zero_shot"]},
    ).json()

    outcome = body["outcomes"][0]
    assert outcome["modal_category"] == "Sports"
    assert outcome["rationale"] == "a cricket tournament report"
    assert outcome["dissenting_category"] == "Technology"
    assert outcome["dissenting_rationale"] == "explains how the camera array works"


def test_compare_has_no_dissent_when_runs_agree(client, monkeypatch):
    monkeypatch.setitem(api._eval_classifiers, "zero_shot", StubClassifier())
    body = client.post(
        "/api/compare",
        json={"article": "text", "runs": 3, "config_keys": ["zero_shot"]},
    ).json()

    assert body["outcomes"][0]["dissenting_category"] is None
    assert body["outcomes"][0]["dissenting_rationale"] is None
