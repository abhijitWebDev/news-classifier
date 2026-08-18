# News Category Classifier 📰

Give it a short news article, get back one of four categories — **Sports**,
**Politics**, **Business**, **Technology** — with a confidence score and a
one-line rationale.

Built on the OpenAI API (`gpt-5.6-terra` by default), with few-shot prompting
as the main lever for consistent labels. Ships with a web UI, a JSON API and a
CLI.

## Setup

```bash
cp .env.example .env      # then put your key in it
uv sync
```

Or export the key directly: `export OPENAI_API_KEY=sk-proj-...`

Set `OPENAI_MODEL` to change model: `gpt-5.6-luna` (cheapest), `gpt-5.6-terra`
(default), `gpt-5.6-sol` (best).

## The assignment's claim, on your own article

The objective is *improve consistency with few-shot prompting*. "Improve" is a
comparative claim, so the deliverable is a before/after — not a classification.

Paste an article, press **Compare prompts**, and the classifier runs it twice
over: once with the few-shot examples and once without, with the system prompt
and everything else held identical. Each is repeated N times, because a single
run cannot show consistency. The result leads with the delta:

```
Few-shot was more consistent: +25 pts agreement

           Configuration   Label     Agreement                Confidence
ZERO-SHOT  Zero-shot       Sports    50%  Sports×2, Tech×2    0.88
FEW-SHOT   Few-shot        Sports    75%  Sports×3, Tech×1    0.81
```

That is a real measured run on the camera-tracking article below. The zero-shot
prompt was a coin-flip on it; the few-shot prompt was not. **Agreement is the
consistency metric** — how often repeated runs of the same article produce the
same label.

Not every article discriminates. On an unambiguous one both prompts score 100%
and the delta is zero, which the UI says plainly. Articles that straddle two
categories are the ones that show the effect — the `Tricky` sample chip is one,
and `hard_cases.py` has twelve more.

Tick **also test on a bare prompt** to add the same comparison with the category
guide removed, which isolates what the examples teach on their own.

## Web UI

```bash
uv run news-classifier-serve          # http://127.0.0.1:8000
uv run news-classifier-serve --reload # auto-reload while editing
```

Paste an article (or click a sample chip), hit **Classify** or `Ctrl/Cmd+Enter`.
The result card shows the category, a confidence meter and the model's one-line
reasoning; results stack into a session history below. Interactive API docs are
at `/docs`.

## JSON API

```bash
curl -X POST http://127.0.0.1:8000/api/classify \
  -H 'Content-Type: application/json' \
  -d '{"article": "The Senate voted 58-42 to advance the bill."}'
```
```json
{"category":"Politics","confidence":0.99,
 "rationale":"A floor vote on a bill.","elapsed_ms":812}
```

| Endpoint | Purpose |
|---|---|
| `POST /api/classify` | One article (max 20,000 chars) |
| `POST /api/classify/batch` | Up to 25 articles in one call |
| `POST /api/compare` | One article, zero-shot vs few-shot, N runs each |
| `GET /api/health` | Model, category list, credential status |
| `GET /api/samples` | Sample articles the UI offers as chips |
| `GET /api/eval/config` | Configurations and the hard eval set |
| `POST /api/eval/run` | One article x N runs x chosen configurations |
| `GET /docs` | OpenAPI / Swagger UI |

Errors come back as standard HTTP codes rather than a 200 with an error body:
`400` empty article, `401` bad key, `422` failed validation, `429` rate limited,
`502` upstream API error, `503` no credentials configured.

## CLI

```bash
uv run news-classifier --demo                      # four built-in samples
uv run news-classifier "Arsenal beat Spurs 2-0..."  # classify one article
uv run news-classifier -f article.txt               # from a file
cat article.txt | uv run news-classifier            # from stdin
uv run news-classifier --demo --json                # JSON Lines output
```

Sample output:

```
Sports       (confidence 0.99)
  why: A match report centred on the result and the scorer.
```

From Python:

```python
from news_classifier import NewsClassifier

classifier = NewsClassifier()          # build once, reuse — the prefix is cached
result = classifier.classify("The Senate voted 58-42 to advance the bill...")

result.category      # <Category.POLITICS: 'Politics'>
result.confidence    # 0.99
result.rationale     # 'Legislative process in the Senate: a floor vote on a bill.'
```

## How consistency is achieved

Four things work together. The few-shot examples are the biggest lever, but they
are not the only one.

**1. Few-shot examples on the boundaries** (`examples.py`) — ten worked examples
replayed as user/assistant turns. Four are unambiguous cases that fix the label
vocabulary and output shape. The other six sit exactly where the model would
otherwise waver from run to run:

| Article | Label | Why it's hard |
|---|---|---|
| Nvidia's quarterly revenue | Business | Tech company, but framed as earnings |
| OpenAI's new model capability | Technology | Same company type, opposite framing |
| Sovereign fund buys a football club | Business | Sport is the asset, not the subject |
| EU platform-regulation vote | Politics | Technology is the object of the law |
| Coach resigns amid ministry inquiry | Sports | Government involved, squad is the story |
| Central bank holds rates | Business | Policy decision, economic framing |

The rule these encode — *classify by the framing of the lede, not by keywords* —
is also stated in the system prompt, but the examples are what make it stick.

**2. A schema-constrained output** — `messages.parse()` with a Pydantic model
means the category is a JSON-schema `enum` of exactly four values. The model
cannot return "Tech", "sports", "Business/Technology", or a paragraph of prose.
Whole classes of inconsistency disappear at the API level rather than being
cleaned up afterwards.

**3. Category definitions that say what they exclude** (`categories.py`) — the
guide states the boundaries in both directions, so overlapping cases resolve the
same way each time instead of being decided fresh per call.

**4. Rationale before category** — `rationale` is the first field in the schema,
so the model states the decisive signal before committing to a label rather than
justifying a label it already picked.

## Measuring it

`news-classifier-eval` runs six *held-out* boundary articles (not in the
few-shot set, so the examples have to generalise) N times each, and reports:

- **agreement** — fraction of runs landing on the modal label; 1.0 means the
  classifier never wavers on that article
- **accuracy** — whether the modal label is the expected one

```bash
uv run news-classifier-eval -n 5              # with few-shot
uv run news-classifier-eval -n 5 --compare    # few-shot vs zero-shot, side by side
```

`--compare` is the honest test of whether the examples earn their tokens. Both
configurations use the same system prompt and the same output schema, so the
only variable is the examples.

### The 2x2: what the examples actually contribute

The first eval could not discriminate — every configuration scored 100%, so
there was no headroom to measure anything. The cause was that `CATEGORY_GUIDE`
in the system prompt already states the boundaries the examples were meant to
teach. The fix was two things: a harder eval set, and an ablation that turns the
guide off so the examples have to stand on their own.

`hard_cases.py` holds 12 articles built to remove the ceiling — lede/body
conflicts, dense vocabulary from the *wrong* category, and genuinely overlapping
events. Each records `hard_because`, so a miss reads as "fell for X" rather than
just a miss. None is reused from the few-shot set, so this measures
generalisation, not recall.

```bash
uv run news-classifier-eval --ablate        # all four configurations
uv run news-classifier-eval --hard          # hard set, shipped config only
```

Measured on 2026-08-18, `gpt-5.6-terra`, 12 articles x 3 runs:

| Configuration | Agreement | Accuracy |
|---|---|---|
| Guide + few-shot *(shipped)* | 94% | **83%** |
| Guide only | 89% | 83% |
| Few-shot only | 94% | 75% |
| Neither | 97% | **67%** |

Two findings, and the second is the interesting one:

**The examples do teach the boundaries.** With no rules in the prompt, adding
them moves accuracy 67% → 75%. That is the few-shot contribution in isolation.

**On top of the guide, they buy consistency rather than accuracy.** Accuracy is
83% either way, but agreement goes 89% → 94% — the same label more often across
repeated runs of the same article. For the stated goal, improving *consistency*,
that 5-point gain is the result.

**Watch the bottom row.** "Neither" has the *highest* agreement (97%) and the
*lowest* accuracy (67%) — it is consistently wrong. Agreement alone is not a
quality metric; it only means something read alongside accuracy. This is why the
UI and the CLI always report both.

> Caveat on statistical power: 12 articles x 3 runs is a small sample, and
> differences of a few points are within noise. Treat the 67% → 75% gap as
> solid and the 89% → 94% one as suggestive; re-run at `-n 10` before quoting
> it as settled.

> Cost: `--ablate -n 3` is 144 API calls.

## Consistency Lab (the UI)

`/eval` runs the same comparison in the browser and shows it as a matrix.

- Pick runs-per-article and which configurations to compare; the call count is
  estimated before you start.
- The browser drives the loop one article per request, so progress appears live
  and no single request approaches a serverless timeout.
- Both tables render as soon as the page loads — the **Comparison** table
  (one row per configuration, with guide/few-shot on-off and agreement and
  accuracy bars) and the per-article matrix, with cells showing `not run` / `—`
  until results arrive. You can see what is about to be compared before
  spending anything.
- Ticking or unticking a configuration adds or removes its column immediately,
  keeping results already collected.
- A verdict line states the few-shot delta in points — including when it is
  zero.
- The matrix is one row per article, one column per configuration. Green is
  correct, red is wrong, and a **striped** cell means the configuration
  disagreed with itself between runs. Each row carries its `hard_because`, and
  hovering a cell shows the full label spread and mean confidence.

## Deploying to Vercel

Vercel's Python runtime detects FastAPI from `pyproject.toml` (it reads
`uv.lock` too) and routes every request to the app, so there is very little to
configure. Two files handle it:

- **`app.py`** — the entrypoint. Vercel looks for a top-level `app` in a known
  filename; this is one of them. It prepends `src/` to `sys.path` before
  importing, so the app loads whether or not the build installed the project
  itself — with a src/ layout, a dependencies-only install would otherwise fail
  at cold start with `ModuleNotFoundError`.
- **`vercel.json`** — sets `maxDuration: 60` and keeps tests and caches out of
  the bundle.

```bash
npm i -g vercel
vercel link
vercel env add OPENAI_API_KEY        # paste the key; repeat for preview/prod
vercel deploy --prod
```

Or push to GitHub and import the repo — same result, with preview deploys per
branch.

### Things that actually bite

**Set the API key as a Vercel environment variable, not in `.env`.** `.env` is
gitignored and never uploaded. `load_dotenv()` is a no-op in production and the
app falls back to the real environment. If you forget, the site still loads and
shows "no credentials" rather than 500ing — check `/api/health`.

**Function duration.** A classification is typically 1–3s, well inside the
60s ceiling. A *batch* is the risk: 25 sequential calls would take ~40s and
could time out. `/api/classify/batch` therefore runs its calls concurrently
(5 at a time, `BATCH_CONCURRENCY` to change) — a 25-article batch that would
take 25s sequentially finishes in ~5s. Raise concurrency only as far as your
OpenAI rate limit allows.

**Cold starts and the prompt cache.** OpenAI caches prompt prefixes
automatically — there is nothing to configure, and the cache lives on their
side, so a cold Vercel instance still benefits. The only thing that breaks it is
a prefix that varies between calls, which is why the system prompt and examples
are built once and never rebuilt per request.

**Static files.** `app.mount("/static", StaticFiles(...))` is promoted to
Vercel's CDN at build time. Don't copy the `excludeFiles` example from Vercel's
docs verbatim — it excludes `static/**`, which would strip this app's UI.

**Cost is per request, and there is no auth.** A public URL means anyone who
finds it can spend your OpenAI credit. Before sharing one, put something in
front of it — Vercel's [Deployment
Protection](https://vercel.com/docs/deployment-protection) is the one-click
option; an API key check or rate limit in `api.py` is the other.

## All the articles in this repo are synthetic

Every article shipped here — the 10 few-shot examples, the 12 hard cases, the 6
first-pass eval articles, the 5 UI sample chips — **was written for this
project**. None is a real news report. Names, figures, quotes and events in them
are invented, and none should be cited as journalism.

That was deliberate: an article that reliably puts its lede in one category and
its body in another is not something you find on demand, and the hard set needs
exactly that to be able to discriminate. But it costs something, and the cost
should be stated:

- **The same author wrote the examples and the eval set.** They are
  stylistically related. `test_hard_cases_are_not_reused_from_the_few_shot_examples`
  forbids literal reuse, but stylistic correlation is not something a test can
  rule out. The examples may be unfairly well matched to these cases.
- **So the measured few-shot gain is a lower-confidence result** than the same
  measurement on independently sourced articles would be. Report it as
  "measured on a synthetic benchmark", not as a general claim about news.

### Swapping in real articles

The evaluation set is data, not code. Point `NEWS_CLASSIFIER_CASES` at your own
file — the measurement does not care where the text came from:

```bash
NEWS_CLASSIFIER_CASES=my_articles.csv uv run news-classifier-eval --hard
NEWS_CLASSIFIER_CASES=my_articles.csv uv run news-classifier-serve
```

**CSV** needs a header row with `text` and `expected`; `hard_because` and
`source` are optional:

```csv
text,expected,hard_because,source
"The finance ministry raised the transaction tax on futures...",Politics,tax rates but the actor is the ministry,Reuters 2026-08-14
```

**JSON** is either a bare list of the same objects, or an object with a `cases`
key plus `name` / `synthetic` / `description`.

Details that will save you time:

- Labels are case-insensitive (`sports`, `SPORTS`, `Sports` all work).
- A file you supply is treated as **real** unless its JSON says
  `"synthetic": true`. The bundled set is the only one that defaults to
  synthetic.
- A bad row fails with the file, the row number, the offending value and the
  valid options — not a `KeyError`.
- A broken file does **not** stop the app starting. It falls back to the
  bundled set, records the error, and reports it in `/api/eval/config`, in the
  Consistency Lab banner, and as a non-zero exit from
  `news-classifier-eval --hard`. It will not silently give you synthetic
  numbers while you think they are real.

Twelve articles is enough to be useful. Pick ones that straddle two
categories — an unambiguous article cannot show a difference between prompts.

## Tests

```bash
uv run pytest
```

All 57 run offline — no API key, no calls. `test_prompt.py` guards the things
that rot quietly: every category still has an example, the examples stay
balanced (a skewed set biases the model), the demonstrated outputs still parse
against the current schema, and the cache breakpoint is still in place.
`test_api.py` drives every endpoint against a stubbed classifier, covering the
validation limits and each error-to-status-code mapping.

## Cost

The system prompt plus examples is a ~1.3K-token prefix that is byte-identical
on every call, so OpenAI's automatic prompt caching serves it at a discount
after the first request. The article and the response are the only per-call
variable cost. On `gpt-5.6-luna` a classification is a fraction of a cent; on
`gpt-5.6-sol` it is roughly 25x that.

## Layout

```
app.py              Vercel entrypoint
vercel.json         Function config for Vercel
src/news_classifier/
├── categories.py   Category enum, boundary definitions, result schema
├── examples.py     The few-shot examples
├── classifier.py   System prompt + the API call
├── cli.py          news-classifier
├── evaluate.py     news-classifier-eval
├── configs.py      The 2x2 of prompt configurations
├── hard_cases.py   Loader for the eval set (JSON/CSV, env-var override)
├── data/
│   └── hard_cases.json   The bundled synthetic set — replace with your own
├── api.py          news-classifier-serve (FastAPI: UI + JSON API)
└── static/         index.html, eval.html, style.css, app.js, eval.js
tests/
├── test_prompt.py  Prompt and few-shot invariants
├── test_cases.py   Eval-set loading, validation and error reporting
└── test_api.py     Endpoints and entrypoint, against a stubbed model
```

The UI is plain HTML/CSS/JS with no build step and no external assets — it is
served straight from `static/`, so editing a file and reloading is the whole
dev loop.
