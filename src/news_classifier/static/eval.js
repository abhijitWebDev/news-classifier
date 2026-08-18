/* Consistency Lab -- drives the eval from the browser, one article per request
   so progress is visible and no request nears a serverless timeout. */

const el = (id) => document.getElementById(id);
const ui = {
  runs: el("runs"),
  picker: el("config-picker"),
  estimate: el("estimate"),
  run: el("run"),
  reset: el("reset"),
  status: el("status"),
  progress: el("progress"),
  progressFill: el("progress-fill"),
  progressLabel: el("progress-label"),
  summary: el("summary"),
  compareBody: el("compare-body"),
  verdictLine: el("verdict-line"),
  matrixWrap: el("matrix-wrap"),
  matrixHead: el("matrix-head"),
  matrixBody: el("matrix-body"),
  footerMeta: el("footer-meta"),
};

let CONFIGS = [];
let CASES = [];
let results = new Map(); // `${caseIndex}:${configKey}` -> outcome
let running = false;

function node(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
}

const selectedKeys = () =>
  [...ui.picker.querySelectorAll("input:checked")].map((i) => i.value);

/* ---------- setup ---------- */

/* A synthetic set is weaker evidence than real reporting, and the person
   reading the numbers is the one who needs to know that -- so it is stated
   above the results, not buried in a README. */
function renderCaseSetBanner(set) {
  if (!set) return;
  const banner = document.getElementById("case-set");
  if (!banner) return;

  banner.replaceChildren();
  banner.classList.toggle("synthetic", Boolean(set.synthetic));
  banner.classList.toggle("broken", Boolean(set.error));

  if (set.error) {
    banner.append(
      node("strong", null, "Your evaluation set failed to load. "),
      node("span", null, `${set.error} Falling back to the bundled synthetic set — fix the file before quoting any numbers.`)
    );
    banner.hidden = false;
    return;
  }

  const title = node("strong", null, `${set.name} — ${set.count} articles`);
  banner.append(title);
  if (set.synthetic) {
    banner.append(
      node(
        "span",
        null,
        " Synthetic: written for this project, not real reporting, and by the " +
          "same hand as the few-shot examples. Treat any measured gain as " +
          "suggestive rather than settled. Point NEWS_CLASSIFIER_CASES at your " +
          "own .json or .csv to use real articles."
      )
    );
  } else if (set.description) {
    banner.append(node("span", null, ` ${set.description}`));
  }
  banner.hidden = false;
}

function renderPicker() {
  ui.picker.replaceChildren(
    ...CONFIGS.map((cfg) => {
      const label = node("label", "config-option");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = cfg.key;
      // Default to the pair that isolates the few-shot effect, plus the
      // shipped configuration for reference.
      box.checked = ["minimal", "examples_only", "full"].includes(cfg.key);
      box.addEventListener("change", () => {
        updateEstimate();
        render();
      });
      const text = node("span");
      text.append(node("span", "name", cfg.label), node("span", "desc", cfg.description));
      label.append(box, text);
      return label;
    })
  );
}

function updateEstimate() {
  const n = Math.max(1, Math.min(10, Number(ui.runs.value) || 1));
  const calls = CASES.length * selectedKeys().length * n;
  ui.estimate.textContent =
    `${CASES.length} articles × ${selectedKeys().length} configs × ${n} runs = ${calls} API calls`;
  ui.run.disabled = running || selectedKeys().length === 0;
}

/* ---------- matrix ---------- */

/* Both tables are rendered from page load, before any run: the articles and
   configurations are known up front, so the page should show what is about to
   be compared rather than an empty panel. Cells fill in as results arrive. */
function render() {
  const keys = selectedKeys();
  buildMatrix(keys);
  renderSummary(keys);
}

function buildMatrix(keys) {
  const head = [node("th", "case-col", "Article")];
  for (const key of keys) {
    head.push(node("th", null, CONFIGS.find((c) => c.key === key).label));
  }
  ui.matrixHead.replaceChildren(...head);

  ui.matrixBody.replaceChildren(
    ...CASES.map((c) => {
      const tr = node("tr");
      tr.dataset.case = c.index;

      const td = node("td", "case-col");
      const snippet = c.text.length > 190 ? c.text.slice(0, 190) + "…" : c.text;
      td.append(node("span", "case-text", snippet));
      const meta = node("span", "case-meta");
      meta.append(
        node("span", "expected", `Expected ${c.expected}. `),
        document.createTextNode(c.hard_because)
      );
      td.append(meta);
      tr.append(td);

      for (const key of keys) {
        const cell = node("td", "cell pending", "—");
        cell.dataset.key = key;
        tr.append(cell);
        const known = results.get(`${c.index}:${key}`);
        if (known) fillCellIn(cell, known);
      }
      return tr;
    })
  );
}

function fillCell(caseIndex, outcome) {
  const row = ui.matrixBody.querySelector(`tr[data-case="${caseIndex}"]`);
  const cell = row?.querySelector(`td[data-key="${outcome.config_key}"]`);
  if (cell) fillCellIn(cell, outcome);
}

function fillCellIn(cell, outcome) {
  cell.className = `cell ${outcome.correct ? "ok" : "bad"}` +
    (outcome.agreement < 1 ? " wavered" : "");
  cell.replaceChildren(
    node("span", "label", outcome.modal_category),
    node("span", "agree", `${Math.round(outcome.agreement * 100)}% agreement`)
  );
  const spread = Object.entries(outcome.spread).map(([k, v]) => `${k}×${v}`).join(", ");
  cell.title = `${spread} · mean confidence ${outcome.mean_confidence.toFixed(2)}`;
}

/* ---------- summary ---------- */

function renderSummary(keys) {
  const cards = keys.map((key) => {
    const cfg = CONFIGS.find((c) => c.key === key);
    const rows = CASES.map((c) => results.get(`${c.index}:${key}`)).filter(Boolean);
    const agreement = rows.length ? rows.reduce((s, r) => s + r.agreement, 0) / rows.length : null;
    const accuracy = rows.length ? rows.filter((r) => r.correct).length / rows.length : null;

    const tr = node("tr");
    const name = node("td");
    name.append(node("span", "name", cfg.label), node("span", "desc", cfg.description));
    tr.append(name);
    tr.append(node("td", "mid", cfg.use_guide ? "on" : "off"));
    tr.append(node("td", "mid", cfg.use_few_shot ? "on" : "off"));

    for (const [value, cls] of [[agreement, ""], [accuracy, "acc"]]) {
      const td = node("td", "metric");
      if (value === null) {
        td.append(node("span", "awaiting", "not run"));
      } else {
        const bar = node("div", "bar");
        const track = node("div", "track");
        const fill = node("div", `fill ${cls}`);
        fill.style.width = `${Math.round(value * 100)}%`;
        track.append(fill);
        bar.append(track, node("span", "num", `${Math.round(value * 100)}%`));
        td.append(bar, node("span", "of", `${rows.length}/${CASES.length} articles`));
      }
      tr.append(td);
    }
    return { key, tr, agreement, accuracy };
  });

  ui.compareBody.replaceChildren(...cards.map((c) => c.tr));
  renderVerdict(cards);
}

/* The interesting comparison is few-shot vs the same prompt without it. Report
   the delta plainly, including when it is zero -- a null result is a result. */
function renderVerdict(cards) {
  const by = Object.fromEntries(
    cards.filter((c) => c.agreement !== null).map((c) => [c.key, c])
  );
  const pairs = [
    ["minimal", "examples_only", "with no boundary rules in the prompt"],
    ["zero_shot", "full", "on top of the full category guide"],
  ].filter(([a, b]) => by[a] && by[b]);

  if (!pairs.length) {
    ui.verdictLine.textContent = Object.keys(by).length
      ? "Select a with/without pair (Neither + Few-shot only, or Guide only + Guide + few-shot) to measure the few-shot effect."
      : "Nothing measured yet — pick your configurations and press Run evaluation. Results fill in article by article.";
    return;
  }

  const lines = pairs.map(([a, b, where]) => {
    const dAgree = Math.round((by[b].agreement - by[a].agreement) * 100);
    const dAcc = Math.round((by[b].accuracy - by[a].accuracy) * 100);
    const fmt = (d) => (d > 0 ? `+${d}` : `${d}`);
    return `Adding few-shot examples ${where}: agreement ${fmt(dAgree)} pts, accuracy ${fmt(dAcc)} pts.`;
  });

  const anyEffect = pairs.some(
    ([a, b]) => by[b].agreement !== by[a].agreement || by[b].accuracy !== by[a].accuracy
  );
  lines.push(
    anyEffect
      ? "A non-zero delta is the evidence that the examples are doing work."
      : "No difference measured — on this set the examples are not changing the outcome."
  );
  ui.verdictLine.textContent = lines.join(" ");
}

/* ---------- run ---------- */

async function runEvaluation() {
  const keys = selectedKeys();
  if (!keys.length || running) return;

  running = true;
  results = new Map();
  ui.run.disabled = true;
  ui.run.querySelector(".btn-label").textContent = "Running";
  ui.progress.hidden = false;
  ui.status.hidden = true;

  const runs = Math.max(1, Math.min(10, Number(ui.runs.value) || 1));
  results = new Map();
  render();

  let done = 0;
  const total = CASES.length;

  for (const c of CASES) {
    ui.progressLabel.textContent = `article ${done + 1} of ${total}`;
    try {
      const response = await fetch("/api/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_index: c.index, runs, config_keys: keys }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = payload?.detail;
        ui.status.textContent =
          typeof detail === "string" ? detail : `Request failed (HTTP ${response.status}).`;
        ui.status.hidden = false;
        break;
      }
      for (const outcome of payload.outcomes) {
        results.set(`${c.index}:${outcome.config_key}`, outcome);
        fillCell(c.index, outcome);
      }
    } catch (err) {
      ui.status.textContent = `Could not reach the server: ${err.message}`;
      ui.status.hidden = false;
      break;
    }

    done += 1;
    ui.progressFill.style.width = `${(done / total) * 100}%`;
    renderSummary(keys);
  }

  ui.progressLabel.textContent = `${done} of ${total} articles`;
  ui.run.querySelector(".btn-label").textContent = "Run evaluation";
  running = false;
  updateEstimate();
}

/* ---------- startup ---------- */

async function init() {
  try {
    const cfg = await fetch("/api/eval/config").then((r) => r.json());
    CONFIGS = cfg.configs;
    CASES = cfg.cases;
    renderCaseSetBanner(cfg.case_set);
    renderPicker();
    updateEstimate();
    render();
  } catch (err) {
    ui.status.textContent = `Could not load the evaluation set: ${err.message}`;
    ui.status.hidden = false;
  }
  try {
    const health = await fetch("/api/health").then((r) => r.json());
    ui.footerMeta.textContent = `${health.model} · ${health.few_shot_examples} few-shot examples`;
    if (health.status !== "ok") {
      ui.status.textContent = health.detail || "The classifier is not configured.";
      ui.status.hidden = false;
    }
  } catch { /* footer is cosmetic */ }
}

ui.runs.addEventListener("input", updateEstimate);
ui.run.addEventListener("click", runEvaluation);
ui.reset.addEventListener("click", () => {
  results = new Map();
  ui.progress.hidden = true;
  ui.progressFill.style.width = "0";
  ui.status.hidden = true;
  render();
});

init();
