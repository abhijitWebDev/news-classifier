/* News Category Classifier -- front end. Talks to /api/classify. */

const CATEGORY_COLORS = {
  Sports: "var(--sports)",
  Politics: "var(--politics)",
  Business: "var(--business)",
  Technology: "var(--technology)",
};

const el = {
  article: document.getElementById("article"),
  compare: document.getElementById("compare"),
  compareRuns: document.getElementById("compare-runs"),
  compareCost: document.getElementById("compare-cost"),
  compareBare: document.getElementById("compare-bare"),
  classify: document.getElementById("classify"),
  clear: document.getElementById("clear"),
  counter: document.getElementById("counter"),
  result: document.getElementById("result"),
  samples: document.getElementById("samples"),
  status: document.getElementById("status"),
  history: document.getElementById("history"),
  historyList: document.getElementById("history-list"),
  footerMeta: document.getElementById("footer-meta"),
};

const history = [];

/* ---------- helpers ---------- */

function updateCounter() {
  const n = el.article.value.trim().length;
  el.counter.textContent = `${n.toLocaleString()} character${n === 1 ? "" : "s"}`;
  el.classify.disabled = n === 0;
  el.compare.disabled = n === 0;
  updateCompareCost();
}

const compareKeys = () =>
  el.compareBare.checked
    ? ["zero_shot", "full", "minimal", "examples_only"]
    : ["zero_shot", "full"];

function updateCompareCost() {
  const runs = Math.max(1, Math.min(10, Number(el.compareRuns.value) || 1));
  const n = compareKeys().length;
  el.compareCost.textContent = `${n} prompts × ${runs} runs = ${n * runs} API calls`;
}

/* One busy-state helper for both buttons: whichever one was pressed shows the
   spinner, and both are disabled so a second request cannot overlap. */
function setBusy(button, busy, runningLabel, idleLabel) {
  const empty = el.article.value.trim().length === 0;
  el.classify.disabled = busy || empty;
  el.compare.disabled = busy || empty;
  button.querySelector(".btn-label").textContent = busy ? runningLabel : idleLabel;
  button.querySelector(".spinner")?.remove();
  if (busy) {
    const s = document.createElement("span");
    s.className = "spinner";
    button.append(s);
  }
}

/* Build DOM nodes rather than assigning innerHTML, so article text and
   server messages can never be interpreted as markup. */
function node(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
}

/* ---------- rendering ---------- */

function renderResult(result) {
  const color = CATEGORY_COLORS[result.category] || "var(--accent)";

  const card = node("div", "card");
  card.style.setProperty("--cat", color);

  const verdict = node("div", "verdict");
  verdict.append(
    node("h2", "category", result.category),
    node("span", "timing", `${result.elapsed_ms} ms`)
  );

  const meter = node("div", "meter");
  const track = node("div", "track");
  const fill = node("div", "fill");
  track.append(fill);
  meter.append(
    node("span", null, "confidence"),
    track,
    node("span", null, result.confidence.toFixed(2))
  );

  const rationale = node("p", "rationale");
  rationale.append(node("strong", null, "Why: "), document.createTextNode(result.rationale));

  card.append(verdict, meter, rationale);
  el.result.replaceChildren(card);
  el.result.hidden = false;

  // Set the width after the node is in the document so the bar animates.
  requestAnimationFrame(() => {
    fill.style.width = `${Math.round(result.confidence * 100)}%`;
  });
}

/* The assignment's claim is "few-shot improves consistency", so the headline
   is the delta between the two runs, not either label on its own. */
function renderComparison(article, data) {
  const card = node("div", "card comparison");
  const effect = data.effects[0];
  card.style.setProperty("--cat", "var(--accent)");

  const pts = (d) => `${d > 0 ? "+" : ""}${Math.round(d * 100)}`;

  const verdict = node("div", "verdict");
  const byKey = Object.fromEntries(data.outcomes.map((o) => [o.config_key, o]));
  let headline;
  if (!effect) {
    headline = "Comparison";
  } else if (effect.changed_label) {
    // A different answer outranks a stability delta: reporting "less
    // consistent" while the label flipped would bury the bigger finding.
    const base = byKey[effect.baseline_key];
    const shot = byKey[effect.few_shot_key];
    headline = `The prompts disagree: zero-shot says ${base.modal_category}, few-shot says ${shot.modal_category}`;
  } else if (effect.agreement_delta > 0) {
    headline = `Few-shot was more consistent: ${pts(effect.agreement_delta)} pts agreement`;
  } else if (effect.agreement_delta < 0) {
    headline = `Few-shot was less consistent: ${pts(effect.agreement_delta)} pts agreement`;
  } else {
    // A zero delta is still a measurement. Say what was measured, not that
    // nothing happened -- otherwise a valid null result reads as a failure.
    headline = `Both prompts agree: ${data.outcomes[0].modal_category} — few-shot effect 0 pts`;
  }
  verdict.append(node("h2", "category", headline), node("span", "timing", `${data.elapsed_ms} ms`));
  card.append(verdict);

  if (effect) {
    card.append(
      node(
        "p",
        "verdict-note",
        effect.changed_label
          ? `The examples changed the answer, not just its stability (agreement ${pts(effect.agreement_delta)} pts). ` +
            "Read the two reasons below and judge which framing is right — a prompt can be perfectly consistent and consistently wrong."
          : effect.agreement_delta === 0
          ? "The comparison ran and both prompts scored identically, so this article does not discriminate between them. Few-shot only shows a gain where the model is unstable — try an article that straddles two categories, such as the Tricky sample above."
          : "Both prompts reached the same label, but one repeated itself more reliably across runs. That stability is what consistency means here."
      )
    );
  }

  const table = node("table", "compare inline");
  const head = node("tr");
  for (const h of ["", "Configuration", "Label", "Agreement", "Conf."]) {
    head.append(node("th", h === "" ? "tag-col" : null, h));
  }
  const thead = node("thead");
  thead.append(head);
  table.append(thead);

  const body = node("tbody");
  for (const o of data.outcomes) {
    const tr = node("tr");

    const tag = node("td", "tag-col");
    tag.append(node("span", `shot-tag ${o.shot_type === "few-shot" ? "few" : "zero"}`, o.shot_type));
    tr.append(tag);

    tr.append(node("td", null, o.label));

    const label = node("td", "label-cell");
    const chip = node("span", "cat-chip", o.modal_category);
    chip.style.setProperty("--cat", CATEGORY_COLORS[o.modal_category] || "var(--accent)");
    label.append(chip);
    const spread = Object.entries(o.spread).map(([k, v]) => `${k}×${v}`).join(", ");
    label.title = spread;
    tr.append(label);

    const agree = node("td", "metric" + (o.agreement < 1 ? " wavered" : ""));
    agree.append(node("span", "num", `${Math.round(o.agreement * 100)}%`));
    if (o.agreement < 1) agree.append(node("span", "of", spread));
    tr.append(agree);

    tr.append(node("td", "metric", o.mean_confidence.toFixed(2)));
    body.append(tr);

    // The reasoning gets its own full-width row rather than a sixth column:
    // it is the longest content here, and as a column it was clipped behind a
    // horizontal scrollbar -- exactly the thing you most want to read.
    const whyRow = node("tr", "why-row");
    const why = node("td", "why");
    why.colSpan = 5;
    why.append(
      node("span", "why-label", `Why ${o.modal_category}:`),
      document.createTextNode(" " + o.rationale)
    );
    if (o.dissenting_rationale) {
      const dissent = node("span", "why-dissent");
      dissent.append(
        node("span", "why-label alt", `Other runs said ${o.dissenting_category}:`),
        document.createTextNode(" " + o.dissenting_rationale)
      );
      why.append(dissent);
    }
    whyRow.append(why);
    body.append(whyRow);
  }
  table.append(body);

  const wrap = node("div", "scroll-x");
  wrap.append(table);
  card.append(wrap);

  if (data.effects.length > 1) {
    const extra = data.effects
      .slice(1)
      .map((e) => `${e.context}: ${pts(e.agreement_delta)} pts agreement`)
      .join(" · ");
    card.append(node("p", "effect-extra", `Also measured — ${extra}`));
  }

  el.result.replaceChildren(card);
  el.result.hidden = false;
}

async function compareConfigurations() {
  const article = el.article.value.trim();
  if (!article) return;

  const runs = Math.max(1, Math.min(10, Number(el.compareRuns.value) || 1));
  setBusy(el.compare, true, "Comparing", "Compare prompts");
  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ article, runs, config_keys: compareKeys() }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      renderError(
        typeof detail === "string" ? detail : `Request failed (HTTP ${response.status}).`
      );
      return;
    }
    renderComparison(article, payload);
  } catch (err) {
    renderError(`Could not reach the server: ${err.message}`);
  } finally {
    setBusy(el.compare, false, "Comparing", "Compare prompts");
  }
}

function renderError(message) {
  const card = node("div", "card error");
  card.append(node("h2", "category", "Could not classify"), node("p", "rationale", message));
  el.result.replaceChildren(card);
  el.result.hidden = false;
}

function pushHistory(article, result) {
  history.unshift({ article, result });
  el.historyList.replaceChildren(
    ...history.slice(0, 8).map(({ article, result }) => {
      const li = node("li");
      li.style.setProperty("--cat", CATEGORY_COLORS[result.category] || "var(--accent)");
      const snippet = article.trim().replace(/\s+/g, " ");
      li.append(
        node("span", "tag", result.category),
        node("span", "snippet", snippet.length > 90 ? snippet.slice(0, 90) + "…" : snippet),
        node("span", "conf", result.confidence.toFixed(2))
      );
      return li;
    })
  );
  el.history.hidden = false;
}

/* ---------- actions ---------- */

async function classify() {
  const article = el.article.value.trim();
  if (!article) return;

  setBusy(el.classify, true, "Classifying", "Classify");
  try {
    const response = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ article }),
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = payload?.detail;
      renderError(
        typeof detail === "string" ? detail : `Request failed (HTTP ${response.status}).`
      );
      return;
    }

    renderResult(payload);
    pushHistory(article, payload);
  } catch (err) {
    renderError(`Could not reach the server: ${err.message}`);
  } finally {
    setBusy(el.classify, false, "Classifying", "Classify");
  }
}

/* ---------- startup ---------- */

async function loadSamples() {
  try {
    const samples = await fetch("/api/samples").then((r) => r.json());
    el.samples.replaceChildren(
      ...samples.map(({ label, text }) => {
        const chip = node("button", "chip", label);
        chip.type = "button";
        chip.addEventListener("click", () => {
          el.article.value = text;
          updateCounter();
          el.article.focus();
        });
        return chip;
      })
    );
  } catch {
    /* Samples are a convenience; the app works fine without them. */
  }
}

async function loadHealth() {
  try {
    const health = await fetch("/api/health").then((r) => r.json());
    el.footerMeta.textContent =
      `${health.model} · ${health.few_shot_examples} few-shot examples`;
    if (health.status !== "ok") {
      el.status.textContent = health.detail || "The classifier is not configured.";
      el.status.hidden = false;
    }
  } catch {
    /* Leave the footer empty rather than blocking the UI. */
  }
}

el.article.addEventListener("input", updateCounter);
el.classify.addEventListener("click", classify);
el.compare.addEventListener("click", compareConfigurations);
el.compareRuns.addEventListener("input", updateCompareCost);
el.compareBare.addEventListener("change", updateCompareCost);
el.clear.addEventListener("click", () => {
  el.article.value = "";
  el.result.hidden = true;
  updateCounter();
  el.article.focus();
});
el.article.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    classify();
  }
});

updateCounter();
loadSamples();
loadHealth();
