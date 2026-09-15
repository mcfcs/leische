/* Leische demo — fetch + DOM, no framework, no build step. */
"use strict";

const $ = (id) => document.getElementById(id);
const CHANNELS = ["conv", "temp", "ret"];
const GROUPS = [
  ["english", "English"],
  ["tagalog", "Tagalog"],
  ["taglish", "Taglish"],
  ["hard", "Hard cases — human and ensemble disagree"],
];

const state = {
  meta: null,
  examples: [],
  current: null,      // the preloaded example currently loaded, if any
  last: null,         // previous calibrated probability, for the ghost mark
  reason: "",         // what caused this run, shown next to the delta
  seq: 0,             // request sequence, so a slow reply cannot overwrite a fast one
  timer: null,
};

/* ------------------------------------------------------------------ setup */
async function init() {
  try {
    const [meta, examples] = await Promise.all([
      fetch("/api/meta").then(okJson),
      fetch("/api/examples").then(okJson),
    ]);
    state.meta = meta;
    state.examples = examples;
    renderMeta(meta);
    renderExamples(examples);
  } catch (err) {
    showError("Could not reach the model server. " + err.message);
  }

  $("run").addEventListener("click", () => run("scored"));
  $("target").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run("scored"); }
  });
  $("target").addEventListener("input", () => {
    countChars();
    if ($("result").hidden) return;             // wait for the first explicit score
    schedule("comment edited", 900);
  });

  for (const ch of CHANNELS) {
    $("t-" + ch).addEventListener("click", (e) => {
      const btn = e.currentTarget;
      const on = btn.getAttribute("aria-pressed") !== "true";
      btn.setAttribute("aria-pressed", String(on));
      paintChannels();
      run((on ? "turned on " : "turned off ") + btn.textContent.trim());
    });
  }

  document.querySelectorAll("[data-add]").forEach((btn) => {
    btn.addEventListener("click", () => {
      addRow(btn.dataset.add);
      const list = $(btn.dataset.add);
      const input = list.lastElementChild.querySelector(".turn-text");
      if (input) input.focus();
    });
  });

  for (const id of ["sub-title", "sub-body"]) {
    $(id).addEventListener("input", () => schedule("context edited", 600));
  }
  for (const id of ["parents", "replies", "history"]) {
    $(id).addEventListener("input", () => schedule("context edited", 600));
  }

  paintChannels();
  countChars();
}

function okJson(r) {
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

function schedule(reason, ms) {
  clearTimeout(state.timer);
  state.timer = setTimeout(() => run(reason), ms);
}

/* ------------------------------------------------------------------- meta */
function renderMeta(meta) {
  $("honesty").textContent = meta.honesty;

  if (!meta.trained) {
    $("untrained").hidden = false;
    $("untrained-detail").textContent =
      "Every number on this page comes from a randomly initialised model. Looking for " +
      meta.checkpoint + " under cache/checkpoints/.";
  }

  const facts = [
    ["decision threshold", pct(meta.threshold_cal) + " calibrated (" + pct(meta.threshold) + " raw)"],
    ["temperature", meta.temperature.toFixed(3)],
    ["retrieval banks", meta.bank_sizes.sarcastic.toLocaleString() + " sarcastic, " +
      meta.bank_sizes.non_sarcastic.toLocaleString() + " not"],
    ["k per bank", String(meta.retrieval_k)],
    ["temporal window", meta.temporal_window_hours ? meta.temporal_window_hours + " h, up to " +
      meta.temporal_k + " posts" : "unbounded"],
    ["running on", meta.device],
  ];
  const dl = $("facts");
  dl.textContent = "";
  for (const [k, v] of facts) {
    const wrap = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    wrap.append(dt, dd);
    dl.append(wrap);
  }

  $("temp-cap").textContent = "Up to " + meta.temporal_k + " posts, within " +
    (meta.temporal_window_hours || "∞") + " hours before the comment";

  const la = meta.label_authority;
  $("foot-authority").textContent = la
    ? "Labels: " + (la.annotators || []).join(", ") +
      (la.adjudicator ? " adjudicated by " + [].concat(la.adjudicator).join(", ") : "") +
      ". Sarcasm Fleiss κ = " + la.sarcastic_fleiss_kappa + " across the three annotators."
    : "No checkpoint loaded, so no label-authority block is attached to this session.";

  if (!meta.sentiment_heads) {
    $("sentiment-missing").hidden = false;
  }
}

/* --------------------------------------------------------------- examples */
function renderExamples(examples) {
  const host = $("example-groups");
  host.textContent = "";
  for (const [key, label] of GROUPS) {
    const rows = examples.filter((e) => e.group === key);
    if (!rows.length) continue;
    const box = document.createElement("div");
    box.className = "group";
    const h = document.createElement("h3");
    h.textContent = label;
    box.append(h);
    for (const ex of rows) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.dataset.id = ex.id;
      btn.append(document.createTextNode(ex.title));
      const em = document.createElement("em");
      const tag = document.createElement("span");
      tag.className = key === "hard" ? "tag-hard"
        : (ex.ensemble_label === "sarcastic" ? "tag-sarc" : "tag-non");
      tag.textContent = key === "hard" ? "disputed" : ex.ensemble_label;
      em.append(tag, document.createTextNode(
        " — " + ex.language + ", " + countContext(ex)));
      btn.append(em);
      btn.addEventListener("click", () => loadExample(ex));
      box.append(btn);
    }
    host.append(box);
  }
}

function countContext(ex) {
  const conv = (ex.submission ? 1 : 0) + ex.parents.length + ex.replies.length;
  return conv + " conversation item" + (conv === 1 ? "" : "s") +
    ", " + ex.history.length + " earlier post" + (ex.history.length === 1 ? "" : "s");
}

function loadExample(ex) {
  state.current = ex;
  state.last = null;                       // a new example is a fresh baseline
  document.querySelectorAll(".chip").forEach((c) =>
    c.setAttribute("aria-current", String(c.dataset.id === ex.id)));

  $("target").value = ex.text;
  $("sub-title").value = ex.submission ? ex.submission.title : "";
  $("sub-body").value = ex.submission ? ex.submission.selftext : "";
  fillTurns("parents", ex.parents);
  fillTurns("replies", ex.replies);
  fillHistory(ex.history);
  countChars();

  const note = $("notes");
  note.textContent = ex.note;
  note.hidden = !ex.note;

  run("loaded a corpus example");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ------------------------------------------------------------- context rows */
function addRow(kind, value) {
  const tpl = $(kind === "history" ? "tpl-hist" : "tpl-turn");
  const node = tpl.content.firstElementChild.cloneNode(true);
  if (value) {
    node.querySelector(".turn-text").value = value.text || "";
    if (kind === "history") {
      node.querySelector(".turn-hours").value = value.hours_ago;
    } else {
      node.querySelector(".turn-op").checked = !!value.is_submitter;
    }
  }
  node.querySelector(".drop").addEventListener("click", () => {
    node.remove();
    run("removed a context item");
  });
  $(kind).append(node);
  return node;
}

function fillTurns(kind, items) {
  $(kind).textContent = "";
  (items || []).forEach((it) => addRow(kind, it));
}

function fillHistory(items) {
  $("history").textContent = "";
  (items || []).forEach((it) => addRow("history", it));
}

function paintChannels() {
  $("ch-conv").classList.toggle("off", !isOn("conv"));
  $("ch-temp").classList.toggle("off", !isOn("temp"));
}

function isOn(ch) { return $("t-" + ch).getAttribute("aria-pressed") === "true"; }

function countChars() {
  $("counter").textContent = $("target").value.length + " / 2000";
}

/* -------------------------------------------------------------- prediction */
function collect() {
  const turns = (id) => Array.from($(id).children).map((li) => ({
    text: li.querySelector(".turn-text").value,
    is_submitter: li.querySelector(".turn-op").checked,
  })).filter((t) => t.text.trim());

  const history = Array.from($("history").children).map((li) => ({
    text: li.querySelector(".turn-text").value,
    hours_ago: parseFloat(li.querySelector(".turn-hours").value),
  })).filter((h) => h.text.trim());

  const title = $("sub-title").value.trim();
  const body = $("sub-body").value.trim();
  const text = $("target").value;

  // the leakage rule only applies while the box still holds the corpus row
  const sameRow = state.current && state.current.text === text.trim();

  return {
    text: text,
    submission: (title || body) ? { title: title, selftext: body } : null,
    parents: turns("parents"),
    replies: turns("replies"),
    history: history,
    use_conv: isOn("conv"),
    use_temp: isOn("temp"),
    use_ret: isOn("ret"),
    exclude_thread: sameRow ? state.current.submission_fullname : null,
  };
}

async function run(reason) {
  clearTimeout(state.timer);
  const body = collect();
  if (!body.text.trim()) {
    showError("Type a comment first.");
    return;
  }
  state.reason = reason || "";
  const seq = ++state.seq;
  $("run").disabled = true;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || "HTTP " + res.status);
    }
    const data = await res.json();
    if (seq !== state.seq) return;          // a newer request is already in flight
    render(data);
  } catch (err) {
    if (seq === state.seq) showError("The server could not score that. " + err.message);
  } finally {
    if (seq === state.seq) $("run").disabled = false;
  }
}

function render(r) {
  $("error").hidden = true;
  $("idle").hidden = true;
  $("result").hidden = false;

  const sarcastic = r.verdict === "sarcastic";
  $("result").style.setProperty("--v-color", sarcastic ? "var(--sarc)" : "var(--non)");
  $("pct").textContent = (r.prob * 100).toFixed(1);
  $("verdict").textContent = sarcastic ? "sarcastic" : "not sarcastic";
  $("figure-sub").textContent =
    "calibrated probability — the model calls it sarcastic at " + pct(r.threshold_cal) +
    " and above (raw probability " + pct(r.prob_raw) + ", margin " + r.margin.toFixed(3) + ")";

  $("fill").style.width = clampPct(r.prob);
  $("mark").style.left = clampPct(r.threshold_cal);

  const ghost = $("ghost");
  const delta = $("delta");
  if (state.last === null) {
    ghost.hidden = true;
    delta.textContent = "Change a toggle or edit the context to see this number move.";
  } else {
    const diff = (r.prob - state.last) * 100;
    const flat = Math.abs(diff) < 0.05;
    ghost.hidden = flat;
    ghost.style.left = clampPct(state.last);
    ghost.dataset.was = pct(state.last);
    delta.textContent = "";
    delta.append(document.createTextNode(state.reason ? capital(state.reason) + ": " : ""));
    const b = document.createElement("b");
    b.className = flat ? "" : (diff >= 0 ? "up" : "down");
    b.textContent = flat
      ? "no visible change"
      : (diff >= 0 ? "+" : "−") + Math.abs(diff).toFixed(1) + " points";
    delta.append(b, document.createTextNode(flat
      ? " — the probability stayed at " + pct(r.prob) + " (" + diff.toFixed(3) + " points)."
      : " — from " + pct(state.last) + " to " + pct(r.prob) + "."));
  }
  state.last = r.prob;

  // sentiment
  if (r.sentiment) {
    $("sentiment-pair").hidden = false;
    $("sentiment-missing").hidden = true;
    $("lit").textContent = r.sentiment.literal;
    $("int").textContent = r.sentiment.intended;
    $("int").title = r.sentiment.flagged
      ? "Flagged sarcastic, so the stage-2 head re-read it over the fused context features."
      : "Not flagged, so the intended reading is the literal one.";
  } else {
    $("sentiment-pair").hidden = true;
    $("sentiment-missing").hidden = false;
  }

  // gates — bars use the exact weights, the printed values are apportioned so
  // the three of them still add up to 1.00 after rounding
  const raw = CHANNELS.map((ch) => r.gates[ch] || 0);
  const shown = apportion(raw);
  CHANNELS.forEach((ch, i) => {
    $("gate-" + ch).style.width = (raw[i] * 100).toFixed(1) + "%";
    $("gate-" + ch + "-val").textContent = shown[i];
  });
  $("gate-conv-items").textContent = itemNote(r.channels_supplied.conv, "conversation item", isOn("conv"));
  $("gate-temp-items").textContent = itemNote(r.channels_supplied.temp, "earlier post", isOn("temp"));
  $("gate-ret-items").textContent = itemNote(r.channels_supplied.ret, "exemplar", isOn("ret"));

  // exemplars
  paintExemplars("ex-sarc", r.exemplars.sarcastic, r.channels_supplied.ret);
  paintExemplars("ex-non", r.exemplars.non_sarcastic, r.channels_supplied.ret);

  // server-side notes (clipped inputs, dropped history items)
  const notes = $("notes");
  const lines = (r.notes || []).slice();
  if (state.current && state.current.text === $("target").value.trim() && state.current.note) {
    lines.unshift(state.current.note);
  }
  notes.textContent = lines.join(" ");
  notes.hidden = !lines.length;
}

function itemNote(n, noun, on) {
  if (!on) return "switched off, so the channel was given no items";
  return n + " " + noun + (n === 1 ? "" : "s") + " supplied";
}

function paintExemplars(id, items, retCount) {
  const ol = $(id);
  ol.textContent = "";
  if (!items || !items.length) {
    const li = document.createElement("li");
    li.className = "ex-empty";
    li.textContent = retCount === 0
      ? "Retrieval is switched off, so no exemplars were retrieved."
      : "No exemplar survived the same-thread exclusion.";
    ol.append(li);
    return;
  }
  for (const it of items) {
    const li = document.createElement("li");
    const p = document.createElement("p");
    p.textContent = it.text;
    const meta = document.createElement("span");
    meta.className = "ex-meta";
    const b = document.createElement("b");
    b.textContent = "cosine " + it.similarity.toFixed(3);
    meta.append(b, document.createTextNode(", labelled " + it.language));
    li.append(p, meta);
    ol.append(li);
  }
}

function showError(message) {
  const box = $("error");
  box.textContent = message;
  box.hidden = false;
  $("idle").hidden = true;
}

/* Two-decimal shares that still sum to 1.00 (largest remainder). */
function apportion(values) {
  const total = values.reduce((a, b) => a + b, 0) || 1;
  const exact = values.map((v) => (v / total) * 100);
  const units = exact.map(Math.floor);
  let left = 100 - units.reduce((a, b) => a + b, 0);
  const order = exact
    .map((v, i) => [v - Math.floor(v), i])
    .sort((a, b) => b[0] - a[0]);
  for (let j = 0; left > 0 && j < order.length; j++, left--) units[order[j][1]] += 1;
  return units.map((u) => (u / 100).toFixed(2));
}

const pct = (p) => (p * 100).toFixed(1) + "%";
const clampPct = (p) => (Math.min(Math.max(p, 0), 1) * 100).toFixed(2) + "%";
const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);

init();
