// Real live-model execution, streamed token-by-token via SSE from
// backend/server.py. Nothing here animates unless a real generate() call is
// actually producing that token right now -- no decorative/idle loops.
//
// Only wired up for the two systems with a real, locally-downloaded model:
// Transformer (Qwen2.5-1.5B-Instruct) and Huginn-0125. BDH-CQ has no public
// checkpoint, so it never gets a live control -- see app.js's static
// mechanism panel for that cell instead.

const LIVE_CAPABLE = {
  "transformer__best_of_n": { kind: "best_of_n", param: "n", min: 1, max: 5, default: 3, label: "N (samples)" },
  "transformer__budget_forcing": { kind: "budget_forcing", param: "budget", min: 0, max: 2, default: 1, label: "Forced continuations" },
  "huginn__latent_compute": { kind: "huginn", param: "r", min: 1, max: 32, default: 4, step_values: [1, 4, 8, 16, 32], label: "r (recurrence depth)" },
};

function liveKeyFor(systemId, methodId) {
  return `${systemId}__${methodId}`;
}

function isLiveCapable(systemId, methodId) {
  return !!LIVE_CAPABLE[liveKeyFor(systemId, methodId)];
}

function liveReady(systemId, methodId) {
  const spec = LIVE_CAPABLE[liveKeyFor(systemId, methodId)];
  if (!spec) return false;
  if (spec.kind === "huginn") return !!BACKEND_HEALTH?.huginn_loaded;
  return !!BACKEND_HEALTH?.qwen_loaded;
}

// ---------- shared live-console UI pieces ----------

function makeSampleCard(label) {
  const card = el("div", { className: "live-sample-card" });
  card.appendChild(el("div", { className: "live-sample-label", text: label }));
  const text = el("pre", { className: "live-sample-text" });
  card.appendChild(text);
  const status = el("div", { className: "live-sample-status", text: "waiting…" });
  card.appendChild(status);
  return { card, text, status };
}

function autoscroll(pre) {
  pre.scrollTop = pre.scrollHeight;
}

// ---------- best-of-n ----------

function renderLiveBestOfN(container, question, n) {
  container.innerHTML = "";
  const badge = el("div", { className: "diagram-badge live", text: "live -- your question, real model, right now" });
  container.appendChild(badge);

  const grid = el("div", { className: "live-sample-grid" });
  const cards = [];
  for (let i = 0; i < n; i++) {
    const c = makeSampleCard(`Sample ${i + 1}`);
    cards.push(c);
    grid.appendChild(c.card);
  }
  container.appendChild(grid);

  const voteBox = el("div", { className: "live-vote-box", text: "Majority vote pending…" });
  container.appendChild(voteBox);

  const url = `${LIVE_BACKEND_URL}/stream/best_of_n?question=${encodeURIComponent(question)}&n=${n}`;
  const es = new EventSource(url);
  _liveCleanup = () => es.close();

  es.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === "token") {
      const c = cards[data.sample];
      if (c) {
        c.status.textContent = "streaming…";
        c.status.classList.add("active");
        c.text.textContent += data.text;
        autoscroll(c.text);
      }
    } else if (data.type === "sample_done") {
      const c = cards[data.sample];
      if (c) {
        c.status.textContent = `answer: ${data.prediction ?? "—"}`;
        c.status.classList.remove("active");
        c.card.classList.add("done");
      }
    } else if (data.type === "done") {
      voteBox.textContent = `Majority vote: ${data.majority_prediction ?? "—"} (unverified -- you typed this question, no gold answer exists)`;
      cards.forEach((c, i) => {
        const pred = c.status.textContent.replace("answer: ", "");
        if (pred === String(data.majority_prediction)) c.card.classList.add("winner");
      });
      es.close();
    } else if (data.type === "error") {
      voteBox.textContent = `Error: ${data.message}`;
      es.close();
    }
  };
  es.onerror = () => {
    voteBox.textContent = "Connection to live backend lost.";
    es.close();
  };
}

// ---------- budget forcing ----------

function renderLiveBudgetForcing(container, question, budget) {
  container.innerHTML = "";
  const badge = el("div", { className: "diagram-badge live", text: "live -- your question, real model, right now" });
  container.appendChild(badge);

  const timeline = el("div", { className: "live-timeline" });
  container.appendChild(timeline);
  const resultBox = el("div", { className: "live-vote-box", text: "Reasoning…" });
  container.appendChild(resultBox);

  let segIndex = 0;
  const makeSegment = () => {
    const seg = el("div", { className: "live-segment" });
    seg.appendChild(el("div", { className: "live-sample-label", text: `Reasoning segment ${segIndex + 1}` }));
    const text = el("pre", { className: "live-sample-text" });
    seg.appendChild(text);
    timeline.appendChild(seg);
    segIndex++;
    return text;
  };

  let currentText = makeSegment();

  const url = `${LIVE_BACKEND_URL}/stream/budget_forcing?question=${encodeURIComponent(question)}&budget=${budget}`;
  const es = new EventSource(url);
  _liveCleanup = () => es.close();

  es.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === "token") {
      currentText.textContent += data.text;
      autoscroll(currentText);
    } else if (data.type === "wait_injected") {
      const marker = el("div", { className: "live-wait-marker", text: '⟲ Stop suppressed -- injecting "Wait" to force continued reasoning' });
      timeline.appendChild(marker);
      currentText = makeSegment();
    } else if (data.type === "done") {
      resultBox.textContent = `Final answer: ${data.prediction ?? "—"} (unverified -- you typed this question, no gold answer exists) · ${data.forced_continuations_used} forced continuation${data.forced_continuations_used === 1 ? "" : "s"} used`;
      es.close();
    } else if (data.type === "error") {
      resultBox.textContent = `Error: ${data.message}`;
      es.close();
    }
  };
  es.onerror = () => {
    resultBox.textContent = "Connection to live backend lost.";
    es.close();
  };
}

// ---------- huginn ----------

function renderLiveHuginn(container, question, r, task) {
  container.innerHTML = "";
  const badge = el("div", { className: "diagram-badge live", text: `live -- r=${r}, real model, right now` });
  container.appendChild(badge);

  const note = el("div", { className: "live-hint", text: r >= 16 ? "Higher r means more recurrent iterations per token -- this can take 30-90s. Watching real compute happen, not a canned wait." : "Watching Huginn's real forward pass, iterated r times per token." });
  container.appendChild(note);

  const card = makeSampleCard(`Huginn output (r=${r})`);
  container.appendChild(card.card);
  const resultBox = el("div", { className: "live-vote-box", text: "Thinking…" });
  container.appendChild(resultBox);

  const url = `${LIVE_BACKEND_URL}/stream/huginn?question=${encodeURIComponent(question)}&r=${r}&task=${task}`;
  const es = new EventSource(url);
  _liveCleanup = () => es.close();

  es.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === "token") {
      card.status.textContent = "streaming…";
      card.status.classList.add("active");
      card.text.textContent += data.text;
      autoscroll(card.text);
    } else if (data.type === "done") {
      card.status.textContent = "done";
      card.status.classList.remove("active");
      card.card.classList.add("done");
      resultBox.textContent = `Extracted answer: ${data.prediction ?? "(none found)"} (unverified -- you typed this question, no gold answer exists)`;
      es.close();
    } else if (data.type === "error") {
      resultBox.textContent = `Error: ${data.message}`;
      es.close();
    }
  };
  es.onerror = () => {
    resultBox.textContent = "Connection to live backend lost.";
    es.close();
  };
}

// ---------- controls: question input + real-parameter slider + run button ----------

let _liveCleanup = null;

function renderLiveControls(hostContainer, diagramContainer, systemId, methodId) {
  const spec = LIVE_CAPABLE[liveKeyFor(systemId, methodId)];
  if (!spec) return;

  const ready = liveReady(systemId, methodId);
  const wrap = el("div", { className: "live-controls" });

  const questionRow = el("div", { className: "live-controls-row" });
  const input = el("input", {
    attrs: { type: "text", placeholder: "Type a math word problem, e.g. \"A store sells pens for $3. If Amy buys 4 pens, how much does she pay?\"" },
  });
  questionRow.appendChild(input);
  wrap.appendChild(questionRow);

  const sliderRow = el("div", { className: "live-controls-row" });
  const sliderLabel = el("span", { className: "live-slider-label" });
  sliderRow.appendChild(sliderLabel);

  let slider;
  if (spec.step_values) {
    slider = el("input", { attrs: { type: "range", min: 0, max: spec.step_values.length - 1, step: 1, value: spec.step_values.indexOf(spec.default) } });
  } else {
    slider = el("input", { attrs: { type: "range", min: spec.min, max: spec.max, step: 1, value: spec.default } });
  }
  const currentValue = () => (spec.step_values ? spec.step_values[Number(slider.value)] : Number(slider.value));
  const updateLabel = () => (sliderLabel.textContent = `${spec.label}: ${currentValue()}`);
  slider.addEventListener("input", updateLabel);
  updateLabel();
  sliderRow.appendChild(slider);
  wrap.appendChild(sliderRow);

  const btn = el("button", { text: ready ? "Run on the real model" : "Live backend offline" });
  if (!ready) btn.disabled = true;
  wrap.appendChild(btn);

  const hint = el("div", { className: "live-hint" });
  if (!ready) hint.textContent = "Start it with: python backend/server.py";
  wrap.appendChild(hint);

  btn.addEventListener("click", () => {
    const question = input.value.trim();
    if (!question) {
      hint.textContent = "Type a question first.";
      return;
    }
    if (_liveCleanup) _liveCleanup();
    const value = currentValue();
    if (spec.kind === "best_of_n") renderLiveBestOfN(diagramContainer, question, value);
    else if (spec.kind === "budget_forcing") renderLiveBudgetForcing(diagramContainer, question, value);
    else if (spec.kind === "huginn") renderLiveHuginn(diagramContainer, question, value, "gsm8k");
  });

  hostContainer.appendChild(wrap);
}
