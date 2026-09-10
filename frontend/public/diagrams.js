// Animated SVG mechanism diagrams -- vanilla SVG + CSS/SMIL, no libraries.
// One diagram shape per METHOD, with system-specific variants where the
// mechanism itself actually differs (latent compute: Huginn's single loop
// vs BDH-CQ's dual loop vs Transformer's fixed stack; best-of-n: transparent
// majority vote vs BDH-CQ's opaque internal ranker; beam search: normal vs
// architecturally blocked).

let _diagramCleanup = null;

function clearDiagram() {
  if (_diagramCleanup) {
    _diagramCleanup();
    _diagramCleanup = null;
  }
}

function escapeHtml(s) {
  return (s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function svgWrap(inner, viewBox, caption, opts = {}) {
  const badges = {
    live: `<span class="diagram-badge live">live -- your question, real model, right now</span>`,
    real: `<span class="diagram-badge real">real sweep data</span>`,
  };
  const badge = badges[opts.tier] || `<span class="diagram-badge illustrative">illustrative -- not real model output</span>`;
  const rawBlock = opts.rawText
    ? `<details class="diagram-raw"><summary>View the real generated text</summary><pre>${escapeHtml(opts.rawText)}</pre></details>`
    : "";
  // viewBox is "minX minY width height" -- give the SVG explicit intrinsic
  // width/height attributes so CSS max-width/max-height can scale it down
  // proportionally instead of the browser falling back to a default
  // replaced-element box (300x150) that ignores the real aspect ratio.
  const [, , vbW, vbH] = viewBox.split(" ").map(Number);
  return `
    <div class="diagram-wrap">
      <div class="diagram-topbar">
        ${badge}
        <button class="diagram-replay" id="diagram-replay">↻ Replay</button>
      </div>
      <svg viewBox="${viewBox}" width="${vbW}" height="${vbH}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--text-dim)" />
          </marker>
        </defs>
        ${inner}
      </svg>
      <p class="diagram-caption">${caption}</p>
      ${rawBlock}
    </div>`;
}

// ---------- Best-of-N ----------

function bestOfNDiagram(variant, realExample) {
  const ys = [20, 60, 100, 140, 180];
  const isReal = variant !== "proprietary" && !!realExample;
  const tier = isReal ? (realExample.live ? "live" : "real") : null;

  let labels, winners, promptLabel, promptTitle, finalAnswer, finalColor, rawText;
  if (isReal) {
    const samples = realExample.samples.slice(0, 5);
    labels = samples.map((s) => s.prediction ?? "—");
    winners = samples
      .map((s, i) => (s.prediction != null && String(s.prediction) === String(realExample.majorityPrediction) ? i : -1))
      .filter((i) => i >= 0);
    promptLabel = "Prompt";
    promptTitle = realExample.question || realExample.problem_id;
    const hasGold = realExample.correct !== undefined && realExample.correct !== null;
    finalAnswer = hasGold
      ? `→ ${realExample.majorityPrediction ?? "—"} ${realExample.correct ? "✓" : "✗"}`
      : `→ ${realExample.majorityPrediction ?? "—"} (unverified -- no gold answer)`;
    finalColor = hasGold ? (realExample.correct ? "var(--green)" : "var(--red)") : "var(--accent)";
    rawText = samples[0]?.raw_text ? `[Sample 1 of ${realExample.samples.length}]\n\n${samples[0].raw_text}` : null;
  } else {
    labels = variant === "proprietary"
      ? ["cand A", "cand B", "cand C", "cand D", "cand E"]
      : ["18", "18", "11", "18", "2"];
    winners = variant === "proprietary" ? [] : [0, 1, 3];
    promptLabel = "Prompt";
    promptTitle = "Illustrative prompt";
    finalAnswer = variant === "proprietary" ? "→ output" : "→ 18";
    finalColor = "var(--green)";
    rawText = null;
  }

  let nodes = "";
  let inLines = "";
  let outLines = "";

  ys.slice(0, labels.length).forEach((y, i) => {
    const delay = (i * 0.12).toFixed(2);
    inLines += `<path class="anim-draw" d="M92,100 C160,100 160,${y} 258,${y}" fill="none" stroke="var(--border)" stroke-width="2" style="animation-delay:${delay}s" />`;
    const isWinner = winners.includes(i);
    const label = String(labels[i]).length > 8 ? String(labels[i]).slice(0, 7) + "…" : labels[i];
    nodes += `
      <g class="anim-fade${isWinner ? " winner-pulse" : ""} diag-node" style="animation-delay:${(0.5 + i * 0.1).toFixed(2)}s">
        <title>${variant === "proprietary" ? `Candidate ${i + 1}, ranked internally` : `Sample ${i + 1} -> answer "${escapeHtml(String(labels[i]))}"${isWinner ? " (part of the majority)" : ""}`}</title>
        <rect x="260" y="${y - 16}" width="80" height="32" rx="6"
          fill="${isWinner ? "var(--green-bg)" : "var(--bg-panel-2)"}"
          stroke="${isWinner ? "var(--green)" : "var(--border)"}" stroke-width="1.5" />
        <text x="300" y="${y + 5}" text-anchor="middle" font-size="13" fill="${isWinner ? "var(--green)" : "var(--text)"}">${escapeHtml(String(label))}</text>
      </g>`;
    outLines += `<path class="anim-draw" d="M342,${y} C410,${y} 410,100 478,100" fill="none" stroke="${isWinner ? "var(--green)" : "var(--border)"}" stroke-width="${isWinner ? 2 : 1.5}" style="animation-delay:${(0.7 + i * 0.08).toFixed(2)}s" />`;
  });

  const voteLabel = variant === "proprietary" ? "Internal ranker" : "Majority vote";
  const voteSub = variant === "proprietary" ? "🔒 opaque" : "";

  const inner = `
    <g class="anim-fade diag-node" style="animation-delay:0s">
      <title>${escapeHtml(promptTitle)}</title>
      <rect x="10" y="80" width="82" height="40" rx="8" fill="var(--bg-panel-2)" stroke="var(--border)" />
      <text x="51" y="104" text-anchor="middle" font-size="13" fill="var(--text)">${promptLabel}</text>
    </g>
    ${inLines}
    ${nodes}
    ${outLines}
    <g class="anim-fade diag-node" style="animation-delay:1.1s">
      <title>${variant === "proprietary" ? "BDH-CQ's ranker is proprietary -- we can observe outputs, not the ranking logic." : "The most frequent final answer across all N samples wins."}</title>
      <rect x="480" y="72" width="150" height="56" rx="10" fill="var(--purple-bg)" stroke="var(--purple)" stroke-width="1.5" />
      <text x="555" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="var(--purple)">${voteLabel}</text>
      <text x="555" y="116" text-anchor="middle" font-size="11" fill="var(--purple)">${voteSub}</text>
    </g>
    <g class="anim-fade" style="animation-delay:1.4s">
      <text x="700" y="104" text-anchor="end" font-size="15" font-weight="700" fill="${finalColor}">${escapeHtml(finalAnswer)}</text>
    </g>
  `;
  let caption;
  if (isReal && realExample.live) {
    caption = `Live: ${realExample.samples.length} samples generated just now for "${(realExample.question || "").slice(0, 60)}${(realExample.question || "").length > 60 ? "…" : ""}" (5 shown) on our real Qwen2.5-1.5B model.`;
  } else if (isReal) {
    caption = `Real: ${realExample.samples.length} samples generated for "${(realExample.question || "").slice(0, 60)}${(realExample.question || "").length > 60 ? "…" : ""}" (5 shown) &middot; gold answer: ${escapeHtml(String(realExample.gold))}`;
  } else {
    caption = variant === "proprietary"
      ? "N candidates generated; ranking happens inside a proprietary pipeline we cannot inspect."
      : "N independent samples at temperature &gt; 0; the most common final answer wins.";
  }
  return svgWrap(inner, "0 0 720 210", caption, { tier, rawText });
}

// ---------- Beam search ----------

function beamSearchDiagram(variant) {
  if (variant === "blocked") {
    const inner = `
      <g class="anim-fade" style="animation-delay:0s">
        <circle cx="330" cy="40" r="20" fill="var(--grey-bg)" stroke="var(--grey)" />
      </g>
      <path class="anim-draw" d="M330,60 L150,110" stroke="var(--grey)" stroke-dasharray="4 4" fill="none" style="animation-delay:.2s" />
      <path class="anim-draw" d="M330,60 L330,110" stroke="var(--grey)" stroke-dasharray="4 4" fill="none" style="animation-delay:.3s" />
      <path class="anim-draw" d="M330,60 L510,110" stroke="var(--grey)" stroke-dasharray="4 4" fill="none" style="animation-delay:.4s" />
      ${[150, 330, 510].map((x, i) => `
        <circle class="anim-fade" cx="${x}" cy="120" r="22" fill="var(--grey-bg)" stroke="var(--grey)" style="animation-delay:${(0.5 + i * 0.1).toFixed(2)}s" />
      `).join("")}
      <g class="anim-fade blocked-x" style="animation-delay:1s">
        <line x1="290" y1="150" x2="370" y2="190" stroke="var(--red)" stroke-width="4" stroke-linecap="round" />
        <line x1="370" y1="150" x2="290" y2="190" stroke="var(--red)" stroke-width="4" stroke-linecap="round" />
      </g>
      <g class="anim-fade" style="animation-delay:1.2s">
        <text x="330" y="225" text-anchor="middle" font-size="12" fill="var(--red)" font-weight="700">no scoreable step exists here</text>
      </g>
    `;
    return svgWrap(inner, "0 0 660 250", "H_r states are opaque latent vectors, never decoded into language -- there is no step to attach a verifier to.");
  }

  const kept1 = [0, 2];
  const level1 = [
    { x: 130, score: "0.8", kept: true },
    { x: 330, score: "0.3", kept: false },
    { x: 530, score: "0.6", kept: true },
  ];
  const level2 = [
    { x: 40, score: "0.9", kept: true, parent: 130 },
    { x: 220, score: "0.4", kept: false, parent: 130 },
    { x: 440, score: "0.5", kept: false, parent: 530 },
    { x: 620, score: "0.85", kept: true, parent: 530 },
  ];

  let svg = `
    <g class="anim-fade" style="animation-delay:0s">
      <rect x="290" y="10" width="80" height="34" rx="8" fill="var(--bg-panel-2)" stroke="var(--border)" />
      <text x="330" y="32" text-anchor="middle" font-size="12" fill="var(--text)">Step 0</text>
    </g>`;

  level1.forEach((n, i) => {
    svg += `<path class="anim-draw" d="M330,44 L${n.x + 40},80" stroke="${n.kept ? "var(--purple)" : "var(--border)"}" stroke-width="${n.kept ? 2 : 1.5}" fill="none" style="animation-delay:${(0.2 + i * 0.1).toFixed(2)}s" />`;
  });
  level1.forEach((n, i) => {
    svg += `
      <g class="anim-fade diag-node${n.kept ? "" : " prune-fade"}" style="animation-delay:${(0.5 + i * 0.1).toFixed(2)}s">
        <title>${n.kept ? "Kept -- among the top N/M scored branches" : "Pruned -- scored too low by the PRM, branch discarded"}</title>
        <rect x="${n.x}" y="80" width="80" height="34" rx="8" fill="${n.kept ? "var(--purple-bg)" : "var(--bg-panel-2)"}" stroke="${n.kept ? "var(--purple)" : "var(--border)"}" />
        <text x="${n.x + 40}" y="102" text-anchor="middle" font-size="12" fill="${n.kept ? "var(--purple)" : "var(--text-dim)"}" text-decoration="${n.kept ? "none" : "line-through"}">score ${n.score}</text>
      </g>`;
  });
  level2.forEach((n, i) => {
    svg += `<path class="anim-draw" d="M${n.parent + 40},114 L${n.x + 40},150" stroke="${n.kept ? "var(--purple)" : "var(--border)"}" stroke-width="${n.kept ? 2 : 1.5}" fill="none" style="animation-delay:${(0.9 + i * 0.1).toFixed(2)}s" />`;
  });
  level2.forEach((n, i) => {
    svg += `
      <g class="anim-fade${n.kept ? "" : " prune-fade"}" style="animation-delay:${(1.2 + i * 0.1).toFixed(2)}s">
        <rect x="${n.x}" y="150" width="80" height="34" rx="8" fill="${n.kept ? "var(--purple-bg)" : "var(--bg-panel-2)"}" stroke="${n.kept ? "var(--purple)" : "var(--border)"}" />
        <text x="${n.x + 40}" y="172" text-anchor="middle" font-size="12" fill="${n.kept ? "var(--purple)" : "var(--text-dim)"}" text-decoration="${n.kept ? "none" : "line-through"}">score ${n.score}</text>
      </g>`;
  });
  svg += `<g class="anim-fade" style="animation-delay:1.8s"><text x="330" y="215" text-anchor="middle" font-size="12" fill="var(--text-dim)">keep top N/M, branch M from each, repeat</text></g>`;

  return svgWrap(svg, "0 0 660 230", "A process reward model scores each partial solution; low-scoring branches are pruned before the next expansion.");
}

// ---------- Latent compute scaling ----------

function latentComputeDiagram(variant, realExample) {
  if (variant === "none") {
    const layers = [0, 1, 2, 3];
    let boxes = layers.map((i) => `
      <g class="anim-fade" style="animation-delay:${(i * 0.15).toFixed(2)}s">
        <rect x="60" y="${20 + i * 55}" width="140" height="38" rx="8" fill="var(--bg-panel-2)" stroke="var(--border)" />
        <text x="130" y="${44 + i * 55}" text-anchor="middle" font-size="12" fill="var(--text)">Layer ${i + 1}${i === 3 ? " (fixed)" : ""}</text>
      </g>
      ${i < 3 ? `<path d="M130,${58 + i * 55} L130,${75 + i * 55}" stroke="var(--text-dim)" marker-end="url(#arrow)" />` : ""}
    `).join("");

    const inner = `
      ${boxes}
      <g class="anim-fade" style="animation-delay:0.8s">
        <circle cx="290" cy="120" r="26" fill="none" stroke="var(--grey)" stroke-width="2" stroke-dasharray="5 4" />
        <path d="M290,94 A26,26 0 0 1 310,105" fill="none" stroke="var(--grey)" stroke-width="2" marker-end="url(#arrow)" />
        <line x1="270" y1="100" x2="310" y2="140" stroke="var(--red)" stroke-width="3" stroke-linecap="round" />
        <text x="290" y="165" text-anchor="middle" font-size="11" fill="var(--red)" font-weight="700">no depth dial</text>
      </g>
    `;
    return svgWrap(inner, "0 0 380 250", "Depth is fixed at training time. There is nothing here to turn up at inference.");
  }

  if (variant === "bdhcq") {
    const inner = `
      <g class="anim-fade" style="animation-delay:0s">
        <rect x="150" y="90" width="110" height="55" rx="10" fill="var(--purple-bg)" stroke="var(--purple)" stroke-width="1.5" />
        <text x="205" y="122" text-anchor="middle" font-size="12" font-weight="700" fill="var(--purple)">H_r</text>
        <text x="205" y="136" text-anchor="middle" font-size="9" fill="var(--purple)">reasoning</text>
      </g>
      <path d="M260,100 A45,35 0 1 1 258,135" fill="none" stroke="var(--purple)" stroke-width="2" marker-end="url(#arrow)" />
      <circle r="4" fill="var(--purple)">
        <animateMotion dur="1.4s" repeatCount="3" path="M260,100 A45,35 0 1 1 258,135" />
      </circle>

      <g class="anim-fade" style="animation-delay:.3s">
        <rect x="150" y="180" width="110" height="45" rx="10" fill="var(--bg-panel-2)" stroke="var(--accent)" stroke-width="1.5" />
        <text x="205" y="206" text-anchor="middle" font-size="12" font-weight="700" fill="var(--accent)">S_t</text>
        <text x="205" y="218" text-anchor="middle" font-size="9" fill="var(--accent)">memory</text>
      </g>
      <path d="M110,100 A130,90 0 1 0 108,205" fill="none" stroke="var(--accent)" stroke-width="2" marker-end="url(#arrow)" />
      <circle r="4" fill="var(--accent)">
        <animateMotion dur="4s" repeatCount="2" path="M110,100 A130,90 0 1 0 108,205" />
      </circle>

      <text x="205" y="20" text-anchor="middle" font-size="11" fill="var(--text-dim)">two recurrences, two timescales</text>
    `;
    return svgWrap(inner, "0 0 410 240", "Fast reasoning loop (H_r, purple) nested inside a slow memory recurrence (S_t, blue) that accumulates across demonstrations.");
  }

  // huginn: real Prelude / Recurrent-block / Coda architecture, Fig. 2 of
  // Geiping et al. -- (l_P, l_R, l_C) = (2, 4, 2), h = 5280 for the large model.
  const isReal = !!realExample;
  const tier = isReal ? (realExample.live ? "live" : "real") : null;
  const rMax = isReal ? realExample.budget : "r";
  const promptTitle = isReal ? (realExample.question || realExample.problem_id) : "Illustrative input";
  const hasGold = isReal && realExample.correct !== undefined && realExample.correct !== null;
  const outputLabel = !isReal
    ? "p (next-token probs)"
    : hasGold
      ? `${realExample.prediction ?? "(no answer extracted)"} ${realExample.correct ? "✓" : "✗"}`
      : `${realExample.prediction ?? "(no answer extracted)"} (unverified)`;
  const outputColor = !isReal ? "var(--text)" : hasGold ? (realExample.correct ? "var(--green)" : "var(--red)") : "var(--accent)";
  const rawText = isReal && realExample.rawOutput ? realExample.rawOutput : null;
  const loopPath = "M290,175 C340,175 340,110 300,95 C270,84 250,90 235,102";

  const inner = `
    <g class="anim-fade diag-node" style="animation-delay:0s">
      <title>${escapeHtml(promptTitle)}</title>
      <text x="130" y="16" text-anchor="middle" font-size="12" fill="var(--text-dim)">x (input tokens)</text>
    </g>
    <path d="M130,22 L130,42" stroke="var(--text-dim)" marker-end="url(#arrow)" />

    <g class="anim-fade diag-node" style="animation-delay:.1s">
      <title>Prelude P: embeds tokens as sigma*E(x), then l_P=2 transformer layers (Sec. 3.2)</title>
      <rect x="60" y="42" width="140" height="46" rx="8" fill="var(--accent)" fill-opacity="0.14" stroke="var(--accent)" stroke-width="1.5" />
      <text x="130" y="61" text-anchor="middle" font-size="12" font-weight="700" fill="var(--accent)">Prelude P (2 layers)</text>
      <text x="130" y="76" text-anchor="middle" font-size="10" fill="var(--accent)">e = &#963;&middot;E(x)</text>
    </g>
    <path d="M130,88 L130,102" stroke="var(--text-dim)" marker-end="url(#arrow)" />
    <path d="M170,65 C 260,65 260,102 235,110" fill="none" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#arrow)" />
    <text x="262" y="60" font-size="10" fill="var(--accent)">e injected every step</text>

    <rect x="30" y="100" width="270" height="130" rx="12" fill="none" stroke="var(--orange)" stroke-width="1.5" stroke-dasharray="2 3" />
    <text x="42" y="94" font-size="10" fill="var(--orange)" font-weight="700">shared recurrent block R</text>

    <g class="anim-fade diag-node" style="animation-delay:.2s">
      <title>Adapter A: R^2h -> R^h, maps concat(s_i-1, e) into hidden dim h (Sec. 3.2)</title>
      <rect x="50" y="112" width="105" height="30" rx="6" fill="var(--orange-bg)" stroke="var(--orange)" />
      <text x="102" y="132" text-anchor="middle" font-size="10" fill="var(--orange)">Adapter A[s,e]</text>
    </g>
    <path d="M102,142 L102,155" stroke="var(--orange)" marker-end="url(#arrow)" />
    <g class="anim-fade diag-node" style="animation-delay:.3s">
      <title>l_R=4 standard transformer layers, applied every recurrence (Sec. 3.2)</title>
      <rect x="50" y="157" width="105" height="30" rx="6" fill="var(--orange-bg)" stroke="var(--orange)" />
      <text x="102" y="177" text-anchor="middle" font-size="10" fill="var(--orange)">4 transformer layers</text>
    </g>
    <path d="M102,187 L102,200" stroke="var(--orange)" marker-end="url(#arrow)" />
    <g class="anim-fade diag-node" style="animation-delay:.4s">
      <title>RMSNorm rescales the block output before it becomes s_i (Sec. 3.2)</title>
      <rect x="50" y="202" width="105" height="24" rx="6" fill="var(--orange-bg)" stroke="var(--orange)" />
      <text x="102" y="218" text-anchor="middle" font-size="10" fill="var(--orange)">RMSNorm -&gt; s_i</text>
    </g>

    <path d="${loopPath}" fill="none" stroke="var(--orange)" stroke-width="2" marker-end="url(#arrow)" />
    <circle r="5" fill="var(--orange)">
      <animateMotion dur="1.6s" repeatCount="3" path="${loopPath}" />
    </circle>
    <text id="latent-counter" x="235" y="215" font-size="12" fill="var(--orange)" font-weight="700">i = 1 of ${rMax}</text>

    <path d="M130,230 L130,246" stroke="var(--text-dim)" marker-end="url(#arrow)" />
    <g class="anim-fade diag-node" style="animation-delay:.6s">
      <title>Coda C: l_C=2 layers, RMSNorm, then projection to vocabulary via tied embeddings E^T (Sec. 3.2)</title>
      <rect x="60" y="248" width="140" height="46" rx="8" fill="var(--purple-bg)" stroke="var(--purple)" stroke-width="1.5" />
      <text x="130" y="267" text-anchor="middle" font-size="12" font-weight="700" fill="var(--purple)">Coda C (2 layers)</text>
      <text x="130" y="282" text-anchor="middle" font-size="10" fill="var(--purple)">RMSNorm + tied E^T</text>
    </g>
    <path d="M130,294 L130,310" stroke="var(--text-dim)" marker-end="url(#arrow)" />
    <text x="130" y="326" text-anchor="middle" font-size="13" font-weight="700" fill="${outputColor}">${escapeHtml(outputLabel)}</text>
  `;
  let caption;
  if (isReal && realExample.live) {
    caption = `Live: r=1 (the only depth that fits in 8GB VRAM) on "${(realExample.question || "").slice(0, 50)}${(realExample.question || "").length > 50 ? "…" : ""}" -- run just now on our real Huginn-0125.`;
  } else if (isReal) {
    caption = `Real: r=${realExample.budget} on "${(realExample.question || "").slice(0, 55)}${(realExample.question || "").length > 55 ? "…" : ""}" &middot; gold: ${escapeHtml(String(realExample.gold))} (only r=1 completed on our GPU -- see Results panel)`;
  } else {
    caption = "Architecture per Geiping et al. Fig. 2: Prelude embeds once, the recurrent block iterates r times, Coda decodes once.";
  }
  return svgWrap(inner, "0 0 340 340", caption, { tier, rawText });
}

// ---------- Budget forcing ----------

function budgetForcingDiagram(variant, realExample) {
  const isReal = variant === "vote" && !!realExample;
  const tier = isReal ? (realExample.live ? "live" : "real") : null;

  if (variant === "blocked") {
    const inner = `
      <g class="anim-fade" style="animation-delay:0s">
        <rect x="10" y="80" width="90" height="40" rx="8" fill="var(--bg-panel-2)" stroke="var(--border)" />
        <text x="55" y="104" text-anchor="middle" font-size="12" fill="var(--text)">H_0 ... H_R</text>
      </g>
      <text x="55" y="66" text-anchor="middle" font-size="10" fill="var(--text-dim)">latent, never decoded</text>
      <path class="anim-draw" d="M100,100 L220,100" stroke="var(--grey)" stroke-dasharray="4 4" fill="none" style="animation-delay:.2s" />
      <g class="anim-fade blocked-x" style="animation-delay:.6s">
        <line x1="230" y1="80" x2="270" y2="120" stroke="var(--red)" stroke-width="4" stroke-linecap="round" />
        <line x1="270" y1="80" x2="230" y2="120" stroke="var(--red)" stroke-width="4" stroke-linecap="round" />
      </g>
      <text x="250" y="140" text-anchor="middle" font-size="11" fill="var(--red)" font-weight="700">no "Wait" to inject here</text>
      <text x="250" y="156" text-anchor="middle" font-size="10" fill="var(--text-dim)">no end-of-thinking token exists</text>
    `;
    return svgWrap(inner, "0 0 500 180", "Budget forcing suppresses a decoded end-of-thinking token and appends text. BDH-CQ's H_r states are never decoded, so there is nothing to suppress and nowhere to inject text.");
  }

  const segments = isReal ? Math.min(realExample.forcedUsed ?? 0, 3) : 2;
  const overflow = isReal && (realExample.forcedUsed ?? 0) > 3;
  let x = 10;
  let svg = "";
  const promptTitle = isReal ? (realExample.question || realExample.problem_id) : "Illustrative prompt";

  svg += `<g class="anim-fade diag-node" style="animation-delay:0s"><title>${escapeHtml(promptTitle)}</title>
    <rect x="${x}" y="70" width="70" height="36" rx="6" fill="var(--bg-panel-2)" stroke="var(--border)" />
    <text x="${x + 35}" y="92" text-anchor="middle" font-size="11" fill="var(--text)">Prompt</text></g>`;
  x += 80;

  for (let i = 0; i <= segments; i++) {
    const delay = (0.2 + i * 0.35).toFixed(2);
    svg += `<path class="anim-draw" d="M${x},88 L${x + 20},88" stroke="var(--border)" stroke-width="2" style="animation-delay:${delay}s" />`;
    x += 20;
    svg += `<g class="anim-fade diag-node" style="animation-delay:${delay}s">
      <title>Reasoning segment ${i + 1}</title>
      <rect x="${x}" y="72" width="70" height="32" rx="16" fill="var(--bg-panel-2)" stroke="var(--border)" stroke-dasharray="3 2" />
      <text x="${x + 35}" y="92" text-anchor="middle" font-size="10" fill="var(--text-dim)">thinking…</text>
    </g>`;
    x += 80;
    if (i < segments) {
      svg += `<path class="anim-draw" d="M${x - 8},88 L${x + 12},88" stroke="var(--red)" stroke-width="2" style="animation-delay:${delay}s" />`;
      svg += `<g class="anim-fade diag-node" style="animation-delay:${(0.2 + (i + 0.5) * 0.35).toFixed(2)}s">
        <title>Stop suppressed -- "Wait" appended to force continued reasoning</title>
        <rect x="${x + 4}" y="66" width="54" height="26" rx="13" fill="var(--orange-bg)" stroke="var(--orange)" />
        <text x="${x + 31}" y="83" text-anchor="middle" font-size="10" font-weight="700" fill="var(--orange)">Wait</text>
      </g>`;
      x += 66;
    }
  }
  if (overflow) {
    svg += `<text x="${x}" y="92" font-size="12" fill="var(--text-dim)">+${realExample.forcedUsed - 3} more</text>`;
    x += 60;
  }
  svg += `<path class="anim-draw" d="M${x},88 L${x + 20},88" stroke="var(--border)" stroke-width="2" style="animation-delay:${(0.2 + (segments + 1) * 0.35).toFixed(2)}s" />`;
  x += 20;
  const hasGold = isReal && realExample.correct !== undefined && realExample.correct !== null;
  const answerLabel = !isReal ? "answer" : hasGold ? `${realExample.prediction ?? "—"} ${realExample.correct ? "✓" : "✗"}` : `${realExample.prediction ?? "—"} (unverified)`;
  const answerColor = !isReal ? "var(--text)" : hasGold ? (realExample.correct ? "var(--green)" : "var(--red)") : "var(--accent)";
  svg += `<g class="anim-fade diag-node" style="animation-delay:${(0.5 + (segments + 1) * 0.35).toFixed(2)}s">
    <title>Final answer, taken after all forced continuations are used</title>
    <rect x="${x}" y="68" width="90" height="40" rx="8" fill="var(--purple-bg)" stroke="var(--purple)" stroke-width="1.5" />
    <text x="${x + 45}" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="${answerColor}">${escapeHtml(answerLabel)}</text>
  </g>`;
  const viewW = x + 100;

  let caption;
  if (isReal && realExample.live) {
    caption = `Live: ${realExample.forcedUsed} forced "Wait"${realExample.forcedUsed === 1 ? "" : "s"} on "${(realExample.question || "").slice(0, 45)}${(realExample.question || "").length > 45 ? "…" : ""}" -- run just now on our real Qwen2.5-1.5B model.`;
  } else if (isReal) {
    caption = `Real: ${realExample.forcedUsed} forced "Wait"${realExample.forcedUsed === 1 ? "" : "s"} on "${(realExample.question || "").slice(0, 45)}${(realExample.question || "").length > 45 ? "…" : ""}" &middot; gold: ${escapeHtml(String(realExample.gold))}`;
  } else {
    caption = "Each time the model tries to stop, we suppress that stop and append \"Wait\" -- up to a fixed budget -- before accepting its final answer.";
  }
  return svgWrap(svg, `0 0 ${viewW} 160`, caption, { tier, rawText: isReal ? realExample.rawOutput : null });
}

// ---------- dispatch ----------

function renderMechanismDiagram(container, systemId, methodId, cell, realExample) {
  clearDiagram();
  let html;
  let animateCounter = false;

  if (methodId === "best_of_n") {
    html = bestOfNDiagram(systemId === "bdh_cq" ? "proprietary" : "vote", realExample);
  } else if (methodId === "beam_search") {
    html = beamSearchDiagram(cell.blocked ? "blocked" : "normal");
  } else if (methodId === "budget_forcing") {
    html = budgetForcingDiagram(cell.blocked ? "blocked" : "vote", realExample);
  } else {
    const variant = systemId === "transformer" ? "none" : systemId === "bdh_cq" ? "bdhcq" : "huginn";
    html = latentComputeDiagram(variant, realExample);
    // Only animate the iteration counter when it's not standing in for a
    // real, specific r value -- a real r=1 result gets a fixed label, never
    // a fake multi-step cycle, so we never imply we ran more than we did.
    animateCounter = variant === "huginn" && !realExample;
  }

  container.innerHTML = html;

  const replayBtn = container.querySelector("#diagram-replay");
  const svgEl = container.querySelector("svg");
  const replay = () => {
    const clone = svgEl.cloneNode(true);
    svgEl.replaceWith(clone);
  };
  if (replayBtn) replayBtn.addEventListener("click", replay);

  if (animateCounter) {
    // Plays one finite cycle (this is an illustrative diagram, no real
    // model backs it here -- Huginn has a real live control elsewhere for
    // this exact cell) then stops, rather than looping forever.
    let i = 1;
    const counterEl = () => container.querySelector("#latent-counter");
    const interval = setInterval(() => {
      i += 1;
      const el = counterEl();
      if (el) el.textContent = `i = ${i} of r`;
      if (i >= 6) clearInterval(interval);
    }, 900);
    _diagramCleanup = () => clearInterval(interval);
  }
}
