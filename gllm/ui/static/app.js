"use strict";

const $ = (id) => document.getElementById(id);
let currentRun = null;
let es = null;

const esc = (s) => (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const pad2 = (n) => String(n).padStart(2, "0");

function setStatus(text, led) {
  $("rd-status").innerHTML = `<i class="led ${led}"></i>${text}`;
}

/* ---------- intake wiring ---------- */
function syncMode() {
  const mode = $("mode").value;
  $("seed-field").classList.toggle("hidden", mode === "generate");
  $("objective-field").classList.toggle("hidden", mode !== "improve");
  $("rd-mode").textContent = mode.toUpperCase();
}
function syncModel() {
  const m = $("model").value;
  $("orm-field").classList.toggle("hidden", m !== "OpenRouter");
  $("rd-model").textContent = m || "—";
}
function syncVericutHint() {
  const hasSetup = !!$("setup_id").value;
  $("vericut-hint").textContent = hasSetup
    ? "will simulate the selected setup"
    : "requires a machine setup — skipped otherwise";
}
$("mode").addEventListener("change", syncMode);
$("model").addEventListener("change", syncModel);
$("setup_id").addEventListener("change", syncVericutHint);

async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const { models, default: def } = await res.json();
    const sel = $("model");
    sel.innerHTML = "";
    for (const m of models) {
      const o = document.createElement("option");
      o.value = m; o.textContent = m.toUpperCase();
      if (m === def) o.selected = true;
      sel.appendChild(o);
    }
  } catch (_) {
    $("model").innerHTML = '<option value="OpenRouter">OPENROUTER</option>';
  }
  syncModel();
}

$("load-setups").addEventListener("click", async () => {
  const path = $("registry_path").value.trim();
  if (!path) { alert("Enter a registry path first."); return; }
  try {
    const res = await fetch(`/api/setups?registry_path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error((await res.json()).error || res.status);
    const setups = await res.json();
    const sel = $("setup_id");
    sel.innerHTML = '<option value="">— NONE · lint-only —</option>';
    for (const s of setups) {
      const o = document.createElement("option");
      o.value = s.id; o.textContent = s.description ? `${s.id} — ${s.description}` : s.id;
      sel.appendChild(o);
    }
    syncVericutHint();
  } catch (err) { alert("Failed to load setups: " + err.message); }
});

/* ---------- rendering ---------- */
function renderAttemptCard(a) {
  $("placeholder").classList.add("hidden");
  let card = $(`attempt-${a.number}`);
  if (!card) {
    card = document.createElement("div");
    card.className = "card";
    card.id = `attempt-${a.number}`;
    $("attempts").appendChild(card);
  }
  const ok = a.blocking_count === 0;
  const findings = a.findings.map((f) =>
    `<li class="sev-${f.severity}"><code>${esc(f.code)}</code>${
      f.line_number != null ? ` <span class="muted">L${f.line_number}</span>` : ""
    } · ${esc(f.message)}</li>`).join("");
  const obj = a.objective_value != null ? ` · obj ${a.objective_value.toFixed(2)}` : "";
  card.innerHTML = `
    <div class="card-head">
      <span class="badge ${ok ? "good" : "bad"}">${ok ? "CLEAN" : a.blocking_count + " BLOCKING"}</span>
      <span class="muted">${a.findings.length} finding(s)${obj}</span>
      <span class="seq">ATTEMPT ${pad2(a.number)}</span>
    </div>
    <div class="card-body">
      <div class="col">
        <h4>Findings</h4>
        <ul class="findings">${findings || '<li class="sev-info">no findings</li>'}</ul>
        <h4 style="margin-top:14px">Toolpath</h4>
        <img class="toolpath" src="/api/runs/${currentRun}/plot/${a.number}" alt="toolpath" onerror="this.style.display='none'" />
      </div>
      <div class="col">
        <h4>G-code</h4>
        <pre class="gcode">${esc(a.gcode)}</pre>
      </div>
    </div>`;
}

function vericutPanel(v) {
  if (!v) return "";
  const map = {
    vericut_accepted: ["ACCEPTED", "good"],
    vericut_rejected: ["REJECTED — simulation found collisions / errors", "bad"],
    vericut_unverified: ["UNVERIFIED — ran but produced no log", "warn"],
    vericut_unavailable: ["UNAVAILABLE — Vericut could not run", "warn"],
  };
  const [label, cls] = map[v.status] || [String(v.status || "").toUpperCase(), "warn"];
  const log = v.vericut || {};
  const findings = (log.findings && log.findings.length) ? log.findings : (v.repair_context || []);
  const stats = [];
  if (log.error_count != null) stats.push(`${log.error_count} error(s)`);
  if (log.warning_count != null) stats.push(`${log.warning_count} warning(s)`);
  if (log.cycle_time) stats.push(`cycle ${log.cycle_time}`);
  if (v.process_returncode != null) stats.push(`exit ${v.process_returncode}`);
  const items = findings.slice(0, 12).map((fd) =>
    `<li class="sev-${esc(fd.severity || "error")}"><code>L${fd.line_number ?? "—"}</code> ${esc(fd.message || "")}</li>`).join("");
  const more = findings.length > 12 ? `<li class="muted">…and ${findings.length - 12} more</li>` : "";
  return `
    <div class="vericut-panel ${cls}">
      <div class="vericut-head">
        <span class="badge ${cls}">VERICUT · ${esc(label)}</span>
        <span class="muted">${esc(stats.join("  ·  "))}</span>
      </div>
      ${items ? `<ul class="findings vericut-findings">${items}${more}</ul>` : ""}
    </div>`;
}

function renderFinal(result) {
  const ok = result.status.startsWith("passed") || result.status === "improved" || result.status === "accepted_vericut";
  const f = $("final");
  f.className = ok ? "ok" : "fail";
  f.innerHTML = `
    <h3><span class="badge ${ok ? "good" : "bad"}">${esc(result.status.replace(/_/g, " ").toUpperCase())}</span></h3>
    <div class="verdict-row">
      <span>action <code>${esc(result.operator_action)}</code></span>
      <span>best <code>attempt ${result.best_attempt_index ?? "—"}</code></span>
      <span>mode <code>${esc(result.mode)}</code></span>
    </div>
    ${vericutPanel(result.vericut)}
    <div class="actions">
      <button id="accept" class="btn">✓ ACCEPT</button>
      <button id="reject" class="btn ghost">✗ REJECT</button>
    </div>
    <div id="decision-msg" class="muted"></div>`;
  $("accept").addEventListener("click", () => decide("accept"));
  $("reject").addEventListener("click", () => decide("reject"));
}

async function decide(action) {
  try {
    await fetch(`/api/runs/${currentRun}/decision`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }),
    });
    $("decision-msg").textContent = `▸ recorded: ${action}`;
  } catch (_) { $("decision-msg").textContent = "failed to record decision"; }
}

function finishRun() {
  $("run-btn").disabled = false;
  $("run-btn").classList.remove("running");
  if (es) { es.close(); es = null; }
}

/* ---------- run ---------- */
$("run-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (es) { es.close(); es = null; }
  $("attempts").innerHTML = "";
  $("final").classList.add("hidden");
  $("placeholder").classList.add("hidden");

  const mode = $("mode").value;
  const payload = {
    prompt: $("prompt").value,
    mode,
    model_name: $("model").value,
    max_attempts: parseInt($("max_attempts").value, 10),
    run_vericut: $("run_vericut").checked,
    registry_path: $("registry_path").value.trim() || null,
    setup_id: $("setup_id").value || null,
  };
  if (payload.model_name === "OpenRouter") {
    const orm = $("openrouter_model_name").value.trim();
    if (orm) payload.openrouter_model_name = orm;
  }
  if (mode !== "generate") payload.seed_gcode = $("seed_gcode").value;
  if (mode === "improve") payload.objective = { metric: $("objective").value };

  $("run-btn").disabled = true;
  $("run-btn").classList.add("running");
  $("rd-attempt").textContent = "00";
  setStatus("DISPATCHING", "run");

  let runId;
  try {
    const res = await fetch("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).error || res.status);
    runId = (await res.json()).run_id;
  } catch (err) {
    setStatus("START FAILED", "fail");
    $("final").classList.remove("hidden");
    $("final").innerHTML = `<h3><span class="badge bad">ERROR</span></h3><div class="verdict-row">${esc(err.message)}</div>`;
    finishRun();
    return;
  }
  currentRun = runId;
  $("final").classList.remove("hidden");
  $("final").innerHTML = "";
  $("final").classList.add("hidden");

  es = new EventSource(`/api/runs/${runId}/events`);
  es.addEventListener("attempt_started", (ev) => {
    const d = JSON.parse(ev.data);
    $("rd-attempt").textContent = pad2(d.payload.attempt);
    setStatus("CUTTING", "run");
  });
  es.addEventListener("findings", (ev) => renderAttemptCard(JSON.parse(ev.data).attempt));
  es.addEventListener("best_updated", (ev) => {
    const a = JSON.parse(ev.data).attempt;
    document.querySelectorAll(".card").forEach((c) => c.classList.remove("best"));
    const card = $(`attempt-${a.number}`);
    if (card) card.classList.add("best");
  });
  es.addEventListener("vericut_started", () => setStatus("VERICUT SIM", "run"));
  es.addEventListener("error", (ev) => {
    if (!ev.data) return; // native connection close — ignore
    try {
      const d = JSON.parse(ev.data);
      setStatus("RUN ERROR", "fail");
      $("final").classList.remove("hidden");
      $("final").className = "fail";
      $("final").innerHTML = `<h3><span class="badge bad">RUN ERROR</span></h3><div class="verdict-row">${esc(d.payload && d.payload.message || "")}</div>`;
      finishRun();
    } catch (_) { /* ignore */ }
  });
  es.addEventListener("done", (ev) => {
    const d = JSON.parse(ev.data);
    const ok = d.result.status.startsWith("passed") || d.result.status === "improved" || d.result.status === "accepted_vericut";
    setStatus(ok ? "COMPLETE" : "REVIEW", ok ? "pass" : "fail");
    $("final").classList.remove("hidden");
    renderFinal(d.result);
    finishRun();
  });
});

/* ---------- init ---------- */
loadModels();
syncMode();
syncVericutHint();
