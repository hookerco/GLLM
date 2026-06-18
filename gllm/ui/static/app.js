"use strict";

const $ = (id) => document.getElementById(id);
let currentRun = null;
let es = null;

function escapeHtml(s) {
  return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function setVisibility() {
  const mode = $("mode").value;
  $("seed-field").classList.toggle("hidden", mode === "generate");
  $("objective-field").classList.toggle("hidden", mode !== "improve");
}
$("mode").addEventListener("change", setVisibility);
setVisibility();

$("load-setups").addEventListener("click", async () => {
  const path = $("registry_path").value.trim();
  if (!path) { alert("Enter a registry path first."); return; }
  try {
    const res = await fetch(`/api/setups?registry_path=${encodeURIComponent(path)}`);
    if (!res.ok) { const e = await res.json(); throw new Error(e.error || res.status); }
    const setups = await res.json();
    const sel = $("setup_id");
    sel.innerHTML = '<option value="">— none (lint-only) —</option>';
    for (const s of setups) {
      const o = document.createElement("option");
      o.value = s.id;
      o.textContent = s.description ? `${s.id} — ${s.description}` : s.id;
      sel.appendChild(o);
    }
    banner(`Loaded ${setups.length} setup(s).`, "good");
  } catch (err) {
    alert("Failed to load setups: " + err.message);
  }
});

function statusClass(blocking) { return blocking > 0 ? "bad" : "good"; }

function banner(text, cls) {
  const b = $("status-banner");
  b.className = cls || "";
  b.classList.remove("hidden");
  b.textContent = text;
}

function renderAttemptCard(a) {
  let card = document.getElementById(`attempt-${a.number}`);
  if (!card) {
    card = document.createElement("div");
    card.className = "card";
    card.id = `attempt-${a.number}`;
    $("attempts").appendChild(card);
  }
  const findings = a.findings
    .map(
      (f) =>
        `<li class="sev-${f.severity}"><code>${escapeHtml(f.code)}</code>${
          f.line_number != null ? ` (line ${f.line_number})` : ""
        }: ${escapeHtml(f.message)}</li>`
    )
    .join("");
  const obj = a.objective_value != null ? ` · objective ${a.objective_value.toFixed(2)}` : "";
  card.innerHTML = `
    <div class="card-head">
      <span class="badge ${statusClass(a.blocking_count)}">Attempt ${a.number}</span>
      <span class="muted">${a.blocking_count} blocking · ${a.findings.length} findings${obj}</span>
    </div>
    <div class="card-body">
      <div class="col">
        <h4>Findings</h4>
        <ul class="findings">${findings || "<li class='sev-info'>none</li>"}</ul>
        <h4>Toolpath</h4>
        <img class="toolpath" src="/api/runs/${currentRun}/plot/${a.number}" alt="toolpath"
             onerror="this.style.display='none'" />
      </div>
      <div class="col">
        <h4>G-code</h4>
        <pre class="gcode">${escapeHtml(a.gcode)}</pre>
      </div>
    </div>`;
}

function renderFinal(result) {
  const f = $("final");
  f.classList.remove("hidden");
  const ok =
    result.status.startsWith("passed") ||
    result.status === "improved" ||
    result.status === "accepted_vericut";
  const v = result.vericut
    ? `<div>Vericut: <code>${escapeHtml(JSON.stringify(result.vericut.status || result.vericut))}</code></div>`
    : "";
  f.innerHTML = `
    <h3>Result: <span class="badge ${ok ? "good" : "bad"}">${escapeHtml(result.status)}</span></h3>
    <div class="muted">operator action: <code>${escapeHtml(result.operator_action)}</code>
      · best attempt: ${result.best_attempt_index ?? "—"}</div>
    ${v}
    <div class="actions">
      <button id="accept">✓ Accept</button>
      <button id="reject" class="ghost">✗ Reject</button>
    </div>
    <div id="decision-msg" class="muted"></div>`;
  $("accept").addEventListener("click", () => decide("accept"));
  $("reject").addEventListener("click", () => decide("reject"));
}

async function decide(action) {
  try {
    await fetch(`/api/runs/${currentRun}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    $("decision-msg").textContent = `Recorded: ${action}.`;
  } catch (err) {
    $("decision-msg").textContent = "Failed to record decision.";
  }
}

$("run-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (es) { es.close(); es = null; }
  $("attempts").innerHTML = "";
  $("final").classList.add("hidden");

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
  if (mode !== "generate") payload.seed_gcode = $("seed_gcode").value;
  if (mode === "improve") payload.objective = { metric: $("objective").value };

  banner("Starting run…", "running");
  let runId;
  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) { const er = await res.json(); throw new Error(er.error || res.status); }
    runId = (await res.json()).run_id;
  } catch (err) {
    banner("Failed to start: " + err.message, "bad");
    return;
  }
  currentRun = runId;

  es = new EventSource(`/api/runs/${runId}/events`);
  es.addEventListener("attempt_started", (ev) => {
    const d = JSON.parse(ev.data);
    banner(`Attempt ${d.payload.attempt} running…`, "running");
  });
  es.addEventListener("findings", (ev) => renderAttemptCard(JSON.parse(ev.data).attempt));
  es.addEventListener("best_updated", (ev) => {
    const a = JSON.parse(ev.data).attempt;
    document.querySelectorAll(".card").forEach((c) => c.classList.remove("best"));
    const card = document.getElementById(`attempt-${a.number}`);
    if (card) card.classList.add("best");
  });
  es.addEventListener("error", (ev) => {
    if (!ev.data) return; // native connection error (e.g. stream closed) — ignore
    try {
      const d = JSON.parse(ev.data);
      banner("Run error: " + (d.payload && d.payload.message ? d.payload.message : ""), "bad");
    } catch (_) { /* ignore */ }
  });
  es.addEventListener("done", (ev) => {
    const d = JSON.parse(ev.data);
    banner("Run complete.", "good");
    renderFinal(d.result);
    if (es) { es.close(); es = null; }
  });
});
