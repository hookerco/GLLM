# Prompt-Driven G-Code Generate / Improve / Repair Loop — Design

**Date:** 2026-06-18
**Status:** Approved (design); ready for implementation planning

## 1. Goal

Let a user **generate, improve, and repair G-code on the fly from a prompt**, as a
single **autonomous closed loop**: the user submits one prompt (optionally with an
existing program and an objective), the system generates a candidate, validates it,
and auto-repairs it across attempts until it passes (or a stop limit is hit), then
surfaces the result for accept/reject.

The work is primarily **wiring and unifying machinery that already exists** in the
repo, not greenfield. `gllm/proof/runner.py:run_proof_scenario` already implements a
static-first → repair → Vericut-final-gate loop; it just has no real LLM wired into
its `candidate_generator` hook, returns the *last* attempt instead of the *best*, has
no improve mode, ignores the `gcode_utils` lint validators, and has no budget or
streaming.

## 2. Decisions (locked)

- **Interaction model:** autonomous closed loop (not conversational, not per-attempt
  approval gates). The loop runs to completion; the UI streams progress.
- **Validation gating:** static-first. Fast local **lint** + machine-policy
  **`static_checks.check_gcode`** gate every attempt. **Vericut runs once** on the
  converged best candidate as a final gate, and **auto-skips** when no license/assets
  are present (`validate_setup_paths`). The loop is fully runnable without Vericut.
- **Entry points (one loop, three starting states):**
  - *generate* — prompt only → fresh candidate.
  - *repair* — prompt + existing (broken) g-code → validate the seed, then repair.
  - *improve* — prompt + existing (passing) g-code + objective → optimize toward the
    objective while keeping all gates green.
- **Stop & return:** converge-or-cap with **best-so-far**. Stop on: converged · max
  attempts (default ~4) · no-improvement for `patience` attempts (default 2) ·
  token/cost budget. Always return the best-scored candidate, with an evidence packet
  explaining anything unresolved.
- **Architecture:** **unified, UI-agnostic loop engine** (`gllm/loop/`) that both the
  corpus CLI and the UI drive. `run_proof_scenario` becomes a thin adapter over it.
- **Front end:** replace Streamlit with **NiceGUI** over a **FastAPI SSE event stream**.
  Engine stays UI-agnostic so the front end is swappable.
- **Build order:** engine first (headless, drives corpus CLI), then service, then UI.

## 3. System shape

Three independently testable layers talking through narrow interfaces:

```
gllm/ui/      (NiceGUI)   prompt · mode · setup · live attempts · gcode+diff ·
                          toolpath · findings · accept/reject
      ▲  LoopEvent stream (SSE)
gllm/service/ (FastAPI)   POST /runs → stream LoopEvent ; GET /runs/{id} → packet
      ▲  plain Python: LoopController.stream(LoopRequest)
gllm/loop/    (engine, no UI/HTTP imports)
   controller.py  LoopController — converge-or-cap + best-so-far
   generator.py   CandidateGenerator (LLM) — wraps model_utils + clean_gcode
   validators.py  LintValidator | StaticPolicyValidator | VericutValidator → [Finding]
   findings.py    Finding (normalizes all three finding shapes)
   scoring.py     score(findings, objective) → float ; objective estimators
   packet.py      LoopResult → EvidencePacket
      ▲  ScenarioRequest → LoopRequest (thin adapter)
gllm/proof/runner.py + corpus_cli   (stay green, delegate to engine)
gllm/vericut/*    reused as-is (check_gcode, run_vericut, verdict)
```

All loop logic lives in `gllm/loop/` as plain Python. The service and UI are dumb
pipes. This is what lets the corpus CLI and the live UI share one code path and lets
the entire loop be tested with zero HTTP and zero Vericut license.

## 4. The normalized `Finding` (linchpin)

Three incompatible finding shapes today — lint `(bool, message)` tuples,
`StaticFinding(code, severity, message, line_number)`, and Vericut parsed-log
findings — are unified behind one schema so the repair prompt can prioritize across
them:

```python
class Severity(Enum): ERROR; WARNING; INFO

@dataclass(frozen=True)
class Finding:
    source: Literal["lint", "static", "vericut"]
    code: str                       # "unsupported_tool", "collision", "rapid_through_material"
    severity: Severity
    message: str
    line_number: int | None = None  # static carries it; lint/vericut when available
    suggested_fix: str | None = None
    raw: dict | None = None         # original payload, for traceability
```

Normalizers `from_lint`, `from_static`, `from_vericut` convert each source
(`StaticFinding` maps ~1:1). The repair prompt sorts ERROR-before-WARNING and
static/vericut-before-lint, replacing `build_repair_prompt`'s flat dump.

### Validator interface

```python
class Validator(Protocol):
    name: str
    tier: Literal["lint", "static", "vericut"]
    blocking: bool                  # does a failure gate the loop?
    def validate(self, gcode: str, ctx: ValidationContext) -> list[Finding]: ...
```

- **LintValidator** wraps the `gcode_utils` validators with a **per-check `blocking`
  flag**; known-buggy checks are downgraded to non-blocking WARNING until fixed.
- **StaticPolicyValidator** wraps `check_gcode(gcode, setup)` — blocking.
- **VericutValidator** wraps `run_vericut` + `parse_vericut_log` — blocking, runs only
  at the final gate, auto-skips when assets/license absent.

Static-first gating = run lint + static every attempt; run Vericut once on the
converged best candidate.

## 5. `CandidateGenerator` + the three modes

The generator is a pure `prompt → gcode` callable (matching today's
`CandidateGenerator = Callable[[str, AttemptContext], str]`); the controller owns
prompt construction. One implementation serves all modes:

```python
class LLMCandidateGenerator:
    """Wraps model_utils.setup_model + setup_langchain_without_rag; returns raw text.
       Exposes last_usage (TokenUsage) for the budget."""
    def __call__(self, prompt: str, ctx: AttemptContext) -> str: ...
    last_usage: TokenUsage
```

| Mode | Attempt-1 candidate | Revision prompt builder | "Better" means |
|------|--------------------|------------------------|----------------|
| generate | `generator(build_generate_prompt(req))` | `build_repair_prompt(ctx)` *(exists)* | fewer/less-severe blocking findings |
| repair | `req.seed_gcode` used directly | `build_repair_prompt(ctx)` | fewer/less-severe blocking findings |
| improve | `req.seed_gcode` used directly (must pass) | `build_improve_prompt(ctx, objective)` *(new)* | gates stay green **and** objective drops |

`build_generate_prompt` = `SYSTEM_MESSAGE` + setup constraints (`_setup_constraints`)
+ task. `build_improve_prompt` instructs "keep every check green; reduce {objective}"
and feeds back the current objective value.

```python
@dataclass(frozen=True)
class Objective:
    metric: Literal["cycle_time", "rapid_distance", "tool_changes", "path_length"]
    direction: Literal["minimize"] = "minimize"
```

`scoring.py` computes cheap static proxies for `rapid_distance` / `path_length` /
`tool_changes` from the parsed toolpath (`parse_gcode` / `parse_coordinates`). True
`cycle_time` comes only from Vericut, so a `cycle_time` objective optimizes a static
proxy and Vericut confirms the real number at the final gate.

**Repair phasing:** Phase 1 = full-program regeneration with prioritized findings fed
back (no source-line mapping needed). `Finding.line_number` is designed in so
**targeted block-range repair** is a clean Phase-2 drop-in.

## 6. `LoopController` algorithm

```python
def stream(req: LoopRequest) -> Iterator[LoopEvent]:
    setup     = load_setup(req)                       # None ⇒ lint-only mode
    budget    = Budget(req.max_attempts, req.token_budget, req.patience)
    blocking  = [LintValidator, StaticPolicyValidator]  # Vericut deferred
    best, no_improve = None, 0
    candidate = req.seed_gcode if req.mode in (REPAIR, IMPROVE) \
                else generator(build_generate_prompt(req), ctx0)

    for n in range(1, budget.max_attempts + 1):
        yield AttemptStarted(n)
        cleaned  = clean_gcode(candidate)
        findings = [f for v in blocking for f in v.validate(cleaned, ctx)]
        score    = scoring.score(findings, req.objective, cleaned)   # lower = better
        attempt  = Attempt(n, cleaned, findings, score)
        yield Findings(attempt)
        best     = pick_best(best, attempt, req.mode, req.objective)
        if best is attempt: yield BestUpdated(attempt)

        if converged(attempt, req.mode):           break
        if budget.exhausted() or no_improve >= budget.patience: break

        prompt    = build_revision_prompt(ctx, findings, req)
        yield RepairPrompt(prompt)
        prev      = best.score
        candidate = generator(prompt, ctx)                          # full regen (Phase 1)
        budget.add(generator.last_usage)
        no_improve = 0 if best.score < prev else no_improve + 1

    verdict = None
    if req.run_vericut and passes_static(best) and vericut_available(setup):
        yield VericutStarted()
        verdict = VericutValidator.validate(best.gcode, ctx)
        yield VericutVerdict(verdict)

    result = LoopResult(best, history, verdict, derive_status(best, verdict, req.mode))
    yield Done(result)                                              # result.to_evidence_packet()
```

**`pick_best`:**
- generate/repair — fewest blocking findings, then lowest weighted score
  (ERROR ≫ WARNING); ties → newest.
- improve — only candidates passing all blocking gates are eligible (regression
  guard); among those, best objective metric. If nothing beats the seed while staying
  green, return the seed (no-op improve is an honest result).

**Status taxonomy** (extends existing strings so `operator_action_for_status` keeps
working): `passed` · `passed_with_warnings` · `improved` / `not_improved` ·
`exhausted_best_effort` (new — the cap/budget/no-improve case the current code hits as
a *silent* break) · `passed_vericut_unavailable` (static-clean, but the requested Vericut
final gate could not run — distinct from "Vericut not requested"; maps to `rerun_vericut`)
· `failed` · plus existing Vericut suffixes.

**Stop criteria** are all explicit: converged · max_attempts · patience · budget. No
more flat-50 or silent break.

`LoopResult.to_evidence_packet()` maps onto the existing `EvidencePacket` /
`ProofAttempt` structures (adding `score`, `mode`, `objective`, `best_attempt_index`)
so `evidence_packet.json/.md`, `operator_action`, and corpus tests keep working.

## 7. Service + UI

**Service (`gllm/service/`, FastAPI)** — thin transport, no loop logic:
- `POST /runs` (body = `LoopRequest`) → `run_id` + SSE stream of `LoopEvent`.
- `GET /runs/{id}` → `EvidencePacket`; `GET /runs/{id}/artifacts/…` → files (reuse
  `output_root` layout).
- `POST /runs/{id}/accept|reject` → records operator action.
- `GET /setups` → registry setups for the picker.

**UI (`gllm/ui/`, NiceGUI)** — replaces `code_generator_streamlit_*.py`:
- Intake: prompt · mode · setup picker · seed-gcode paste/upload · objective · model
  picker · `run_vericut` toggle · advanced (max_attempts, token budget).
- Live run view: a timeline that grows per `LoopEvent`; each attempt card shows
  status, findings grouped by severity/source, candidate g-code, toolpath plot.
- G-code + diff: code view + diff vs. previous attempt and vs. seed (Phase 1 text
  diff; Monaco later).
- Toolpath: Phase 1 embeds `plot_gcode` PNG; Phase 2 three.js custom component.
- Final packet: status · `operator_action` · best attempt · Vericut verdict ·
  blockers; Accept / Reject / "Run Vericut now" / "Edit & rerun".

Model lives in the engine/service process, so the UI just subscribes to the stream —
no fragile `session_state` model caching (removes the `test_streamlit_model_state.py`
class of problem).

## 8. Validator reliability cleanup

The loop trusts these signals, so fix or quarantine:
- `validate_functional_correctness` — stop returning `True` on exception; emit a
  non-blocking WARNING `Finding` instead of a silent pass.
- `validate_continuity` — fix the false-positive on valid linear paths, else ship
  non-blocking until fixed.
- `validate_z_levels` — guard the `.gcodes[0].params` `IndexError`.
- All lint checks behind `LintValidator` with a per-check `blocking` flag; unreliable
  ones default to non-blocking. Each fix gets a unit test (none exist today).

## 9. Testing

The whole loop is exercisable with no LLM and no Vericut license:
- Engine units: `FakeCandidateGenerator` + `FakeValidator` assert converge / cap /
  patience / budget / best-so-far (both comparators) / seed handling / status
  derivation — deterministic.
- Normalizer units: lint tuple, `StaticFinding`, Vericut payload → `Finding`.
- Adapter test: refactored `run_proof_scenario` still emits the same `EvidencePacket`;
  the 9 corpus fixtures + `test_proof_corpus.py` / `test_proof_run.py` stay green
  (mocked Vericut). Add a broken→repairs fixture and an improve fixture.
- Service: SSE event-sequence test via `TestClient`.
- UI: smoke/manual.

## 10. Migration (engine-first, each step shippable)

1. Build `gllm/loop/` + Fake-driven tests. No UI/service.
2. Refactor `run_proof_scenario` → thin adapter over `LoopController`; wire
   `build_repair_prompt` as default builder + the LLM generator. The CLI/corpus path
   now actually closes the loop end-to-end.
3. Validator cleanup + tests.
4. `gllm/service/` FastAPI event stream.
5. `gllm/ui/` NiceGUI to parity; retire the Streamlit file.
6. Phase 2 (out of v1 scope): Monaco diff, three.js viewer, targeted block repair.

## 11. Non-goals (YAGNI)

Line-level/targeted repair, 3D toolpath viewer, multi-user/auth, RAG/fine-tuning
changes — explicitly deferred.

## 12. Key reused interfaces (verified against current code)

- `gllm/proof/runner.py`: `run_proof_scenario(request, *, candidate_generator,
  repair_prompt_builder=None, run_vericut_fn=run_vericut) -> EvidencePacket`,
  `build_repair_prompt(ctx)`, `operator_action_for_status(status)`, `ScenarioRequest`,
  `AttemptContext`, `ProofAttempt`, `EvidencePacket`.
- `gllm/vericut/static_checks.py`: `check_gcode(gcode, setup) -> StaticCheckReport`,
  `StaticFinding(code, severity, message, line_number)`; `passed` = no error-severity
  findings.
- `gllm/vericut/runner.py`: `prepare_job_workspace`, `run_vericut`,
  `build_vericut_command`. `gllm/vericut/verdict.py`: `parse_vericut_log`,
  `write_verdict_packet`. `gllm/vericut/registry.py`: `load_setup_registry`,
  `VericutSetup`, `validate_setup_paths`.
- `gllm/utils/gcode_utils.py`: `clean_gcode`, `parse_gcode`, `parse_coordinates`, the
  validator family. `gllm/utils/plot_utils.py`: `plot_gcode`.
- `gllm/utils/model_utils.py`: `setup_model`, `setup_langchain_without_rag`.
- `gllm/utils/prompts_utils.py`: `SYSTEM_MESSAGE`, `REQUIRED_PARAMETERS`.
