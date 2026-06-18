# Vericut-Driven Repair Loop — Design

**Date:** 2026-06-18
**Status:** Approved (design); implementing
**Extends:** `2026-06-18-prompt-driven-gcode-loop-design.md`

## 1. Goal

Today the loop runs Vericut **once** as a final gate on the converged best candidate
(`LoopController.stream`, the block at the end of the static repair loop). If Vericut
**rejects** the toolpath, the loop stops with status `rejected_vericut` and never acts
on the failure.

This feature feeds a Vericut rejection back into the generator: convert the reported
Vericut toolpath errors into repair findings, regenerate the program, re-validate
statically, and re-simulate — repeating up to a small bounded cap or until Vericut
accepts. This deliberately revises the v1 "Vericut runs once" decision.

The work is mostly **wiring machinery that already exists**:
- `VericutVerdict.to_dict()` already emits a `repair_context` — the first 10 parsed
  Vericut findings as dicts (`gllm/vericut/verdict.py`).
- `from_vericut(payload)` already normalizes a Vericut finding dict into a loop
  `Finding(source="vericut", ...)` (`gllm/loop/findings.py`).
- `build_repair_prompt(request, gcode, findings, constraints)` already accepts any
  `Finding` iterable (`gllm/loop/generator.py`).

## 2. Decisions (locked)

- **Depth:** a **bounded mini-loop**. After a rejection, repair → regenerate →
  static-validate → re-simulate, repeating up to a small cap or until accepted/budget.
- **Trigger:** **`vericut_rejected` only** — i.e. positive evidence that Vericut
  simulated the toolpath and found real errors. `vericut_unverified` (failed to
  launch / no log / abnormal exit) and `vericut_unavailable` (no setup/assets/license/
  deps) stay terminal — they carry no toolpath findings to act on, so a repair would be
  a blind regenerate with no signal.
- **Integration:** a distinct, bounded **Vericut-repair phase** appended after the
  existing static repair loop. The static-phase budget logic (`max_attempts`,
  `patience`) is untouched. Rejected the alternative of re-entering the unified main
  loop with Vericut findings as blocking findings (entangles the static-attempt budget
  with Vericut rounds; risks re-running expensive Vericut on intermediate attempts).
- **Cost bound:** at most `vericut_max_rounds` additional Vericut runs after the first
  gate, also halted by the shared `token_budget`. Default `vericut_max_rounds = 2`
  (repair is ON by default); `0` restores exact pre-feature behavior.

## 3. New request knob

Add one field to `LoopRequest` (`gllm/loop/types.py`):

```python
vericut_max_rounds: int = 2
```

- Caps the number of repair rounds **after** the initial gate rejection.
- `0` ⇒ no repair (single gate, today's behavior).
- Repair rounds share the existing `token_budget` (a tight budget halts them) but do
  **not** consume `max_attempts` (that governs the static phase only).

## 4. Algorithm

Replaces the single final-gate block at the end of `LoopController.stream`. Pseudocode
(`final_gate` signature unchanged: `(gcode, ctx, request) -> dict | None`):

```python
vericut = None
if final_gate and request.run_vericut and best and not best.blocking_findings:
    yield LoopEvent("vericut_started")
    verdict = final_gate(best.gcode, ctx, request) or {"status": "vericut_unavailable"}
    yield LoopEvent("vericut_verdict", payload=verdict)

    best_pair = (best, verdict)          # (Attempt, verdict dict)
    n = len(history)                     # continue attempt numbering
    rounds = 0
    while (
        verdict.get("status") == "vericut_rejected"
        and rounds < request.vericut_max_rounds
        and not budget.exhausted()
    ):
        rounds += 1
        vfindings = [from_vericut(p) for p in verdict.get("repair_context", [])]
        # carry any static findings from the prior repaired candidate forward too
        prompt = build_repair_prompt(request, best_pair[0].gcode,
                                     vfindings + carried_static, constraints)
        yield LoopEvent("repair_prompt", prompt=prompt)
        candidate = generator(prompt, ctx); budget.add(generator.last_usage)

        n += 1
        cleaned   = clean_gcode(candidate)
        sfindings = self._validate(cleaned, ctx)
        attempt   = Attempt(n, cleaned, tuple(sfindings),
                            findings_penalty(sfindings),
                            objective_value(cleaned, request.objective))
        history.append(attempt)
        yield LoopEvent("findings", attempt=attempt)

        if attempt.blocking_findings:
            # Regressed on static gates — do NOT spend a Vericut run on it.
            carried_static = list(attempt.findings)   # feed forward next round
            continue                                  # round still consumed

        carried_static = []
        yield LoopEvent("vericut_started")
        verdict = final_gate(attempt.gcode, ctx, request) or {"status": "vericut_unavailable"}
        yield LoopEvent("vericut_verdict", payload=verdict)
        best_pair = _pick_best_vericut(best_pair, (attempt, verdict))

    best, vericut = best_pair

status = derive_status(best, request.mode, vericut)
```

`carried_static` starts as `[]` before the loop.

### `_pick_best_vericut(a, b)` — module-level helper

Given two `(attempt, verdict)` pairs, return the better:
1. An **accepted** verdict (`status == "vericut_accepted"`, i.e. `passed is True`) beats
   a non-accepted one.
2. Among rejected, **fewer** `verdict["vericut"]["error_count"]` wins.
3. Tie ⇒ the **later** pair (`b`).

This keeps `result.best` and `result.vericut` consistent: the reported best candidate
is exactly the one its verdict describes.

## 5. Statuses & events

No new statuses or event types.

- `derive_status(best, mode, vericut)` already maps the final verdict:
  accepted ⇒ `accepted_vericut`, rejected ⇒ `rejected_vericut`,
  `vericut_unverified` ⇒ `blocked_vericut_unverified`,
  `vericut_unavailable` ⇒ `passed_vericut_unavailable`. A successful repair therefore
  yields `accepted_vericut` for free; an exhausted one stays `rejected_vericut`.
- `vericut_started` / `vericut_verdict` fire once per round, so a streaming consumer
  shows each repair simulation with no consumer changes. Repair prompts reuse the
  existing `repair_prompt` event.

## 6. Testing

All deterministic, no Vericut license — mirrors the fake-driven style in
`tests/loop/test_controller.py`. A scripted `final_gate` returns a verdict sequence by
call count; a scripted generator returns successive G-code candidates.

Helper: a `_verdict(status, errors=0, findings=())` factory producing a verdict dict
shaped like `VericutVerdict.to_dict()` (with `vericut.error_count` and `repair_context`).

Cases:
- **Repairs to acceptance:** gate returns reject→accept; generator returns a fixed
  candidate. Status flips to `accepted_vericut`; `history` grew by one; final
  `result.vericut` is the accepted one.
- **Exhausts the cap:** gate always rejects; `vericut_max_rounds=2`. Exactly 1 initial
  + 2 repair Vericut calls; status `rejected_vericut`; best pair = fewest errors.
- **`vericut_max_rounds=0`:** no repair attempted; behavior identical to today
  (one gate call, `rejected_vericut`).
- **Token budget halts mid-repair:** `token_budget` exhausts after the first repair;
  loop stops before the cap.
- **Static regression skips simulation:** a repaired candidate has a blocking static
  finding ⇒ that round runs no Vericut call; findings carried into the next prompt.
- **Best-by-fewest-errors:** reject(3 errors)→reject(1 error) ⇒ reported best is the
  1-error pair.
- **Only rejection triggers repair:** an initial `vericut_unverified` /
  `vericut_unavailable` verdict does **not** enter the repair loop.

## 7. Non-goals (YAGNI)

- Repairing `vericut_unverified` / `vericut_unavailable` (no toolpath signal).
- A separate Vericut token/time budget distinct from `token_budget`.
- New statuses or event types.
- Targeted line-level repair (still full-program regeneration, per the parent spec).

## 8. Touched files

- `gllm/loop/types.py` — add `vericut_max_rounds` to `LoopRequest`.
- `gllm/loop/controller.py` — replace the single-gate block with the bounded loop;
  add `_pick_best_vericut`; import `from_vericut`, `build_repair_prompt` (already
  imported).
- `tests/loop/test_controller.py` (or a new `tests/loop/test_vericut_repair.py`) —
  the cases above.
- Possibly update existing Vericut-status tests that assume a rejection is terminal to
  pass `vericut_max_rounds=0` where they intend the single-gate path.
