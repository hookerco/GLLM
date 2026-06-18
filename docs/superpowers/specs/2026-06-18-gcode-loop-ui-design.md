# G-Code Loop UI + Service — Design

**Date:** 2026-06-18
**Status:** Approved direction (continuation of the loop-engine spec, Section 5); building under user authorization ("spec and run it; I can adjust the frontend")

## 1. Goal

Put a usable front end on the `gllm/loop` engine so a user can **generate / improve / repair G-code on the fly from a prompt**: type a prompt, pick a mode and machine setup, hit Run, watch the autonomous loop stream its attempts (findings, candidate G-code, toolpath) live, and accept or reject the final result.

This is the second sub-project of the prompt-driven-gcode work. The engine (`gllm/loop/`) is complete, tested, and emits `LoopEvent`s. This spec covers the **service layer** (a FastAPI app exposing the run lifecycle, including an SSE event stream) and the **NiceGUI UI** that drives it.

## 2. Decisions (locked, adjustable)

- **One process.** NiceGUI is built on FastAPI; the UI and the API routes live in one app/process (`from nicegui import app, ui` — `app` is the underlying FastAPI app). Splitting into separate services is a later option; the `RunManager` seam makes it cheap.
- **The real boundary is `RunManager`, not HTTP.** Both the bundled UI (in-process, async) and external clients (over HTTP/SSE) drive runs through one `RunManager`. The UI consumes events in-process; the SSE HTTP endpoint exists for external consumers and for testing.
- **Runs execute in a background thread.** `LoopController.stream()` is a synchronous generator with blocking LLM calls, so each run executes on a worker thread that pushes serialized events into a per-run buffer; consumers (UI async loop, SSE endpoint) drain the buffer.
- **Injectable controller factory.** `RunManager` is constructed with a `controller_factory(LoopRequest) -> LoopController`. The default factory wires a real `LLMCandidateGenerator`; tests inject a factory returning a controller with a fake generator + scripted validators, so the entire service is testable with no LLM and no Vericut license.
- **SSE via plain `StreamingResponse`** (`text/event-stream`), no extra dependency.
- **Toolpath = matplotlib PNG** (reuse `plot_gcode`), Phase 1. A three.js viewer is a later enhancement.
- **In-memory run registry.** Runs live in a dict keyed by `run_id`; artifacts (gcode, plots) under `output_root`. Persistence is out of scope for v1.

## 3. Module layout

```
gllm/service/                         (new — transport + run lifecycle)
  serialize.py    LoopEvent/Attempt/Finding/LoopResult -> JSON-safe dicts; SSE frame formatting
  runs.py         RunManager: start_run, per-run event buffer, async/iter drain, decision, registry
  api.py          create_api(run_manager) -> APIRouter/route registrations (FastAPI)
gllm/ui/                              (new — NiceGUI front end; replaces the Streamlit app)
  components.py   small render helpers (attempt card, findings list, final panel)
  app.py          build_ui(run_manager): intake form + live run view + final panel
  __main__.py     launcher: build RunManager + API + UI, ui.run(...)
tests/service/    test_serialize.py, test_runs.py, test_api.py
tests/ui/         test_ui_smoke.py
```

Reused verbatim from the engine: `gllm.loop` (`LoopController`, `LoopRequest`, `Mode`, `Objective`, `LLMCandidateGenerator`, `VericutFinalGate`, `result_to_dict`, `LoopEvent`, `Attempt`), `gllm.vericut.registry.load_setup_registry`, `gllm.utils.plot_utils.plot_gcode`.

## 4. Serialization (`serialize.py`)

`LoopEvent` carries `Attempt`/`LoopResult` objects that must become JSON. Functions:

- `finding_to_dict(f)` → use `Finding.to_dict()`.
- `attempt_to_dict(a)` → `{number, gcode, score, objective_value, findings: [...], blocking_count}`.
- `event_to_dict(ev)` → `{type, attempt?, prompt?, result?, payload?}` where `result` uses `result_to_dict` and `attempt` uses `attempt_to_dict`.
- `sse_frame(ev_dict)` → `"event: {type}\ndata: {json}\n\n"`.

All pure and unit-tested.

## 5. `RunManager` (`runs.py`)

State per run: `id`, `request`, a thread-safe event list, a `done` flag, the final `LoopResult` dict, and a `decision` (`accept`/`reject`/None).

API:
- `start_run(request: LoopRequest) -> str` — create `run_id`, spawn a worker thread that runs `for ev in controller.stream(request): self._append(run_id, event_to_dict(ev))`, sets the final result on the `done` event, marks done. Returns immediately.
- `iter_events(run_id, from_index=0)` — generator yielding event dicts from `from_index` onward, blocking-polling until the `done` event is seen (used by the SSE endpoint, run in a threadpool).
- `async aiter_events(run_id, from_index=0)` — async wrapper (`await asyncio.sleep` poll over the buffer) for the in-process UI.
- `get_result(run_id) -> dict | None` — final `result_to_dict`, or None while running.
- `set_decision(run_id, action)` / `get_status(run_id)`.
- `list_setups(registry_path) -> [{id, description}]` — wraps `load_setup_registry`.

Concurrency: a worker thread appends to a `list` guarded by a `threading.Lock`; consumers poll by index (no cross-thread async queues — simplest correct approach). A run is bounded by the engine's own stop criteria, so the buffer is finite.

Tests: inject a `controller_factory` returning a controller with a scripted fake generator/validator; assert events stream in order, `done` carries the result, `get_result` returns the packet, decisions record. No LLM.

## 6. API (`api.py`)

Routes (prefix `/api`):
- `GET /api/setups?registry_path=...` → `[{id, description}]` (400 on bad registry).
- `POST /api/runs` (JSON body → `LoopRequest`: `prompt`, `mode`, `registry_path?`, `setup_id?`, `seed_gcode?`, `objective?`, `model_name?`, `openrouter_model_name?`, `run_vericut?`, `max_attempts?`, `token_budget?`) → `{run_id}`.
- `GET /api/runs/{run_id}/events` → SSE `StreamingResponse` (drains `iter_events` via a threadpool generator).
- `GET /api/runs/{run_id}` → final result dict, or `{status: "running"}` (202) while in progress.
- `POST /api/runs/{run_id}/decision` (`{action: "accept"|"reject"}`) → `{ok: true}`.

Request→`LoopRequest` mapping is a small validated builder (`build_loop_request(payload)`), unit-tested (objective dict → `Objective`, mode string → `Mode`, defaults). Tested with FastAPI `TestClient` + a `RunManager` whose factory uses a fake generator; SSE read via `client.stream(...)`.

## 7. UI (`app.py`, NiceGUI)

`build_ui(run_manager)` registers the page(s):
- **Intake** (left/top): prompt textarea; mode select (Generate/Repair/Improve); registry path input + setup select (populated from `RunManager.list_setups`); seed-gcode textarea (shown for Repair/Improve); objective select (shown for Improve); model select (OpenRouter default, etc.); `run_vericut` switch; `max_attempts` number; **Run** button.
- **Live run view**: on Run, build a `LoopRequest`, `start_run`, then `async for ev in run_manager.aiter_events(...)`: append an **attempt card** per `findings` event (status derived from blocking findings; findings grouped by severity/source; candidate G-code in a code block; toolpath image via `plot_gcode` → PNG saved under the run's artifact dir and shown with `ui.image`). Show a spinner between attempts.
- **Final panel**: on `done`, render status, `operator_action`, best attempt index, Vericut verdict (if any), and **Accept / Reject** buttons that call `set_decision`.

The model lives in the engine/service process, so there is no `st.session_state` model caching — this removes the entire `test_streamlit_model_state.py` class of fragility. The old `code_generator_streamlit_*.py` is retired once this reaches parity (left in place for now; not deleted in this sub-project).

UI verification: an import/smoke test (the page builder runs without a live server) plus a manual launch. The frontend is explicitly the adjustable layer.

## 8. Launcher (`__main__.py`)

Builds the default `RunManager` (real controller factory), creates the API, builds the UI, and calls `ui.run(host="127.0.0.1", port=8080, reload=False, show=False)`. Documented in the README. A detached-launch note mirrors the existing Streamlit guidance.

## 9. Testing & verification

- **serialize**: pure unit tests (event/attempt/finding/result dicts; SSE frame format).
- **runs**: `RunManager` with a fake controller factory — event ordering, `done` result, `get_result`, decisions, `list_setups` against a temp registry.
- **api**: `TestClient` — `/setups`, `POST /runs` + `GET /runs/{id}/events` (SSE) + `GET /runs/{id}` + decision, all with a fake-generator factory. `build_loop_request` unit tests.
- **ui**: import/smoke (build the page function against a fake `RunManager`; assert no exception).
- **end-to-end manual**: launch the app, hit `/api/setups` with the sample registry, run a Generate against a sample setup, watch attempts stream, accept.

## 10. Non-goals (v1)

Auth/multi-user, run persistence, three.js/Monaco, websockets (SSE only), deleting the old Streamlit app, the `cycle_time`/`rapid_distance` objectives (engine Phase 2). All deferred.

## 11. Build order (each step shippable/testable)

1. Add `nicegui` dependency.
2. `serialize.py` + tests.
3. `runs.py` (`RunManager`) + tests (fake factory).
4. `api.py` (`build_loop_request` + routes) + `TestClient` tests.
5. `ui/` (components + `build_ui`) + smoke test.
6. `__main__.py` launcher + README + manual end-to-end verification.
