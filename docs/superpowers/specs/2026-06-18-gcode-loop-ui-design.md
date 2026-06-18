# G-Code Loop UI + Service — Design

**Date:** 2026-06-18
**Status:** Approved direction (continuation of the loop-engine spec, Section 5); building under user authorization ("spec and run it; I can adjust the frontend")

## 1. Goal

Put a usable front end on the `gllm/loop` engine so a user can **generate / improve / repair G-code on the fly from a prompt**: type a prompt, pick a mode and machine setup, hit Run, watch the autonomous loop stream its attempts (findings, candidate G-code, toolpath) live, and accept or reject the final result.

This is the second sub-project of the prompt-driven-gcode work. The engine (`gllm/loop/`) is complete, tested, and emits `LoopEvent`s. This spec covers the **service layer** (the run lifecycle + an SSE event stream) and the **web UI** that drives it.

## 2. Framework decision — Starlette + static HTML/JS (NOT NiceGUI)

The original direction was NiceGUI + FastAPI. **That is blocked by a hard dependency conflict:** the project runs on **pydantic 1.10 (v1)** (streamlit 1.31, langchain 0.2, and a `manifest-ml` pin that forces `pydantic <2`), while NiceGUI/FastAPI require **pydantic v2**. Forcing a pydantic-v2 upgrade would risk destabilizing the entire working stack.

Resolution (best for the project): use a **pydantic-free** stack.
- **Service:** **Starlette** (the ASGI core under FastAPI — *no* pydantic dependency) + **uvicorn**. Installed cleanly with zero disruption to the pydantic-v1 stack.
- **Frontend:** a **static single-page app** (hand-written `index.html` + `app.js` + `style.css`) served by Starlette, consuming the SSE stream via the browser's native `EventSource`. No build step, no JS framework, no pydantic — and trivially adjustable (it's just files).

This keeps the same architecture as the engine spec's Section 5 (RunManager seam, background-thread runs, SSE event stream); only the web framework changes. The user explicitly authorized adjusting the frontend.

## 3. Decisions (locked, adjustable)

- **The real boundary is `RunManager`, not HTTP.** Runs are driven through one `RunManager`; the SSE HTTP endpoint and the result/decision endpoints are thin wrappers.
- **Runs execute in a background thread.** `LoopController.stream()` is synchronous with blocking LLM calls, so each run runs on a worker thread that appends serialized events to a per-run buffer; consumers drain the buffer by index.
- **Injectable controller factory.** `RunManager(controller_factory)` — default factory wires a real `LLMCandidateGenerator` (+ optional `VericutFinalGate`); tests inject a factory returning a controller with a fake generator + scripted validators, so the whole service is testable with no LLM and no Vericut license.
- **SSE via Starlette `StreamingResponse`** (`text/event-stream`); no `sse-starlette` dependency.
- **Toolpath = matplotlib PNG** (reuse `plot_gcode`), served per attempt. A three.js viewer is a later enhancement.
- **In-memory run registry**, artifacts under `output_root`. Persistence out of scope for v1.

## 4. Module layout

```
gllm/service/                         (new — transport + run lifecycle)
  serialize.py    LoopEvent/Attempt/Finding/LoopResult -> JSON-safe dicts; SSE frame formatting
  runs.py         RunManager: start_run, per-run event buffer, iter drain, get_result, decision, list_setups, plot_png
  app.py          create_app(run_manager=None) -> Starlette app (API routes + static UI mount)
  __main__.py     launcher: uvicorn.run(create_app(), host, port)
gllm/ui/static/                       (new — the frontend, plain files)
  index.html      intake form + live run view + final panel
  app.js          fetch() + EventSource wiring
  style.css       minimal styling
tests/service/    test_serialize.py, test_runs.py, test_app.py
```

Reused verbatim: `gllm.loop` (`LoopController`, `LoopRequest`, `Mode`, `Objective`, `LLMCandidateGenerator`, `VericutFinalGate`, `result_to_dict`, `LoopEvent`, `Attempt`), `gllm.vericut.registry.load_setup_registry`, `gllm.utils.plot_utils.plot_gcode`. The old `code_generator_streamlit_*.py` is left in place (retired later, not deleted here).

## 5. Serialization (`serialize.py`)

- `attempt_to_dict(a)` → `{number, gcode, score, objective_value, blocking_count, findings: [f.to_dict()...]}`.
- `event_to_dict(ev)` → `{type, attempt?, prompt?, result?, payload?}` (result via `result_to_dict`, attempt via `attempt_to_dict`).
- `sse_frame(ev_dict)` → `"event: {type}\ndata: {json}\n\n"`.

Pure, unit-tested.

## 6. `RunManager` (`runs.py`)

Per-run state: `id`, `request`, event list (guarded by a `threading.Lock`), `done` flag, final result dict, `decision`.

- `start_run(request: LoopRequest) -> str` — new `run_id`; spawn a daemon worker thread running `for ev in controller.stream(request): append(event_to_dict(ev))`; on the `done` event store `result_to_dict(ev.result)`; mark done. Returns immediately.
- `iter_events(run_id, from_index=0)` — generator yielding buffered event dicts from `from_index`, polling (`time.sleep`) until the `done` event is emitted. Used by the SSE route (driven in a threadpool).
- `get_result(run_id) -> dict | None` — final result, or None while running.
- `set_decision(run_id, action)` / `get_run(run_id)`.
- `list_setups(registry_path) -> [{id, description}]` — wraps `load_setup_registry`.
- `plot_png(run_id, attempt_number) -> bytes | None` — render the attempt's cleaned gcode via `plot_gcode` to a PNG (stdout suppressed), cache to disk.

Default `controller_factory(request)` builds `LoopController(generator=LLMCandidateGenerator(model_name=request.model_name, openrouter_model_name=request.openrouter_model_name), final_gate=VericutFinalGate())`. Tests pass a fake factory.

Tests: fake factory with a scripted generator/validator — event ordering, `done` carries the result, `get_result`, decisions, `list_setups` against a temp registry. No LLM.

## 7. Service app (`app.py`)

`create_app(run_manager=None) -> Starlette` (defaults to a real `RunManager`). Routes:
- `GET /` → serve `gllm/ui/static/index.html`.
- `GET /static/...` → `StaticFiles` mount (`app.js`, `style.css`).
- `GET /api/setups?registry_path=...` → `JSONResponse([{id, description}])` (400 on bad registry).
- `POST /api/runs` (JSON body) → `build_loop_request(payload)` → `start_run` → `{run_id}`.
- `GET /api/runs/{run_id}/events` → `StreamingResponse(text/event-stream)` draining `iter_events` (via `anyio.to_thread`/threadpool generator).
- `GET /api/runs/{run_id}` → result dict, or `{status: "running"}` (HTTP 202) while running, 404 if unknown.
- `POST /api/runs/{run_id}/decision` (`{action}`) → `{ok: true}`.
- `GET /api/runs/{run_id}/plot/{n}.png` → `Response(plot_png, media_type="image/png")`, 404 if unavailable.

`build_loop_request(payload: dict) -> LoopRequest` is a small pydantic-free validated builder (mode string → `Mode`, objective dict → `Objective`, sensible defaults, rejects unknown mode). Unit-tested.

Tests: Starlette `TestClient` + a `RunManager` with a fake-generator factory — `/setups`, `POST /runs` + `GET .../events` (SSE read via `client.stream`) + `GET /runs/{id}` + decision + a plot fetch. `build_loop_request` unit tests.

## 8. Frontend (`gllm/ui/static/`)

`index.html` + `app.js` + `style.css`, vanilla JS:
- **Intake form:** prompt textarea; mode select (Generate/Repair/Improve); registry-path input + setup select (populated via `fetch /api/setups`); seed-gcode textarea (shown for Repair/Improve); objective select (Improve); model select (OpenRouter default); `run_vericut` checkbox; `max_attempts` number; **Run** button.
- **Live run view:** on Run → `POST /api/runs` → open `EventSource('/api/runs/{id}/events')`. On `attempt_started` add a card; on `findings` fill it (status from blocking-findings count, findings grouped by severity, candidate G-code in `<pre>`, toolpath `<img src=/api/runs/{id}/plot/{n}.png>`); on `best_updated` mark the best; on `done` render the **final panel** (status, operator action, best index, Vericut verdict) with **Accept / Reject** buttons → `POST .../decision`.
- Minimal CSS; no framework. This is the adjustable layer.

Frontend verification: a smoke test asserting `create_app()` builds and `GET /` returns the HTML + `/static/app.js` loads; plus manual launch.

## 9. Launcher (`__main__.py`)

`uvicorn.run(create_app(), host="127.0.0.1", port=8080)`. `python -m gllm.service` starts it. README documents it alongside (not replacing) the Streamlit instructions.

## 10. Testing & verification

- **serialize / runs / app**: unit + `TestClient` (all with fakes; no LLM, no Vericut, no pydantic).
- **end-to-end manual**: launch `python -m gllm.service`, `GET /api/setups` with the sample registry (`config/vericut_setups.example.json`), run a Generate, watch SSE attempts stream, fetch a toolpath PNG, accept.
- Automated end-to-end: a `TestClient` test that drives a full fake run start→events→result→decision.

## 11. Non-goals (v1)

Auth/multi-user, run persistence, three.js/Monaco, websockets (SSE only), deleting the Streamlit app, pydantic-v2 migration, `cycle_time`/`rapid_distance` objectives. Deferred.

## 12. Build order (each step shippable/testable)

1. `starlette` + `uvicorn` dependencies (done).
2. `serialize.py` + tests.
3. `runs.py` (`RunManager`) + tests (fake factory).
4. `app.py` (`build_loop_request` + routes) + `TestClient` tests.
5. `gllm/ui/static/` frontend (`index.html`/`app.js`/`style.css`) + smoke test.
6. `__main__.py` launcher + README + manual end-to-end verification.
