from __future__ import annotations

from pathlib import Path

import anyio
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from gllm.loop.types import LoopRequest, Mode, Objective
from gllm.service.runs import RunManager
from gllm.service.serialize import sse_frame

_STATIC_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"
_SENTINEL = object()


def build_loop_request(payload: dict) -> LoopRequest:
    mode_str = str(payload.get("mode", "generate")).lower()
    try:
        mode = Mode(mode_str)
    except ValueError as exc:
        raise ValueError(f"Unknown mode: {mode_str!r}") from exc

    objective = None
    obj = payload.get("objective")
    if obj:
        metric = obj.get("metric") if isinstance(obj, dict) else obj
        if metric:
            objective = Objective(metric=str(metric))

    return LoopRequest(
        prompt=str(payload.get("prompt", "")),
        mode=mode,
        registry_path=payload.get("registry_path") or None,
        setup_id=payload.get("setup_id") or None,
        seed_gcode=payload.get("seed_gcode") or None,
        objective=objective,
        model_name=payload.get("model_name") or "OpenRouter",
        openrouter_model_name=payload.get("openrouter_model_name") or None,
        run_vericut=bool(payload.get("run_vericut", False)),
        max_attempts=int(payload.get("max_attempts", 4)),
        token_budget=payload.get("token_budget"),
        scenario_id=payload.get("scenario_id") or None,
    )


async def _event_stream(run_manager: RunManager, run_id: str):
    gen = run_manager.iter_events(run_id)
    while True:
        ev = await anyio.to_thread.run_sync(next, gen, _SENTINEL)
        if ev is _SENTINEL:
            break
        yield sse_frame(ev)


async def index(request):
    index_file = _STATIC_DIR / "index.html"
    if not index_file.exists():
        return PlainTextResponse("GLLM loop UI is not built yet.")
    return FileResponse(index_file)


async def setups(request):
    registry_path = request.query_params.get("registry_path")
    if not registry_path:
        return JSONResponse({"error": "registry_path is required"}, status_code=400)
    try:
        return JSONResponse(request.app.state.run_manager.list_setups(registry_path))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def models(request):
    from gllm.utils.model_utils import DEFAULT_MODEL, MODEL_OPTIONS

    return JSONResponse({"models": list(MODEL_OPTIONS), "default": DEFAULT_MODEL})


async def create_run(request):
    payload = await request.json()
    try:
        loop_request = build_loop_request(payload)
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    run_id = request.app.state.run_manager.start_run(loop_request)
    return JSONResponse({"run_id": run_id})


async def run_events(request):
    run_id = request.path_params["run_id"]
    rm = request.app.state.run_manager
    if rm.get_run(run_id) is None:
        return JSONResponse({"error": "unknown run"}, status_code=404)
    return StreamingResponse(_event_stream(rm, run_id), media_type="text/event-stream")


async def run_result(request):
    run_id = request.path_params["run_id"]
    rm = request.app.state.run_manager
    if rm.get_run(run_id) is None:
        return JSONResponse({"error": "unknown run"}, status_code=404)
    result = rm.get_result(run_id)
    if result is None:
        return JSONResponse({"status": "running"}, status_code=202)
    return JSONResponse(result)


async def run_decision(request):
    run_id = request.path_params["run_id"]
    payload = await request.json()
    action = payload.get("action")
    if action not in ("accept", "reject"):
        return JSONResponse({"error": "action must be 'accept' or 'reject'"}, status_code=400)
    if not request.app.state.run_manager.set_decision(run_id, action):
        return JSONResponse({"error": "unknown run"}, status_code=404)
    return JSONResponse({"ok": True})


async def run_plot(request):
    run_id = request.path_params["run_id"]
    try:
        n = int(request.path_params["n"])
    except (ValueError, TypeError):
        return JSONResponse({"error": "bad attempt number"}, status_code=400)
    png = request.app.state.run_manager.plot_png(run_id, n)
    if png is None:
        return JSONResponse({"error": "no plot available"}, status_code=404)
    return Response(png, media_type="image/png")


def create_app(run_manager: RunManager | None = None) -> Starlette:
    rm = run_manager if run_manager is not None else RunManager()
    routes = [
        Route("/", index),
        Route("/api/setups", setups),
        Route("/api/models", models),
        Route("/api/runs", create_run, methods=["POST"]),
        Route("/api/runs/{run_id}/events", run_events),
        Route("/api/runs/{run_id}", run_result),
        Route("/api/runs/{run_id}/decision", run_decision, methods=["POST"]),
        Route("/api/runs/{run_id}/plot/{n}", run_plot),
    ]
    if _STATIC_DIR.exists():
        routes.append(Mount("/static", app=StaticFiles(directory=str(_STATIC_DIR)), name="static"))
    app = Starlette(routes=routes)
    app.state.run_manager = rm
    return app
