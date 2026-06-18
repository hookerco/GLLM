from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import contextlib  # noqa: E402
import io  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

from gllm.loop.controller import LoopController  # noqa: E402
from gllm.loop.generator import LLMCandidateGenerator  # noqa: E402
from gllm.loop.types import LoopRequest  # noqa: E402
from gllm.loop.vericut_gate import VericutFinalGate  # noqa: E402
from gllm.service.serialize import event_to_dict  # noqa: E402
from gllm.vericut.registry import load_setup_registry  # noqa: E402


def default_controller_factory(request: LoopRequest) -> LoopController:
    generator = LLMCandidateGenerator(
        model_name=request.model_name,
        openrouter_model_name=request.openrouter_model_name,
    )
    return LoopController(generator=generator, final_gate=VericutFinalGate())


class _Run:
    def __init__(self, run_id: str, request: LoopRequest):
        self.id = run_id
        self.request = request
        self.events: list[dict] = []
        self.done = False
        self.result: dict | None = None
        self.decision: str | None = None
        self.error: str | None = None
        self.lock = threading.Lock()


class RunManager:
    def __init__(self, controller_factory=default_controller_factory, output_root: str | Path = ".loop-runs"):
        self._factory = controller_factory
        self._runs: dict[str, _Run] = {}
        self._output_root = Path(output_root)

    def start_run(self, request: LoopRequest) -> str:
        run_id = request.scenario_id or uuid.uuid4().hex
        run = _Run(run_id, request)
        self._runs[run_id] = run
        threading.Thread(target=self._execute, args=(run,), daemon=True).start()
        return run_id

    def _execute(self, run: _Run) -> None:
        try:
            controller = self._factory(run.request)
            for event in controller.stream(run.request):
                ev_dict = event_to_dict(event)
                with run.lock:
                    run.events.append(ev_dict)
                    if event.type == "done":
                        run.result = ev_dict.get("result")
        except Exception as exc:  # surface engine failures as a terminal error event
            with run.lock:
                run.error = f"{type(exc).__name__}: {exc}"
                run.events.append({"type": "error", "payload": {"message": run.error}})
        finally:
            with run.lock:
                run.done = True

    def get_run(self, run_id: str) -> _Run | None:
        return self._runs.get(run_id)

    def iter_events(self, run_id: str, from_index: int = 0, poll_interval: float = 0.03, timeout: float = 600.0):
        run = self._runs.get(run_id)
        if run is None:
            return
        i, waited = from_index, 0.0
        while True:
            with run.lock:
                n = len(run.events)
                done = run.done
            while i < n:
                with run.lock:
                    ev = run.events[i]
                yield ev
                i += 1
                if ev["type"] in ("done", "error"):
                    return
            if done and i >= n:
                return
            time.sleep(poll_interval)
            waited += poll_interval
            if waited >= timeout:
                return

    def get_result(self, run_id: str) -> dict | None:
        run = self._runs.get(run_id)
        if run is None or not run.done:
            return None
        return run.result

    def set_decision(self, run_id: str, action: str) -> bool:
        run = self._runs.get(run_id)
        if run is None:
            return False
        with run.lock:
            run.decision = action
        return True

    def list_setups(self, registry_path: str | Path) -> list[dict]:
        registry = load_setup_registry(registry_path)
        return [{"id": sid, "description": registry.get(sid).description} for sid in registry.ids()]

    def plot_png(self, run_id: str, attempt_number: int) -> bytes | None:
        run = self._runs.get(run_id)
        if run is None:
            return None
        gcode = None
        with run.lock:
            for ev in run.events:
                if ev["type"] == "findings" and ev.get("attempt", {}).get("number") == attempt_number:
                    gcode = ev["attempt"]["gcode"]
                    break
        if not gcode or not gcode.strip():
            return None
        try:
            from gllm.utils.plot_utils import plot_gcode
            buf = io.BytesIO()
            with contextlib.redirect_stdout(io.StringIO()):
                plt = plot_gcode(gcode)
            plt.savefig(buf, format="png")
            plt.close("all")
            return buf.getvalue()
        except Exception:
            return None
