import time
import unittest
from gllm.service.runs import RunManager
from gllm.loop.controller import LoopController
from gllm.loop.types import LoopRequest, TokenUsage
from gllm.loop.findings import Finding, Severity


def _err():
    return Finding("static", "unsupported_tool", Severity.ERROR, "T99 not allowed", blocking=True)


class _Gen:
    def __init__(self, outs):
        self._outs, self._i, self.last_usage = outs, -1, TokenUsage(1, 1)

    def __call__(self, prompt, ctx=None):
        self._i += 1
        return self._outs[min(self._i, len(self._outs) - 1)]


class _Val:
    name, tier = "scripted", "static"

    def __init__(self, per):
        self._per, self._i = per, -1

    def validate(self, gcode, ctx):
        self._i += 1
        return list(self._per[min(self._i, len(self._per) - 1)])


def _fake_factory(request):
    # attempt 1 fails (bad), attempt 2 clean -> status "passed", best index 2
    return LoopController(
        generator=_Gen(["G1 X1 Y1 F50\nM30", "G1 X2 Y2 F50\nM30"]),
        blocking_validators=[_Val([[_err()], []])],
        final_gate=None,
    )


def _drain(mgr, run_id, timeout=5.0):
    out, deadline = [], time.time() + timeout
    for ev in mgr.iter_events(run_id):
        out.append(ev)
        if time.time() > deadline:
            break
    return out


class RunManagerTests(unittest.TestCase):
    def test_run_streams_events_and_completes(self):
        mgr = RunManager(controller_factory=_fake_factory)
        run_id = mgr.start_run(LoopRequest(prompt="mill a square", max_attempts=4))
        events = _drain(mgr, run_id)
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "attempt_started")
        self.assertEqual(types[-1], "done")
        self.assertIn("findings", types)
        result = mgr.get_result(run_id)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["best_attempt_index"], 2)

    def test_iter_events_from_index(self):
        mgr = RunManager(controller_factory=_fake_factory)
        run_id = mgr.start_run(LoopRequest(prompt="p", max_attempts=4))
        _drain(mgr, run_id)  # let it finish
        tail = list(mgr.iter_events(run_id, from_index=1))
        full = list(mgr.iter_events(run_id, from_index=0))
        self.assertEqual(len(full) - len(tail), 1)

    def test_decision_recorded(self):
        mgr = RunManager(controller_factory=_fake_factory)
        run_id = mgr.start_run(LoopRequest(prompt="p", max_attempts=4))
        _drain(mgr, run_id)
        self.assertTrue(mgr.set_decision(run_id, "accept"))
        self.assertEqual(mgr.get_run(run_id).decision, "accept")

    def test_unknown_run_is_safe(self):
        mgr = RunManager(controller_factory=_fake_factory)
        self.assertIsNone(mgr.get_result("nope"))
        self.assertEqual(list(mgr.iter_events("nope")), [])
        self.assertFalse(mgr.set_decision("nope", "accept"))

    def test_list_setups_from_example_registry(self):
        mgr = RunManager(controller_factory=_fake_factory)
        setups = mgr.list_setups("config/vericut_setups.example.json")
        self.assertTrue(len(setups) >= 1)
        self.assertIn("id", setups[0])
        self.assertIn("description", setups[0])

    def test_plot_png_returns_png_bytes(self):
        mgr = RunManager(controller_factory=_fake_factory)
        run_id = mgr.start_run(LoopRequest(prompt="p", max_attempts=4))
        _drain(mgr, run_id)
        png = mgr.plot_png(run_id, 2)
        self.assertIsInstance(png, (bytes, bytearray))
        self.assertEqual(bytes(png[:4]), b"\x89PNG")
