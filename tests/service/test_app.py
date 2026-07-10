import time
import unittest
from starlette.testclient import TestClient
from gllm.service.app import create_app, build_loop_request
from gllm.service.runs import RunManager
from gllm.loop.controller import LoopController
from gllm.loop.types import TokenUsage, Mode
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
    return LoopController(
        generator=_Gen(["G1 X1 Y1 F50\nM30", "G1 X2 Y2 F50\nM30"]),
        blocking_validators=[_Val([[_err()], []])],
        final_gate=None,
    )


def _client():
    return TestClient(create_app(RunManager(controller_factory=_fake_factory)))


def _wait_done(client, run_id, tries=200):
    for _ in range(tries):
        res = client.get(f"/api/runs/{run_id}")
        if res.status_code == 200:
            return res
        time.sleep(0.03)
    return res


class BuildLoopRequestTests(unittest.TestCase):
    def test_defaults(self):
        req = build_loop_request({"prompt": "mill"})
        self.assertEqual(req.mode, Mode.GENERATE)
        self.assertEqual(req.model_name, "OpenRouter")
        self.assertEqual(req.max_attempts, 4)

    def test_improve_objective(self):
        req = build_loop_request({"prompt": "p", "mode": "improve", "objective": {"metric": "path_length"}})
        self.assertEqual(req.mode, Mode.IMPROVE)
        self.assertEqual(req.objective.metric, "path_length")

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            build_loop_request({"prompt": "p", "mode": "bogus"})

    def test_vericut_max_rounds_from_payload(self):
        self.assertEqual(build_loop_request({"prompt": "p"}).vericut_max_rounds, 2)
        self.assertEqual(
            build_loop_request({"prompt": "p", "vericut_max_rounds": 0}).vericut_max_rounds, 0
        )


class ApiTests(unittest.TestCase):
    def test_create_run_and_get_result(self):
        c = _client()
        r = c.post("/api/runs", json={"prompt": "mill", "max_attempts": 4})
        self.assertEqual(r.status_code, 200)
        run_id = r.json()["run_id"]
        res = _wait_done(c, run_id)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "passed")
        self.assertEqual(res.json()["best_attempt_index"], 2)

    def test_events_stream(self):
        c = _client()
        run_id = c.post("/api/runs", json={"prompt": "mill", "max_attempts": 4}).json()["run_id"]
        types = []
        with c.stream("GET", f"/api/runs/{run_id}/events") as s:
            for line in s.iter_lines():
                if line.startswith("event: "):
                    types.append(line[len("event: "):].strip())
        self.assertEqual(types[0], "attempt_started")
        self.assertEqual(types[-1], "done")
        self.assertIn("findings", types)

    def test_setups(self):
        c = _client()
        r = c.get("/api/setups", params={"registry_path": "config/vericut_setups.example.json"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(len(r.json()) >= 1)
        self.assertIn("id", r.json()[0])

    def test_setups_missing_param_400(self):
        self.assertEqual(_client().get("/api/setups").status_code, 400)

    def test_models(self):
        r = _client().get("/api/models")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("OpenRouter", body["models"])
        self.assertEqual(body["default"], "OpenRouter")

    def test_decision(self):
        c = _client()
        run_id = c.post("/api/runs", json={"prompt": "p", "max_attempts": 4}).json()["run_id"]
        _wait_done(c, run_id)
        r = c.post(f"/api/runs/{run_id}/decision", json={"action": "accept"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_decision_bad_action_400(self):
        c = _client()
        run_id = c.post("/api/runs", json={"prompt": "p", "max_attempts": 4}).json()["run_id"]
        _wait_done(c, run_id)
        self.assertEqual(c.post(f"/api/runs/{run_id}/decision", json={"action": "nope"}).status_code, 400)

    def test_unknown_run_404(self):
        self.assertEqual(_client().get("/api/runs/nope").status_code, 404)

    def test_plot_endpoint_returns_png(self):
        c = _client()
        run_id = c.post("/api/runs", json={"prompt": "p", "max_attempts": 4}).json()["run_id"]
        _wait_done(c, run_id)
        r = c.get(f"/api/runs/{run_id}/plot/2")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/png")
        self.assertEqual(r.content[:4], b"\x89PNG")
