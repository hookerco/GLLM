import json
import unittest
from gllm.service.serialize import attempt_to_dict, event_to_dict, sse_frame
from gllm.loop.types import Attempt, LoopEvent, LoopResult, LoopRequest, Mode
from gllm.loop.findings import Finding, Severity


def _attempt():
    f = Finding("static", "unsupported_tool", Severity.ERROR, "T99 not allowed", line_number=2)
    return Attempt(1, "G1 X1 Y1\nM30", (f,), 1000.0, None)


class SerializeTests(unittest.TestCase):
    def test_attempt_to_dict(self):
        d = attempt_to_dict(_attempt())
        self.assertEqual(d["number"], 1)
        self.assertEqual(d["gcode"], "G1 X1 Y1\nM30")
        self.assertEqual(d["score"], 1000.0)
        self.assertIsNone(d["objective_value"])
        self.assertEqual(d["blocking_count"], 1)
        self.assertEqual(d["findings"][0]["code"], "unsupported_tool")

    def test_event_findings_carries_attempt(self):
        d = event_to_dict(LoopEvent("findings", attempt=_attempt()))
        self.assertEqual(d["type"], "findings")
        self.assertEqual(d["attempt"]["number"], 1)
        self.assertNotIn("result", d)

    def test_event_attempt_started_payload(self):
        d = event_to_dict(LoopEvent("attempt_started", payload={"attempt": 3}))
        self.assertEqual(d["type"], "attempt_started")
        self.assertEqual(d["payload"], {"attempt": 3})

    def test_event_done_has_result(self):
        req = LoopRequest(prompt="p", mode=Mode.GENERATE)
        res = LoopResult(req, "sc-1", "passed", best=_attempt(), history=(_attempt(),))
        d = event_to_dict(LoopEvent("done", result=res))
        self.assertEqual(d["type"], "done")
        self.assertEqual(d["result"]["status"], "passed")
        self.assertEqual(d["result"]["best_attempt_index"], 1)

    def test_event_dict_is_json_serializable(self):
        d = event_to_dict(LoopEvent("findings", attempt=_attempt()))
        json.dumps(d)  # must not raise

    def test_sse_frame_format(self):
        frame = sse_frame({"type": "attempt_started", "payload": {"attempt": 1}})
        self.assertTrue(frame.startswith("event: attempt_started\ndata: "))
        self.assertTrue(frame.endswith("\n\n"))
        # the data line must be valid JSON
        data_line = frame.split("\ndata: ", 1)[1].rstrip("\n")
        self.assertEqual(json.loads(data_line)["type"], "attempt_started")
