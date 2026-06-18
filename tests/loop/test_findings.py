import unittest
from gllm.loop.findings import (
    Finding, Severity, from_lint, from_static, from_vericut, prioritize, format_findings,
)
from gllm.vericut.static_checks import StaticFinding


class FindingsTests(unittest.TestCase):
    def test_from_lint_ok_returns_none(self):
        self.assertIsNone(from_lint("safety", (True, None)))

    def test_from_lint_failure_builds_blocking_error(self):
        f = from_lint("safety", (False, "rapid through material"))
        self.assertEqual(f.source, "lint")
        self.assertEqual(f.code, "safety")
        self.assertEqual(f.severity, Severity.ERROR)
        self.assertTrue(f.blocking)
        self.assertIn("rapid", f.message)

    def test_from_lint_can_be_nonblocking_warning(self):
        f = from_lint("continuity", (False, "discontinuity"), blocking=False, severity=Severity.WARNING)
        self.assertFalse(f.blocking)
        self.assertEqual(f.severity, Severity.WARNING)

    def test_from_static_maps_error_to_blocking(self):
        sf = StaticFinding(code="unsupported_tool", severity="error", message="T99 not allowed", line_number=2)
        f = from_static(sf)
        self.assertEqual(f.source, "static")
        self.assertEqual(f.code, "unsupported_tool")
        self.assertEqual(f.line_number, 2)
        self.assertTrue(f.blocking)

    def test_from_vericut_payload(self):
        f = from_vericut({"code": "collision", "severity": "error", "message": "X exceeded", "line_number": 4})
        self.assertEqual(f.source, "vericut")
        self.assertEqual(f.code, "collision")
        self.assertTrue(f.blocking)

    def test_prioritize_orders_errors_and_static_first(self):
        warn = from_lint("continuity", (False, "x"), blocking=False, severity=Severity.WARNING)
        err_static = from_static(StaticFinding("unsupported_tool", "error", "x", 5))
        ordered = prioritize([warn, err_static])
        self.assertEqual(ordered[0].code, "unsupported_tool")

    def test_format_findings_handles_empty(self):
        self.assertIn("No concrete findings", format_findings([]))
