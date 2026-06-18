import unittest
from gllm.loop.scoring import findings_penalty, objective_value, path_length, tool_changes
from gllm.loop.findings import Finding, Severity
from gllm.loop.types import Objective


class ScoringTests(unittest.TestCase):
    def test_no_findings_is_zero_penalty(self):
        self.assertEqual(findings_penalty([]), 0.0)

    def test_blocking_error_outweighs_nonblocking_warning(self):
        err = Finding("static", "x", Severity.ERROR, "m", blocking=True)
        warn = Finding("lint", "continuity", Severity.WARNING, "m", blocking=False)
        self.assertGreater(findings_penalty([err]), findings_penalty([warn]))

    def test_tool_changes_counts_t_words(self):
        self.assertEqual(tool_changes("T1 M06\nG0 X1\nT2 M06\nM30"), 2.0)

    def test_path_length_sums_segments(self):
        # parse_gcode omits the implicit (0,0) origin -> points [(3,0),(3,4)] -> one 4-unit segment.
        gcode = "G1 X3 Y0 F50\nG1 X3 Y4 F50\nM30"
        self.assertAlmostEqual(path_length(gcode), 4.0, places=3)

    def test_objective_value_none_objective(self):
        self.assertIsNone(objective_value("G1 X1 Y1\nM30", None))

    def test_objective_value_robust_to_garbage(self):
        # Must return a float or None, never raise.
        self.assertIsInstance(objective_value("not gcode", Objective("path_length")), (float, type(None)))
