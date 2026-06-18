import unittest
from gllm.loop.validators import LintValidator, StaticPolicyValidator
from gllm.loop.types import ValidationContext
from gllm.loop.findings import Severity
from gllm.vericut.registry import load_setup_registry


GOOD = "G90\nT1 M06\nS800\nG0 Z1.0\nG1 Z-0.1 F50\nM30\n"


class LintValidatorTests(unittest.TestCase):
    def test_clean_program_has_no_blocking_findings(self):
        findings = LintValidator().validate(GOOD, ValidationContext())
        self.assertEqual([f for f in findings if f.blocking], [])

    def test_rapid_through_material_is_blocking(self):
        bad = "G1 X1 Y1 F50\nG0 X5 Y5\nM30\n"
        findings = LintValidator().validate(bad, ValidationContext())
        codes = {f.code for f in findings if f.blocking}
        self.assertIn("safety", codes)

    def test_continuity_is_nonblocking_warning(self):
        # validate_continuity is known-noisy; it must never block the loop.
        findings = LintValidator().validate("G1 X1 Y1\nG1 X9 Y9\nM30\n", ValidationContext())
        for f in findings:
            if f.code == "continuity":
                self.assertFalse(f.blocking)
                self.assertEqual(f.severity, Severity.WARNING)

    def test_validator_never_raises(self):
        # Degenerate input must yield findings, not an exception.
        findings = LintValidator().validate("(comment only)\n%\n", ValidationContext())
        self.assertIsInstance(findings, list)


class StaticPolicyValidatorTests(unittest.TestCase):
    def test_no_setup_means_no_findings(self):
        self.assertEqual(StaticPolicyValidator().validate(GOOD, ValidationContext()), [])
