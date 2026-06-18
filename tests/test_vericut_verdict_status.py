import unittest

from gllm.vericut.verdict import ParsedVericutLog, VericutArtifact, _status_for
from gllm.vericut.runner import VericutRunResult
from gllm.vericut.static_checks import StaticCheckReport


def _report(passed=True):
    # An empty findings tuple means no error-severity findings -> passed is True.
    return StaticCheckReport(())


def _run(returncode=0):
    return VericutRunResult(command=("vericut.bat",), returncode=returncode, stdout="", stderr="")


def _log(errors=0, warnings=0):
    return ParsedVericutLog(error_count=errors, warning_count=warnings, findings=())


def _artifacts(has_log=True):
    return (VericutArtifact(name="vericut.log", path="x", size_bytes=1),) if has_log else ()


class StatusForTests(unittest.TestCase):
    def test_ran_clean_is_accepted(self):
        self.assertEqual(_status_for(_report(), _run(0), _log(0), _artifacts(True)), "vericut_accepted")

    def test_ran_with_errors_is_rejected(self):
        self.assertEqual(_status_for(_report(), _run(0), _log(5), _artifacts(True)), "vericut_rejected")

    def test_nonzero_exit_with_error_log_is_still_rejected(self):
        # Vericut reported toolpath errors then exited non-zero -> a real rejection.
        self.assertEqual(_status_for(_report(), _run(1), _log(3), _artifacts(True)), "vericut_rejected")

    def test_launch_failure_no_log_is_unverified_not_rejected(self):
        # Process failed to run and produced no log -> could not verify, NOT a rejection.
        self.assertEqual(_status_for(_report(), _run(2), _log(0), _artifacts(False)), "vericut_unverified")

    def test_nonzero_exit_no_errors_is_unverified(self):
        # Abnormal exit but the log reports no toolpath errors -> unverified, not rejected.
        self.assertEqual(_status_for(_report(), _run(1), _log(0), _artifacts(True)), "vericut_unverified")

    def test_clean_exit_no_log_is_unverified(self):
        self.assertEqual(_status_for(_report(), _run(0), _log(0), _artifacts(False)), "vericut_unverified")


if __name__ == "__main__":
    unittest.main()
