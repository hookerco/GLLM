from pathlib import Path
import unittest


def harness_scripts() -> str:
    paths = [
        Path("scripts/start_streamlit_detached.ps1"),
        Path("scripts/streamlit-detached-worker.ps1"),
    ]
    return "\n".join(path.read_text() for path in paths if path.exists())


class DevHarnessTests(unittest.TestCase):
    def test_detached_streamlit_launcher_records_process_and_logs(self):
        script = harness_scripts()

        self.assertIn("Start-Process", script)
        self.assertIn("-WindowStyle Hidden", script)
        self.assertIn("-PassThru", script)
        self.assertIn("-RedirectStandardOutput", script)
        self.assertIn("-RedirectStandardError", script)
        self.assertIn("streamlit-detached.pid.txt", script)

    def test_detached_streamlit_launcher_waits_for_port_with_timeout(self):
        script = harness_scripts()

        self.assertIn("Get-NetTCPConnection", script)
        self.assertIn("$TimeoutSeconds", script)
        self.assertIn("Start-Sleep", script)

    def test_detached_streamlit_launcher_falls_back_to_netstat_port_detection(self):
        script = harness_scripts()

        self.assertIn("Get-PortListener", script)
        self.assertIn("netstat -ano", script)

    def test_detached_streamlit_launcher_uses_short_lived_worker(self):
        script = Path("scripts/start_streamlit_detached.ps1").read_text()

        self.assertIn("streamlit-detached-worker.ps1", script)
        self.assertIn("Start-StreamlitWorker", script)
        self.assertIn("Worker launcher PID", script)
        self.assertIn("exit 0", script)


if __name__ == "__main__":
    unittest.main()
