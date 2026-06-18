import json
import shutil
import unittest
from pathlib import Path

from gllm.loop.vericut_gate import VericutFinalGate
from gllm.loop.types import LoopRequest, ValidationContext
from gllm.vericut.runner import VericutRunResult


class VericutGateTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / ".loop-vericut-tmp" / self._testMethodName
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def _registry(self):
        support = self.root / "support"
        support.mkdir()
        (support / "fixture.stl").write_text("fixture", encoding="utf-8")
        (support / "tools.tls").write_text("tools", encoding="utf-8")
        (support / "control.ctl").write_text("control", encoding="utf-8")
        project = self.root / "template.vcproject"
        project.write_text(
            '<VcProject Version="9.5" Unit="inch"><Setup Name="d">'
            '<Component Name="F"><STL><File>fixture.stl</File></STL></Component>'
            '<NCPrograms Type="gcode"><NCProgram Use="on"><File>old.mcd</File></NCProgram></NCPrograms>'
            '<APT><Library>tools.tls</Library></APT><Control><File>control.ctl</File></Control>'
            "</Setup></VcProject>",
            encoding="utf-8",
        )
        bat = self.root / "vericut.bat"
        bat.write_text("@echo off", encoding="utf-8")
        reg = self.root / "setups.json"
        reg.write_text(json.dumps({"setups": [{
            "id": "s1", "description": "d", "vericut_bat": str(bat),
            "project_template": str(project), "library_search_paths": [str(support)],
            "allowed_tools": ["T1"], "required_modes": ["G90"], "safe_z_min": 0.25,
        }]}), encoding="utf-8")
        return reg

    def test_rejected_verdict_from_log(self):
        reg = self._registry()
        from gllm.vericut.registry import load_setup_registry
        setup = load_setup_registry(reg).get("s1")
        ctx = ValidationContext.from_setup(setup)
        req = LoopRequest(prompt="p", registry_path=reg, setup_id="s1", run_vericut=True,
                          output_root=self.root / "runs")

        def fake_run_vericut(setup, job, *, batch=True, timeout_seconds=None):
            log = job.output_dir / "vericut.log"
            log.write_text("Error for line 4\n Error: Component \"X\" exceeded limit at line: (4)\n"
                           "Number of Errors: 1\nNumber of Warnings: 0", encoding="utf-8")
            return VericutRunResult(command=("vericut.bat",), returncode=0, stdout="", stderr="", artifacts=(log,))

        gate = VericutFinalGate(run_vericut_fn=fake_run_vericut)
        verdict = gate("G90\nT1 M06\nG0 Z1.0\nM30\n", ctx, req)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["status"], "vericut_rejected")

    def test_missing_setup_returns_none(self):
        ctx = ValidationContext()  # no setup
        req = LoopRequest(prompt="p", run_vericut=True)
        self.assertIsNone(VericutFinalGate().__call__("G90\nM30\n", ctx, req))


if __name__ == "__main__":
    unittest.main()
