import json
import shutil
import unittest
from pathlib import Path

from gllm.proof.runner import run_proof_with_llm


class _StubChain:
    """Stands in for a LangChain chain: returns scripted G-code by attempt."""
    def __init__(self):
        self._calls = 0

    def invoke(self, _inputs):
        from langchain_core.messages.ai import AIMessage
        self._calls += 1
        # First call: bad tool; second call (repair): valid.
        text = "G90\nT99 M06\nG0 Z1.0\nM30\n" if self._calls == 1 else "G90\nT1 M06\nG0 Z1.0\nM30\n"
        return AIMessage(content=text)


class ProofLLMBridgeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / ".proof-llm-tmp" / self._testMethodName
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def _registry(self):
        support = self.root / "support"
        support.mkdir()
        (support / "fixture.stl").write_text("f", encoding="utf-8")
        (support / "tools.tls").write_text("t", encoding="utf-8")
        (support / "control.ctl").write_text("c", encoding="utf-8")
        project = self.root / "template.vcproject"
        project.write_text(
            '<VcProject Version="9.5" Unit="inch"><Setup Name="d">'
            '<Component Name="F"><STL><File>fixture.stl</File></STL></Component>'
            '<NCPrograms Type="gcode"><NCProgram Use="on"><File>old.mcd</File></NCProgram></NCPrograms>'
            '<APT><Library>tools.tls</Library></APT><Control><File>control.ctl</File></Control>'
            "</Setup></VcProject>", encoding="utf-8")
        bat = self.root / "vericut.bat"
        bat.write_text("@echo off", encoding="utf-8")
        reg = self.root / "setups.json"
        reg.write_text(json.dumps({"setups": [{
            "id": "s1", "description": "d", "vericut_bat": str(bat),
            "project_template": str(project), "library_search_paths": [str(support)],
            "allowed_tools": ["T1"], "required_modes": ["G90"], "safe_z_min": 0.25,
        }]}), encoding="utf-8")
        return reg

    def test_llm_bridge_repairs_static_rejection_into_acceptance(self):
        from gllm.proof.runner import ScenarioRequest
        reg = self._registry()
        request = ScenarioRequest(
            prompt="Mill a square pocket.", registry_path=reg, setup_id="s1",
            output_root=self.root / "runs", scenario_id="bridge-1", max_repair_attempts=1,
        )
        packet = run_proof_with_llm(request, chain=_StubChain())
        self.assertEqual(packet.attempts[0].status, "rejected_static")
        self.assertEqual(packet.status, "accepted_static_only")
        self.assertEqual(packet.final_attempt, 2)
