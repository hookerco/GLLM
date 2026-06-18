import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.messages.ai import AIMessage
from gllm.proof.runner import ScenarioRequest, run_proof_with_llm


class ProofLLMModelDefaultTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / ".proof-model-tmp" / self._testMethodName
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

    def test_default_unknown_model_resolves_to_openrouter(self):
        reg = self._registry()
        # Default ScenarioRequest.model_name is "unknown"; no chain passed.
        request = ScenarioRequest(
            prompt="Mill a square pocket.", registry_path=reg, setup_id="s1",
            output_root=self.root / "runs", scenario_id="model-default-1",
        )

        class _StubChain:
            def invoke(self, _inputs):
                return AIMessage(content="G90\nT1 M06\nG0 Z1.0\nM30\n")

        captured = {}

        def fake_setup_model(model_name, openrouter_model_name=None):
            captured["model_name"] = model_name
            return object()  # a sentinel "model"

        def fake_setup_chain(_model):
            return _StubChain()

        with mock.patch("gllm.loop.generator.setup_model", side_effect=fake_setup_model), \
             mock.patch("gllm.loop.generator.setup_langchain_without_rag", side_effect=fake_setup_chain):
            packet = run_proof_with_llm(request)  # no chain -> lazy build -> patched setup_model

        self.assertEqual(captured["model_name"], "OpenRouter")
        self.assertEqual(packet.status, "accepted_static_only")
