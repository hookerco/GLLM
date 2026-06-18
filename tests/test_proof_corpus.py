import json
import shutil
import unittest
from pathlib import Path

from gllm.proof.corpus import load_prompt_corpus, run_prompt_corpus


class PromptCorpusTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / ".proof-test-tmp" / self._testMethodName
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def write_registry(self):
        support = self.root / "support"
        support.mkdir()
        (support / "fixture.stl").write_text("fixture", encoding="utf-8")
        (support / "tools.tls").write_text("tools", encoding="utf-8")
        (support / "control.ctl").write_text("control", encoding="utf-8")
        project = self.root / "template.vcproject"
        project.write_text(
            """
<VcProject Version="9.5" Unit="inch">
  <Setup Name="proof-demo">
    <Component Name="Fixture">
      <STL><File>fixture.stl</File></STL>
    </Component>
    <NCPrograms Type="gcode">
      <NCProgram Use="on"><File>old_program.mcd</File></NCProgram>
    </NCPrograms>
    <APT><Library>tools.tls</Library></APT>
    <Control><File>control.ctl</File></Control>
  </Setup>
</VcProject>
""".strip(),
            encoding="utf-8",
        )
        vericut_bat = self.root / "vericut.bat"
        vericut_bat.write_text("@echo off", encoding="utf-8")
        registry_path = self.root / "setups.json"
        registry_path.write_text(
            json.dumps(
                {
                    "setups": [
                        {
                            "id": "sample_haas",
                            "description": "Sample Haas setup",
                            "vericut_bat": str(vericut_bat),
                            "project_template": str(project),
                            "library_search_paths": [str(support)],
                            "allowed_tools": ["T1"],
                            "required_modes": ["G90"],
                            "expected_units": "inch",
                            "feed_rate_min": 1.0,
                            "feed_rate_max": 200.0,
                            "spindle_speed_min": 1.0,
                            "spindle_speed_max": 10000.0,
                            "work_envelope": {
                                "x_min": -10.0,
                                "x_max": 10.0,
                                "y_min": -10.0,
                                "y_max": 10.0,
                                "z_min": -2.0,
                                "z_max": 6.0,
                            },
                            "safe_z_min": 0.0,
                        },
                        {
                            "id": "sample_router",
                            "description": "Second sample setup",
                            "vericut_bat": str(vericut_bat),
                            "project_template": str(project),
                            "library_search_paths": [str(support)],
                            "allowed_tools": ["T2"],
                            "required_modes": ["G90"],
                            "expected_units": "inch",
                            "work_envelope": {
                                "x_min": -20.0,
                                "x_max": 20.0,
                                "y_min": -20.0,
                                "y_max": 20.0,
                                "z_min": -3.0,
                                "z_max": 8.0,
                            },
                            "safe_z_min": 0.0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return registry_path

    def test_prompt_corpus_loads_relative_candidates_and_runs_multiple_setups(self):
        registry_path = self.write_registry()
        pass_candidate = self.root / "pass.nc"
        pass_candidate.write_text("G20\nG90\nT1 M06\nS2500 M03\nF20\nG0 Z1.0\nM30\n", encoding="utf-8")
        bad_tool_candidate = self.root / "bad_tool.nc"
        bad_tool_candidate.write_text("G20\nG90\nT99 M06\nG0 Z1.0\nM30\n", encoding="utf-8")
        second_setup_candidate = self.root / "second_setup.nc"
        second_setup_candidate.write_text("G20\nG90\nT2 M06\nG0 Z1.0\nM30\n", encoding="utf-8")
        corpus_path = self.root / "corpus.json"
        corpus_path.write_text(
            json.dumps(
                {
                    "name": "sample proof corpus",
                    "scenarios": [
                        {
                            "id": "sample-pass",
                            "setup_id": "sample_haas",
                            "prompt": "Mill a one inch square pocket.",
                            "candidate_gcode_file": "pass.nc",
                            "expected_status": "accepted_static_only",
                            "tags": ["easy_pass"],
                        },
                        {
                            "id": "sample-bad-tool",
                            "setup_id": "sample_haas",
                            "prompt": "Use an unsupported tool.",
                            "candidate_gcode_file": "bad_tool.nc",
                            "expected_status": "rejected_static",
                            "tags": ["bad_tool"],
                        },
                        {
                            "id": "second-setup-pass",
                            "setup_id": "sample_router",
                            "prompt": "Run the same proof loop on a second setup.",
                            "candidate_gcode_file": "second_setup.nc",
                            "expected_status": "accepted_static_only",
                            "tags": ["second_setup"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        corpus = load_prompt_corpus(corpus_path)
        result = run_prompt_corpus(
            corpus,
            registry_path=registry_path,
            output_root=self.root / "proof-runs",
        )

        self.assertEqual(corpus.name, "sample proof corpus")
        self.assertEqual(len(corpus.scenarios), 3)
        self.assertTrue(result.passed)
        self.assertEqual(result.total, 3)
        self.assertEqual(result.statuses["sample-pass"], "accepted_static_only")
        self.assertEqual(result.statuses["sample-bad-tool"], "rejected_static")
        self.assertEqual(result.statuses["second-setup-pass"], "accepted_static_only")
        self.assertEqual(result.setup_ids, ("sample_haas", "sample_router"))
        self.assertTrue(result.summary_file.exists())
