import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from gllm.proof.runner import AttemptContext, ScenarioRequest, build_repair_prompt, run_proof_scenario
from gllm.vericut.runner import VericutRunResult


class ProofRunTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / ".proof-test-tmp" / self._testMethodName
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def write_registry(self, *, safe_z_min=0.25):
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
                            "safe_z_min": safe_z_min,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return registry_path

    def request(self, registry_path, *, scenario_id="scenario-001", run_vericut=False, max_repair_attempts=0):
        return ScenarioRequest(
            prompt="Mill a simple square pocket.",
            registry_path=registry_path,
            setup_id="sample_haas",
            output_root=self.root / "proof-runs",
            scenario_id=scenario_id,
            model_name="FakeModel",
            run_vericut=run_vericut,
            max_repair_attempts=max_repair_attempts,
        )

    def test_static_only_run_writes_evidence_packet_and_prepares_job(self):
        registry_path = self.write_registry()

        packet = run_proof_scenario(
            self.request(registry_path),
            candidate_generator=lambda prompt, context: "G90\nT1 M06\nG0 Z1.0\nM30\n",
        )

        self.assertEqual(packet.status, "accepted_static_only")
        self.assertEqual(packet.final_attempt, 1)
        self.assertTrue(packet.packet_file.exists())
        self.assertTrue(packet.summary_file.exists())

        packet_payload = json.loads(packet.packet_file.read_text(encoding="utf-8"))
        self.assertEqual(packet_payload["status"], "accepted_static_only")
        self.assertEqual(packet_payload["operator_action"], "rerun_vericut")
        self.assertEqual(packet_payload["request"]["setup_id"], "sample_haas")
        self.assertEqual(packet_payload["attempts"][0]["static_report"]["passed"], True)
        self.assertEqual(packet_payload["attempts"][0]["operator_action"], "rerun_vericut")
        self.assertEqual(packet_payload["attempts"][0]["vericut"]["status"], "skipped")
        self.assertTrue(Path(packet_payload["attempts"][0]["job"]["generated_nc"]).exists())

    def test_static_findings_feed_repair_attempt(self):
        registry_path = self.write_registry()
        repair_contexts = []

        def candidate_generator(prompt, context):
            if context.attempt_number == 1:
                return "G90\nT99 M06\nG0 Z0.1\nM30\n"
            return "G90\nT1 M06\nG0 Z1.0\nM30\n"

        def repair_prompt_builder(context):
            repair_contexts.append(context.repair_context)
            return "Repair the G-code from the concrete findings."

        packet = run_proof_scenario(
            self.request(registry_path, scenario_id="scenario-repair", max_repair_attempts=1),
            candidate_generator=candidate_generator,
            repair_prompt_builder=repair_prompt_builder,
        )

        self.assertEqual(packet.status, "accepted_static_only")
        self.assertEqual(packet.final_attempt, 2)
        self.assertEqual(packet.attempts[0].status, "rejected_static")
        self.assertEqual(packet.attempts[1].status, "accepted_static_only")
        self.assertIn("unsupported_tool", [finding["code"] for finding in repair_contexts[0]])
        self.assertIn("rapid_below_safe_z", [finding["code"] for finding in repair_contexts[0]])

    def test_vericut_log_rejection_becomes_repair_context(self):
        registry_path = self.write_registry()

        def fake_run_vericut(setup, job, *, batch=True, timeout_seconds=None):
            log_path = job.output_dir / "vericut.log"
            log_path.write_text(
                """
Error for line 4, Filename: generated.nc
G0 Z1.0
 Error: Component "X" exceeded maximum limit at line: (4)
Number of Errors: 1
Number of Warnings: 0
""".strip(),
                encoding="utf-8",
            )
            return VericutRunResult(
                command=("vericut.bat", "BATCH", str(job.project_file)),
                returncode=0,
                stdout="",
                stderr="",
                artifacts=(log_path,),
            )

        packet = run_proof_scenario(
            self.request(registry_path, scenario_id="scenario-vericut", run_vericut=True),
            candidate_generator=lambda prompt, context: "G90\nT1 M06\nG0 Z1.0\nM30\n",
            run_vericut_fn=fake_run_vericut,
        )

        self.assertEqual(packet.status, "rejected_vericut")
        self.assertEqual(packet.operator_action, "fix_prompt")
        self.assertEqual(packet.attempts[0].vericut["status"], "vericut_rejected")
        self.assertEqual(packet.repair_context[0]["line_number"], 4)
        self.assertIn("Component", packet.repair_context[0]["message"])

    def test_vericut_unverified_output_is_not_a_prompt_rejection(self):
        registry_path = self.write_registry()

        def fake_run_vericut(setup, job, *, batch=True, timeout_seconds=None):
            return VericutRunResult(
                command=("vericut.bat", "BATCH", str(job.project_file)),
                returncode=0,
                stdout="",
                stderr="",
                artifacts=(),
            )

        packet = run_proof_scenario(
            self.request(registry_path, scenario_id="scenario-vericut-unverified", run_vericut=True),
            candidate_generator=lambda prompt, context: "G90\nT1 M06\nG0 Z1.0\nM30\n",
            run_vericut_fn=fake_run_vericut,
        )

        self.assertEqual(packet.status, "blocked_vericut_unverified")
        self.assertEqual(packet.operator_action, "rerun_vericut")
        self.assertEqual(packet.attempts[0].vericut["status"], "vericut_unverified")

    def test_repair_prompt_includes_prior_candidate_evidence_and_setup_constraints(self):
        registry_path = self.write_registry()
        request = self.request(registry_path, scenario_id="scenario-repair-prompt")
        previous_candidate_file = self.root / "candidate.nc"
        previous_candidate_file.write_text(
            "G90\nT99 M06\nG0 Z0.1\nM30\n",
            encoding="utf-8",
        )
        context = AttemptContext(
            request=request,
            scenario_id="scenario-repair-prompt",
            attempt_number=2,
            repair_context=(
                {
                    "code": "unsupported_tool",
                    "line_number": 2,
                    "message": "Tool T99 is not allowed by setup sample_haas.",
                },
            ),
            previous_gcode="G90\nT99 M06\nG0 Z0.1\nM30\n",
            previous_status="rejected_static",
            previous_cleaned_gcode_file=previous_candidate_file,
            setup_constraints=(
                "allowed_tools: T1",
                "required_modes: G90",
                "safe_z_min: 0.25",
            ),
        )

        prompt = build_repair_prompt(context)

        self.assertIn("Repair attempt: 2", prompt)
        self.assertIn("Previous candidate file:", prompt)
        self.assertIn(str(previous_candidate_file), prompt)
        self.assertIn("allowed_tools: T1", prompt)
        self.assertIn("unsupported_tool line 2", prompt)
        self.assertIn("G0 Z0.1", prompt)

    def test_missing_setup_paths_block_before_generation(self):
        registry_path = self.root / "missing_paths_setups.json"
        registry_path.write_text(
            json.dumps(
                {
                    "setups": [
                        {
                            "id": "sample_haas",
                            "description": "Broken setup",
                            "vericut_bat": str(self.root / "missing-vericut.bat"),
                            "project_template": str(self.root / "missing.vcproject"),
                            "library_search_paths": [str(self.root / "missing-support")],
                            "allowed_tools": ["T1"],
                            "required_modes": ["G90"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        packet = run_proof_scenario(
            self.request(registry_path, scenario_id="scenario-blocked"),
            candidate_generator=lambda prompt, context: self.fail("generation should not run"),
        )

        self.assertEqual(packet.status, "blocked_setup_paths")
        self.assertEqual(packet.attempts, ())
        self.assertIn("vericut_bat", packet.blockers[0])

    def test_cli_can_prove_existing_candidate_file(self):
        registry_path = self.write_registry()
        candidate_file = self.root / "candidate.nc"
        candidate_file.write_text("G90\nT1 M06\nG0 Z1.0\nM30\n", encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "gllm.proof.cli",
                "--registry",
                str(registry_path),
                "--setup-id",
                "sample_haas",
                "--prompt",
                "Mill a simple square pocket.",
                "--candidate-gcode-file",
                str(candidate_file),
                "--output-root",
                str(self.root / "cli-proof-runs"),
                "--scenario-id",
                "scenario-cli",
            ],
            capture_output=True,
            check=True,
            text=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "accepted_static_only")
        self.assertTrue(Path(payload["packet_file"]).exists())


if __name__ == "__main__":
    unittest.main()
