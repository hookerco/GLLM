import json
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path

from gllm.vericut.registry import load_setup_registry
from gllm.vericut.runner import (
    VericutJob,
    VericutRunResult,
    build_vericut_command,
    prepare_job_workspace,
)
from gllm.vericut.static_checks import StaticCheckReport, check_gcode
from gllm.vericut.verdict import parse_vericut_log, write_verdict_packet


@contextmanager
def workspace_temp_dir(name: str):
    root = Path.cwd() / ".proof-test-tmp" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.exists():
            shutil.rmtree(root)


class VericutIntegrationTests(unittest.TestCase):
    def test_setup_registry_loads_sample_setup(self):
        with workspace_temp_dir("registry-loads-sample-setup") as root:
            project = root / "sample.vcproject"
            project.write_text("<VcProject />", encoding="utf-8")
            vericut_bat = root / "vericut.bat"
            vericut_bat.write_text("@echo off", encoding="utf-8")
            registry_path = root / "setups.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "setups": [
                            {
                                "id": "sample_haas",
                                "description": "Sample Haas setup",
                                "vericut_bat": str(vericut_bat),
                                "project_template": str(project),
                                "library_search_paths": [str(root)],
                                "allowed_tools": ["T1", "T2"],
                                "required_modes": ["G20", "G90"],
                                "safe_z_min": 0.25,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            registry = load_setup_registry(registry_path)

        setup = registry.get("sample_haas")
        self.assertEqual(setup.description, "Sample Haas setup")
        self.assertEqual(setup.allowed_tools, frozenset({"T1", "T2"}))
        self.assertEqual(setup.required_modes, frozenset({"G20", "G90"}))

    def test_setup_registry_loads_static_policy_fields(self):
        with workspace_temp_dir("registry-loads-static-policy") as root:
            project = root / "sample.vcproject"
            project.write_text("<VcProject />", encoding="utf-8")
            vericut_bat = root / "vericut.bat"
            vericut_bat.write_text("@echo off", encoding="utf-8")
            registry_path = root / "setups.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "setups": [
                            {
                                "id": "sample_haas",
                                "description": "Sample Haas setup",
                                "vericut_bat": str(vericut_bat),
                                "project_template": str(project),
                                "library_search_paths": [str(root)],
                                "allowed_tools": ["T1", "T2"],
                                "required_modes": ["G90"],
                                "expected_units": "inch",
                                "feed_rate_min": 2.0,
                                "feed_rate_max": 120.0,
                                "spindle_speed_min": 500.0,
                                "spindle_speed_max": 8000.0,
                                "work_envelope": {
                                    "x_min": -10.0,
                                    "x_max": 10.0,
                                    "y_min": -8.0,
                                    "y_max": 8.0,
                                    "z_min": -2.0,
                                    "z_max": 6.0,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            setup = load_setup_registry(registry_path).get("sample_haas")

        self.assertEqual(setup.expected_units, "inch")
        self.assertEqual(setup.feed_rate_min, 2.0)
        self.assertEqual(setup.feed_rate_max, 120.0)
        self.assertEqual(setup.spindle_speed_min, 500.0)
        self.assertEqual(setup.spindle_speed_max, 8000.0)
        self.assertEqual(setup.work_envelope.x_min, -10.0)
        self.assertEqual(setup.work_envelope.x_max, 10.0)
        self.assertEqual(setup.work_envelope.y_min, -8.0)
        self.assertEqual(setup.work_envelope.y_max, 8.0)
        self.assertEqual(setup.work_envelope.z_min, -2.0)
        self.assertEqual(setup.work_envelope.z_max, 6.0)

    def test_static_check_reports_missing_modes_and_unsupported_tool(self):
        with workspace_temp_dir("static-check-missing-modes") as root:
            project = root / "sample.vcproject"
            project.write_text("<VcProject />", encoding="utf-8")
            vericut_bat = root / "vericut.bat"
            vericut_bat.write_text("@echo off", encoding="utf-8")
            registry_path = root / "setups.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "setups": [
                            {
                                "id": "sample_haas",
                                "description": "Sample Haas setup",
                                "vericut_bat": str(vericut_bat),
                                "project_template": str(project),
                                "library_search_paths": [str(root)],
                                "allowed_tools": ["T1"],
                                "required_modes": ["G20", "G90"],
                                "safe_z_min": 0.25,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            setup = load_setup_registry(registry_path).get("sample_haas")

            report = check_gcode("T99 M06\nG0 X1.0 Y1.0 Z0.1\nM30\n", setup)

        self.assertFalse(report.passed)
        self.assertIn("missing_required_mode", [finding.code for finding in report.findings])
        self.assertIn("unsupported_tool", [finding.code for finding in report.findings])
        self.assertIn("rapid_below_safe_z", [finding.code for finding in report.findings])

    def test_static_check_enforces_setup_policy_bounds(self):
        with workspace_temp_dir("static-check-policy-bounds") as root:
            project = root / "sample.vcproject"
            project.write_text("<VcProject />", encoding="utf-8")
            vericut_bat = root / "vericut.bat"
            vericut_bat.write_text("@echo off", encoding="utf-8")
            registry_path = root / "setups.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "setups": [
                            {
                                "id": "sample_haas",
                                "description": "Sample Haas setup",
                                "vericut_bat": str(vericut_bat),
                                "project_template": str(project),
                                "library_search_paths": [str(root)],
                                "allowed_tools": ["T1"],
                                "required_modes": ["G90"],
                                "expected_units": "inch",
                                "feed_rate_min": 2.0,
                                "feed_rate_max": 120.0,
                                "spindle_speed_min": 500.0,
                                "spindle_speed_max": 8000.0,
                                "work_envelope": {
                                    "x_min": -10.0,
                                    "x_max": 10.0,
                                    "y_min": -8.0,
                                    "y_max": 8.0,
                                    "z_min": -2.0,
                                    "z_max": 6.0,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            setup = load_setup_registry(registry_path).get("sample_haas")

            rejected = check_gcode(
                "G21 G90\nT1 M06\nS12000 M03\nF0.5\nG1 X12.5 Y0.0 Z-3.0\nM30\n",
                setup,
            )
            missing_units = check_gcode(
                "G90\nT1 M06\nS2500 M03\nF25.0\nG1 X1.0 Y1.0 Z-0.5\nM30\n",
                setup,
            )
            accepted = check_gcode(
                "G20 G90\nT1 M06\nS2500 M03\nF25.0\nG1 X1.0 Y1.0 Z-0.5\nM30\n",
                setup,
            )

        rejected_codes = [finding.code for finding in rejected.findings]
        self.assertFalse(rejected.passed)
        self.assertIn("unexpected_units", rejected_codes)
        self.assertIn("feed_rate_out_of_range", rejected_codes)
        self.assertIn("spindle_speed_out_of_range", rejected_codes)
        self.assertIn("axis_out_of_envelope", rejected_codes)
        self.assertFalse(missing_units.passed)
        self.assertIn("missing_expected_units", [finding.code for finding in missing_units.findings])
        self.assertTrue(accepted.passed)

    def test_static_check_parses_tokens_without_spaces(self):
        with workspace_temp_dir("static-check-token-parsing") as root:
            project = root / "sample.vcproject"
            project.write_text("<VcProject />", encoding="utf-8")
            vericut_bat = root / "vericut.bat"
            vericut_bat.write_text("@echo off", encoding="utf-8")
            registry_path = root / "setups.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "setups": [
                            {
                                "id": "sample_haas",
                                "description": "Sample Haas setup",
                                "vericut_bat": str(vericut_bat),
                                "project_template": str(project),
                                "library_search_paths": [str(root)],
                                "allowed_tools": ["T1"],
                                "required_modes": ["G90"],
                                "safe_z_min": 0.25,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            setup = load_setup_registry(registry_path).get("sample_haas")

            report = check_gcode("N0020T99M6\nN0030G90G0X5.Y4.Z0.1\nM30\n", setup)

        self.assertFalse(report.passed)
        self.assertNotIn("missing_required_mode", [finding.code for finding in report.findings])
        self.assertIn("unsupported_tool", [finding.code for finding in report.findings])
        self.assertIn("rapid_below_safe_z", [finding.code for finding in report.findings])

    def test_prepare_job_workspace_repoints_project_to_generated_nc(self):
        with workspace_temp_dir("prepare-job-workspace") as root:
            support = root / "support"
            support.mkdir()
            (support / "fixture.stl").write_text("fixture", encoding="utf-8")
            (support / "tools.tls").write_text("tools", encoding="utf-8")
            (support / "control.ctl").write_text("control", encoding="utf-8")
            project = root / "template.vcproject"
            project.write_text(
                """
<VcProject Version="9.5" Unit="inch">
  <ProjectReport><File>results.htm</File></ProjectReport>
  <Report><File>results.htm</File></Report>
  <Picture><File>vericut</File></Picture>
  <Setup Name="fixture-demo">
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
            vericut_bat = root / "vericut.bat"
            vericut_bat.write_text("@echo off", encoding="utf-8")
            registry_path = root / "setups.json"
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
                                "required_modes": ["G20", "G90"],
                                "safe_z_min": 0.25,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            setup = load_setup_registry(registry_path).get("sample_haas")

            job = prepare_job_workspace(
                setup=setup,
                gcode="G20 G90\nT1 M06\nG0 Z1.0\nM30\n",
                output_root=root / "runs",
                job_id="job-001",
            )

            self.assertEqual(job.job_id, "job-001")
            self.assertTrue(job.generated_nc.exists())
            self.assertTrue((job.input_dir / "fixture.stl").exists())
            self.assertTrue((job.input_dir / "tools.tls").exists())
            self.assertTrue((job.input_dir / "control.ctl").exists())
            self.assertIn("generated.nc", job.project_file.read_text(encoding="utf-8"))
            self.assertNotIn("old_program.mcd", job.project_file.read_text(encoding="utf-8"))
            self.assertEqual(job.missing_dependencies, ())

            command = build_vericut_command(setup, job, batch=True)
            self.assertEqual(command[0], str(vericut_bat))
            self.assertIn("BATCH", command)
            self.assertIn(str(job.project_file), command)

            (job.input_dir / "fixture.stl").chmod(0o444)
            rerun_job = prepare_job_workspace(
                setup=setup,
                gcode="G20 G90\nT1 M06\nG0 Z1.0\nM30\n",
                output_root=root / "runs",
                job_id="job-001",
            )
            self.assertTrue((rerun_job.input_dir / "fixture.stl").exists())

    def test_parse_vericut_log_extracts_error_counts_and_findings(self):
        log_text = """
VERICUT-log
******************** TOOLPATH ERROR REPORT ********************
Error for line 21, Filename: generated.nc
N0075G72I.625J45L6
 Error: Component "X" exceeded maximum limit of 0 by 0.2155 at line: (101) X POSX Y POSY
Number of Errors: 13
Number of Warnings: 1
Total cycle time since last rewind: 0:13:21
Total air time since last rewind: 0:05:04 (38)%
""".strip()

        parsed = parse_vericut_log(log_text)

        self.assertEqual(parsed.error_count, 13)
        self.assertEqual(parsed.warning_count, 1)
        self.assertEqual(parsed.cycle_time, "0:13:21")
        self.assertEqual(parsed.air_time, "0:05:04")
        self.assertEqual(parsed.air_time_percent, 38)
        self.assertEqual(parsed.findings[0].severity, "error")
        self.assertEqual(parsed.findings[0].line_number, 21)
        self.assertEqual(parsed.findings[0].filename, "generated.nc")
        self.assertIn("Component \"X\" exceeded", parsed.findings[0].message)

    def test_verdict_packet_rejects_zero_returncode_when_log_has_errors(self):
        with workspace_temp_dir("verdict-packet-log-errors") as root:
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            request_file = root / "request.json"
            request_file.write_text('{"setup_id": "sample_haas"}', encoding="utf-8")
            generated_nc = input_dir / "generated.nc"
            generated_nc.write_text("G90\nM30\n", encoding="utf-8")
            project_file = input_dir / "sample.vcproject"
            project_file.write_text("<VcProject />", encoding="utf-8")
            log_file = output_dir / "vericut.log"
            log_file.write_text(
                """
******************** TOOLPATH ERROR REPORT ********************
Error for line 2, Filename: generated.nc
G90
 Error: Lower case letters (comments included) at line: (2) in file "generated.nc"
Number of Errors: 1
Number of Warnings: 0
""".strip(),
                encoding="utf-8",
            )
            job = VericutJob(
                job_id="job-001",
                workspace=root,
                input_dir=input_dir,
                output_dir=output_dir,
                request_file=request_file,
                generated_nc=generated_nc,
                project_file=project_file,
                copied_dependencies=(),
                missing_dependencies=(),
            )
            run_result = VericutRunResult(
                command=("vericut.bat", "BATCH", str(project_file)),
                returncode=0,
                stdout="",
                stderr="",
            )

            verdict = write_verdict_packet(
                job=job,
                static_report=StaticCheckReport(()),
                run_result=run_result,
            )

            payload = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
            summary = (output_dir / "verdict.md").read_text(encoding="utf-8")

        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.status, "vericut_rejected")
        self.assertEqual(verdict.vericut.error_count, 1)
        self.assertEqual(payload["status"], "vericut_rejected")
        self.assertEqual(payload["vericut"]["error_count"], 1)
        self.assertIn("vericut.log", [artifact["name"] for artifact in payload["artifacts"]])
        self.assertIn("Vericut Verdict: Rejected", summary)


if __name__ == "__main__":
    unittest.main()
