# Vericut Integration

This integration keeps Vericut setup data local. The app or agent can choose a setup ID, run conservative static checks, and stage a Vericut project copy without sending machine definitions, fixtures, tool libraries, or simulation output to a model provider.

## Local Setup Registry

Setups live in JSON files with one entry per machine or pilot template:

```json
{
  "setups": [
    {
      "id": "vericut96_haas_minimill_sample",
      "description": "Vericut 9.6 sample Haas MiniMill project.",
      "vericut_bat": "C:\\Program Files\\CGTech\\Vericut 9.6\\windows64\\commands\\vericut.bat",
      "project_template": "C:\\Program Files\\CGTech\\Vericut 9.6\\samples\\Haas\\haas_minimill.vcproject",
      "library_search_paths": [
        "C:\\Program Files\\CGTech\\Vericut 9.6\\samples\\Support_files",
        "C:\\Program Files\\CGTech\\Vericut 9.6\\samples\\TLS",
        "C:\\Program Files\\CGTech\\Vericut 9.6\\library"
      ],
      "allowed_tools": ["T1", "T2", "T3"],
      "required_modes": ["G90"],
      "expected_units": "inch",
      "feed_rate_min": 1.0,
      "feed_rate_max": 300.0,
      "spindle_speed_min": 1.0,
      "spindle_speed_max": 10000.0,
      "work_envelope": {
        "x_min": -100.0,
        "x_max": 100.0,
        "y_min": -100.0,
        "y_max": 100.0,
        "z_min": -10.0,
        "z_max": 100.0
      },
      "safe_z_min": 0.0
    }
  ]
}
```

`config/vericut_setups.example.json` is wired to the Vericut 9.6 Haas sample tree discovered on this machine. It is still a sample, but it now acts like a real onboarding contract: a candidate must declare inch units, stay inside feed and spindle limits, use an approved tool, and keep axis words inside the setup envelope before Vericut staging is accepted. The file includes both a Haas MiniMill sample with a static-safe live-control fixture and a Haas VF3 pocket-cycle sample so setup #2 can prove the abstraction without code changes. Copy it to an untracked local registry before adding proprietary shop setups or tighter production limits.

The static policy fields are:

- `allowed_tools`: optional tool whitelist such as `T1`, `T2`, or `T3`.
- `required_modes`: modal G-codes that must appear, such as `G90`.
- `expected_units`: optional `inch` or `metric`, enforced through explicit `G20` or `G21`.
- `feed_rate_min` / `feed_rate_max`: optional bounds for `F` words.
- `spindle_speed_min` / `spindle_speed_max`: optional bounds for `S` words.
- `work_envelope`: optional `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, and `z_max` bounds.
- `safe_z_min`: optional minimum Z value for rapid `G0` moves.

## Prepare A Job

Create or choose a G-code file, then stage a Vericut run:

```powershell
poetry run python -m gllm.vericut.cli --registry config/vericut_setups.example.json --setup-id vericut96_haas_minimill_sample --gcode-file tests\fixtures\haas_minimill_sample_control.nc --output-root .vericut-runs
```

The command writes a job folder containing:

- `input/generated.nc`
- a copied `.vcproject` template with its NC program reference pointed at `generated.nc`
- copied local dependencies that were referenced by the project template
- `request.json` with a local manifest and the Vericut command to run

By default, the CLI does not launch Vericut. It prints the command so an operator or a later approval-gated agent step can run it, typically:

```powershell
& "C:\Program Files\CGTech\Vericut 9.6\windows64\commands\vericut.bat" BATCH ".vericut-runs\<job-id>\input\haas_minimill.vcproject"
```

To let the CLI launch Vericut after staging, add `--run-vericut`. This may consume a Vericut license or open CGTech tooling, so keep it as an explicit operator-approved action:

```powershell
poetry run python -m gllm.vericut.cli --registry config/vericut_setups.example.json --setup-id vericut96_haas_minimill_sample --gcode-file tests\fixtures\haas_minimill_safe_noop.nc --output-root .vericut-runs --run-vericut --timeout-seconds 900
```

After a run, the CLI writes `output/verdict.json` and `output/verdict.md`. The verdict is stricter than the process return code: a zero exit code is accepted only when static checks pass and the Vericut log reports zero errors. If Vericut logs errors while returning `0`, the CLI reports `vericut_rejected` and exits nonzero so an agent or operator cannot miss it.

## Prompt-To-Verdict Packets

Use `gllm.proof.cli` when a natural-language prompt and setup ID should produce a complete evidence packet. It wraps generation or an existing candidate file, local static checks, Vericut staging, optional Vericut execution, parsed verdicts, and repair context into `.proof-runs/<scenario-id>/evidence_packet.json`.

```powershell
poetry run python -m gllm.proof.cli --registry config/vericut_setups.example.json --setup-id vericut96_haas_minimill_sample --prompt "Run the shipped Haas MiniMill sample control program." --candidate-gcode-file tests\fixtures\haas_minimill_sample_control.nc --output-root .proof-runs --scenario-id sample-haas-minimill-control-static
```

See `docs/prompt_to_verdict.md` for packet statuses and repair-loop behavior.

To run the canonical prompt corpus across the checked-in sample setups:

```powershell
poetry run python -m gllm.proof.corpus_cli --corpus config/proof_prompt_corpus.example.json --registry config/vericut_setups.example.json --output-root .proof-runs\corpus-smoke
```

The corpus intentionally includes passing fixtures and expected static rejects. A successful corpus run means the expected outcomes were reproduced and the second setup staged through the same proof path.

## Proprietary Boundary

Keep the registry, copied fixtures, tool libraries, machine files, reports, `.vericut-runs`, and `.proof-runs` folders local. If this becomes an agent loop, pass only setup IDs, static-check codes, and redacted report summaries back into model prompts unless an operator explicitly approves richer context.
