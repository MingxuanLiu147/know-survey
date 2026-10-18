# Artifact Layout

Skill orchestration metadata lives under `artifacts/<skill>/`. Training and
evaluation outputs continue to use the Hydra-resolved directory under `saves/`.

```text
artifacts/
  plan-surgery/surgery_plan.json
  inspect-repo/repo_inventory.json
  setup-env/environment_report.json
  <skill>/<trace-id>.json
saves/<resolved-mode>/<task-name>/
  command.txt
  resolved_config.yaml
  environment.json
  input_manifest.json
  run_trace.json
  *_SUMMARY.json or operation-specific metrics
```

Execution traces record the generated command, process status and evidence level;
data conversion helpers write their own counts and source manifest. These files
are created when the user invokes a workflow and are not required package inputs.
Use each helper's `--output` option to select another location; report generation
uses `--json-output` and `--markdown-output` instead.

Reports may contain local paths inherited from the input. Review them before
sharing. Do not copy credentials, full environment dumps or model weights into
workflow metadata. A source checkout needs no historical run or audit directory
to plan and preview a new operation.
