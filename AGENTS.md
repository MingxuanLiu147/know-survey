# Know-Surgery workbench instructions

Use the shared Skills in `.claude/skills/`; `.agents/skills` points to the same workflows.
Read `README.md`, `docs/architecture.md` and `docs/agent_mode.md` before changing interfaces.

- Public operations are exactly `unlearning`, `inject` and `edit`.
- Use `know-surgery` or `python src/surgery.py` for configuration and execution.
- Preview the resolved Hydra job before loading a model; run a small example before a larger experiment.
- Use registered models, datasets and methods; extend the existing interfaces and configs.
- Keep target knowledge and held-out locality or retention probes separate.
- Preserve commands, configs, input sources, predictions, metrics and failures for reproducibility.
- Missing metrics are unavailable, never substitute zero. Process success is not benchmark success.
- Read provider credentials from environment variables; never include them in exports or logs.
- Preserve user changes and keep private data, model weights and generated runs outside source control.
- Validate affected commands and document model-specific compatibility limits.
