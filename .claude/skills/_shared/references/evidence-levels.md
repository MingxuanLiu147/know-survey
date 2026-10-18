# Evidence Levels

Use one of these levels verbatim in skill outputs and UI records.

| Level | Minimum evidence | Allowed wording |
|---|---|---|
| `planned` | Intent or roadmap only | Planned, proposed |
| `configured` | Resolvable config and registered interfaces | Configured, composable |
| `local_validation` | Bounded local validation with command and artifacts | Locally validated |
| `verified_smoke` | Real model execution with input manifest, trace, and expected artifacts | Verified smoke |
| `verified_result` | Complete protocol, data, seed, environment, metrics, and report | Verified result |

Never promote a level from filenames alone. Inspect the command, resolved config,
input manifest, run trace, metric files, and scope. Treat missing evidence as
`null`, never as zero or success.

