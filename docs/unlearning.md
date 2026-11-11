# Knowledge unlearning / 知识遗忘

[Documentation](README.md) · [Injection](injection.md) · [Editing](editing.md)

Use `operation=unlearning` with separate **forget** and **retain** sets. Start
from a checkpoint that has learned the target knowledge; a foundation model and
a benchmark-trained Vanilla checkpoint are different inputs.
/ 遗忘既要衡量目标知识变化，也要衡量保留能力，不能只看遗忘集上的下降。

## Text: TOFU and other benchmarks

```bash
know-surgery operation=unlearning experiment=unlearn/tofu/default \
  trainer=NPO forget_split=forget10 retain_split=retain90 \
  task_name=tofu_npo --cfg job
```

Choose the matching checkpoint and dataset version. Text methods include
GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL and other registered trainers.
Their losses and reference requirements differ; changing `trainer` does not make
their objectives or budgets equivalent. See [the catalog](algorithm-model-catalog.md).

Use `--preflight`, inspect parameters and input paths, then execute with
`--reviewed-config YOUR_REVIEW_HASH`. A new architecture may require different
RMU layers/modules. TOFU forget-quality evaluation needs the appropriate
retain-only reference logs; missing reference results are not inferred.

## Multimodal: the paper's MLLMU-Bench case study

The manuscript uses **Qwen2.5-VL-3B-Instruct**, MLLMU-Bench forget-5%, seed 42,
and one common benchmark-trained Vanilla checkpoint. The split contains 25
forget and 475 retain identities. Training selects QA from `ft_Data` by identity:
381 forget QA and 7,323 retain QA. Evaluation questions are separate.

Set `model.model_path` to Vanilla, `model.processor_path` to its saved processor,
and `data.data_dir` to the complete MLLMU-Bench directory. The existing Vanilla
is the common starting point; these recipes do not reproduce another paper's
Vanilla training.

| Method | Experiment below `unlearn/multimodal/` | Training budget |
|---|---|---|
| GA | `mllmu_ga_qwen25vl3b` | Answer-only gradient ascent; lr `2e-5`, 4 epochs, 384 updates |
| GD | `mllmu_mip_qwen25vl3b` | Forget then retain training; lr `2e-5`, 4 epochs, 7,708 updates |
| NPO+RT | `mllmu_npo_rt_qwen25vl3b` | lr `1e-5`, 10 epochs, 120 updates; frozen Vanilla reference |
| RetainFT | `mllmu_retainft_mip_qwen25vl3b` | Continue Vanilla on retain QA; lr `2e-5`, 4 epochs, 7,324 updates |
| MANU | `manu_mip_qwen25vl3b_mllmu` | Activation statistics and persistent pruning; no optimizer updates |
| VKD | `vkd_qwen25vl3b_mllmu` | Visual-module updates; lr `2e-5`, 4 epochs, 384 updates |

These are method-specific budgets. RetainFT continues Vanilla, rather than
retraining the foundation model on retain-only data. NPO+RT uses a sequence
log-ratio plus retain NLL; its physical batch 32 preserves 4-QA logical loss
groups. Do not substitute the older scalar-loss-ratio NPO objective.

GA/GD/RetainFT use language-projection LoRA (rank 8, alpha 16, dropout 0.1).
MANU uses the released MANUPruner scoring convention. VKD freezes the language
model and updates visual encoder/projector weights, with LoRA disabled. Its
local calibration selects at most 1% of candidate neurons and balances losses
using eight training batches; calibration results are saved. These are documented
adaptations, not claims of identical upstream-model reproduction.

```bash
know-surgery operation=unlearning modality=multimodal \
  experiment=unlearn/multimodal/mllmu_ga_qwen25vl3b \
  model.model_path=/path/to/Vanilla \
  model.processor_path=/path/to/Vanilla \
  data.data_dir=/path/to/MLLMU-Bench \
  output_dir=saves/unlearning/mllmu_ga --cfg job
```

Preview and review as for text tasks. Methods have different trainable modules
and reference memory costs; a small LoRA batch does not establish the memory
needed for MANU statistics or VKD calibration.

## Evaluate forgetting and retention

The paper uses `eval=mllmu_paper`: native prompts/scoring, one-shot classification,
two-profile cloze examples, seed 42 and 50 generated tokens. It scores Forget,
Shared-Retain, Retain-Celebrity and Test; the main table selects Forget and
Shared-Retain. The saved Qwen processor supplies the chat template.

```bash
python src/mm_eval.py model=Qwen2.5-VL-3B eval=mllmu_paper \
  model.model_path=/path/to/saved_update \
  model.processor_path=/path/to/saved_update \
  eval.data_dir=/path/to/MLLMU-Bench \
  eval.output_dir=saves/unlearning/mllmu_ga/eval --cfg job
```

Remove `--cfg job` after checking the inputs to run evaluation.

| Main-table column | Native output | Direction |
|---|---|---|
| VQA | Image–text classification accuracy | Forget ↓, Retain ↑ |
| VGEN | Image–text generation ROUGE-L | Forget ↓, Retain ↑ |
| QA | Text-only classification accuracy | Forget ↓, Retain ↑ |

JSON outputs include per-example predictions, denominators and scoring protocol.
Few-shot exemplars are excluded from scored classification records; do not use
raw identity counts as denominators. Optional LLM factuality judging is separate
and is not required for these native columns. A lower forget score describes
measured behavior, not proof of irreversible deletion.

Text outputs normally use `saves/unlearn/<task_name>/`; multimodal recipes use
their explicit `output_dir`. Keep checkpoints, configurations, processor,
selection state and evaluation JSON together.
