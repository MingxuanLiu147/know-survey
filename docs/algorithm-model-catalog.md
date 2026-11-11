# Algorithms and models

Know-Surgery organizes knowledge updates into three operations: **Inject**, **Unlearning**, and **Edit**. The tables link each public implementation to its configuration and source. Model-specific preparation is listed alongside the method; evaluation results belong to individual runs.

This catalog follows the public repository at [revision b920df0](https://github.com/MingxuanLiu147/know-survey/tree/b920df08f0f702bc48378b6ce965464d64c2c566). It contains 43 operation–method–modality entries: 6 for injection, 19 for unlearning, and 18 for editing. Text and image–text adapters of the same method appear separately.

[Injection](#inject) · [Unlearning](#unlearning) · [Editing](#edit) · [Models](#models)

## Inject

CLI operation: `operation=inject`. Inputs are supervised knowledge examples, with separate evaluation data for target learning and retention.

### Full fine-tuning

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| Full FT | Text | Supervised updates to the base model parameters. | [Config](../configs/trainer/finetune.yaml) · [Code](../src/trainer/base.py) | Disable adapters and check the intended trainable parameters. |

### Parameter-efficient fine-tuning

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| [LoRA](https://arxiv.org/abs/2106.09685) | Text | Low-rank weight increments with frozen base weights. | [Config](../configs/trainer/inject/LoRA.yaml) · [Code](../src/trainer/inject/lora.py) | Model-specific target modules and adapter rank. |
| [AdaLoRA](https://arxiv.org/abs/2303.10512) | Text | Adaptive allocation of the low-rank parameter budget. | [Config](../configs/trainer/inject/AdaLoRA.yaml) · [Code](../src/trainer/inject/adalora.py) | Rank-allocation schedule matched to the training budget. |
| [DoRA](https://proceedings.mlr.press/v235/liu24bn.html) | Text | Separate weight magnitude and direction, with low-rank direction updates. | [Config](../configs/trainer/inject/DoRA.yaml) · [Code](../src/trainer/inject/dora.py) | Compatible PEFT modules and model layer names. |
| MM-LoRA | Image–text | LoRA training on image-conditioned answer targets. | [Config](../configs/trainer/inject/MM_LoRA.yaml) · [Code](../src/trainer/inject/mm_lora.py) | Repository adapter of [LoRA](https://arxiv.org/abs/2106.09685); image–text processor, training records, and enabled LoRA. |

### Representation fine-tuning

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| [LoReFT](https://arxiv.org/abs/2404.03592) | Text | Low-rank interventions on hidden representations. | [Config](../configs/trainer/inject/LoReFT.yaml) · [Code](../src/trainer/inject/loreft.py) | PyReFT optional dependency and intervention-layer settings. |

## Unlearning

CLI operation: `operation=unlearning`. Inputs identify forget examples and, when required by the objective, a separate retain set and reference model.

### Gradient-based

This group covers loss shaping, reweighting, retention constraints, output distillation, and masked gradients. MMRetainFT supplies a retain-only reference baseline.

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| GradAscent | Text | Increases loss on forget examples. | [Config](../configs/trainer/GradAscent.yaml) · [Code](../src/trainer/unlearn/grad_ascent.py) | Generic baseline used in [TOFU](https://arxiv.org/abs/2401.06121); select the intended starting checkpoint. |
| GradDiff | Text | Combines forget ascent with retain loss minimization. | [Config](../configs/trainer/GradDiff.yaml) · [Code](../src/trainer/unlearn/grad_diff.py) | [TOFU](https://arxiv.org/abs/2401.06121) baseline; local retention options include NLL and KL. |
| [CE-U](https://arxiv.org/abs/2503.01224) | Text | Cross-entropy against an adjusted distribution that suppresses the original target token. | [Config](../configs/trainer/CEU.yaml) · [Code](../src/trainer/unlearn/ceu.py) | Trainer ID: `CEU`; answer-token masking follows the configuration. |
| PDU | Text | Native primal–dual extension of GradDiff with a retain-loss constraint. | [Config](../configs/trainer/PDU.yaml) · [Code](../src/trainer/unlearn/pdu.py) | Set `retain_loss_eps`. [Framework reference](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f346824d37ce8d0ab8f5920514934b8f-Abstract-Conference.html); the local forget objective is inherited from GradDiff. |
| [SatImp](https://arxiv.org/abs/2505.11953) | Text | Combines saturation-based and importance-based loss weights. | [Config](../configs/trainer/SatImp.yaml) · [Code](../src/trainer/unlearn/satimp.py) | Loss-weight and retention settings. |
| [UnDIAL](https://aclanthology.org/2025.naacl-long.444/) | Text | Self-distillation from logits adjusted to suppress forget targets. | [Config](../configs/trainer/UNDIAL.yaml) · [Code](../src/trainer/unlearn/undial.py) | Trainer ID: `UNDIAL`; frozen reference outputs. |
| [WGA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9ad307ab459fb93163248a0a56959307-Abstract-Conference.html) | Text | Confidence-weighted token-wise gradient ascent. | [Config](../configs/trainer/WGA.yaml) · [Code](../src/trainer/unlearn/wga.py) | Weighting and retention settings. |
| MMGradAscent | Image–text | Gradient ascent on image-conditioned forget answers. | [Config](../configs/trainer/MMGradAscent.yaml) · [Code](../src/trainer/unlearn/mm_grad_ascent.py) | [MLLMU-Bench](https://aclanthology.org/2025.naacl-long.207/) / [CLEAR](https://aclanthology.org/2025.findings-acl.1058/) adapter. |
| MMGradDiff | Image–text | Forget ascent combined with retain descent. | [Config](../configs/trainer/MMGradDiff.yaml) · [Code](../src/trainer/unlearn/mm_grad_diff.py) | Select the [MLLMU-Bench](https://aclanthology.org/2025.naacl-long.207/) or [CLEAR](https://aclanthology.org/2025.findings-acl.1058/) schedule. |
| MMKLMin | Image–text | Forget objective with KL-based preservation of retain outputs. | [Config](../configs/trainer/MMKLMin.yaml) · [Code](../src/trainer/unlearn/mm_kl_min.py) | Frozen reference model, retain set, and benchmark-specific objective settings. Sources: [MLLMU-Bench](https://aclanthology.org/2025.naacl-long.207/), [CLEAR](https://aclanthology.org/2025.findings-acl.1058/). |
| MMRetainFT | Image–text | Trains on retain examples to construct a reference or comparison checkpoint. | [Config](../configs/trainer/MMRetainFT.yaml) · [Code](../src/trainer/unlearn/mm_retain_ft.py) | Retain set and intended initialization; reference baseline associated with [CLEAR](https://aclanthology.org/2025.findings-acl.1058/). |
| MMSCRUB | Image–text | Teacher–student unlearning with forget and retain objectives. | [Config](../configs/trainer/MMSCRUB.yaml) · [Code](../src/trainer/unlearn/mm_scrub.py) | [CLEAR](https://aclanthology.org/2025.findings-acl.1058/) adaptation of [SCRUB](https://papers.nips.cc/paper_files/paper/2023/hash/062d711fb777322e2152435459e6e9d9-Abstract-Conference.html); frozen teacher and retain data. |
| [MMUnlearner](https://aclanthology.org/2025.findings-acl.375/) | Image–text | Saliency-masked updates target visual entity knowledge while preserving its textual counterpart. | [Config](../configs/trainer/MMUnlearner.yaml) · [Code](../src/trainer/unlearn/mmunlearner.py) | Checkpoint-specific saliency masks, retain data, full-parameter training, and separate text-preservation evaluation. |

### Preference-based

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| IDK-DPO | Text | Prefers an alternate refusal answer over the original answer. | [Config](../configs/trainer/DPO.yaml) · [Code](../src/trainer/unlearn/dpo.py) | Local [DPO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) adaptation; paired answers and reference model. Trainer ID: `DPO`. |
| [NPO](https://arxiv.org/abs/2404.05868) | Text | Negative-preference loss reduces target likelihood relative to a reference. | [Config](../configs/trainer/NPO.yaml) · [Code](../src/trainer/unlearn/npo.py) | Frozen reference model; retention settings selected by the recipe. |
| [SimNPO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/02443e4e008231e0af0855f3cc70ed17-Abstract-Conference.html) | Text | Reference-free negative-preference objective with response-length normalization. | [Config](../configs/trainer/SimNPO.yaml) · [Code](../src/trainer/unlearn/simnpo.py) | Forget answers and the configured retention loss. |
| MMNPO | Image–text | Benchmark-specific image–text adaptation of [NPO](https://arxiv.org/abs/2404.05868). | [Config](../configs/trainer/MMNPO.yaml) · [Code](../src/trainer/unlearn/mm_npo.py) | Public registry requires a retain-only oracle; select the matching loss-ratio or probability-ratio recipe. |

### Representation-based

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| [RMU](https://proceedings.mlr.press/v235/li24bc.html) | Text | Redirects forget representations while preserving retain representations. | [Config](../configs/trainer/RMU.yaml) · [Code](../src/trainer/unlearn/rmu.py) | Model-specific layer mappings, retain set, and reference representations. |
| MANU adapter | Image–text | Ranks language-model neurons from paired text/image–text activations and attenuates selected weight columns. | [Config](../configs/trainer/MMMANU.yaml) · [Code](../src/trainer/unlearn/mm_manu.py) | Port of the simplified MANU baseline in [MIP-Editor](https://github.com/PreckLi/MIP-Editor), informed by the [MANU paper](https://arxiv.org/abs/2502.15910). Canonical trainer: `MMMANU`. |

## Edit

CLI operation: `operation=edit`. Requests specify the prompt and new target, with paraphrase, locality, and other probes supplied separately.

### Gradient-based

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| FT | Text | Fine-tunes selected layers on the requested new answer. | [Config](../configs/trainer/edit/FT.yaml) · [Code](../src/trainer/edit/ft.py) | Editable layers and optimization budget. |
| MM-LoRA | Image–text | Optimizes a LoRA adapter on image-conditioned edit targets. | [Config](../configs/trainer/edit/MM_LORA.yaml) · [Code](../src/trainer/edit/mm_lora.py) | Repository adapter of [LoRA](https://arxiv.org/abs/2106.09685); requires an attached PEFT adapter. |

### Locate-then-edit

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| [ROME](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) | Text | Rank-one update to a feed-forward projection for a factual association. | [Config](../configs/trainer/edit/ROME.yaml) · [Code](../src/trainer/edit/rome.py) | Subject span, one rewrite layer, and preservation statistics. |
| [MEMIT](https://arxiv.org/abs/2210.07229) | Text | Distributes factual updates across multiple feed-forward layers. | [Config](../configs/trainer/edit/MEMIT.yaml) · [Code](../src/trainer/edit/memit.py) | Subject spans, layer mappings, and preservation statistics. |
| [AlphaEdit](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html) | Text | Projects weight changes into an estimated null space of preserved knowledge. | [Config](../configs/trainer/edit/AlphaEdit.yaml) · [Code](../src/trainer/edit/alphaedit.py) | Projection statistics, layer mappings, and continuation state. |
| [AnyEdit](https://proceedings.mlr.press/v267/jiang25b.html) | Text | Splits a target into token chunks, optimizes boundary representations autoregressively, and writes their updates jointly. | [Config](../configs/trainer/edit/AnyEdit.yaml) · [Code](../src/trainer/edit/anyedit.py) | Public backends: `MEMIT_ARE` and `AlphaEdit_ARE`; prepare target chunks and preservation statistics. |

### Learned editor

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| [MEND](https://arxiv.org/abs/2110.11309) | Text | Factor network transforms activation/backpropagation factors into weight updates. | [Config](../configs/trainer/edit/MEND_official.yaml) · [Code](../src/trainer/edit/mend_official.py) | Native auxiliary checkpoint matched to the exact base model. |
| [InstructEdit](https://www.ijcai.org/proceedings/2024/733) | Text | Compact instruction-conditioned MEND-style update network. | [Config](../configs/trainer/edit/InstructEdit.yaml) · [Code](../src/trainer/edit/instructedit.py) | Native checkpoint matched to the model and instruction template. |
| [MALMEN](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f074097f6f186947726ef28819d589e7-Abstract-Conference.html) | Text | Compact learned shift aggregation with regularized least squares. | [Config](../configs/trainer/edit/MALMEN.yaml) · [Code](../src/trainer/edit/malmen.py) | Native auxiliary checkpoint matched to the exact base model. |
| MM-MEND | Image–text | Image-conditioned gradient collection for the native [MEND](https://arxiv.org/abs/2110.11309) network. | [Config](../configs/trainer/edit/MM_MEND.yaml) · [Code](../src/trainer/edit/mm_mend.py) | Compatible native edit network, processor, and target-layer mapping. |

### Memory-based

| Method | Modality | Mechanism | Configuration / code | Preparation |
|---|---|---|---|---|
| [SERAC](https://proceedings.mlr.press/v162/mitchell22a.html) | Text | Routes requests through a trained scope encoder and separate counterfactual model. | [Config](../configs/trainer/edit/SERAC.yaml) · [Code](../src/trainer/edit/serac.py) | Trained auxiliary archive and compatible causal tokenizer. |
| [GRACE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html) | Text | Stores edits in a discrete key–value adapter codebook. | [Config](../configs/trainer/edit/GRACE.yaml) · [Code](../src/trainer/edit/grace.py) | Adapter location and codebook settings. |
| [IKE](https://aclanthology.org/2023.emnlp-main.296/) | Text | Retrieves demonstrations for in-context editing with frozen model weights. | [Config](../configs/trainer/edit/IKE.yaml) · [Code](../src/trainer/edit/ike.py) | Demonstration memory and retrieval backend; select the intended protocol. |
| [WISE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/60960ad78868fce5c165295fbd895060-Abstract-Conference.html) | Text | Native WISE-Merge separates main/side memories with routing and TIES merging. | [Config](../configs/trainer/edit/WISE.yaml) · [Code](../src/trainer/edit/wise_merge.py) | Projection mapping, context templates, and independent scope negatives. |
| MM-SERAC | Image–text | Compact text-scope and prefix routing while retaining the image input. | [Config](../configs/trainer/edit/MM_SERAC.yaml) · [Code](../src/trainer/edit/mm_serac.py) | Repository adaptation inspired by [SERAC](https://proceedings.mlr.press/v162/mitchell22a.html); uses compact routing without a separate counterfactual model. |
| MM-GRACE | Image–text | Image-conditioned training of a discrete key–value adapter. | [Config](../configs/trainer/edit/MM_GRACE.yaml) · [Code](../src/trainer/edit/mm_grace.py) | Repository [GRACE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html) adapter; processor and layer mapping. |
| MM-IKE | Image–text | Text-demonstration retrieval for image–text edit requests. | [Config](../configs/trainer/edit/MM_IKE.yaml) · [Code](../src/trainer/edit/mm_ike.py) | Repository [IKE](https://aclanthology.org/2023.emnlp-main.296/) adapter; retrieval encodes the textual part of each fact. |
| MM-WISE | Image–text | Image-conditioned editing with the native WISE adapter and shard routing. | [Config](../configs/trainer/edit/MM_WISE.yaml) · [Code](../src/trainer/edit/mm_wise.py) | Repository [WISE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/60960ad78868fce5c165295fbd895060-Abstract-Conference.html) adapter; differs from the text WISE-Merge path. |

## Implementation profiles and dependencies

- ROME, MEMIT, and AlphaEdit have explicit reference profiles: [ROME_Official](../configs/trainer/edit/ROME_Official.yaml), [MEMIT_Official](../configs/trainer/edit/MEMIT_Official.yaml), and [AlphaEdit_Official](../configs/trainer/edit/AlphaEdit_Official.yaml). They reuse this repository's model mappings and shared utilities. A reference profile identifies an implementation setting; performance depends on the experiment.
- [AnyEditOfficial](../configs/trainer/edit/AnyEditOfficial.yaml) selects the reference ARE profile. [IKE_official](../configs/trainer/edit/IKE_official.yaml) uses an independent demonstration corpus and the copy/update/retain protocol. These are configurations of the listed methods.
- MEND has a factor-network profile (`MEND_official`) and a separate compact profile (`MEND`). MALMEN and InstructEdit use compact native networks. Checkpoints are implementation-specific; supply weights matching the model, matrices and editor.
- [SERACCompact](../configs/trainer/edit/SERACCompact.yaml) and [WISEAdapted](../configs/trainer/edit/WISEAdapted.yaml) expose alternative local implementations. Their checkpoints and mechanisms differ from the default SERAC and WISE-Merge configurations.
- Multimodal editor names identify repository adapters. [MMEdit](https://aclanthology.org/2023.emnlp-main.856/) provides the image–text editing context; each adapter's processor, layer mapping, retrieval, or auxiliary-network requirements remain method-specific.

## Additional records

The public [multimodal registry](../src/mm_registry.py) includes VKD as `MMVKD`, a paper-based Qwen visual adaptation with training-only calibration. See [unlearning](unlearning.md) for its recipe and scope. UnKE and the `UnKE_ARE` backend are excluded from the public distribution, as documented in the [AnyEdit source](../src/trainer/edit/anyedit.py).

For execution settings, see the [editing guide](../docs/editing.md), [experiment configuration guide](../docs/experiments.md), and [evaluation guide](../docs/evaluation.md). Preview the composed configuration before starting a run.

## Models

The tables below list the model configurations shipped with Know-Surgery. Use a configuration's filename without `.yaml` as the `model=<name>` value. Each model–method pairing also requires the appropriate layer mappings, adapter targets and auxiliary assets; see the [editing guide](../docs/editing.md) and [multimodal registry](../src/mm_registry.py).

### Text models

| Family | Model variants | Configuration files | Notes |
|---|---|---|---|
| GPT-2 | GPT-2; GPT-2 XL (1.5B) | [GPT-2](../configs/model/gpt2.yaml), [GPT-2 XL](../configs/model/gpt2-xl.yaml) | Plain-text prompt formatting. |
| GPT-J | GPT-J 6B | [GPT-J 6B](../configs/model/gpt-j-6b.yaml) | Plain-text prompt formatting. |
| Llama 2 | 7B; 7B Chat; 13B | [7B](../configs/model/Llama-2-7b-hf.yaml), [7B Chat](../configs/model/Llama-2-7b-chat-hf.yaml), [13B](../configs/model/Llama-2-13b-hf.yaml) | Separate base and chat prompt formats. |
| Llama 3 | 3-8B-Instruct; 3.1-8B-Instruct; 3.2-1B-Instruct; 3.2-3B-Instruct | [3 / 8B](../configs/model/Llama-3-8B-Instruct.yaml), [3.1 / 8B](../configs/model/Llama-3.1-8B-Instruct.yaml), [3.2 / 1B](../configs/model/Llama-3.2-1B-Instruct.yaml), [3.2 / 3B](../configs/model/Llama-3.2-3B-Instruct.yaml) | Each version and size has a separate configuration. |
| Qwen2.5 | 1.5B base; 1.5B, 3B and 7B Instruct | [1.5B base](../configs/model/Qwen2.5-1.5B.yaml), [1.5B Instruct](../configs/model/Qwen2.5-1.5B-Instruct.yaml), [3B Instruct](../configs/model/Qwen2.5-3B-Instruct.yaml), [7B Instruct](../configs/model/Qwen2.5-7B-Instruct.yaml) | Base and instruction checkpoints use separate prompt settings. |
| DeepSeek-R1 distillations | Distill-Qwen-1.5B; Distill-Qwen-7B; Distill-Llama-8B | [Qwen 1.5B](../configs/model/DeepSeek-R1-Distill-Qwen-1.5B.yaml), [Qwen 7B](../configs/model/DeepSeek-R1-Distill-Qwen-7B.yaml), [Llama 8B](../configs/model/DeepSeek-R1-Distill-Llama-8B.yaml) | Includes model-specific chat and answer-prefix settings. |
| Phi | phi-1_5; Phi-3.5-mini-instruct | [phi-1_5](../configs/model/phi-1_5.yaml), [Phi-3.5 mini](../configs/model/Phi-3.5-mini-instruct.yaml) | QA formatting for phi-1_5; chat formatting for Phi-3.5 mini. |
| Gemma | gemma-7b-it | [7B IT](../configs/model/gemma-7b-it.yaml) | Instruction-tuned checkpoint with chat formatting. |
| Zephyr | zephyr-7b-beta | [7B beta](../configs/model/zephyr-7b-beta.yaml) | Chat checkpoint with its own prompt template. |

### Vision–language models

| Family | Model variant | Configuration file | Notes |
|---|---|---|---|
| Qwen2-VL | 2B Instruct | [Qwen2VL-2B](../configs/model/Qwen2VL-2B.yaml) | Method bindings for MMEdit/VLKEB editing and MLLMU-Bench/CLEAR unlearning. |
| Qwen2-VL | 7B Instruct | [Qwen2VL-7B](../configs/model/Qwen2VL-7B.yaml) | Method bindings for MLLMU-Bench/CLEAR unlearning. |
| Qwen2.5-VL | 3B Instruct | [Qwen2.5-VL-3B](../configs/model/Qwen2.5-VL-3B.yaml) | Includes an [image/text injection recipe](../docs/components.md#imagetext-injection), plus editing and unlearning method bindings. |
| Qwen3-VL | 4B Instruct | [Qwen3-VL-4B](../configs/model/Qwen3-VL-4B.yaml) | Model and processor settings are provided; no method bindings are declared. |
| LLaVA 1.5 | 7B | [LLaVA-1.5-7B](../configs/model/LLaVA-1.5-7B.yaml) | Method bindings for MMEdit/VLKEB editing and MLLMU-Bench/CLEAR unlearning. |
| BLIP-2 | OPT-2.7B | [BLIP2-OPT-2.7B](../configs/model/BLIP2-OPT-2.7B.yaml) | The model configuration binds `MM_IKE` only. |

For multimodal benchmark unlearning, select the benchmark-trained starting checkpoint required by the registry. Model weights and processors are supplied separately. See [custom models and data](../docs/components.md#models) for path overrides and extension instructions.
