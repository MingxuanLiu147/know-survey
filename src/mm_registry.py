"""Registries for multimodal unlearning methods and benchmarks.

The registry deliberately separates runnable implementations from paper-only
extension points.  A paper or benchmark mentioning a method is not enough to
make it runnable in Know-Surgery.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class MMAlgorithmRegistration:
    """Metadata for one multimodal unlearning method."""

    name: str
    family: str
    implementation: str | None
    status: str
    supported_benchmarks: tuple[str, ...]
    benchmark_origin: tuple[str, ...] = ()
    paper_arxiv: str | None = None
    aliases: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    notes: str = ""

    @property
    def runnable(self) -> bool:
        return self.implementation is not None


@dataclass(frozen=True)
class MMDatasetRegistration:
    """Metadata for one multimodal unlearning benchmark."""

    key: str
    name: str
    loader: str
    evaluator: str
    status: str
    paper_arxiv: str
    source: str
    supported_splits: tuple[str, ...]
    original_baselines: tuple[str, ...]
    original_models: tuple[str, ...]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class MMModelMethodCompatibility:
    """One model/method execution contract, not a run-success claim."""

    method: str
    supported_benchmarks: tuple[str, ...]
    update_policy: str
    lora_target_policy: str
    memory_tier: str
    single_rtx3090_status: str
    requirements: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class MMModelRegistration:
    """Model compatibility metadata for multimodal unlearning."""

    key: str
    name: str
    config_name: str
    model_family: str
    source_model_id: str
    status: str
    supported_benchmarks: tuple[str, ...]
    paper_original_benchmarks: tuple[str, ...]
    minimum_transformers_version: str
    method_compatibility: tuple[MMModelMethodCompatibility, ...]
    aliases: tuple[str, ...] = ()
    notes: str = ""

    @property
    def configured_methods(self) -> tuple[str, ...]:
        """Return configured methods without implying successful GPU runs."""

        return tuple(profile.method for profile in self.method_compatibility)


MM_UNLEARNING_ALGORITHMS: dict[str, MMAlgorithmRegistration] = {
    # Shared, benchmark-native baselines implemented by the sidecar pipeline.
    "MMGradAscent": MMAlgorithmRegistration(
        name="MMGradAscent",
        family="benchmark_baseline",
        implementation="trainer.unlearn.mm_grad_ascent.MMGradAscent",
        status="Prototype",
        supported_benchmarks=("mllmu", "clear"),
        benchmark_origin=("MLLMU-Bench", "CLEAR"),
        aliases=("MMGA",),
        notes="Gradient-ascent baseline.",
    ),
    "MMGradDiff": MMAlgorithmRegistration(
        name="MMGradDiff",
        family="benchmark_baseline",
        implementation="trainer.unlearn.mm_grad_diff.MMGradDiff",
        status="Prototype",
        supported_benchmarks=("mllmu", "clear"),
        benchmark_origin=("MLLMU-Bench", "CLEAR"),
        aliases=("MMGD", "MMGADiff"),
        requirements=("retain_set",),
    ),
    "MMKLMin": MMAlgorithmRegistration(
        name="MMKLMin",
        family="benchmark_baseline",
        implementation="trainer.unlearn.mm_kl_min.MMKLMin",
        status="Prototype",
        supported_benchmarks=("mllmu", "clear"),
        benchmark_origin=("MLLMU-Bench", "CLEAR"),
        aliases=("MMKL",),
        requirements=("retain_set", "frozen_reference_model"),
    ),
    "MMNPO": MMAlgorithmRegistration(
        name="MMNPO",
        family="benchmark_baseline",
        implementation="trainer.unlearn.mm_npo.MMNPO",
        status="Prototype",
        supported_benchmarks=("mllmu", "clear"),
        benchmark_origin=("MLLMU-Bench", "CLEAR"),
        paper_arxiv="2404.05868",
        requirements=("retain_only_oracle_checkpoint",),
    ),
    "MMRetainFT": MMAlgorithmRegistration(
        name="MMRetainFT",
        family="reference_baseline",
        implementation="trainer.unlearn.mm_retain_ft.MMRetainFT",
        status="Prototype",
        supported_benchmarks=("mllmu", "clear"),
        benchmark_origin=("CLEAR",),
        requirements=("retain_set",),
        notes="Builds a retain-only reference/oracle checkpoint.",
    ),
    # Later proposed method: not an original MLLMU-Bench or CLEAR baseline.
    "MMUnlearner": MMAlgorithmRegistration(
        name="MMUnlearner",
        family="later_method",
        implementation="trainer.unlearn.mmunlearner.MMUnlearner",
        status="Prototype",
        supported_benchmarks=("mllmu", "clear"),
        paper_arxiv="2502.11051",
        requirements=(
            "saliency_mask",
            "retain_set",
            "full_parameter_training",
            "target_text_preservation_evaluation",
        ),
        notes=(
            "Later visual-entity unlearning method: delete target visual patterns "
            "while preserving the same entity's text knowledge. Do not compare it "
            "as ordinary full multimodal deletion without modality-specific metrics."
        ),
    ),
    "MMMANU": MMAlgorithmRegistration(
        name="MMMANU",
        family="later_method",
        implementation="trainer.unlearn.mm_manu.MMMANU",
        status="Prototype",
        supported_benchmarks=("mllmu", "clear"),
        benchmark_origin=("MLLMU-Bench",),
        paper_arxiv="2502.15910",
        aliases=("MANU", "MMManu"),
        requirements=(
            "paired_text_multimodal_inputs",
            "retain_set",
            "full_parameter_weight_edit",
        ),
        notes=(
            "Port of the simplified MANUPruner baseline bundled with MIP-Editor, "
            "not a reproduction of the official MANU repository. The original "
            "MANU paper targets MLLMU-Bench with LLaVA/Idefics2; CLEAR and Qwen are "
            "portability extensions. Know-Surgery materializes down-projection input "
            "scaling into equivalent weight-column edits for serializable checkpoints."
        ),
    ),
    # CLEAR-native baselines. Only entries with an implementation are runnable;
    # the remaining registrations are explicit extension contracts.
    "MMIDK": MMAlgorithmRegistration(
        name="MMIDK",
        family="benchmark_baseline",
        implementation=None,
        status="Planned",
        supported_benchmarks=("clear",),
        benchmark_origin=("CLEAR",),
        requirements=("idk_target_adapter", "retain_set"),
    ),
    "MMDPO": MMAlgorithmRegistration(
        name="MMDPO",
        family="benchmark_baseline",
        implementation=None,
        status="Planned",
        supported_benchmarks=("clear",),
        benchmark_origin=("CLEAR",),
        requirements=("preference_pair_adapter", "idk_reference_model"),
    ),
    "MMLLMU": MMAlgorithmRegistration(
        name="MMLLMU",
        family="benchmark_baseline",
        implementation=None,
        status="Planned",
        supported_benchmarks=("clear",),
        benchmark_origin=("CLEAR",),
        requirements=("idk_target_adapter", "frozen_reference_model", "retain_set"),
    ),
    "MMSCRUB": MMAlgorithmRegistration(
        name="MMSCRUB",
        family="benchmark_baseline",
        implementation="trainer.unlearn.mm_scrub.MMSCRUB",
        status="Prototype",
        supported_benchmarks=("clear",),
        benchmark_origin=("CLEAR",),
        paper_arxiv="2410.18057",
        requirements=(
            "frozen_teacher_model",
            "retain_set",
            "dual_model_memory_or_offload",
        ),
        notes=(
            "CLEAR-only prototype with the paper's KL direction by default."
        ),
    ),
    "MMRMU": MMAlgorithmRegistration(
        name="MMRMU",
        family="benchmark_baseline",
        implementation=None,
        status="Planned",
        supported_benchmarks=("clear",),
        benchmark_origin=("CLEAR",),
        requirements=("multimodal_layer_mapping", "retain_set"),
        notes="Requires multimodal layer mapping; the text-only RMU trainer is insufficient.",
    ),
    "MMSKU": MMAlgorithmRegistration(
        name="MMSKU",
        family="benchmark_baseline",
        implementation=None,
        status="Planned",
        supported_benchmarks=("clear",),
        benchmark_origin=("CLEAR",),
        requirements=("related_answer_adapter", "weight_arithmetic_stage"),
    ),
    "MMPrompting": MMAlgorithmRegistration(
        name="MMPrompting",
        family="benchmark_baseline",
        implementation=None,
        status="Planned",
        supported_benchmarks=("mllmu",),
        benchmark_origin=("MLLMU-Bench",),
        requirements=("inference_only_prompt_policy",),
        notes="MLLMU-Bench prompting baseline; it does not update model weights.",
    ),
    # Later methods and adjacent tasks. They remain non-runnable until their
    # architecture-specific code and data contracts are integrated and tested.
    "MMVKD": MMAlgorithmRegistration(
        name="MMVKD",
        family="later_method",
        implementation=None,
        status="Planned",
        supported_benchmarks=("mllmu", "clear"),
        paper_arxiv="2512.11325",
        aliases=("VKD",),
        requirements=("intermediate_visual_teacher_features", "vision_only_updates"),
        notes="The paper names the method VKD; MMVKD is retained only as a user-facing alias.",
    ),
    "MIPEditor": MMAlgorithmRegistration(
        name="MIPEditor",
        family="later_method",
        implementation=None,
        status="Planned",
        supported_benchmarks=("mllmu", "clear"),
        paper_arxiv="2511.06793",
        aliases=("MIP-Editor",),
        requirements=("influential_path_checkpoint_or_discovery", "qwen25vl_hooks"),
    ),
    "CLIPErase": MMAlgorithmRegistration(
        name="CLIPErase",
        family="adjacent_multimodal_method",
        implementation=None,
        status="Planned",
        supported_benchmarks=("cifar100", "flickr30k"),
        paper_arxiv="2410.23330",
        requirements=("clip_model_adapter",),
        notes="CLIP association unlearning, not an MLLM/CLEAR/MLLMU baseline.",
    ),
    "SingleImageUnlearning": MMAlgorithmRegistration(
        name="SingleImageUnlearning",
        family="later_method",
        implementation=None,
        status="Planned",
        supported_benchmarks=("mmubench",),
        paper_arxiv="2405.12523",
        aliases=("SIU",),
        requirements=("mmubench_adapter", "dual_masked_kl"),
    ),
}


MM_UNLEARNING_DATASETS: dict[str, MMDatasetRegistration] = {
    "mllmu": MMDatasetRegistration(
        key="mllmu",
        name="MLLMU-Bench",
        loader="data.multimodal.get_mm_data",
        evaluator="evals.mllmu.MLLMUEvaluator",
        status="Implemented",
        paper_arxiv="2410.22108",
        source="MLLMMU/MLLMU-Bench",
        supported_splits=(
            "forget_5",
            "forget_10",
            "forget_15",
            "retain_95",
            "retain_90",
            "retain_85",
            "Retain_Set",
            "Test_Set",
        ),
        original_baselines=(
            "MMGradAscent",
            "MMGradDiff",
            "MMKLMin",
            "MMNPO",
            "MMPrompting",
        ),
        original_models=("LLaVA-1.5-7B", "Idefics2-8B"),
        aliases=("mllmu-bench", "mllmu_bench"),
    ),
    "clear": MMDatasetRegistration(
        key="clear",
        name="CLEAR",
        loader="data.clear_dataset.get_clear_data",
        evaluator="evals.clear.CLEAREvaluator",
        status="Implemented",
        paper_arxiv="2410.18057",
        source="therem/CLEAR",
        supported_splits=(
            "forget01",
            "forget05",
            "forget10",
            "retain99",
            "retain95",
            "retain90",
            "real_faces",
            "real_world",
        ),
        original_baselines=(
            "MMRetainFT",
            "MMIDK",
            "MMDPO",
            "MMLLMU",
            "MMSCRUB",
            "MMRMU",
            "MMGradAscent",
            "MMGradDiff",
            "MMKLMin",
            "MMNPO",
            "MMSKU",
        ),
        original_models=("LLaVA-1.5-7B", "Qwen2-VL-2B"),
    ),
}


_AUTO_LINEAR_LORA_POLICY = "auto-linear-excluding-embeddings-lm-head-patch-embed"


def _seven_b_method_compatibility(
    *, retain_ft_single_rtx3090_status: str = "configured"
) -> tuple[MMModelMethodCompatibility, ...]:
    """Return conservative execution contracts for configured 7B methods."""

    benchmarks = ("mllmu", "clear")
    return (
        MMModelMethodCompatibility(
            method="MMGradAscent",
            supported_benchmarks=benchmarks,
            update_policy="lora",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier="single-7B-model",
            single_rtx3090_status="configured",
            requirements=("benchmark_trained_base_checkpoint",),
        ),
        MMModelMethodCompatibility(
            method="MMGradDiff",
            supported_benchmarks=benchmarks,
            update_policy="lora",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier="single-7B-model-benchmark-aware-retain-schedule",
            single_rtx3090_status="configured",
            requirements=("benchmark_trained_base_checkpoint", "retain_set"),
        ),
        MMModelMethodCompatibility(
            method="MMKLMin",
            supported_benchmarks=benchmarks,
            update_policy="lora-with-frozen-reference",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier="dual-7B-model-or-offload",
            single_rtx3090_status="requires-dual-model-memory",
            requirements=(
                "benchmark_trained_base_checkpoint",
                "retain_set",
                "frozen_reference_model",
                "multi_gpu_or_model_offload",
            ),
            notes=(
                "Requires memory for the trainable and frozen reference models; "
                "use additional devices or offload when needed."
            ),
        ),
        MMModelMethodCompatibility(
            method="MMNPO",
            supported_benchmarks=benchmarks,
            update_policy="lora-with-frozen-oracle",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier="dual-7B-model-or-offload",
            single_rtx3090_status="requires-dual-model-memory",
            requirements=(
                "benchmark_trained_base_checkpoint",
                "retain_only_oracle_checkpoint",
                "multi_gpu_or_model_offload",
            ),
            notes=(
                "Requires a retain-only oracle checkpoint and memory for both models."
            ),
        ),
        MMModelMethodCompatibility(
            method="MMRetainFT",
            supported_benchmarks=benchmarks,
            update_policy="lora",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier="single-7B-model",
            single_rtx3090_status=retain_ft_single_rtx3090_status,
            requirements=("retain_set",),
            notes=(
                "Reference/gold construction stage; it is a CLEAR baseline but not "
                "one of the four trainable MLLMU-Bench unlearning baselines."
            ),
        ),
        MMModelMethodCompatibility(
            method="MMSCRUB",
            supported_benchmarks=("clear",),
            update_policy="lora-with-frozen-teacher",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier="dual-7B-model-or-offload",
            single_rtx3090_status="requires-dual-model-memory",
            requirements=(
                "benchmark_trained_base_checkpoint",
                "retain_set",
                "frozen_teacher_model",
                "multi_gpu_or_model_offload",
            ),
            notes=(
                "CLEAR-only prototype requiring a frozen teacher, retain data and "
                "sufficient memory for both models."
            ),
        ),
        MMModelMethodCompatibility(
            method="MMUnlearner",
            supported_benchmarks=benchmarks,
            update_policy="full-parameter-masked-alternating-updates",
            lora_target_policy="not-applicable-full-parameter",
            memory_tier="full-parameter-7B-high-risk",
            single_rtx3090_status="requires-full-parameter-mask-memory",
            requirements=(
                "benchmark_trained_base_checkpoint",
                "saliency_mask",
                "retain_set",
                "architecture_mask_selector",
                "target_text_preservation_evaluation",
            ),
            notes=(
                "Requires full-parameter updates and checkpoint-bound saliency masks. "
                "Its target is visual deletion with same-entity text preservation; "
                "mask construction needs a separate memory budget."
            ),
        ),
        MMModelMethodCompatibility(
            method="MMMANU",
            supported_benchmarks=benchmarks,
            update_policy="full-parameter-materialized-neuron-scaling",
            lora_target_policy="not-applicable-full-parameter",
            memory_tier="single-7B-model-activation-statistics",
            single_rtx3090_status="configured",
            requirements=(
                "benchmark_trained_base_checkpoint",
                "paired_text_multimodal_inputs",
                "retain_set",
            ),
            notes=(
                "The adapter materializes neuron scaling as down_proj column edits "
                "for saved checkpoints."
            ),
        ),
    )


def _small_qwen_method_compatibility(
    parameter_scale: str,
    *,
    ga_single_rtx3090_status: str = "configured",
    manu_single_rtx3090_status: str = "configured",
) -> tuple[MMModelMethodCompatibility, ...]:
    """Return conservative contracts for Qwen2/2.5-VL 2B--3B targets."""

    benchmarks = ("mllmu", "clear")
    single_model = f"single-{parameter_scale}-model"
    dual_model = f"dual-{parameter_scale}-model"
    return (
        MMModelMethodCompatibility(
            method="MMGradAscent",
            supported_benchmarks=benchmarks,
            update_policy="lora",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier=single_model,
            single_rtx3090_status=ga_single_rtx3090_status,
            requirements=("benchmark_trained_base_checkpoint",),
        ),
        MMModelMethodCompatibility(
            method="MMGradDiff",
            supported_benchmarks=benchmarks,
            update_policy="lora",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier=single_model,
            single_rtx3090_status="configured",
            requirements=("benchmark_trained_base_checkpoint", "retain_set"),
        ),
        MMModelMethodCompatibility(
            method="MMKLMin",
            supported_benchmarks=benchmarks,
            update_policy="lora-with-frozen-reference",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier=dual_model,
            single_rtx3090_status="requires-dual-model-memory",
            requirements=(
                "benchmark_trained_base_checkpoint",
                "retain_set",
                "frozen_reference_model",
            ),
        ),
        MMModelMethodCompatibility(
            method="MMNPO",
            supported_benchmarks=benchmarks,
            update_policy="lora-with-frozen-oracle",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier=dual_model,
            single_rtx3090_status="requires-dual-model-memory",
            requirements=(
                "benchmark_trained_base_checkpoint",
                "retain_only_oracle_checkpoint",
            ),
        ),
        MMModelMethodCompatibility(
            method="MMRetainFT",
            supported_benchmarks=benchmarks,
            update_policy="lora",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier=single_model,
            single_rtx3090_status="configured",
            requirements=("retain_set",),
            notes="CLEAR-native reference construction; not an MLLMU baseline.",
        ),
        MMModelMethodCompatibility(
            method="MMSCRUB",
            supported_benchmarks=("clear",),
            update_policy="lora-with-frozen-teacher",
            lora_target_policy=_AUTO_LINEAR_LORA_POLICY,
            memory_tier=dual_model,
            single_rtx3090_status="requires-dual-model-memory",
            requirements=("retain_set", "frozen_teacher_model"),
        ),
        MMModelMethodCompatibility(
            method="MMUnlearner",
            supported_benchmarks=benchmarks,
            update_policy="full-parameter-masked-alternating-updates",
            lora_target_policy="not-applicable-full-parameter",
            memory_tier=f"full-parameter-{parameter_scale}",
            single_rtx3090_status="requires-full-parameter-mask-memory",
            requirements=(
                "benchmark_trained_base_checkpoint",
                "saliency_mask",
                "retain_set",
                "architecture_mask_selector",
                "target_text_preservation_evaluation",
            ),
        ),
        MMModelMethodCompatibility(
            method="MMMANU",
            supported_benchmarks=benchmarks,
            update_policy="full-parameter-materialized-neuron-scaling",
            lora_target_policy="not-applicable-full-parameter",
            memory_tier=f"{single_model}-activation-statistics",
            single_rtx3090_status=manu_single_rtx3090_status,
            requirements=(
                "benchmark_trained_base_checkpoint",
                "paired_text_multimodal_inputs",
                "retain_set",
            ),
        ),
    )


MM_UNLEARNING_MODELS: dict[str, MMModelRegistration] = {
    "llava15_7b": MMModelRegistration(
        key="llava15_7b",
        name="LLaVA-1.5-7B",
        config_name="LLaVA-1.5-7B",
        model_family="llava",
        source_model_id="llava-hf/llava-1.5-7b-hf",
        status="Configured",
        supported_benchmarks=("mllmu", "clear"),
        paper_original_benchmarks=("mllmu", "clear"),
        minimum_transformers_version="4.45.0",
        method_compatibility=_seven_b_method_compatibility(),
        aliases=("llava-1.5-7b", "llava15-7b", "llava"),
        notes=(
            "Paper-aligned only when model_path is overridden with the benchmark's "
            "full-data-finetuned vanilla/base checkpoint."
        ),
    ),
    "qwen2vl_7b": MMModelRegistration(
        key="qwen2vl_7b",
        name="Qwen2-VL-7B",
        config_name="Qwen2VL-7B",
        model_family="qwen2vl",
        source_model_id="Qwen/Qwen2-VL-7B-Instruct",
        status="Configured",
        supported_benchmarks=("mllmu", "clear"),
        paper_original_benchmarks=(),
        minimum_transformers_version="4.45.0",
        method_compatibility=_seven_b_method_compatibility(
            retain_ft_single_rtx3090_status="configured"
        ),
        aliases=("qwen2-vl-7b", "qwen2vl-7b", "qwen2-vl-7b-instruct"),
        notes=(
            "Portability extension: MLLMU-Bench did not use Qwen2-VL, and CLEAR's "
            "appendix used Qwen2-VL-2B rather than 7B."
        ),
    ),
    "qwen2vl_2b": MMModelRegistration(
        key="qwen2vl_2b",
        name="Qwen2-VL-2B",
        config_name="Qwen2VL-2B",
        model_family="qwen2vl",
        source_model_id="Qwen/Qwen2-VL-2B-Instruct",
        status="Configured",
        supported_benchmarks=("mllmu", "clear"),
        paper_original_benchmarks=("clear",),
        minimum_transformers_version="4.45.0",
        method_compatibility=_small_qwen_method_compatibility(
            "2B",
            ga_single_rtx3090_status="configured",
            manu_single_rtx3090_status="configured",
        ),
        aliases=("qwen2-vl-2b", "qwen2vl-2b", "qwen2-vl-2b-instruct"),
        notes=(
            "CLEAR appendix architecture and MLLMU portability target; use "
            "model-specific method settings."
        ),
    ),
    "qwen25vl_3b": MMModelRegistration(
        key="qwen25vl_3b",
        name="Qwen2.5-VL-3B",
        config_name="Qwen2.5-VL-3B",
        model_family="qwen2_5_vl",
        source_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        status="Configured",
        supported_benchmarks=("mllmu", "clear"),
        paper_original_benchmarks=(),
        minimum_transformers_version="4.49.0",
        method_compatibility=_small_qwen_method_compatibility(
            "3B",
            ga_single_rtx3090_status="configured",
            manu_single_rtx3090_status="configured",
        ),
        aliases=(
            "qwen2.5-vl-3b",
            "qwen25vl-3b",
            "qwen2.5-vl-3b-instruct",
        ),
        notes=(
            "Portability target for CLEAR and MLLMU-Bench; it is not an original "
            "model of either benchmark."
        ),
    ),
}


def _normalize_name(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def _algorithm_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name, registration in MM_UNLEARNING_ALGORITHMS.items():
        for alias in (name, *registration.aliases):
            aliases[_normalize_name(alias)] = name
    return aliases


def _dataset_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, registration in MM_UNLEARNING_DATASETS.items():
        for alias in (key, registration.name, *registration.aliases):
            aliases[_normalize_name(alias)] = key
    return aliases


def _model_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, registration in MM_UNLEARNING_MODELS.items():
        for alias in (
            key,
            registration.name,
            registration.config_name,
            *registration.aliases,
        ):
            aliases[_normalize_name(alias)] = key
    return aliases


def get_mm_algorithm(
    name: str, *, require_runnable: bool = True
) -> MMAlgorithmRegistration:
    """Resolve an algorithm name/alias and optionally require local code."""

    canonical = _algorithm_aliases().get(_normalize_name(name))
    if canonical is None:
        available = ", ".join(sorted(MM_UNLEARNING_ALGORITHMS))
        raise KeyError(
            f"Unknown multimodal unlearning method `{name}`. Available: {available}"
        )
    registration = MM_UNLEARNING_ALGORITHMS[canonical]
    if require_runnable and not registration.runnable:
        raise NotImplementedError(
            f"{registration.name} is registered as {registration.status} but has no "
            "faithful runnable implementation. Requirements: "
            f"{', '.join(registration.requirements) or 'not recorded'}."
        )
    return registration


def get_mm_dataset(name: str) -> MMDatasetRegistration:
    """Resolve a benchmark key/name/alias."""

    canonical = _dataset_aliases().get(_normalize_name(name))
    if canonical is None:
        available = ", ".join(sorted(MM_UNLEARNING_DATASETS))
        raise KeyError(
            f"Unknown multimodal unlearning benchmark `{name}`. Available: {available}"
        )
    return MM_UNLEARNING_DATASETS[canonical]


def get_mm_model_registration(name: str) -> MMModelRegistration:
    """Resolve a configured multimodal model without loading its weights."""

    canonical = _model_aliases().get(_normalize_name(name))
    if canonical is None:
        available = ", ".join(sorted(MM_UNLEARNING_MODELS))
        raise KeyError(
            f"Unknown multimodal unlearning model `{name}`. Available: {available}"
        )
    return MM_UNLEARNING_MODELS[canonical]


def get_mm_model_method_compatibility(
    model_name: str, method_name: str, benchmark_name: str
) -> MMModelMethodCompatibility:
    """Resolve a model/method/benchmark contract without claiming run success."""

    model = get_mm_model_registration(model_name)
    method = get_mm_algorithm(method_name, require_runnable=False)
    benchmark = get_mm_dataset(benchmark_name)
    if benchmark.key not in model.supported_benchmarks:
        raise ValueError(
            f"{model.name} is not configured for {benchmark.name}; configured: "
            f"{list(model.supported_benchmarks)}"
        )
    if benchmark.key not in method.supported_benchmarks:
        raise ValueError(
            f"{method.name} is not registered for {benchmark.name}; registered: "
            f"{list(method.supported_benchmarks)}"
        )
    for profile in model.method_compatibility:
        if profile.method == method.name:
            if benchmark.key not in profile.supported_benchmarks:
                break
            return profile
    raise ValueError(
        f"{model.name} has no configured {method.name} contract for {benchmark.name}"
    )


def import_symbol(dotpath: str) -> Any:
    """Import a class or callable from a registry dot path."""

    module_path, symbol_name = dotpath.rsplit(".", 1)
    return getattr(import_module(module_path), symbol_name)
