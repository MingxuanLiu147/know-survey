"""Shared evaluation inputs and reference-model lifetime for training and CLI."""

import json
import os
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf, open_dict


def _to_plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _load_structured_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation input file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext in {".yaml", ".yml"}:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    raise ValueError(
        f"Unsupported evaluation input format for {path}. Use .json/.jsonl/.yaml/.yml."
    )


def _get_optional_evaluator_input(eval_cfg: DictConfig, key: str):
    value = eval_cfg.get(key, None)
    if value is not None:
        return _to_plain(value)
    path = eval_cfg.get(f"{key}_path", None)
    if path is None:
        return None
    return _load_structured_data(path)


class EvaluationRuntime:
    """Resolve explicit evaluator inputs and reuse references within one run.

    Close after the evaluation/training run to release references. Missing inputs
    stay absent; each evaluator decides whether to fail or record unavailability.
    """

    def __init__(self, model_cfg, model_loader=None):
        from model.checkpoint import load_checkpoint

        self.model_cfg = deepcopy(model_cfg)
        self.model_loader = model_loader or load_checkpoint
        self.original_models = {}

    def resolve(self, evaluator_cfg):
        args = {}
        for key in ("edit_data", "eval_data", "benchmark_data"):
            value = _get_optional_evaluator_input(evaluator_cfg, key)
            if value is not None:
                args[key] = value
        original = evaluator_cfg.get("original_model")
        path = evaluator_cfg.get("original_model_path")
        if original is not None and path is not None:
            raise ValueError(
                "Use either original_model or original_model_path, not both."
            )
        if original is not None:
            args["original_model"] = original
        elif path is not None:
            path = str(path)
            if path not in self.original_models:
                cfg = deepcopy(self.model_cfg)
                with open_dict(cfg.model_args):
                    cfg.model_args.pretrained_model_name_or_path = path
                reference, _ = self.model_loader(cfg)
                reference.eval()
                reference.requires_grad_(False)
                self.original_models[path] = reference
            args["original_model"] = self.original_models[path]
        return args

    def close(self):
        self.original_models.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
