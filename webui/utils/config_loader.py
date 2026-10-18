"""
配置加载工具
=============

扫描 configs/ 目录，提取可用的模型、方法、数据集、评测配置。
为 Gradio 下拉框提供数据源。
"""

import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional
import yaml


PUBLIC_OPERATION_BY_MODE = {
    "unlearn": "unlearning",
    "unlearning": "unlearning",
    "inject": "inject",
    "edit": "edit",
}
EXPERIMENTAL_CHOICE = "(Experimental manual composition / 实验性手工组合)"


class ConfigLoader:
    """Hydra 配置加载器，为 GUI 提供配置选项"""

    def __init__(self, configs_dir: str = None):
        """初始化配置加载器

        Args:
            configs_dir: configs 目录路径，默认为项目根目录下的 configs/
        """
        if configs_dir is None:
            # 默认从 webui 目录向上找 configs
            base_dir = Path(__file__).parent.parent.parent
            configs_dir = base_dir / "configs"
        self.configs_dir = Path(configs_dir)
        self.project_root = self.configs_dir.parent
        self.saves_dir = self.project_root / "saves"
        self._catalog_error = ""
        self._catalog_cache: tuple[Any, ...] | None = None

    def _scan_yaml_files(self, subdir: str, recursive: bool = False) -> List[str]:
        """扫描子目录下的 yaml 文件

        Args:
            subdir: 子目录名（相对于 configs/）
            recursive: 是否递归扫描子目录

        Returns:
            配置名列表（不含 .yaml 后缀）
        """
        target_dir = self.configs_dir / subdir
        if not target_dir.exists():
            return []

        configs = []
        if recursive:
            for yaml_file in target_dir.rglob("*.yaml"):
                # 生成相对路径作为配置名
                rel_path = yaml_file.relative_to(target_dir)
                config_name = str(rel_path).replace(".yaml", "").replace(os.sep, "/")
                configs.append(config_name)
        else:
            for yaml_file in target_dir.glob("*.yaml"):
                configs.append(yaml_file.stem)

        return sorted(configs)

    def _load_yaml(self, filepath: Path) -> Optional[Dict]:
        """加载单个 yaml 文件"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def _catalog_records(self) -> tuple[Any, ...]:
        """Return one catalog snapshot; explicit Refresh invalidates it."""
        if self._catalog_cache is not None:
            return self._catalog_cache
        try:
            from src.extension_registry import discover_extensions

            records = tuple(discover_extensions(self.project_root))
            self._catalog_error = ""
            self._catalog_cache = records
        except Exception as exc:
            self._catalog_error = f"Extension catalog unavailable: {exc}"
            self._catalog_cache = ()
        return self._catalog_cache

    def refresh_catalog(self) -> tuple[Any, ...]:
        """Invalidate cached configs/evidence and create one consistent snapshot."""
        self._catalog_cache = None
        return self._catalog_records()

    def get_catalog_records(self) -> tuple[Any, ...]:
        """Expose the current immutable snapshot to other UI catalog views."""
        return self._catalog_records()

    @staticmethod
    def _manifest_dict(record: Any) -> Dict[str, Any]:
        manifest = getattr(record, "manifest", None)
        if manifest is None:
            return {}
        if hasattr(manifest, "to_dict"):
            return manifest.to_dict()
        return dict(manifest) if isinstance(manifest, dict) else {}

    @staticmethod
    def _component_name(config_path: str | None, group: str) -> str | None:
        if not config_path:
            return None
        path = PurePosixPath(config_path)
        prefix = PurePosixPath("configs") / group
        try:
            relative = path.relative_to(prefix)
        except ValueError:
            return None
        return str(relative.with_suffix(""))

    def get_recipe_records(
        self, mode: str | None = None, *, experimental: bool = False
    ) -> tuple[Any, ...]:
        """Return text recipes in their computed catalog lane."""
        operation = PUBLIC_OPERATION_BY_MODE.get(mode, mode) if mode else None
        records = []
        for record in self._catalog_records():
            manifest = self._manifest_dict(record)
            record_operation = manifest.get("operation") or getattr(
                record, "operation", None
            )
            if operation and record_operation != operation:
                continue
            if not getattr(record, "experiment", None):
                continue
            is_default = bool(getattr(record, "default_runnable", False))
            if experimental == is_default:
                continue
            records.append(record)
        return tuple(sorted(records, key=lambda item: (item.id, str(item.source))))

    def get_recipe_metadata(self, experiment: str | None) -> Dict[str, Any]:
        """Resolve the component bindings for one trusted recipe."""
        if not experiment or experiment.startswith("("):
            return {}
        for record in self._catalog_records():
            if getattr(record, "experiment", None) != experiment:
                continue
            manifest = self._manifest_dict(record)
            recipe = manifest.get("recipe", {})
            datasets = {
                role: self._component_name(path, "data/datasets")
                for role, path in recipe.get("datasets", {}).items()
            }
            return {
                "id": record.id,
                "status": record.status,
                "reasons": list(record.reasons),
                "experiment": experiment,
                "operation": manifest.get("operation"),
                "model": self._component_name(recipe.get("model"), "model"),
                "trainer": self._component_name(recipe.get("method"), "trainer"),
                "datasets": {key: value for key, value in datasets.items() if value},
                "evaluation": self._component_name(recipe.get("evaluation"), "eval"),
                "default_runnable": bool(record.default_runnable),
            }
        return {}

    def get_catalog_diagnostics(self) -> str:
        """Summarize default and experimental lanes for visible UI feedback."""
        records = self._catalog_records()
        default_count = sum(bool(record.default_runnable) for record in records)
        experimental_count = len(records) - default_count
        prefix = f"{self._catalog_error} " if self._catalog_error else ""
        return (
            f"{prefix}{default_count} default recipes; "
            f"{experimental_count} experimental/disabled entries."
        ).strip()

    def _bound_component_names(
        self, component: str, mode: str | None = None
    ) -> List[str]:
        values = []
        for record in self.get_recipe_records(mode):
            metadata = self.get_recipe_metadata(record.experiment)
            if component == "datasets":
                values.extend(metadata.get("datasets", {}).values())
            elif metadata.get(component):
                values.append(metadata[component])
        return sorted(set(values))

    def _text_model_configs(self) -> List[str]:
        models = []
        for name in self._scan_yaml_files("model"):
            config = self.get_model_config(name)
            # Multimodal configs use model_path/processor_path rather than the
            # text loader's model_args/tokenizer_args contract.
            if config.get("model_args") and config.get("tokenizer_args"):
                models.append(name)
        return models

    def get_models(
        self, mode: str | None = None, *, experimental: bool = False
    ) -> List[str]:
        """获取可用模型列表

        扫描 configs/model/*.yaml
        """
        if experimental:
            return self._text_model_configs()
        return self._bound_component_names("model", mode)

    def get_trainers(
        self, mode: str = None, *, experimental: bool = False
    ) -> List[str]:
        """获取可用训练方法列表

        Args:
            mode: 训练模式 (unlearn/inject/edit)，用于过滤方法

        Returns:
            方法名列表
        """
        if not experimental:
            return self._bound_component_names("trainer", mode)

        trainer_dir = self.configs_dir / "trainer"
        if not trainer_dir.exists():
            return []

        trainers = []

        # 根据 mode 过滤
        if mode == "unlearn":
            # Unlearn 方法在 trainer/ 顶层
            for yaml_file in trainer_dir.glob("*.yaml"):
                if yaml_file.stem not in ["finetune"]:  # 排除非 unlearn 方法
                    trainers.append(yaml_file.stem)
        elif mode == "inject":
            # Inject 方法在 trainer/inject/
            inject_dir = trainer_dir / "inject"
            if inject_dir.exists():
                for yaml_file in inject_dir.glob("*.yaml"):
                    if yaml_file.stem != "base_inject":
                        trainers.append(f"inject/{yaml_file.stem}")
        elif mode == "edit":
            # Edit 方法在 trainer/edit/
            edit_dir = trainer_dir / "edit"
            if edit_dir.exists():
                for yaml_file in edit_dir.glob("*.yaml"):
                    if yaml_file.stem != "base_editor":
                        trainers.append(f"edit/{yaml_file.stem}")
        else:
            # 返回所有方法
            trainers = self._scan_yaml_files("trainer", recursive=True)
            # 过滤掉 base 配置
            trainers = [
                t
                for t in trainers
                if not t.endswith("base_inject") and not t.endswith("base_editor")
            ]

        return sorted(
            trainer
            for trainer in trainers
            if not trainer.split("/")[-1].startswith("MM")
        )

    def get_datasets(
        self, mode: str = None, *, experimental: bool = False
    ) -> Dict[str, List[str]]:
        """获取数据集配置

        Args:
            mode: 训练模式，决定返回的数据集类型

        Returns:
            数据集配置字典，包含 forget/retain/edit/train 等键
        """
        if not experimental:
            result: Dict[str, List[str]] = {}
            for record in self.get_recipe_records(mode):
                metadata = self.get_recipe_metadata(record.experiment)
                for role, name in metadata.get("datasets", {}).items():
                    result.setdefault(role, []).append(name)
            return {role: sorted(set(names)) for role, names in result.items()}

        datasets_dir = self.configs_dir / "data" / "datasets"
        if not datasets_dir.exists():
            return {}

        all_datasets = []
        for yaml_file in datasets_dir.glob("*.yaml"):
            lowered_name = yaml_file.stem.lower()
            if lowered_name.startswith(("mm", "mllmu", "vlkeb", "clear")):
                continue
            config = self._load_yaml(yaml_file) or {}
            handlers = [
                item.get("handler", "")
                for item in config.values()
                if isinstance(item, dict)
            ]
            if any(str(handler).startswith("MM") for handler in handlers):
                continue
            all_datasets.append(yaml_file.stem)

        result = {}

        if mode == "unlearn":
            # Unlearn 需要 forget 和 retain 数据集
            result["forget"] = [d for d in all_datasets if "forget" in d.lower()]
            result["retain"] = [d for d in all_datasets if "retain" in d.lower()]
        elif mode == "edit":
            # Edit 需要编辑数据集
            result["edit"] = [d for d in all_datasets if "edit" in d.lower()]
        elif mode == "inject":
            # Inject 需要训练数据集
            result["train"] = [
                d
                for d in all_datasets
                if "inject" in d.lower()
                or "alpaca" in d.lower()
                or "custom" in d.lower()
            ]
        else:
            result["all"] = sorted(all_datasets)

        return result

    def get_evals(self, mode: str = None, *, experimental: bool = False) -> List[str]:
        """获取评测套件列表

        Args:
            mode: 训练模式，用于智能排序（推荐的套件排前面）
        """
        if not experimental:
            return self._bound_component_names("evaluation", mode)

        eval_dir = self.configs_dir / "eval"
        if not eval_dir.exists():
            return []

        # 扫描所有评估配置
        all_evals = []
        for yaml_file in eval_dir.glob("*.yaml"):
            all_evals.append(yaml_file.stem)

        # 根据模式智能排序：推荐的评估套件排在前面
        recommended = {
            "unlearn": ["tofu", "muse"],
            "edit": ["edit"],
            "inject": ["inject"],
        }

        if mode and mode in recommended:
            # 推荐套件在前，其他在后
            rec_list = [e for e in recommended[mode] if e in all_evals]
            other_list = sorted([e for e in all_evals if e not in rec_list])
            return rec_list + other_list

        return sorted(all_evals)

    def get_experiments(
        self, mode: str = None, *, experimental: bool = False
    ) -> List[str]:
        """获取实验模板列表

        Args:
            mode: 训练模式，用于过滤实验模板
        """
        if not experimental:
            experiments = [
                record.experiment for record in self.get_recipe_records(mode)
            ]
            return sorted(set(experiments))

        exp_dir = self.configs_dir / "experiment"
        if not exp_dir.exists():
            return ["(无模板 - 自定义配置)"]

        experiments = [EXPERIMENTAL_CHOICE]

        if mode:
            mode_dir = exp_dir / mode
            if mode_dir.exists():
                for yaml_file in mode_dir.rglob("*.yaml"):
                    rel_path = yaml_file.relative_to(exp_dir)
                    exp_name = str(rel_path).replace(".yaml", "").replace(os.sep, "/")
                    if "/multimodal/" not in f"/{exp_name}/":
                        experiments.append(exp_name)
        else:
            for yaml_file in exp_dir.rglob("*.yaml"):
                rel_path = yaml_file.relative_to(exp_dir)
                exp_name = str(rel_path).replace(".yaml", "").replace(os.sep, "/")
                if "/multimodal/" not in f"/{exp_name}/":
                    experiments.append(exp_name)

        return experiments

    def get_trainer_config(self, trainer_name: str) -> Dict:
        """获取特定 trainer 的配置内容

        Args:
            trainer_name: trainer 名称（如 "SimNPO" 或 "inject/LoRA"）

        Returns:
            配置字典
        """
        trainer_path = self.configs_dir / "trainer" / f"{trainer_name}.yaml"
        return self._load_yaml(trainer_path) or {}

    def get_model_config(self, model_name: str) -> Dict:
        """获取特定模型的配置内容"""
        model_path = self.configs_dir / "model" / f"{model_name}.yaml"
        return self._load_yaml(model_path) or {}

    def get_default_config(self, mode: str) -> Dict:
        """获取某个模式的默认入口配置

        Args:
            mode: 模式名 (unlearn/inject/edit/eval)
        """
        config_path = self.configs_dir / f"{mode}.yaml"
        return self._load_yaml(config_path) or {}

    def get_saved_models(self) -> List[str]:
        """获取 saves/ 目录下已保存的模型路径

        扫描 saves/unlearn/, saves/finetune/, saves/edit/ 等目录，
        查找包含 model.safetensors 或 pytorch_model.bin 的目录

        Returns:
            模型路径列表（相对于项目根目录）
        """
        saved_models = []

        if not self.saves_dir.exists():
            return saved_models

        # 扫描的子目录
        subdirs = ["unlearn", "finetune", "edit", "inject"]

        for subdir in subdirs:
            subdir_path = self.saves_dir / subdir
            if not subdir_path.exists():
                continue

            # 遍历所有子目录
            for model_dir in subdir_path.iterdir():
                if not model_dir.is_dir():
                    continue

                # 检查是否包含模型文件
                has_model = (
                    (model_dir / "model.safetensors").exists()
                    or (model_dir / "pytorch_model.bin").exists()
                    or (model_dir / "adapter_model.safetensors").exists()
                )

                if has_model:
                    # 返回相对路径
                    rel_path = model_dir.relative_to(self.project_root)
                    saved_models.append(str(rel_path))

        return sorted(saved_models)

    def get_eval_suites(self) -> List[str]:
        """获取可用的评测套件列表（用于 eval 模式）

        Returns:
            评测套件名列表
        """
        eval_dir = self.configs_dir / "eval"
        if not eval_dir.exists():
            return []

        suites = []
        for yaml_file in eval_dir.glob("*.yaml"):
            suites.append(yaml_file.stem)

        return sorted(suites)
