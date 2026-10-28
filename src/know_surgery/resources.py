"""Locate the single-source Hydra tree in a checkout or installed wheel."""

from pathlib import Path

import yaml


def resource_root() -> Path:
    """Return the directory containing bundled ``configs`` resources.

    A source checkout keeps configs beside src. A wheel stores the same files
    inside know_surgery, without copying user data or experiment outputs.
    """
    package_dir = Path(__file__).resolve().parent
    checkout = package_dir.parents[1]
    if (checkout / "src" / "surgery.py").is_file() and (
        checkout / "configs" / "surgery.yaml"
    ).is_file():
        return checkout
    if (package_dir / "configs" / "surgery.yaml").is_file():
        return package_dir
    raise FileNotFoundError(
        "Know-Surgery Hydra resources are missing; reinstall the package."
    )


def config_dir() -> Path:
    """Return the absolute Hydra config search directory."""
    return resource_root() / "configs"


def reference_trainer_for_experiment(experiment: str | None, trainer: str) -> str:
    """Keep a recipe's reference profile when a plan names its base algorithm.

    This resolves generated Skill/UI commands only. A different algorithm remains
    an explicit override, and direct CLI overrides retain their usual semantics.
    """
    if not experiment:
        return trainer
    root = config_dir() / "experiment"
    path = (root / f"{experiment}.yaml").resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return trainer
    recipe = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in recipe.get("defaults", []):
        if isinstance(item, dict):
            selected = item.get("override /trainer", item.get("/trainer"))
            if selected == f"{trainer}_Official":
                return selected
    return trainer
