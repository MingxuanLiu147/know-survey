"""Locate the single-source Hydra tree in a checkout or installed wheel."""

from pathlib import Path


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
