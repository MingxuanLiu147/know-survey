from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).parent


def require_public_source():
    """Keep restricted implementation modules out of built distributions."""
    if any(
        (ROOT / "src/trainer/edit" / name).exists()
        for name in ("unke_editor.py", "preservation.py", "research_unke_backend.py")
    ):
        raise RuntimeError(
            "Distribution cannot include restricted implementation modules: "
            "unke_editor.py, preservation.py or research_unke_backend.py. "
            "Remove these modules from the source being packaged."
        )


class SourceDistribution(sdist):
    """Apply the same source boundary to source and binary distributions."""

    def run(self):
        require_public_source()
        super().run()


class BuildPy(build_py):
    """Bundle the repository's skill definitions without copying run outputs."""

    def run(self):
        if not getattr(self, "editable_mode", False):
            require_public_source()
        super().run()
        skill_root = ROOT / ".claude" / "skills"
        resource_root = Path(self.build_lib) / "know_surgery" / "_skill_resources"
        for source in skill_root.rglob("*"):
            if source.is_file() and source.suffix in {".md", ".py", ".json", ".yaml"}:
                destination = resource_root / "skills" / source.relative_to(skill_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                self.copy_file(str(source), str(destination))
        runtime = ROOT / "scripts" / "skill_runtime.py"
        if runtime.is_file():
            destination = resource_root / "scripts" / runtime.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(runtime), str(destination))


with (ROOT / "requirements.txt").open(encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")
    ]

lm_eval_extra = ["lm-eval==0.4.8"]
optional_groups = {
    "gpu": {"bitsandbytes", "deepspeed"},
    "reft": {"pyreft"},
    "webui": {"gradio", "fastapi", "uvicorn", "python-multipart"},
}


def dependency_name(requirement):
    """Read the distribution name from the project's pinned requirements."""
    import re

    return re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].lower()


optional_names = set().union(*optional_groups.values())
extras = {
    group: [item for item in requirements if dependency_name(item) in names]
    for group, names in optional_groups.items()
}
extras.update(
    {
        "lm-eval": lm_eval_extra,
        "lm_eval": lm_eval_extra,
        "dev": [
            "build>=1.2.0",
            "ruff==0.6.9",
        ],
    }
)

setup(
    cmdclass={"build_py": BuildPy, "sdist": SourceDistribution},
    name="know-surgery",
    version="0.1.0",
    author="Know-Surgery contributors",
    description="Know-Surgery: a toolkit for controlled LLM knowledge unlearning, injection, editing, and evaluation.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/MingxuanLiu147/know-survey",
    license="MIT",
    license_files=["LICENSE", "THIRD_PARTY_NOTICES.md", "LICENSES/*.txt"],
    packages=find_packages("src") + ["know_surgery.configs"],
    package_dir={"": "src", "know_surgery.configs": "configs"},
    py_modules=[path.stem for path in sorted((ROOT / "src").glob("*.py"))],
    package_data={
        "know_surgery.configs": [
            path.relative_to(ROOT / "configs").as_posix()
            for path in sorted((ROOT / "configs").rglob("*.yaml"))
        ],
    },
    include_package_data=False,
    entry_points={"console_scripts": ["know-surgery=know_surgery.cli:main"]},
    install_requires=[
        item for item in requirements if dependency_name(item) not in optional_names
    ],
    extras_require=extras,
    python_requires=">=3.11",
)
