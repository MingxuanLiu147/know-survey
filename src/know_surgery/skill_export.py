"""Export shared Agent Skills into a writable source workbench."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from know_surgery.resources import resource_root


def export_skills(destination: str | Path) -> list[Path]:
    """Export skills and their runtime into an existing source checkout.

    Existing nonidentical files are preserved and reported before any write.
    Runtime execution needs src/configs and writes artifacts in this workspace;
    skill definitions alone are not an installed execution environment.
    """
    destination = Path(destination).expanduser().resolve()
    if (
        not (destination / "src" / "surgery.py").is_file()
        or not (destination / "configs" / "surgery.yaml").is_file()
    ):
        raise ValueError(
            "Skill execution requires a Know-Surgery source workbench with src/ and configs/. "
            "Clone the repository first and pass that directory with --output."
        )
    resources = Path(__file__).parent / "_skill_resources"
    if resources.is_dir():
        skills = resources / "skills"
        runtime = resources / "scripts" / "skill_runtime.py"
    else:
        root = resource_root()
        skills = root / ".claude" / "skills"
        runtime = root / "scripts" / "skill_runtime.py"
    files = [
        (source, destination / ".claude" / "skills" / source.relative_to(skills))
        for source in sorted(skills.rglob("*"))
        if source.is_file() and source.suffix in {".md", ".py", ".json", ".yaml"}
    ]
    if not files or not runtime.is_file():
        raise FileNotFoundError(
            "The installed distribution is missing Agent Skill resources."
        )
    files.append((runtime, destination / "scripts" / "skill_runtime.py"))
    conflicts = [
        target
        for source, target in files
        if target.exists() and source.read_bytes() != target.read_bytes()
    ]
    if conflicts:
        raise FileExistsError(
            "Existing skill files differ; preserve/review them before export: "
            + ", ".join(str(path.relative_to(destination)) for path in conflicts)
        )
    agent_link = destination / ".agents" / "skills"
    if (agent_link.exists() or agent_link.is_symlink()) and (
        not agent_link.is_symlink()
        or agent_link.resolve() != (destination / ".claude" / "skills").resolve()
    ):
        raise FileExistsError(
            "Existing .agents/skills does not reference .claude/skills."
        )
    for source, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    if not agent_link.is_symlink():
        agent_link.parent.mkdir(parents=True, exist_ok=True)
        agent_link.symlink_to("../.claude/skills", target_is_directory=True)
    return [target for _, target in files]


def main(argv: list[str] | None = None) -> None:
    """Expose skill export through ``know-surgery export-skills``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Existing Know-Surgery source checkout",
    )
    args = parser.parse_args(argv)
    try:
        files = export_skills(args.output)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Exported {len(files)} Agent Skill files into {args.output}.")
    print("Skills execute in this source workbench; previews are not GPU verification.")
