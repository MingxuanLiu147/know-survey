"""Installed entrypoint for the canonical surgery router."""

import sys


def main(argv: list[str] | None = None) -> int | None:
    """Run the same router used by ``python src/surgery.py``."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "agent":
        from know_surgery.agent import main as agent_main

        return agent_main(argv[1:])
    if argv and argv[0] == "export-skills":
        from know_surgery.skill_export import main as export_main

        export_main(argv[1:])
        return
    from surgery import main as surgery_main

    surgery_main(argv, command_prefix=["know-surgery"])
