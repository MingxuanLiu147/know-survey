#!/usr/bin/env python3
"""Validate a Know-Surgery custom extension bundle or installed extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extension_registry import validate_bundle, validate_installed  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely validate an extension ZIP/directory, or validate a statically "
            "installed extension. Uploaded Python is never imported or executed."
        )
    )
    parser.add_argument("bundle", nargs="?", help="extension ZIP or directory")
    parser.add_argument(
        "--installed",
        metavar="EXTENSION_ID",
        help="validate an extension listed in configs/extensions/registry.yaml",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run the installed recipe's restricted one-sample smoke and write "
            "saves/extension_validation/<id>/validation.json"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if bool(args.bundle) == bool(args.installed):
        parser.error("provide exactly one bundle path or --installed EXTENSION_ID")
    if args.smoke and not args.installed:
        parser.error("--smoke is allowed only with --installed")

    if args.installed:
        result = validate_installed(
            args.installed, project_root=args.project_root, smoke=args.smoke
        )
    else:
        result = validate_bundle(args.bundle, project_root=args.project_root)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 1 if result.status == "disabled" else 0


if __name__ == "__main__":
    raise SystemExit(main())
