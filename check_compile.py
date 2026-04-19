#!/usr/bin/env python3
"""Compile project Python sources while excluding VCS and virtualenv directories."""

from __future__ import annotations

import argparse
import compileall
import pathlib
import re
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile Python files in the repository while ignoring directories "
            "that are not part of the source tree (e.g. .git and .venv)."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to compile (default: current directory).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=1,
        help="Decrease output verbosity (can be passed multiple times).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.path).resolve()

    # Exclude hidden tooling/environment directories that are not Python source.
    exclude = re.compile(r"(^|/)(\.git|\.venv|__pycache__|\.mypy_cache|\.pytest_cache)(/|$)")

    ok = compileall.compile_dir(
        str(root),
        quiet=args.quiet,
        rx=exclude,
        force=False,
        workers=0,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
