#!/usr/bin/env python3
"""Compile project Python sources while excluding VCS and virtualenv directories."""

from __future__ import annotations

import argparse
import compileall
import pathlib
import re
import sys

CONFLICT_MARKERS = ("<" * 7 + " ", "=" * 7, ">" * 7 + " ")
TEXT_EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".txt", ".toml", ".json", ".sh"}


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


def has_merge_conflict_markers(root: pathlib.Path, exclude: re.Pattern[str]) -> bool:
    has_conflict = False
    for source_file in root.rglob("*"):
        if not source_file.is_file() or source_file.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        normalized = source_file.as_posix()
        if exclude.search(normalized):
            continue
        try:
            content = source_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Warning: cannot read {source_file}: {exc}", file=sys.stderr)
            continue
        if any(marker in content for marker in CONFLICT_MARKERS):
            print(f"Merge conflict marker found: {source_file}", file=sys.stderr)
            has_conflict = True
    return has_conflict


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
    conflict_free = not has_merge_conflict_markers(root, exclude)
    return 0 if ok and conflict_free else 1


if __name__ == "__main__":
    sys.exit(main())
