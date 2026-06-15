# SPDX-License-Identifier: MIT
"""Module dispatcher for `python -m yfs_xrd_refinement`."""

from __future__ import annotations

import runpy
import sys

_COMMANDS = {
    "standard": "yfs_xrd_refinement.standard",
    "ql": "yfs_xrd_refinement.qlearning",
    "qlearning": "yfs_xrd_refinement.qlearning",
    "batch": "yfs_xrd_refinement.batch",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        commands = ", ".join(sorted(_COMMANDS))
        print("Usage: python -m yfs_xrd_refinement <command> [args...]")
        print(f"Commands: {commands}")
        raise SystemExit(0)

    command = sys.argv.pop(1)
    try:
        module_name = _COMMANDS[command]
    except KeyError:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Use --help to list available commands.", file=sys.stderr)
        raise SystemExit(2)

    sys.argv[0] = f"python -m yfs_xrd_refinement {command}"
    runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    main()
