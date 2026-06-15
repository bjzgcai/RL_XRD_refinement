# SPDX-License-Identifier: MIT
"""Compatibility wrapper for `yfs_xrd_refinement.qlearning`.

The implementation lives under `src/` so the project can be installed as a
normal Python package. This wrapper preserves the original script command.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    runpy.run_module("yfs_xrd_refinement.qlearning", run_name="__main__")


if __name__ == "__main__":
    main()
