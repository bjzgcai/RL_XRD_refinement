# SPDX-License-Identifier: MIT
"""Compatibility wrapper for `yfs_xrd_refinement.batch`."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from yfs_xrd_refinement.batch import main as batch_main
    batch_main()


if __name__ == "__main__":
    main()
