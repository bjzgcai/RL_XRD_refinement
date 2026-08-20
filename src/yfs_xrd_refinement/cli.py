# SPDX-License-Identifier: MIT
"""Console entry points for yfs_xrd_refinement."""

from __future__ import annotations


def standard() -> None:
    from .standard import cli_main
    cli_main()


def qlearning() -> None:
    from .qlearning import cli_main
    cli_main()


def batch() -> None:
    from .batch import main
    main()
