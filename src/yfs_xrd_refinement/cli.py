# SPDX-License-Identifier: MIT
"""Console entry points for yfs_xrd_refinement."""

from __future__ import annotations

import runpy
import sys


def _run_module(module_name: str) -> None:
    sys.argv[0] = module_name
    runpy.run_module(module_name, run_name="__main__")


def standard() -> None:
    _run_module("yfs_xrd_refinement.standard")


def qlearning() -> None:
    _run_module("yfs_xrd_refinement.qlearning")


def batch() -> None:
    _run_module("yfs_xrd_refinement.batch")
