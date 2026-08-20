#!/usr/bin/env python3
"""Fail-closed input validation for the four occupancy lambda-sweep cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from pymatgen.core import Structure

COMMENT_PREFIXES = ("#", "%", ";", "!", "@", "'")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_target(text: str) -> Dict[str, float]:
    target: Dict[str, float] = {}
    for item in text.split(","):
        if ":" not in item:
            raise ValueError(f"invalid stoichiometry item: {item!r}")
        element, raw_value = (part.strip() for part in item.split(":", 1))
        if not element or element in target:
            raise ValueError(f"invalid or duplicate target element: {element!r}")
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"target value must be finite and positive: {item!r}")
        target[element] = value
    if not target:
        raise ValueError("stoichiometry target is empty")
    return target


def normalize_to_anion(
    composition: Dict[str, float], target: Dict[str, float], anion: str
) -> Dict[str, float]:
    if anion not in target or target[anion] <= 0.0:
        raise ValueError(f"target does not contain a positive {anion} reference")
    actual_anion = float(composition.get(anion, 0.0))
    if not math.isfinite(actual_anion) or actual_anion <= 0.0:
        raise ValueError(f"parsed CIF does not contain a positive {anion} amount")
    factor = target[anion] / actual_anion
    return {key: float(value) * factor for key, value in composition.items()}


def maximum_formula_deviation(
    normalized: Dict[str, float], target: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    keys = sorted(set(normalized) | set(target))
    differences = {
        key: float(normalized.get(key, 0.0) - target.get(key, 0.0))
        for key in keys
    }
    return max((abs(value) for value in differences.values()), default=0.0), differences


def read_xy_metadata(path: Path) -> Dict[str, object]:
    x_values = []
    observed = []
    column_counts = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(COMMENT_PREFIXES):
                continue
            fields = stripped.split()
            try:
                numeric = [float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(f"non-numeric XY row at line {line_number}") from exc
            if len(numeric) < 2:
                raise ValueError(f"XY row {line_number} has fewer than two columns")
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"non-finite XY value at line {line_number}")
            column_counts.add(len(numeric))
            x_values.append(numeric[0])
            observed.append(numeric[1])
    if len(x_values) < 2:
        raise ValueError("XY file must contain at least two numeric rows")
    if any(right <= left for left, right in zip(x_values, x_values[1:])):
        raise ValueError("XY 2theta values must be strictly increasing")
    if not any(abs(value) > 0.0 for value in observed):
        raise ValueError("XY observed intensity is identically zero")
    return {
        "rows": len(x_values),
        "source_column_counts": sorted(column_counts),
        "used_columns_zero_based": [0, 1],
        "ignored_columns_zero_based": sorted(
            {index for count in column_counts for index in range(2, count)}
        ),
        "two_theta_min": min(x_values),
        "two_theta_max": max(x_values),
        "observed_min": min(observed),
        "observed_max": max(observed),
    }


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def validate_inputs(args: argparse.Namespace) -> Dict[str, object]:
    cif_path = Path(args.cif).resolve(strict=True)
    xy_path = Path(args.xy).resolve(strict=True)
    physical_source_dir = Path(args.physical_source_dir).resolve(strict=True)
    if not cif_path.is_file() or not xy_path.is_file():
        raise ValueError("CIF and XY inputs must be regular files")
    target = parse_target(args.target)
    if args.anion not in target:
        raise ValueError(f"target is missing reference anion {args.anion}")
    cif_text = cif_path.read_text(encoding="utf-8", errors="replace")
    if args.expected_marker and args.expected_marker not in cif_text:
        raise ValueError(
            f"CIF identity marker {args.expected_marker!r} was not found for {args.sample}"
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        structure = Structure.from_file(cif_path)
    parsed = {
        key: float(value)
        for key, value in structure.composition.get_el_amt_dict().items()
        if float(value) > 0.0
    }
    if not parsed or not all(math.isfinite(value) for value in parsed.values()):
        raise ValueError("CIF produced an empty or non-finite composition")
    normalized = normalize_to_anion(parsed, target, args.anion)
    max_deviation, differences = maximum_formula_deviation(normalized, target)
    if max_deviation > args.max_formula_deviation:
        normalized_text = ", ".join(
            f"{key}={normalized[key]:.6g}" for key in sorted(normalized)
        )
        raise ValueError(
            f"{args.sample} CIF composition is incompatible with its target: "
            f"anion-normalized [{normalized_text}], maximum deviation "
            f"{max_deviation:.6g} > {args.max_formula_deviation:.6g}. "
            "Provide a composition-consistent CIF before rerunning."
        )
    xy_metadata = read_xy_metadata(xy_path)
    return {
        "schema_version": 1,
        "logical_sample_id": args.sample,
        "physical_source_dir": str(physical_source_dir),
        "cif_path": str(cif_path),
        "cif_sha256": sha256_file(cif_path),
        "xy_path": str(xy_path),
        "xy_sha256": sha256_file(xy_path),
        "expected_identity_marker": args.expected_marker,
        "stoich_target": target,
        "normalization_anion": args.anion,
        "parsed_composition": parsed,
        "anion_normalized_composition": normalized,
        "composition_differences_from_target": differences,
        "maximum_formula_deviation": max_deviation,
        "maximum_allowed_formula_deviation": args.max_formula_deviation,
        "cif_parser_warnings": [str(item.message) for item in caught],
        "xy": xy_metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--physical-source-dir", required=True)
    parser.add_argument("--cif", required=True)
    parser.add_argument("--xy", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--anion", required=True)
    parser.add_argument("--expected-marker", default="")
    parser.add_argument("--max-formula-deviation", type=float, default=0.02)
    parser.add_argument("--metadata-output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_formula_deviation) or args.max_formula_deviation < 0.0:
        parser.error("--max-formula-deviation must be finite and non-negative")
    try:
        metadata = validate_inputs(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.metadata_output is not None:
        write_json_atomic(args.metadata_output.resolve(), metadata)
    print(
        f"VALID: {args.sample} -> {metadata['physical_source_dir']} | "
        f"max formula deviation={metadata['maximum_formula_deviation']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
