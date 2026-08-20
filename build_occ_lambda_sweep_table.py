#!/usr/bin/env python3
"""Build Table S8 from validated occupancy lambda-sweep outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "major_revision_consolidated_20260820"
    / "tables"
    / "Table_S8_occupancy_lambda_stoich_sweep.csv"
)
LAMBDAS = (0.3, 1.0, 2.0)


@dataclass(frozen=True)
class Case:
    physical_dir: str
    xy_filename: str
    target: Dict[str, float]
    anion: str
    formula_order: Tuple[str, ...]
    pending_status: str = "pending_corrected_rerun"


CASES: Dict[str, Case] = {
    "Li2HfCl6": Case(
        "Li2HfCl6", "Li2HfCl6.xy", {"Li": 2.0, "Hf": 1.0, "Cl": 6.0},
        "Cl", ("Li", "Hf", "Cl"),
    ),
    "LiMn1.5Ni0.5O4": Case(
        "LiMn1.5Ni0.5O4", "Refined.xy",
        {"Li": 1.0, "Mn": 1.5, "Ni": 0.5, "O": 4.0},
        "O", ("Li", "Mn", "Ni", "O"),
    ),
    "Na0.67Ni0.33Mn0.67O2": Case(
        "Na0.56Ni0.333Mn0.667O1.95", "Refined.xy",
        {"Na": 0.67, "Ni": 0.33, "Mn": 0.67, "O": 2.0},
        "O", ("Na", "Ni", "Mn", "O"),
        pending_status="blocked_source_cif_mismatch",
    ),
    "Na0.58Ni0.33Mn0.67O1.95": Case(
        "Na0.67Ni0.33Mn0.67O2", "Refined.xy",
        {"Na": 0.58, "Ni": 0.33, "Mn": 0.67, "O": 1.95},
        "O", ("Na", "Ni", "Mn", "O"),
    ),
}


def portable_text(value: object) -> str:
    """Render repository paths without publishing machine-specific prefixes."""
    return str(value).replace(str(PROJECT_DIR), ".")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values_match(left: Dict[str, float], right: Dict[str, float]) -> bool:
    return set(left) == set(right) and all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-9)
        for key in left
    )


def command_tokens(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    command = " ".join(line for line in lines if line and not line.startswith("#!"))
    return shlex.split(command)


def flag_value(tokens: Sequence[str], flag: str) -> str:
    try:
        index = tokens.index(flag)
    except ValueError as exc:
        raise ValueError(f"command is missing {flag}") from exc
    if index + 1 >= len(tokens):
        raise ValueError(f"command has no value after {flag}")
    return tokens[index + 1]


def parse_target(text: str) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for item in text.split(","):
        key, value = item.split(":", 1)
        parsed[key] = float(value)
    return parsed


def normalize_formula(
    composition: Dict[str, float], case: Case
) -> Dict[str, float]:
    anion_amount = float(composition.get(case.anion, 0.0))
    if not math.isfinite(anion_amount) or anion_amount <= 0.0:
        raise ValueError(f"refined composition has no positive {case.anion}")
    factor = case.target[case.anion] / anion_amount
    normalized = {
        key: float(value) * factor for key, value in composition.items()
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in normalized.values()):
        raise ValueError("refined composition contains invalid values")
    return normalized


def format_formula(normalized: Dict[str, float], case: Case) -> str:
    return "".join(
        f"{element}{normalized.get(element, 0.0):.2f}"
        for element in case.formula_order
    )


def validate_candidate(
    run_dir: Path, sample: str, case: Case, requested_lambda: float
) -> Tuple[str, float, str, str]:
    summary_path = run_dir / "run_summary.json"
    command_path = run_dir / "run_command.sh"
    if not summary_path.is_file() or not command_path.is_file():
        raise ValueError("missing run_summary.json or run_command.sh")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tokens = command_tokens(command_path)
    if "--single-phase" not in tokens:
        raise ValueError("command did not use --single-phase")
    if flag_value(tokens, "--occupancy-objective") != "stoich":
        raise ValueError("command did not use the active stoich objective")
    command_target = parse_target(flag_value(tokens, "--stoich"))
    if not values_match(command_target, case.target):
        raise ValueError("command stoichiometry target does not match logical sample")
    if not math.isclose(
        float(flag_value(tokens, "--lambda-stoich")), requested_lambda,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError("command lambda does not match table row")
    if "--wl" in tokens and not math.isclose(
        float(flag_value(tokens, "--wl")), 1.5406, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("command wavelength is not 1.5406 Angstrom")

    expected_source = (PROJECT_DIR / "Occ refinement" / case.physical_dir).resolve()
    input_manifest_path = run_dir / "input_manifest.json"
    if input_manifest_path.is_file():
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        manifest_source = Path(input_manifest["physical_source_dir"]).resolve()
        if manifest_source != expected_source:
            raise ValueError(
                f"physical source mismatch: {manifest_source} != {expected_source}"
            )
        if input_manifest.get("logical_sample_id") != sample:
            raise ValueError("input manifest logical sample mismatch")
        if not values_match(input_manifest.get("stoich_target", {}), case.target):
            raise ValueError("input manifest target mismatch")
        if float(input_manifest.get("maximum_formula_deviation", math.inf)) > float(
            input_manifest.get("maximum_allowed_formula_deviation", -math.inf)
        ):
            raise ValueError("input manifest formula validation failed")
        main_path = Path(flag_value(tokens, "--main")).resolve()
        xy_path = Path(flag_value(tokens, "--xy")).resolve()
        if main_path.parent != (run_dir / "inputs").resolve() or main_path.name != "data.cif":
            raise ValueError("command did not use the isolated CIF snapshot")
        if xy_path.parent != (run_dir / "inputs").resolve() or xy_path.name != "obs.xy":
            raise ValueError("command did not use the isolated XY snapshot")
        source_note = portable_text(input_manifest_path)
    else:
        main_path = Path(flag_value(tokens, "--main")).resolve()
        xy_path = Path(flag_value(tokens, "--xy")).resolve()
        if main_path != expected_source / "data.cif":
            raise ValueError(
                f"legacy command used wrong physical CIF: {main_path}"
            )
        if xy_path != expected_source / case.xy_filename:
            raise ValueError(
                f"legacy command used wrong physical XY: {xy_path}"
            )
        source_note = "legacy command-path provenance"

    if summary.get("occupancy_objective") != "stoich":
        raise ValueError("summary occupancy objective is not stoich")
    if not math.isclose(
        float(summary.get("lambda_stoich", math.nan)), requested_lambda,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError("summary lambda mismatch")
    if not values_match(summary.get("stoich_target", {}), case.target):
        raise ValueError("summary stoichiometry target mismatch")
    if not math.isclose(
        float(summary.get("wavelength_angstrom", math.nan)), 1.5406,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError("summary wavelength is not 1.5406 Angstrom")
    stage_values = summary.get("lambda_applied_by_stage", {})
    expected_stages = {"Stage 1", "Stage 2", "Stage 3"}
    if set(stage_values) != expected_stages or any(
        not math.isclose(float(value), requested_lambda, rel_tol=0.0, abs_tol=1e-9)
        for value in stage_values.values()
    ):
        raise ValueError("lambda was not applied consistently in all three stages")
    loops = summary.get("stage_loops", [])
    if len(loops) != 3 or any(int(value) <= 0 for value in loops):
        raise ValueError("summary stage loop counts are invalid")
    if int(summary.get("rl_update_count", 0)) <= 0:
        raise ValueError("summary contains no Q-learning updates")
    if summary.get("device") != "cuda":
        raise ValueError("summary device is not cuda")
    rwp = float(summary.get("final_rwp_percent", math.nan))
    penalty = float(summary.get("final_stoich_penalty_l2_squared", math.nan))
    objective = float(summary.get("final_search_objective", math.nan))
    if not all(math.isfinite(value) and value >= 0.0 for value in (rwp, penalty, objective)):
        raise ValueError("summary metrics are non-finite or negative")
    expected_objective = rwp + 100.0 * requested_lambda * penalty
    if not math.isclose(objective, expected_objective, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("summary objective does not recompute from Rwp, lambda, and L2^2")
    phase_key = summary.get("stoich_phase")
    composition = summary.get("refined_compositions", {}).get(phase_key)
    if not isinstance(composition, dict):
        raise ValueError("summary does not contain the target phase composition")
    normalized = normalize_formula(composition, case)
    return format_formula(normalized, case), rwp, str(summary_path), source_note


def discover_run_roots(explicit: Sequence[Path]) -> List[Path]:
    if explicit:
        return [path.resolve() for path in explicit]
    occ_dir = PROJECT_DIR / "Occ refinement"
    roots = [occ_dir / "lambda_stoich_sweep_cuda0"]
    roots.extend(sorted(occ_dir.glob("lambda_stoich_sweep_cuda0_corrected_*")))
    return [path.resolve() for path in roots if path.is_dir()]


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_table(run_roots: Sequence[Path], allow_pending: bool) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    table_rows: List[Dict[str, str]] = []
    validation_rows: List[Dict[str, str]] = []
    pending_count = 0
    for sample, case in CASES.items():
        for requested_lambda in LAMBDAS:
            lambda_label = str(requested_lambda).replace(".", "p")
            candidates = [
                root / f"lambda_{lambda_label}" / sample for root in run_roots
            ]
            valid = []
            rejected = []
            for run_dir in candidates:
                if not run_dir.exists():
                    continue
                try:
                    valid.append(
                        (run_dir,) + validate_candidate(
                            run_dir, sample, case, requested_lambda
                        )
                    )
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    rejected.append(portable_text(f"{run_dir}: {exc}"))
            if valid:
                run_dir, formula, rwp, summary_path, source_note = valid[-1]
                table_rows.append({
                    "target_formula": sample,
                    "lambda_stoich": f"{requested_lambda:.2f}",
                    "refined_formula_anion_normalized": formula,
                    "final_Rwp_percent": f"{rwp:.2f}",
                })
                validation_rows.append({
                    "target_formula": sample,
                    "lambda_stoich": f"{requested_lambda:.2f}",
                    "validation_status": "valid",
                    "reason": "validated summary, command, input mapping, wavelength, and three-stage lambda evidence",
                    "selected_run": portable_text(run_dir),
                    "summary_sha256": sha256_file(Path(summary_path)),
                    "provenance": source_note,
                    "rejected_candidates": " | ".join(rejected),
                })
            else:
                pending_count += 1
                reason = (
                    "source CIF composition is incompatible with Na0.67Ni0.33Mn0.67O2; "
                    "replace/repair and rerun with corrected physical mapping"
                    if case.pending_status == "blocked_source_cif_mismatch"
                    else "no output passed the corrected physical-directory and target validation; corrected rerun required"
                )
                table_rows.append({
                    "target_formula": sample,
                    "lambda_stoich": f"{requested_lambda:.2f}",
                    "refined_formula_anion_normalized": "NA",
                    "final_Rwp_percent": "NA",
                })
                validation_rows.append({
                    "target_formula": sample,
                    "lambda_stoich": f"{requested_lambda:.2f}",
                    "validation_status": case.pending_status,
                    "reason": reason,
                    "selected_run": "",
                    "summary_sha256": "",
                    "provenance": "",
                    "rejected_candidates": " | ".join(rejected),
                })
    if pending_count and not allow_pending:
        raise RuntimeError(
            f"{pending_count} table row(s) are pending/invalid; rerun corrected inputs or pass --allow-pending to emit NA placeholders"
        )
    return table_rows, validation_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", action="append", type=Path, default=[],
        help="result root; repeat to provide legacy and corrected roots (later valid roots win)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-pending", action="store_true",
        help="emit NA placeholders plus validation sidecar instead of failing",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    roots = discover_run_roots(args.run_root)
    table_rows, validation_rows = build_table(roots, args.allow_pending)
    output = args.output.resolve()
    validation_output = output.with_name(f"{output.stem}.validation.csv")
    write_csv_atomic(
        output,
        ("target_formula", "lambda_stoich", "refined_formula_anion_normalized", "final_Rwp_percent"),
        table_rows,
    )
    write_csv_atomic(
        validation_output,
        (
            "target_formula", "lambda_stoich", "validation_status", "reason",
            "selected_run", "summary_sha256", "provenance", "rejected_candidates",
        ),
        validation_rows,
    )
    valid_count = sum(row["validation_status"] == "valid" for row in validation_rows)
    print(f"Wrote {output} ({valid_count}/12 valid rows)")
    print(f"Wrote {validation_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
