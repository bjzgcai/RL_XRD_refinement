import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = load_module("occ_validator", ROOT / "validate_occ_sweep_inputs.py")
builder = load_module("occ_table_builder", ROOT / "build_occ_lambda_sweep_table.py")


class OccupancySweepToolTests(unittest.TestCase):
    def test_corrected_na_physical_mappings_are_locked(self):
        self.assertEqual(
            builder.CASES["Na0.67Ni0.33Mn0.67O2"].physical_dir,
            "Na0.56Ni0.333Mn0.667O1.95",
        )
        self.assertEqual(
            builder.CASES["Na0.58Ni0.33Mn0.67O1.95"].physical_dir,
            "Na0.67Ni0.33Mn0.67O2",
        )

    def test_formula_guard_detects_the_known_na067_mismatch(self):
        target = validator.parse_target("Na:0.67,Ni:0.33,Mn:0.67,O:2")
        parsed = {"Na": 4.02, "Ni": 0.99, "Mn": 2.01, "O": 12.0}
        normalized = validator.normalize_to_anion(parsed, target, "O")
        maximum, _ = validator.maximum_formula_deviation(normalized, target)
        self.assertAlmostEqual(normalized["Na"], 0.67)
        self.assertAlmostEqual(normalized["Ni"], 0.165)
        self.assertAlmostEqual(normalized["Mn"], 0.335)
        self.assertGreater(maximum, 0.02)

    def test_pending_rows_never_reuse_invalid_numbers(self):
        rows, validation = builder.build_table([], allow_pending=True)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(validation), 12)
        self.assertTrue(all(row["final_Rwp_percent"] == "NA" for row in rows))
        self.assertTrue(all(row["selected_run"] == "" for row in validation))

    def test_runner_uses_explicit_cu_wavelength_and_no_overwrite(self):
        source = (ROOT / "run_occ_lambda_sweep_cuda0.sh").read_text()
        self.assertIn('--wl "${WAVELENGTH}"', source)
        self.assertIn('WAVELENGTH="${WAVELENGTH:-1.5406}"', source)
        self.assertIn('refusing to overwrite existing output root', source)
        self.assertIn('scheduling:      sequential', source)


if __name__ == "__main__":
    unittest.main()
