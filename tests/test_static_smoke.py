# SPDX-License-Identifier: MIT
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StaticSmokeTests(unittest.TestCase):
    def test_python_sources_parse(self):
        paths = [
            ROOT / "yfs_XRD.py",
            ROOT / "QL_yfs_XRD.py",
            ROOT / "parallel_batch_refine.py",
        ]
        paths.extend((ROOT / "src" / "yfs_xrd_refinement").glob("*.py"))
        for path in paths:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path.relative_to(ROOT)))

    def test_license_is_declared(self):
        self.assertTrue((ROOT / "LICENSE").exists())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("MIT", readme)
        self.assertIn("LICENSE", readme)

    def test_pyproject_package_metadata(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "yfs-xrd-refinement"', text)
        self.assertIn('[project.scripts]', text)
        self.assertIn('yfs-xrd-refine', text)

    def test_batch_runner_keeps_shell_disabled(self):
        source = (ROOT / "src" / "yfs_xrd_refinement" / "batch.py").read_text(encoding="utf-8")
        self.assertIn("subprocess.run", source)
        self.assertNotIn("shell=True", source)


    def test_qlearning_runs_three_stages_with_active_updates(self):
        source = (ROOT / "src" / "yfs_xrd_refinement" / "qlearning.py").read_text(encoding="utf-8")
        self.assertIn("\"粗调 (Stage 1)\"", source)
        self.assertIn("\"微调 (Stage 2)\"", source)
        self.assertIn("\"精调 (Stage 3)\"", source)
        self.assertNotIn("stage_settings = stage_settings[:1]", source)
        self.assertIn("agent.choose_action(current_state)", source)
        self.assertIn("agent.learn(current_state, action_selected, reward, next_state)", source)
        self.assertIn("e, l2 = 250, 0.30", source)
        self.assertIn("e, l2 = 200, 0.30", source)
        self.assertIn("e, l2 = 150, 0.25", source)
        self.assertIn("lbfgs_max_iter=50", source)


if __name__ == "__main__":
    unittest.main()
