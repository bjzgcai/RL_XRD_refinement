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


if __name__ == "__main__":
    unittest.main()
