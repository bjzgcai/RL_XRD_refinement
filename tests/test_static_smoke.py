# SPDX-License-Identifier: MIT
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StaticSmokeTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for rel in ("yfs_XRD.py", "QL_yfs_XRD.py", "parallel_batch_refine.py"):
            source = (ROOT / rel).read_text(encoding="utf-8")
            ast.parse(source, filename=rel)

    def test_license_is_declared(self):
        self.assertTrue((ROOT / "LICENSE").exists())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("MIT", readme)
        self.assertIn("LICENSE", readme)

    def test_batch_runner_keeps_shell_disabled(self):
        source = (ROOT / "parallel_batch_refine.py").read_text(encoding="utf-8")
        self.assertIn("subprocess.run", source)
        self.assertNotIn("shell=True", source)


if __name__ == "__main__":
    unittest.main()
