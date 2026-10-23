"""CPU-only regression checks: reference delivery and reduction semantics."""
import ast
import hashlib
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/cannbench/workers"))
sys.path.insert(0, str(ROOT / "integrations/cannbench/submission"))
import prompts
from source_contract import check_high_level_source


class PerformanceGuidanceTests(unittest.TestCase):
    def test_each_tuning_task_gets_exact_reference_and_digest(self):
        # No API client, credentials, worker invocation, or hardware is involved.
        data = prompts.TUNING_REFERENCE.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        for task in sorted(prompts.TASKS_DIR.glob("*/proto.yaml")):
            with self.subTest(operator=task.parent.name), patch.object(
                prompts, "timings_table", return_value=("archived fixture", "0.5", "fixture")
            ):
                text = prompts.build_tuning_prompt(task.parent.name, "callable", "module")
                self.assertIn(data.decode("utf-8"), text)
                self.assertIn(digest, text)
                self.assertNotRegex(text, r"\{\{[A-Z_]+\}\}")
                cases, count = prompts.cases_summary(task.parent.name)
                self.assertGreater(count, 0)
                self.assertIn(cases, text)

    def test_reference_changes_are_delivered_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.md"
            with patch.object(prompts, "TUNING_REFERENCE", reference), patch.object(
                prompts, "timings_table", return_value=("", "", "")
            ):
                for data in (b"first controlled reference", b"second controlled reference"):
                    reference.write_bytes(data)
                    text = prompts.build_tuning_prompt("gelu", "gelu", "gelu")
                    self.assertIn(data.decode(), text)
                    self.assertIn(hashlib.sha256(data).hexdigest(), text)
                reference.unlink()
                with self.assertRaises(FileNotFoundError):
                    prompts.build_tuning_prompt("gelu", "gelu", "gelu")

    def test_public_structure_example_passes_source_gate(self):
        ast.parse(prompts.PUBLIC_API_REFERENCE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.py"
            path.write_text(prompts.PUBLIC_API_REFERENCE)
            self.assertEqual(check_high_level_source(path), [])

    def test_documented_reduction_geometry_preserves_axis(self):
        reference = ROOT / "skills/pyasc-api-patterns/references/reduction-tiling.md"
        blocks = re.findall(r"```python\n(.*?)\n```", reference.read_text(), re.S)
        self.assertEqual(len(blocks), 1)
        namespace = {}
        exec(compile(blocks[0], str(reference), "exec"), namespace)
        geometry = namespace["last_axis_geometry"]
        for shape, expected in [
            ((32, 1), (32, 1)), ((1500, 1, 61), (1500, 61)),
            ((1, 1), (1, 1)), ((7,), (1, 7)), ((0, 7), (0, 7)),
            ((2, 0), (2, 0)), ((2, 1, 3, 1), (6, 1)),
            ((2**32, 2, 9), (2**33, 9)),
        ]:
            with self.subTest(shape=shape):
                self.assertEqual(geometry(shape), expected)
                rows, columns = geometry(shape)
                self.assertEqual(rows * columns, math.prod(shape))
                self.assertEqual(columns, shape[-1])
        for shape in ((), (-1, 2), (True, 3), (1.5, 2)):
            with self.subTest(invalid=shape), self.assertRaises(ValueError):
                geometry(shape)


if __name__ == "__main__":
    unittest.main()
