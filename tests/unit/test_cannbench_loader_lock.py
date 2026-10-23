"""The official loader may initialize torch_npu even during catalog discovery."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations/cannbench/workers"))
import run_camodel_cases as runner

class LoaderLockTests(unittest.TestCase):
    def check_loader(self, fail=False):
        events = []
        @contextmanager
        def lock(kind):
            self.assertEqual(kind, "execution")
            events.append("acquired")
            try:
                yield
            finally:
                events.append("released")
        def constructor(root):
            self.assertEqual(events, ["acquired"])
            self.assertEqual(root, "/catalog/tasks")
            events.append("probe")
            if fail:
                raise RuntimeError("probe failed")
            return "loader"
        kem = SimpleNamespace(modules={"kernel_eval.data.case_loader": SimpleNamespace(CaseLoader=constructor)})
        with patch.dict(sys.modules, {"vm_locks": SimpleNamespace(shared_locks=lock)}):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "probe failed"):
                    runner.create_case_loader(kem, Path("/catalog/tasks"))
            else:
                self.assertEqual(runner.create_case_loader(kem, Path("/catalog/tasks")), "loader")
        self.assertEqual(events, ["acquired", "probe", "released"])
    def test_probe_serialized_then_lock_released(self):
        self.check_loader()
    def test_probe_exception_releases_lock(self):
        self.check_loader(True)
