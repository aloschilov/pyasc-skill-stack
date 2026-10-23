#!/usr/bin/env python3
"""Unit tests for verify_native_runtime.analyze (no runtime imports)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]
        / "integrations" / "cannbench" / "workers"),
)

import verify_native_runtime as vnr  # noqa: E402

PIN = "0a631f70968c3cb7c33ce45330a85768dd5a6f06"


class AnalyzeTest(unittest.TestCase):
    def _analyze(self, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site-packages"
            asc_dir = site / "asc"
            asctile_dir = site / "asctile"
            asc_dir.mkdir(parents=True)
            asctile_dir.mkdir(parents=True)
            asc = asc_dir / "__init__.py"
            asctile = asctile_dir / "__init__.py"
            asc.write_text("")
            asctile.write_text("")
            arguments = dict(
                pin=PIN,
                source_head=PIN,
                site_packages=str(site),
                asc_file=str(asc),
                asctile_file=str(asctile),
                dist_version="1.1.1+g0a631f70",
                editable_files=[],
                fork_leak_files=[],
                asc_extensions={"libasc.so": "aa"},
                asctile_extensions={"libasctile.so": "bb"},
            )
            arguments.update(overrides)
            return vnr.analyze(**arguments)

    def test_fully_qualified(self):
        record, reasons = self._analyze()
        self.assertEqual(reasons, [])
        self.assertTrue(record["qualified"])
        self.assertTrue(record["dist_version_contains_pin"])

    def test_source_head_mismatch_fails(self):
        record, reasons = self._analyze(source_head="deadbeef" * 5)
        self.assertFalse(record["qualified"])
        self.assertTrue(any("source HEAD" in reason for reason in reasons))

    def test_dist_version_without_pin_fails_even_though_record_has_pin(self):
        """Guards against the old tautology (pin[:8] in json.dumps(record))."""
        record, reasons = self._analyze(dist_version="1.1.1")
        self.assertFalse(record["dist_version_contains_pin"])
        self.assertFalse(record["qualified"])
        self.assertTrue(any("does not contain pin" in reason
                            for reason in reasons))

    def test_missing_native_extension_fails_closed(self):
        record, reasons = self._analyze(asc_extensions={},
                                        asctile_extensions={})
        self.assertFalse(record["has_native_extension"])
        self.assertFalse(record["qualified"])
        self.assertTrue(any("no native .so" in reason for reason in reasons))

    def test_import_outside_private_venv_fails(self):
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / "asc" / "__init__.py"
            outside.parent.mkdir(parents=True)
            outside.write_text("")
            record, reasons = self._analyze(asc_file=str(outside))
        self.assertFalse(record["asc_under_private_venv"])
        self.assertFalse(record["qualified"])
        self.assertTrue(any("outside private venv" in reason
                            for reason in reasons))

    def test_legacy_editable_and_fork_leaks_fail(self):
        record, reasons = self._analyze(
            editable_files=["__editable__.pyasc.pth"],
            fork_leak_files=["pyasc.pth"])
        self.assertFalse(record["qualified"])
        self.assertTrue(any("editable" in reason for reason in reasons))
        self.assertTrue(any("source checkout" in reason for reason in reasons))

    def test_asctile_only_extension_is_acceptable_when_asc_has_none(self):
        record, _ = self._analyze(asc_extensions={})
        self.assertTrue(record["has_native_extension"])
        self.assertTrue(record["qualified"])


if __name__ == "__main__":
    unittest.main()
