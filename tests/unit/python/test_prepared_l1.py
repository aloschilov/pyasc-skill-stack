"""Verify that a two-operator submission cannot change one task or its visibility."""

from hashlib import sha256
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "integrations/cannbench/ci"))
import prepared_l1 as l1


class PreparedL1Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "prepared"
        self.run.mkdir()
        self.ledger = self.run.parent / "ledger"
        self.ledger.mkdir()
        self.q = Mock(benchmark_slug="official-tasks", target_hardware="950pr")
        self.q._client.list_jobs.return_value = {"jobs": []}
        self.q._submit_streaming.return_value = {
            "job_id": "job_123",
            "submission_id": "sub_abc",
        }

        def task(queue, op, fn):
            return {
                "benchmark_version": "2.1.0",
                "benchmark_slug": "official-tasks",
                "operators": [{"operator": op, "function_name": fn, "case_count": 20,
                               "rel_path": f"level1/{fn}",
                               "case_ids": [f"level1/{fn}_{i}" for i in range(1, 21)]}],
            }, {"contract": fn}

        self.snapshot = patch.object(l1.smoke, "task_snapshot", side_effect=task)
        self.snapshot.start()
        self.addCleanup(self.snapshot.stop)
        catalog, self.tasks = l1.smoke.group_snapshot(self.q, l1.OPERATORS)
        (self.run / "submission.zip").write_bytes(b"opaque, qualified source archive")
        self.save(self.tasks, catalog)

    def save(self, tasks, catalog):
        l1.smoke.save(self.run / "task.json", tasks)
        self.manifest = {
            "catalog": catalog,
            "scope": "Level 1 Exp and SwiGLU, all standard public cases",
            "visibility": "private",
            "target_hardware": "950pr",
            "aggregation_token": "example",
            "files": {
                f: l1.smoke.sha(self.run / f) for f in ("task.json", "submission.zip")
            },
        }
        l1.smoke.save(self.run / "manifest.json", self.manifest)

    def test_upload_size_boundary_and_get_only_recovery(self):
        digest = "a" * 64
        home = self.run.parent
        staged = home / ".local/state/pyasc-cannbench-l1" / digest
        staged.mkdir(parents=True)
        archive = staged / "submission.zip"
        limit = 10 * 1024 * 1024
        with archive.open("wb") as f:
            f.truncate(limit)

        def invoke(action, value=digest):
            with patch.object(l1.sys, "argv", ["prepared_l1", action, value]):
                l1.main()

        with (patch.object(l1.Path, "home", return_value=home),
              patch.object(l1, "ci_client", return_value=self.q) as client):
            invoke("check-size")
            with archive.open("ab") as f:
                f.write(b"x")
            for action in ("check-size", "stage-private", "stage-public", "submit-private",
                           "submit-public", "retry-partial"):
                with self.subTest(action=action), self.assertRaisesRegex(
                    ValueError, "WEB-ZIP-001.*10485761.*10485760"
                ):
                    invoke(action, str(staged) if action.startswith("stage-") else digest)
            client.assert_not_called()
            self.assertFalse((home / ".local/state/pyasc-cannbench-smoke").exists())
            self.assertEqual(list(staged.iterdir()), [archive])

            with (patch.object(l1, "verify_package", return_value=self.manifest) as verify,
                  patch.object(l1.smoke, "collect", return_value=(0, {})) as collect,
                  patch.object(l1.smoke, "submit_once") as submit,
                  patch.object(l1, "result_receipt", return_value={"score": 161, "job": "job_a"})):
                invoke("collect")
            verify.assert_called_once_with(staged, digest)
            collect.assert_called_once()
            submit.assert_not_called()

    def test_changed_second_task_stops_before_intent(self):
        self.save(
            {**self.tasks, "swi_glu": {"contract": "old"}}, self.manifest["catalog"]
        )
        with self.assertRaisesRegex(ValueError, "changed"):
            l1.smoke.submit_once(self.q, self.run, self.ledger, selections=l1.OPERATORS)
        self.q._submit_streaming.assert_not_called()
        self.assertEqual(list(self.ledger.iterdir()), [])

    def test_source_proof_must_match_the_submitted_operator_module(self):
        sources = {"exp": b"retained Exp", "swi_glu": b"retained SwiGLU"}
        with ZipFile(self.run / "submission.zip", "w") as z:
            z.writestr("build.sh", b"build")
            z.writestr("notes/old_exp.py", sources["exp"])
            for fn, content in sources.items():
                z.writestr(f"cann_bench/{fn}.py", content)
        q = {
            "build_passed": True,
            "package_sha256": l1.smoke.sha(self.run / "submission.zip"),
            "wheel_sha256": "3" * 64,
            "operators": {
                fn: {
                    "package_source": f"cann_bench/{fn}.py",
                    "source_sha256": sha256(content).hexdigest(),
                    "passed_cases": 20,
                    "skills_revision": "a" * 40,
                }
                for fn, content in sources.items()
            },
        }

        def pin():
            l1.smoke.save(self.run / "qualification.json", q)
            for name in ("qualification.json", "submission.zip"):
                self.manifest["files"][name] = l1.smoke.sha(self.run / name)
            l1.smoke.save(self.run / "manifest.json", self.manifest)
            return l1.smoke.sha(self.run / "manifest.json")

        self.assertEqual(l1.verify_package(self.run, pin()), self.manifest)
        q["operators"]["exp"]["package_source"] = "notes/old_exp.py"
        with self.assertRaisesRegex(ValueError, "source for Exp"):
            l1.verify_package(self.run, pin())

    def test_camodel_can_qualify_for_first_hardware_measurement(self):
        source_sha = sha256(b"generated source").hexdigest()
        model = {
            "source_sha256": source_sha,
            "evidence_sha256": sha256(b"simulator evidence").hexdigest(),
            "compiled_case_ids": list(range(1, 21)),
            "numerical_pairs": [
                {"seed": s, "launch_matches_golden": True, "suppressed_rejected": True}
                for s in (19, 23)
            ],
        }
        source = {"source_sha256": source_sha, "camodel": model}
        self.assertTrue(l1.qualified_source(source))
        for change in (
            {"source_sha256": "0" * 64}, {"compiled_case_ids": [1] * 20},
            {"numerical_pairs": [model["numerical_pairs"][0]]},
            {"numerical_pairs": [*model["numerical_pairs"], {"seed": 29, "suppressed_rejected": False}]},
        ):
            with self.subTest(change=change):
                self.assertFalse(l1.qualified_source({**source, "camodel": {**model, **change}}))

    def aot_fixture(self, origins=None):
        origins = origins or {fn: ("retained", "fallback") for _, fn in l1.OPERATORS}
        files = {"build.sh": b"build", "build_device.py": b"build device sources",
                 "pyasc_source/fallback.py": b"general fallback"}
        flags, entries, native, operators = {}, {}, {}, {}
        for _, fn in l1.OPERATORS:
            source = f"pyasc_source/{fn}.py"
            files[source] = ("retained generated " + fn).encode()
            files[f"cann_bench/{fn}.py"] = ("public adapter " + fn).encode()
            operators[fn] = {"package_source": source, "source_sha256": sha256(files[source]).hexdigest(),
                             "passed_cases": 20, "skills_revision": "a" * 40}
            for origin in origins[fn]:
                name, symbol = fn + "-" + origin, fn + "_" + origin
                cpp = f"csrc/{name}.cpp"
                files[cpp] = ("AscendC for " + name).encode()
                binary = sha256(("ELF for " + name).encode()).hexdigest()
                flags[name] = {"source": cpp, "commands": [["bisheng", "@SRC@"]], "binary_sha256": binary}
                entries[json.dumps([fn, symbol, [], []])] = {
                    "file": name + ".o", "symbol": symbol, "binary_sha256": binary, "ub_bytes": 65536,
                }
                native[name] = {"operator": fn, "origin": origin,
                                "pyasc_source": source if origin == "retained" else "pyasc_source/fallback.py",
                                "native_sha256": binary, "rebuilt_sha256": binary}
                if origin == "reference":
                    reference = f"pyasc_source/{fn}_reference.py"
                    files[reference] = ("public reference " + fn).encode()
                    native[name].update(pyasc_source=reference,
                                        source_sha256=sha256(files[reference]).hexdigest())
        files["flags.json"] = json.dumps(flags).encode()
        files["cann_bench/specializations.json"] = json.dumps(entries).encode()
        inputs = {name: sha256(content).hexdigest() for name, content in files.items()}
        wheel = {name: value for name, value in inputs.items() if name.startswith("cann_bench/")}
        wheel.update({"cann_bench/device/" + name + ".o": proof["native_sha256"] for name, proof in native.items()})
        wheel.update({f"cann_bench-0.1.0.dist-info/{name}": "7" * 64 for name in ("METADATA", "WHEEL", "RECORD")})
        q = {
            "packaging_mode": "source-aot-v1", "build_passed": True, "wheel_sha256": "3" * 64,
            "operators": operators,
            "aot": {"inputs": inputs, "wheel_files": wheel, "native_binaries": native,
                    "runtime_revision": "e" * 40, "runtime_extension_sha256": "e" * 64,
                    "compiler_sha256": {"bisheng": "c" * 64, "ld.lld": "d" * 64},
                    "build_evidence_sha256": "b" * 64},
        }
        binding = l1.aot_binding_sha256(q)
        q["packaging_binding_sha256"] = binding
        q["host_adapter"] = {
            "binding_sha256": binding, "wheel_sha256": q["wheel_sha256"], "evidence_sha256": "f" * 64,
            "dispatch_matches_retained": True, "caller_stream_preserved": True, "launch_grid_preserved": True,
            "operators": {fn: {
                "source_sha256": inputs[f"cann_bench/{fn}.py"], "evidence_sha256": "f" * 64,
                "compiled_case_ids": list(range(1, 21)),
                "numerical_pairs": [{"seed": seed, "scope": scope,
                                     "launch_matches_golden": True, "suppressed_rejected": True,
                                     **({"key": json.dumps([fn, fn + "_reference", [], []]),
                                         "binary_sha256": native[fn + "-reference"]["native_sha256"]}
                                        if scope == "reference" else {})}
                                    for scope in origins[fn] for seed in (19, 47)],
            } for _, fn in l1.OPERATORS},
        }
        for _, fn in l1.OPERATORS:
            if "fallback" not in origins[fn]:
                q["host_adapter"]["operators"][fn]["missing_specialization_check"] = {
                    "key": json.dumps([fn, fn + "_" + origins[fn][0], [], []]),
                    "rejected": True, "launches": 0, "evidence_sha256": "9" * 64,
                }
        return files, q

    def verify_aot_fixture(self, files, q):
        with ZipFile(self.run / "submission.zip", "w") as z:
            for name, content in files.items():
                z.writestr(name, content)
        q["package_sha256"] = l1.smoke.sha(self.run / "submission.zip")
        l1.smoke.save(self.run / "qualification.json", q)
        for name in ("qualification.json", "submission.zip"):
            self.manifest["files"][name] = l1.smoke.sha(self.run / name)
        l1.smoke.save(self.run / "manifest.json", self.manifest)
        return l1.verify_package(self.run, l1.smoke.sha(self.run / "manifest.json"))

    def test_aot_keeps_generated_source_and_installed_adapter_proofs_separate(self):
        files, q = self.aot_fixture()
        self.assertEqual(self.verify_aot_fixture(files, q), self.manifest)
        changed = deepcopy(q)
        changed["operators"]["exp"]["package_source"] = "cann_bench/exp.py"
        changed["operators"]["exp"]["source_sha256"] = q["aot"]["inputs"]["cann_bench/exp.py"]
        with self.assertRaisesRegex(ValueError, "source for Exp"):
            self.verify_aot_fixture(files, changed)
        for mutate in (
            lambda v: v["operators"]["exp"].update(passed_cases=0),
            lambda v: v["host_adapter"]["operators"]["exp"].update(passed_cases=20, numerical_pairs=[]),
            lambda v: v["host_adapter"]["operators"]["exp"].update(source_sha256=q["operators"]["exp"]["source_sha256"]),
            lambda v: v["host_adapter"]["operators"]["swi_glu"]["numerical_pairs"].pop(),
            lambda v: v["host_adapter"]["operators"]["exp"]["numerical_pairs"][2].update(scope="retained"),
            lambda v: v["host_adapter"].update(launch_grid_preserved=False),
        ):
            changed = deepcopy(q)
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.verify_aot_fixture(files, changed)

    def test_aot_rejects_unbound_build_wheel_and_binary_map(self):
        files, q = self.aot_fixture()
        for mutate in (
            lambda v: v["aot"]["inputs"].update({"build_device.py": "0" * 64}),
            lambda v: v["aot"]["wheel_files"].update({"cann_bench/exp.py": "0" * 64}),
            lambda v: v["aot"]["wheel_files"].update({"asc/libpyasc.so": "0" * 64}),
            lambda v: v["aot"]["native_binaries"].pop("exp-fallback"),
            lambda v: v["aot"]["native_binaries"]["exp-retained"].update(rebuilt_sha256="0" * 64),
            lambda v: v["aot"]["native_binaries"]["swi_glu-fallback"].update(operator="exp"),
            lambda v: v["host_adapter"].update(binding_sha256="0" * 64),
            lambda v: v.update(wheel_sha256="0" * 64),
            lambda v: v.update(packaging_mode="guess"),
        ):
            changed = deepcopy(q)
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.verify_aot_fixture(files, changed)
        # Rehashing a changed build input does not requalify the installed adapter.
        files["build_device.py"] += b" changed"
        q["aot"]["inputs"]["build_device.py"] = sha256(files["build_device.py"]).hexdigest()
        q["packaging_binding_sha256"] = l1.aot_binding_sha256(q)
        with self.assertRaisesRegex(ValueError, "unbound host adapter"):
            self.verify_aot_fixture(files, q)

    def test_aot_rejects_binary_and_unsafe_source_members(self):
        for name, content in (("hidden.cpp", b"\x7fELFbinary"), ("nested.whl", b"wheel"),
                              ("../escape.py", b"source"), ("cann_bench/module.pyc", b"code")):
            files, q = self.aot_fixture()
            files[name] = content
            q["aot"]["inputs"][name] = sha256(content).hexdigest()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "source-aot-v1"):
                self.verify_aot_fixture(files, q)

    def reference_aot_fixture(self):
        return self.aot_fixture({"exp": ("retained", "fallback"),
                                 "swi_glu": ("retained", "reference")})

    def test_aot_qualifies_only_shipped_origin_scopes(self):
        for origins in ({"exp": ("retained", "fallback"), "swi_glu": ("retained",)},
                        {"exp": ("retained", "fallback"), "swi_glu": ("retained", "reference")}):
            files, q = self.aot_fixture(origins)
            with self.subTest(origins=origins):
                self.assertEqual(self.verify_aot_fixture(files, q), self.manifest)
        files, q = self.aot_fixture({"exp": ("retained", "fallback"), "swi_glu": ("reference",)})
        with self.assertRaisesRegex(ValueError, "missing retained kernels"):
            self.verify_aot_fixture(files, q)

    def test_reference_source_requires_separate_exact_attribution(self):
        files, q = self.reference_aot_fixture()
        for mutate in (
            lambda p: p.pop("pyasc_source"),
            lambda p: p.update(pyasc_source="pyasc_source/missing.py"),
            lambda p: p.update(pyasc_source="pyasc_source/swi_glu.py",
                               source_sha256=q["aot"]["inputs"]["pyasc_source/swi_glu.py"]),
            lambda p: p.pop("source_sha256"),
            lambda p: p.update(source_sha256="0" * 64),
            lambda p: p.update(origin="retained"),
            lambda p: p.update(origin="undeclared"),
        ):
            changed = deepcopy(q)
            mutate(changed["aot"]["native_binaries"]["swi_glu-reference"])
            self.rebind_aot(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.verify_aot_fixture(files, changed)
        # Rehashing a changed reference does not preserve the adapter's old binding.
        name = "pyasc_source/swi_glu_reference.py"
        files[name] += b" changed"
        digest = sha256(files[name]).hexdigest()
        q["aot"]["inputs"][name] = digest
        q["aot"]["native_binaries"]["swi_glu-reference"]["source_sha256"] = digest
        q["packaging_binding_sha256"] = l1.aot_binding_sha256(q)
        with self.assertRaisesRegex(ValueError, "unbound host adapter"):
            self.verify_aot_fixture(files, q)

    def test_origin_scopes_require_distinct_valid_seeds_and_no_extra_scope(self):
        files, q = self.reference_aot_fixture()
        for mutate in (
            lambda pairs: pairs.pop(),
            lambda pairs: pairs[-1].update(seed=pairs[-2]["seed"]),
            lambda pairs: pairs[-1].update(seed=True),
            lambda pairs: pairs[-1].update(launch_matches_golden=False),
            lambda pairs: pairs[-1].update(suppressed_rejected=False),
            lambda pairs: pairs[-1].update(scope="retained"),
            lambda pairs: pairs[-1].update(scope="unknown"),
            lambda pairs: pairs.append({**pairs[0], "scope": "fallback"}),
        ):
            changed = deepcopy(q)
            mutate(changed["host_adapter"]["operators"]["swi_glu"]["numerical_pairs"])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.verify_aot_fixture(files, changed)

    def test_reference_pairs_bind_same_operator_reference_key_and_binary(self):
        files, q = self.aot_fixture({"exp": ("retained", "reference", "fallback"),
                                     "swi_glu": ("retained", "reference")})
        for change in (
            {"key": None}, {"key": "missing"},
            {"key": json.dumps(["exp", "exp_reference", [], []]),
             "binary_sha256": q["aot"]["native_binaries"]["exp-reference"]["native_sha256"]},
            {"key": json.dumps(["swi_glu", "swi_glu_retained", [], []]),
             "binary_sha256": q["aot"]["native_binaries"]["swi_glu-retained"]["native_sha256"]},
            {"binary_sha256": "0" * 64},
        ):
            changed = deepcopy(q)
            changed["host_adapter"]["operators"]["swi_glu"]["numerical_pairs"][-1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify_aot_fixture(files, changed)

    def test_no_fallback_requires_installed_missing_key_rejection(self):
        files, q = self.reference_aot_fixture()
        good = q["host_adapter"]["operators"]["swi_glu"]["missing_specialization_check"]
        checks = [None, {}, *({**good, **change} for change in (
            {"rejected": False}, {"rejected": 1}, {"launches": 1}, {"launches": False},
            {"evidence_sha256": None}, {"key": "missing"},
            {"key": json.dumps(["exp", "exp_retained", [], []])},
        ))]
        for check in checks:
            changed = deepcopy(q)
            changed["host_adapter"]["operators"]["swi_glu"]["missing_specialization_check"] = check
            with self.subTest(check=check), self.assertRaises(ValueError):
                self.verify_aot_fixture(files, changed)

    def add_target_dispatch(self, files, q):
        entries = json.loads(files["cann_bench/specializations.json"])
        rows = []
        for _, fn in l1.OPERATORS:
            key = next(k for k, v in entries.items() if v["symbol"] == fn + "_retained")
            rows.extend({"operator": fn, "case_id": case, "key": key,
                         "binary_sha256": entries[key]["binary_sha256"], "grid": 32}
                        for case in range(1, 21))
        q["aot"]["target_dispatch"] = {
            "hardware": "950pr", "required_effective_aivs": [64, 72],
            "evidence_sha256": "d" * 64,
            "profiles": [{"effective_aivs": n, "usable_ub_bytes": 253952,
                          "cases": deepcopy(rows)} for n in (64, 72)],
        }
        self.rebind_aot(q)

    def rebind_aot(self, q):
        q["packaging_binding_sha256"] = l1.aot_binding_sha256(q)
        q["host_adapter"]["binding_sha256"] = q["packaging_binding_sha256"]

    def test_target_dispatch_binds_all_cases_to_declared_capacity_and_binary(self):
        files, q = self.aot_fixture()
        self.add_target_dispatch(files, q)
        self.assertEqual(self.verify_aot_fixture(files, q), self.manifest)
        # Even a well-formed new grid must invalidate the original adapter receipt.
        changed = deepcopy(q)
        changed["aot"]["target_dispatch"]["profiles"][0]["cases"][0]["grid"] = 31
        with self.assertRaisesRegex(ValueError, "unbound host adapter"):
            self.verify_aot_fixture(files, changed)

        entries = json.loads(files["cann_bench/specializations.json"])
        fallback = next(k for k, v in entries.items() if v["symbol"] == "exp_fallback")
        other = next(k for k, v in entries.items() if v["symbol"] == "swi_glu_retained")
        mutations = [
            lambda t: t["profiles"].pop(0),  # the original 72-only blind spot
            lambda t: t.update(required_effective_aivs=[64]),
            lambda t: t.update(required_effective_aivs=[True, 72]),
            lambda t: t.update(required_effective_aivs=[64, 64]),
            lambda t: t.update(hardware="other"),
            lambda t: t.update(evidence_sha256="unknown"),
            lambda t: t["profiles"][0].update(effective_aivs=72),
            lambda t: t["profiles"][0].update(effective_aivs=True),
            lambda t: t["profiles"][0].update(usable_ub_bytes=0),
            lambda t: t["profiles"][0]["cases"].pop(),
            lambda t: t["profiles"][0]["cases"][0].update(case_id=2),
            lambda t: t["profiles"][0]["cases"][0].update(case_id=True),
            lambda t: t["profiles"][0]["cases"][0].update(operator="other"),
            lambda t: t["profiles"][0]["cases"][0].update(key="missing"),
            lambda t: t["profiles"][0]["cases"][0].update(key=other),
            lambda t: t["profiles"][0]["cases"][0].update(
                key=fallback, binary_sha256=entries[fallback]["binary_sha256"]),
            lambda t: t["profiles"][0]["cases"][0].update(binary_sha256="0" * 64),
        ]
        mutations.extend(lambda t, n=n: t["profiles"][0]["cases"][0].update(grid=n)
                         for n in (0, 65, True, 1.5))
        for mutate in mutations:
            changed = deepcopy(q)
            mutate(changed["aot"]["target_dispatch"])
            self.rebind_aot(changed)
            with self.subTest(mutation=mutate), self.assertRaisesRegex(ValueError, "target dispatch"):
                self.verify_aot_fixture(files, changed)

    def test_target_reference_routes_expose_proof_derived_origin_counts(self):
        files, q = self.reference_aot_fixture()
        self.add_target_dispatch(files, q)
        entries = json.loads(files["cann_bench/specializations.json"])
        key = json.dumps(["swi_glu", "swi_glu_reference", [], []])
        for profile in q["aot"]["target_dispatch"]["profiles"]:
            row = next(r for r in profile["cases"] if r["operator"] == "swi_glu" and r["case_id"] == 12)
            row.update(key=key, binary_sha256=entries[key]["binary_sha256"], origin="reference")
        self.rebind_aot(q)
        self.assertEqual(self.verify_aot_fixture(files, q), self.manifest)
        counts = l1.verify_target_dispatch(q["aot"], entries)
        for capacity in (64, 72):
            self.assertEqual(counts[capacity]["swi_glu"], {"retained": 19, "reference": 1, "fallback": 0})
            self.assertEqual(counts[capacity]["exp"], {"retained": 20, "reference": 0, "fallback": 0})
        for origin in (None, "retained", "fallback"):
            changed = deepcopy(q)
            row = next(r for r in changed["aot"]["target_dispatch"]["profiles"][0]["cases"]
                       if r["operator"] == "swi_glu" and r["case_id"] == 12)
            if origin is None:
                row.pop("origin")
            else:
                row["origin"] = origin
            self.rebind_aot(changed)
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "case origin"):
                self.verify_aot_fixture(files, changed)

    def test_target_dispatch_checks_selected_ub_against_each_profile(self):
        files, q = self.aot_fixture()
        self.add_target_dispatch(files, q)
        entries = json.loads(files["cann_bench/specializations.json"])
        # Exact capacity is valid; a lower declaration must not certify this binding.
        for profile in q["aot"]["target_dispatch"]["profiles"]:
            profile["usable_ub_bytes"] = 65536
        self.rebind_aot(q)
        self.assertEqual(self.verify_aot_fixture(files, q), self.manifest)
        changed = deepcopy(q)
        changed["aot"]["target_dispatch"]["profiles"][1]["usable_ub_bytes"] = 65535
        self.rebind_aot(changed)
        with self.assertRaisesRegex(ValueError, "valid UB allocation"):
            self.verify_aot_fixture(files, changed)
        for value in (None, 0, -1, True, 65536.0, "65536", 65537):
            for origin in ("retained", "reference"):
                with self.subTest(value=value, origin=origin):
                    modified = deepcopy(entries)
                    key = q["aot"]["target_dispatch"]["profiles"][0]["cases"][0]["key"]
                    if value is None:
                        modified[key].pop("ub_bytes")
                    else:
                        modified[key]["ub_bytes"] = value
                    aot = deepcopy(q["aot"])
                    name = modified[key]["file"].removesuffix(".o")
                    aot["native_binaries"][name]["origin"] = origin
                    for profile in aot["target_dispatch"]["profiles"]:
                        for row in profile["cases"]:
                            if row["key"] == key:
                                row["origin"] = origin
                    with self.assertRaisesRegex(ValueError, "valid UB allocation"):
                        l1.verify_target_dispatch(aot, modified)
        # Historical collection without a target declaration does not acquire new requirements.
        self.assertEqual(l1.verify_target_dispatch({}, {}), {})

    def test_legacy_aot_collects_but_cannot_start_a_new_submission(self):
        files, q = self.aot_fixture()
        self.verify_aot_fixture(files, q)
        digest = l1.smoke.sha(self.run / "manifest.json")
        home = self.run.parent
        staged = home / ".local/state/pyasc-cannbench-l1" / digest
        staged.parent.mkdir(parents=True)
        self.run.rename(staged)
        self.run = staged

        def invoke(action, value=digest):
            with patch.object(l1.sys, "argv", ["prepared_l1", action, value]):
                l1.main()

        with (patch.object(l1.Path, "home", return_value=home),
              patch.object(l1, "ci_client", return_value=self.q) as client,
              patch.object(l1.smoke, "submit_once") as submit):
            for action in ("submit-private", "submit-public", "retry-partial",
                           "stage-private", "stage-public"):
                with self.subTest(action=action), self.assertRaisesRegex(
                    ValueError, "target-capacity dispatch evidence"
                ):
                    invoke(action, str(staged) if action.startswith("stage-") else digest)
            client.assert_not_called()
            submit.assert_not_called()
            self.assertFalse((home / ".local/state/pyasc-cannbench-smoke").exists())
            with (patch.object(l1.smoke, "collect", return_value=(0, {})) as collect,
                  patch.object(l1, "result_receipt", return_value={"score": 150, "job": "job_a"})):
                invoke("collect")
            collect.assert_called_once()
            submit.assert_not_called()

    def full_build_fixture(self):
        files, q = self.aot_fixture()
        self.add_target_dispatch(files, q)
        self.verify_aot_fixture(files, q)
        q["source_build_validation"] = {
            "schema_version": 1, "scope": "full-source-cold-build", "status": "passed",
            "package_sha256": q["package_sha256"], "build_command": ["bash", "build.sh"],
            "exit_code": 0, "elapsed_seconds": 12.5,
            "budget": {"seconds": 30, "basis": "explicitly-declared-local",
                       "reference": {"reason": "bounded offline fixture"}},
            "compiled_keys": sorted(json.loads(files["flags.json"])),
            "inherited_keys": [], "initial_native_outputs": [],
            "compiler_sha256": deepcopy(q["aot"]["compiler_sha256"]),
            "environment": {"image_digest": "example/image@sha256:" + "1" * 64,
                            "cpu_limit": 2, "memory_limit_bytes": 8 * 1024**3,
                            "compiler_workers": 2, "network": "none", "fresh_container": True},
            "output_wheel_sha256": "8" * 64,
            "output_wheel_files_sha256": sha256(json.dumps(
                q["aot"]["wheel_files"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()).hexdigest(),
            "evidence_sha256": "9" * 64,
        }
        l1.smoke.save(self.run / "qualification.json", q)
        return q

    def test_full_build_receipt_keeps_numerical_binding_and_accepts_equal_wheel_members(self):
        q = self.full_build_fixture()
        binding = l1.aot_binding_sha256(q)
        self.assertEqual(binding, q["host_adapter"]["binding_sha256"])
        # Different wheel ZIP metadata is allowed; installed member bytes must match.
        self.assertNotEqual(q["wheel_sha256"], q["source_build_validation"]["output_wheel_sha256"])
        for budget in (
            q["source_build_validation"]["budget"],
            {"seconds": 40, "basis": "observed-service-runner",
             "reference": {"job_id": "job_example", "compile_log_sha256": "a" * 64, "line": 4}},
            {"seconds": 50, "basis": "verified-service-contract",
             "reference": {"source": "retained contract version", "sha256": "b" * 64}},
        ):
            with self.subTest(budget=budget):
                q["source_build_validation"]["budget"] = budget
                l1.smoke.save(self.run / "qualification.json", q)
                l1.require_source_aot_full_build(self.run)
                self.assertEqual(l1.aot_binding_sha256(q), binding)

    def test_full_build_rejects_partial_mismatched_and_unattributed_receipts(self):
        q = self.full_build_fixture()
        mutations = [
            lambda p: p.update(schema_version=True),
            lambda p: p.update(scope="incremental-rebuild"),
            lambda p: p.update(status="failed"),
            lambda p: p.update(package_sha256="0" * 64),
            lambda p: p.update(build_command=["python3", "partial-build.py"]),
            lambda p: p.update(exit_code=False),
            lambda p: p.update(exit_code=1),
            lambda p: p["compiled_keys"].pop(),
            lambda p: p["compiled_keys"].append(p["compiled_keys"][0]),
            lambda p: p["compiled_keys"].append("unrelated-key"),
            lambda p: p["compiled_keys"].append({"key": "unhashable"}),
            lambda p: p.update(inherited_keys=["exp-retained"]),
            lambda p: p.update(initial_native_outputs=["old.whl"]),
            lambda p: p["compiler_sha256"].update(bisheng="0" * 64),
            lambda p: p["environment"].update(image_digest="mutable:latest"),
            lambda p: p["environment"].update(cpu_limit=True),
            lambda p: p["environment"].update(memory_limit_bytes=True),
            lambda p: p["environment"].update(compiler_workers=0),
            lambda p: p["environment"].update(fresh_container=False),
            lambda p: p["environment"].update(network="host"),
            lambda p: p.update(output_wheel_sha256="unbound"),
            lambda p: p.update(output_wheel_files_sha256="0" * 64),
            lambda p: p.update(evidence_sha256="unbound"),
            lambda p: p["budget"].update(reference={}),
            lambda p: p["budget"].update(basis="assumed-universal-service-limit"),
            lambda p: p.update(budget={"seconds": 30, "basis": "observed-service-runner",
                                      "reference": {"job_id": "job_example"}}),
            lambda p: p.update(elapsed_seconds=30.1),
        ]
        for value in (True, 0, -1, float("nan"), float("inf"), "12"):
            mutations.extend((lambda p, v=value: p.update(elapsed_seconds=v),
                              lambda p, v=value: p["budget"].update(seconds=v)))
        for index, mutate in enumerate(mutations):
            changed = deepcopy(q)
            mutate(changed["source_build_validation"])
            l1.smoke.save(self.run / "qualification.json", changed)
            with self.subTest(mutation=index), self.assertRaisesRegex(ValueError, "full cold build"):
                l1.require_source_aot_full_build(self.run)

    def test_full_build_can_stage_without_changing_its_qualification(self):
        q = self.full_build_fixture()
        base = self.run.parent / "staged"
        with (patch.object(l1, "ci_client", return_value=self.q),
              patch.dict(l1.os.environ, {"CANNBENCH_AGGREGATION_TOKEN": "example"}),
              patch("sys.stdout", new_callable=io.StringIO) as output):
            l1.stage(self.run, "private", base)
        digest = output.getvalue().strip()
        self.assertEqual(len(digest), 64)
        self.assertEqual(json.loads((base / digest / "qualification.json").read_bytes()), q)
        l1.require_source_aot_full_build(base / digest)

    def test_missing_full_build_blocks_new_actions_but_not_historical_collect(self):
        q = self.full_build_fixture()
        del q["source_build_validation"]
        l1.smoke.save(self.run / "qualification.json", q)
        self.manifest["files"]["qualification.json"] = l1.smoke.sha(self.run / "qualification.json")
        l1.smoke.save(self.run / "manifest.json", self.manifest)
        digest = l1.smoke.sha(self.run / "manifest.json")
        home = self.run.parent
        staged = home / ".local/state/pyasc-cannbench-l1" / digest
        staged.parent.mkdir(parents=True)
        self.run.rename(staged)

        def invoke(action):
            value = str(staged) if action.startswith("stage-") else digest
            with patch.object(l1.sys, "argv", ["prepared_l1", action, value]):
                l1.main()

        with (patch.object(l1.Path, "home", return_value=home),
              patch.object(l1, "ci_client", return_value=self.q) as client,
              patch.object(l1.smoke, "submit_once") as submit):
            for action in ("stage-private", "stage-public", "submit-private", "submit-public", "retry-partial"):
                with self.subTest(action=action), self.assertRaisesRegex(ValueError, "full cold build"):
                    invoke(action)
            client.assert_not_called()
            submit.assert_not_called()
            self.assertFalse((home / ".local/state/pyasc-cannbench-smoke").exists())
            with (patch.object(l1.smoke, "collect", return_value=(0, {})) as collect,
                  patch.object(l1, "result_receipt", return_value={"score": 150, "job": "job_example"})):
                invoke("collect")
            collect.assert_called_once()
            submit.assert_not_called()
            self.assertEqual(l1.smoke.sha(staged / "manifest.json"), digest)

    def test_source_jit_does_not_acquire_aot_build_requirements(self):
        l1.smoke.save(self.run / "qualification.json", {"packaging_mode": "source-jit"})
        l1.require_source_aot_full_build(self.run)

    def test_private_qualification_never_becomes_public_by_dispatch(self):
        with self.assertRaisesRegex(ValueError, "visibility"):
            l1.smoke.submit_once(
                self.q, self.run, self.ledger, selections=l1.OPERATORS, public=True
            )
        self.q._submit_streaming.assert_not_called()

    def test_replaying_retains_the_two_operator_job(self):
        for _ in range(2):
            state = l1.smoke.submit_once(
                self.q, self.run, self.ledger, selections=l1.OPERATORS
            )
            self.assertEqual(state["job_id"], "job_123")
        self.q._check_credit.assert_called_once_with(2)
        self.q._submit_streaming.assert_called_once()
        args, kwargs = self.q._submit_streaming.call_args
        self.assertEqual(args[1], ["exp", "swi_glu"])
        self.assertEqual(kwargs["benchmark_version"], "2.1.0")
        self.assertIs(kwargs["public"], False)

    def add_active_job(self, number):
        digest = sha256(str(number).encode()).hexdigest()
        job = {"id": f"job_{number}", "submission_id": f"sub_{number}",
               "job_tag": "pyasc-smoke-" + digest[:24], "status": "queued"}
        state = {"phase": "submitted", "archive_sha256": digest, "job_id": job["id"],
                 "submission_id": job["submission_id"], "job_tag": job["job_tag"]}
        l1.smoke.save(self.ledger / (digest + ".json"), state)
        return job

    def test_accepts_different_package_alongside_verified_job(self):
        current = self.add_active_job(1)
        self.q._client.get_job.return_value = {"job": current}
        self.q._client.list_jobs.return_value = {"jobs": [{"id": "job_1", "status": "queued"}]}
        with self.assertRaisesRegex(ValueError, "Reconcile"):
            l1.smoke.submit_once(self.q, self.run, self.ledger, selections=l1.OPERATORS)
        for _ in range(2):
            state = l1.smoke.submit_once(self.q, self.run, self.ledger,
                selections=l1.OPERATORS, allow_confirmed_jobs=True)
            self.assertEqual(state["other_submitted_jobs"], ["job_1"])
        self.q._submit_streaming.assert_called_once()
        self.q._client.get_job.assert_called_once_with("job_1")

    def test_unknown_delivery_untracked_job_and_capacity_stop_new_upload(self):
        def submit():
            l1.smoke.submit_once(self.q, self.run, self.ledger,
                selections=l1.OPERATORS, allow_confirmed_jobs=True)
        untracked = {"id": "job_external", "status": "queued"}
        self.q._client.list_jobs.return_value = {"jobs": [untracked]}
        with self.assertRaisesRegex(ValueError, "active or unknown"):
            submit()
        first = self.add_active_job(1)
        originals = {first["id"]: first}
        self.q._client.get_job.side_effect = lambda ident: {"job": originals[ident]}
        self.q._client.list_jobs.return_value = {"jobs": list(originals.values())}
        for field in ("id", "submission_id", "job_tag"):
            with self.subTest(field=field):
                originals["job_1"] = {**first, field: "different"}
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    submit()
        originals["job_1"] = first
        path = self.ledger / "unknown.json"
        l1.smoke.save(path, {"phase": "ambiguous"})
        with self.assertRaisesRegex(ValueError, "Reconcile"):
            submit()
        path.unlink()
        for i in (2, 3):
            job = self.add_active_job(i)
            originals[job["id"]] = job
        with self.assertRaisesRegex(ValueError, "three active jobs"):
            submit()
        self.q._submit_streaming.assert_not_called()

    def test_overlapping_upload_returns_accepted_id_without_holding_runner(self):
        manifest = "f" * 64
        staged = self.run.parent / ".local/state/pyasc-cannbench-l1" / manifest
        staged.mkdir(parents=True)
        (staged / "submission.zip").write_bytes(b"already verified")
        l1.smoke.save(staged / "qualification.json", {"packaging_mode": "source-jit"})
        state = {"phase": "submitted", "job_id": "job_a", "other_submitted_jobs": ["job_b"]}
        summary = self.run / "summary.md"
        self.q._client._get_json.return_value = {"maintenance": {"enabled": False}}
        with (
            patch.dict(l1.os.environ, GITHUB_STEP_SUMMARY=str(summary)),
            patch.object(l1.sys, "argv", ["prepared_l1", "submit-public", manifest]),
            patch.object(l1.Path, "home", return_value=self.run.parent),
            patch.object(l1, "ci_client", return_value=self.q),
            patch.object(l1, "verify_package", return_value={"visibility": "public"}),
            patch.object(l1.smoke, "submit_once", return_value=state),
            patch.object(l1.smoke, "collect") as collect,
        ):
            l1.main()
        collect.assert_not_called()
        for expected in ("job_a", "pending", "l1-collect", manifest):
            self.assertIn(expected, summary.read_text())

    def test_retry_partial_preserves_original_attempt_and_reconciles_tag_first(self):
        from evalqueue import RemoteError
        size = (self.run / "submission.zip").stat().st_size
        def submit(retry_partial=False):
            return l1.smoke.submit_once(self.q, self.run, self.ledger,
                selections=l1.OPERATORS, allow_confirmed_jobs=True, retry_partial=retry_partial)
        self.q._client.token = "test-token"
        self.q._submit_streaming.side_effect = RemoteError(
            "CANNBench upload failed (HTTP 000, curl rc=28, 8 multipart bytes in 3600s): timeout")
        with self.assertRaises(RemoteError):
            submit()
        state = submit()
        self.assertEqual(state["phase"], "ambiguous")
        self.q._client.list_jobs.return_value = {"jobs": [{"job_tag": state["job_tag"]}]}
        self.assertEqual(submit(True), state)
        self.q._submit_streaming.assert_called_once()
        self.q._client.list_jobs.return_value = {"jobs": []}
        error = self.run / "upload-error.json"
        original = json.loads(error.read_bytes())
        l1.smoke.save(error, dict(original, detail=original["detail"].replace("8 multipart", f"{size} multipart")))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            submit(True)
        l1.smoke.save(error, dict(original, intent_created_at="previous transfer"))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            submit(True)
        l1.smoke.save(error, original)
        self.q._submit_streaming.side_effect = None
        accepted = submit(True)
        self.assertEqual(accepted["phase"], "submitted")
        prior, = accepted["previous_intents"]
        self.assertEqual(prior["partial_upload"]["error"], original)
        self.assertEqual(prior["created_at"], state["created_at"])
        self.assertEqual(submit(True), accepted)
        self.assertEqual(self.q._submit_streaming.call_count, 2)

    def test_legacy_incomplete_upload_requires_bound_final_read_counters(self):
        run, ledger = self.run, self.ledger
        record = ledger / "original.json"
        state = {"phase": "ambiguous", "created_at": "2026-11-17T13:27:00+00:00"}
        l1.smoke.save(record, state)
        error = run / "upload-error.json"
        l1.smoke.save(error, {"type": "RemoteError", "at": "2026-11-17T14:27:00+00:00",
                              "detail": "CANNBench upload failed (HTTP 000, curl rc=28): timeout"})
        size = (run / "submission.zip").stat().st_size
        start = dict(event="before", at="2026-11-17T14:20:00+00:00", parent_rchar=1000,
                     expected_zip_bytes=size, zip_offset=2, child_rchar=5)
        trace = run / "transfer-trace.jsonl"
        def bind(end):
            tail = dict(event="parent_after_wait", at="2026-11-17T14:26:59+00:00", parent_rchar=1000+end,
                        upper_bound_child_read_bytes=end, expected_zip_bytes=size)
            trace.write_text(json.dumps(start)+"\n"+json.dumps(tail)+"\n")
            l1.smoke.save(run/"partial-upload.json", dict(intent_sha256=l1.smoke.sha(record),
                error_sha256=l1.smoke.sha(error), trace_sha256=l1.smoke.sha(trace)))
        bind(size)
        with self.assertRaisesRegex(ValueError, "incomplete ZIP"):
            l1.smoke.verify_partial_upload(run, record, state)
        bind(9)
        self.assertEqual(l1.smoke.verify_partial_upload(run, record, state)["trace_sha256"], l1.smoke.sha(trace))
        l1.smoke.save(record, {"phase": "another attempt"})
        with self.assertRaisesRegex(ValueError, "original attempt"):
            l1.smoke.verify_partial_upload(run, record, state)

    def test_public_receipt_requires_public_service_visibility(self):
        self.manifest["visibility"] = "public"
        job = {
            "selected_operators": ["exp", "swi_glu"],
            "case_set": "standard",
            "is_private": True,
            "is_independent_environment": False,
            "result_score": 166,
        }
        l1.smoke.save(self.run / "job.json", {"job": job})
        with self.assertRaisesRegex(ValueError, "visibility"):
            l1.result_receipt(self.run, self.manifest, {}, self.q)

    def completed_job(self):
        job = {
            "id": "job_abc", "submission_id": "sub_def", "case_set": "standard",
            "selected_operators": ["exp", "swi_glu"], "target_hardware": "950pr",
            "runner_hardware": "950pr",
            "benchmark_slug": "official-tasks", "benchmark_version": "2.1.0", "is_private": False,
            "is_independent_environment": False, "result_score": 161.5, "status": "succeeded",
            "queued_at": "2026-11-17", "claimed_at": "2026-11-17", "finished_at": "2026-11-17",
            "submission_download_url": "/api/submissions/sub_def/download",
            "results": {"result_stage": "final", "contains_performance": True,
                        "anti_cheat_failed_cases": 0,
                        "summary": {"total_cases": 40, "passed_cases": 40},
                        "operators": []},
        }
        for spec in self.manifest["catalog"]["operators"]:
            job["results"]["operators"].append({
                "operator": spec["operator"], "rel_path": spec["rel_path"],
                "total_cases": 20, "passed_cases": 20, "anti_cheat_failed_cases": 0,
                "cases": [{"case_id": cid, "status": "success", "elapsed_us": 4.5,
                           "baseline_perf_us": 3.0, "accuracy": {"passed": True},
                           "op_times": {"device_kernels": {"compute": 4.5}}}
                          for cid in spec["case_ids"]],
            })
        self.manifest["visibility"] = "public"
        self.manifest["files"]["qualification.json"] = "b" * 64
        self.q._client._url.return_value = "https://cannbench.example/api/jobs/job_abc"
        return job

    def retained_report(self, job):
        l1.smoke.save(self.run / "job.json", {"job": job})
        report = l1.smoke.evaluate(self.manifest["catalog"], [{"job": job}], "950pr")
        report.update(scope=self.manifest["scope"],
                      archive_sha256=self.manifest["files"]["submission.zip"],
                      job_id=job["id"], submission_id=job["submission_id"],
                      **{key: job.get(key) for key in
                         ("queued_at", "claimed_at", "started_at", "finished_at")})
        return report

    def assert_receipt_refused(self, report, message):
        with (patch.object(l1, "download") as download,
              patch.object(l1.urllib.request, "urlopen") as anonymous,
              self.assertRaisesRegex(ValueError, message)):
            l1.result_receipt(self.run, self.manifest, report, self.q)
        download.assert_not_called()
        anonymous.assert_not_called()

    def test_receipt_rejects_missing_or_false_accuracy_in_either_operator(self):
        for op in range(2):
            for accuracy in (None, {}, {"passed": False}, {"passed": 1}, {"passed": "true"}, True):
                with self.subTest(operator=op, accuracy=accuracy):
                    job = self.completed_job()
                    case = job["results"]["operators"][op]["cases"][-1]
                    if accuracy is None:
                        del case["accuracy"]
                    else:
                        case["accuracy"] = accuracy
                    report = self.retained_report(job)
                    self.assertTrue(report["gate_passed"])  # The shared legacy gate is unchanged.
                    self.assert_receipt_refused(report, "explicit numerical pass")

    def test_receipt_rejects_absent_or_malformed_device_evidence_in_either_operator(self):
        malformed = [None, {}, [], "compute", {"": 1.0}, {" ": 1.0}, {"compute": True},
                     {"compute": "4.5"}, {"compute": {}}, {"compute": 0}, {"compute": -1},
                     {"compute": float("nan")}, {"compute": float("inf")},
                     {"compute": 4.5, "copy": None}]
        for op in range(2):
            for kernels in malformed:
                with self.subTest(operator=op, kernels=kernels):
                    job = self.completed_job()
                    job["results"]["operators"][op]["cases"][-1]["op_times"] = {"device_kernels": kernels}
                    # Service JSON can be malformed; retain the bytes so the receipt must reject them.
                    report = self.retained_report(job)
                    self.assert_receipt_refused(report, "device-kernel measurements")
            for timings in (None, {}, "compute", ["compute"]):
                with self.subTest(operator=op, op_times=timings):
                    job = self.completed_job()
                    case = job["results"]["operators"][op]["cases"][-1]
                    if timings is None:
                        del case["op_times"]
                    else:
                        case["op_times"] = timings
                    self.assert_receipt_refused(self.retained_report(job), "device-kernel measurements")

    def test_receipt_rejects_explicit_profile_failures_before_any_download(self):
        case_flags = ("perf_integrity_failed", "perf_collection_failed", "io_cleanup_failed", "profile_exception_type")
        scopes = [("operator", key) for key in
                  ("zeroed_by_integrity", "zeroed_by_no_npu_perf", "zeroed_by_cpu_fallback", "profile_exception_type")]
        scopes += [(scope, key) for scope in ("case", "_perf_result", "metadata", "nested metadata") for key in case_flags]
        for op_index in range(2):
            for scope, key in scopes:
                for value in (True, 0, None, "false", []):
                    if key == "profile_exception_type" and value is None:
                        continue  # An absent exception is explicitly valid.
                    with self.subTest(operator=op_index, scope=scope, flag=key, value=value):
                        job = self.completed_job()
                        op = job["results"]["operators"][op_index]
                        case = op["cases"][-1]
                        if scope == "operator": target = op
                        elif scope == "case": target = case
                        elif scope == "nested metadata":
                            case["_perf_result"] = {"metadata": {}}
                            target = case["_perf_result"]["metadata"]
                        else:
                            case[scope] = {}; target = case[scope]
                        target[key] = value
                        report = self.retained_report(job)
                        self.assertTrue(report["gate_passed"])  # Legacy gate alone misses these flags.
                        self.assert_receipt_refused(report, "profiling or integrity failure")

    def test_receipt_rejects_malformed_performance_containers(self):
        for key in ("_perf_result", "metadata"):
            for value in (False, 0, "unknown", [], {"metadata": "unknown"}):
                with self.subTest(key=key, value=value):
                    job = self.completed_job()
                    job["results"]["operators"][1]["cases"][-1][key] = value
                    self.assert_receipt_refused(self.retained_report(job), "profiling or integrity failure")

    def test_receipt_rechecks_failed_incomplete_and_mismatched_saved_job(self):
        for defect in ("failed", "missing second operator", "missing second-operator case",
                       "failed second-operator case", "duplicate case", "wrong case",
                       "wrong catalog", "wrong hardware", "anti-cheat"):
            with self.subTest(defect=defect):
                job = self.completed_job()
                report = self.retained_report(job)
                results = job["results"]
                cases = results["operators"][1]["cases"]
                if defect == "failed": job["status"] = "correctness_failed"
                elif defect == "missing second operator": results["operators"].pop()
                elif defect == "missing second-operator case": cases.pop()
                elif defect == "failed second-operator case":
                    cases[-1]["status"] = "accuracy_failed"
                    results["operators"][1]["passed_cases"] = 19
                    results["summary"]["passed_cases"] = 39
                elif defect == "duplicate case": cases[-1]["case_id"] = cases[0]["case_id"]
                elif defect == "wrong case": cases[-1]["case_id"] = "level1/swi_glu_21"
                elif defect == "wrong catalog": job["benchmark_version"] = "2.0.0"
                elif defect == "wrong hardware": job["runner_hardware"] = "910b"
                elif defect == "anti-cheat": results["anti_cheat_failed_cases"] = 1
                l1.smoke.save(self.run / "job.json", {"job": job})
                self.assert_receipt_refused(report, "exact 40-case pair")

    def test_receipt_refuses_failed_incomplete_or_stale_input_report(self):
        job = self.completed_job()
        valid = self.retained_report(job)
        for field, value in (("gate_passed", False), ("correct_cases", 39), ("operators", []),
                             ("errors", ["failed"]), ("job_id", "job_other"),
                             ("archive_sha256", "f" * 64), ("scope", "other cohort"),
                             ("benchmark_version", "2.0.0")):
            with self.subTest(field=field):
                self.assert_receipt_refused({**valid, field: value}, "report differs")
        changed = deepcopy(valid)
        changed["operators"][1]["cases"][-1]["elapsed_us"] = 1.0
        for report in ({}, None, {k: v for k, v in valid.items() if k != "operators"}, changed):
            self.assert_receipt_refused(report, "report differs")

    def test_receipt_rejects_changed_job_scope(self):
        for field, value in (("selected_operators", ["exp"]),
                             ("selected_operators", ["exp", "swi_glu", "exp"]),
                             ("case_set", "custom"), ("is_independent_environment", True)):
            with self.subTest(field=field, value=value):
                job = self.completed_job()
                report = self.retained_report(job)
                job[field] = value
                l1.smoke.save(self.run / "job.json", {"job": job})
                self.assert_receipt_refused(report, "scope or visibility")

    def test_published_score_matches_downloaded_package_and_anonymous_job(self):
        job = self.completed_job()
        for op in job["results"]["operators"]:
            op.update(zeroed_by_integrity=False, zeroed_by_no_npu_perf=False, zeroed_by_cpu_fallback=False)
            for case in op["cases"]:
                case.update(perf_integrity_failed=False, perf_collection_failed=False, io_cleanup_failed=False,
                            profile_exception_type=None,
                            _perf_result={"elapsed_us": 4.5, "metadata": {"samples": 100,
                                          "perf_collection_failed": False, "profile_exception_type": ""}})
        # A valid multi-kernel path must remain acceptable; the receipt does not
        # assume a particular compiler symbol or a single launch per case.
        job["results"]["operators"][1]["cases"][-1]["op_times"]["device_kernels"] = {
            "compute_specialization": 3.5, "copy_output": 1.0,
        }
        report = self.retained_report(job)
        with patch.object(l1, "download", return_value=b"different source"), \
                self.assertRaisesRegex(ValueError, "prepared ZIP"):
            l1.result_receipt(self.run, self.manifest, report, self.q)
        qualified = (self.run / "submission.zip").read_bytes()
        for field, value in [("result_score", 160), ("is_private", True), ("submission_id", "sub_abc"),
                             ("selected_operators", ["exp"])]:
            public = io.BytesIO(json.dumps({"job": {**job, field: value}}).encode())
            with self.subTest(field=field), patch.object(l1, "download", return_value=qualified), \
                    patch.object(l1.urllib.request, "urlopen", return_value=public), \
                    self.assertRaisesRegex(ValueError, f"differs in {field}"):
                l1.result_receipt(self.run, self.manifest, report, self.q)
        with patch.object(l1, "download", return_value=qualified), \
                patch.object(l1.urllib.request, "urlopen",
                             return_value=io.BytesIO(json.dumps({"job": job}).encode())) as anonymous:
            receipt = l1.result_receipt(self.run, self.manifest, report, self.q)
        anonymous.assert_called_once_with(self.q._client._url.return_value, timeout=30)
        self.assertEqual(receipt["score"], job["result_score"])
        self.assertEqual(receipt["job_sha256"], l1.smoke.sha(self.run / "job.json"))
        self.assertEqual(receipt["operators"], report["operators"])
        self.assertEqual(receipt["package_sha256"], sha256(qualified).hexdigest())
