import ast
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/cannbench/workers"))
driver = importlib.import_module("driver")
contracts = importlib.import_module("contracts")
prompts = importlib.import_module("prompts")
CATALOG = ROOT / "integrations/cannbench/catalogs/official-tasks-1.1.2-20261001"


class DriverTests(unittest.TestCase):
    def test_all_1060_case_arguments_bind_to_official_public_contract(self):
        manifest = json.loads((CATALOG / "manifest.json").read_text())
        count = 0
        for op in manifest["operators"]:
            task = CATALOG / "tasks" / op["function_name"]
            arguments = copy.deepcopy(contracts.public_signature(task, op["function_name"]))
            for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
                arg.annotation = None
            definition = ast.FunctionDef(name="fn", args=arguments, body=[ast.Pass()], decorator_list=[], returns=None, type_params=[])
            tree = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
            namespace = {}
            exec(compile(tree, "<contract>", "exec"), namespace)
            proto = contracts.load_contract(task)
            cases = json.loads((task / "cases.yaml").read_text())["cases"]
            for case in cases:
                with self.subTest(operator=op["operator"], case=case["case_id"]):
                    kwargs = contracts.case_arguments(proto, case, lambda shape, dtype: (tuple(shape), dtype))
                    inspect.signature(namespace["fn"]).bind(**kwargs)
                    count += 1
        self.assertEqual(count, 1060)

    def test_prompt_includes_exact_full_contract_without_old_module(self):
        task = CATALOG / "tasks/lstm"
        text = prompts.build_generation_prompt("lstm", "lstm", "lstm", "-", task)
        for name in ("proto.yaml", "golden.py", "cases.yaml", "desc.md"):
            self.assertIn((task / name).read_text(), text)
        self.assertNotIn("asc2.jit", text)

    def test_worker_preserves_provider_while_pinning_local_skills(self):
        configuration = {"provider": {"gateway": {"options": {"apiKey": "test-key"}, "models": {"glm-5.3": {}}}}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENCODE_CONFIG_CONTENT": json.dumps(configuration)}), patch.object(driver, "REPO_ROOT", Path(directory)):
            path = Path(directory) / "integrations/cannbench/workers/.scratch/run/exp-generate"
            path.mkdir(parents=True)
            with patch.object(driver.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
                driver.run_worker(path, "implement", path / "log", "gateway/glm-5.3")
            passed = json.loads(run.call_args.kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
            self.assertEqual(passed["provider"], configuration["provider"])
            self.assertEqual(passed["skills"]["paths"], [str(driver.SKILLS_ROOT)])
            edits = passed["permission"]["edit"]
            import fnmatch
            relative = ".ci-campaigns/scoped/integrations/cannbench/workers/.scratch/run/exp-generate/design.md"
            self.assertTrue(any(action == "allow" and fnmatch.fnmatch(relative, pattern) for pattern, action in edits.items()))
            self.assertFalse(any(action == "allow" and fnmatch.fnmatch("skills/pyasc-cannbench-kernel/SKILL.md", pattern) for pattern, action in edits.items()))
            self.assertNotIn("test-key", (path / "log").read_text())

    def test_asctile_and_contract_signature_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            candidate.write_text('import asctile\nfrom ._pyasc_runtime import ensure_npu_platform\n@asctile.jit\ndef kernel(): pass\ndef exp(x, base=-1.0, scale=1.0, shift=0.0): return x\n')
            self.assertEqual(driver.static_check(candidate, "exp", CATALOG / "tasks/exp"), [])
            candidate.write_text(candidate.read_text().replace("base=-1.0", "base=2.0"))
            self.assertIn("Public callable defaults differ from official golden", driver.static_check(candidate, "exp", CATALOG / "tasks/exp"))

    def test_required_skills_are_reachable(self):
        loaded = set().union(*(set(driver.PHASE_SKILLS[p]) for p in driver.WORKFLOW_PHASES))
        self.assertTrue(set(driver.REQUIRED_SKILLS) <= loaded)

    def test_review_receives_measured_failure_and_changed_source_is_recompiled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root / "run"; run.mkdir()
            compiled = []

            def phase(scratch, name, iteration, models, index, attempts):
                if name == "design":
                    (scratch / "design.md").write_text("design")
                elif name == "implement":
                    (scratch / "candidate.py").write_text("before review")
                else:
                    report = json.loads((scratch / "compile_report.json").read_text())
                    self.assertEqual(report["measured_feedback"]["status"], "failed")
                    self.assertEqual(len(compiled), 1)
                    (scratch / "candidate.py").write_text("after measured repair")
                return driver.WorkerResult(0, name, driver.PHASE_SKILLS[name], (), name.upper() + "_DONE"), models[index]

            def compile_candidate(op, candidate, iteration, task):
                compiled.append(candidate.read_text())
                passed = len(compiled) == 2
                (iteration / "local_compile.json").write_text(json.dumps({"status": "passed" if passed else "failed"}))
                return {"hard_failure": not passed, "status": "passed" if passed else "failed", "passed": 20 if passed else 0, "total": 20}

            with patch.object(driver, "SCRATCH_ROOT", root / "scratch"), patch.object(driver, "run_phase", side_effect=phase), patch.object(driver, "local_evaluate", side_effect=compile_candidate), patch.object(driver, "static_check", return_value=[]), patch.object(driver, "incumbent_score", return_value=None), patch.object(driver.subprocess, "run", return_value=SimpleNamespace(stdout="1.18.33")):
                result = driver.process_item(driver.WorkItem("exp", "generate", task_dir=CATALOG / "tasks/exp"), None, run, 1, False, "local", ("glm", "qwen"), 1)
            self.assertTrue(result.status.startswith("locally qualified"))
            self.assertEqual(compiled, ["before review", "after measured repair"])
            provenance = json.loads((run / "exp-generate/iter1/provenance.json").read_text())
            self.assertNotEqual(provenance["reviewed_compile"]["candidate_sha256"], provenance["candidate_sha256"])


if __name__ == "__main__":
    unittest.main()
