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
import time
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
            with patch.object(driver.subprocess, "Popen", return_value=SimpleNamespace(returncode=0, communicate=lambda **kw: ("", ""))) as run:
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

    def test_worker_timeout_kills_descendant_and_retains_full_trace(self):
        # A TERM-ignoring child keeps the output pipe open after its parent exits.
        # The timeout must kill the whole group before collecting output.
        real_popen = subprocess.Popen
        child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('CHILD_READY',flush=True); time.sleep(30)"
        script = ("import subprocess,sys,time; print('TRACE_BEGIN'+('x'*10000),flush=True); "
                  "subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(30)")
        processes = []
        def launch(command, **kwargs):
            proc = real_popen([sys.executable, '-c', script], **kwargs)
            processes.append(proc)
            return proc
        with tempfile.TemporaryDirectory() as directory, patch.object(driver, 'REPO_ROOT', Path(directory)):
            scratch = Path(directory) / 'integrations/cannbench/workers/.scratch/test/exp'
            scratch.mkdir(parents=True)
            try:
                with patch.object(driver.subprocess, 'Popen', side_effect=launch):
                    result = driver.run_worker(scratch, 'test', scratch / 'log', 'gateway/glm-5.3', timeout_s=1)
                self.assertEqual(result.returncode, 124)
                log = (scratch / 'log').read_text()
                self.assertIn('TRACE_BEGIN' + 'x'*10000, log)
                self.assertIn('CHILD_READY', log)
                self.assertIsNotNone(processes[0].poll())
            finally:
                import signal
                for proc in processes:
                    try: os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    proc.wait(timeout=5)

    def test_asctile_and_contract_signature_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            candidate.write_text('import asctile\nfrom ._pyasc_runtime import ensure_npu_platform\n@asctile.jit\ndef kernel(): pass\ndef exp(x, base=-1.0, scale=1.0, shift=0.0): return x\n')
            self.assertEqual(driver.static_check(candidate, "exp", CATALOG / "tasks/exp"), [])
            candidate.write_text(candidate.read_text().replace("base=-1.0", "base=2.0"))
            self.assertIn("Public callable defaults differ from official golden", driver.static_check(candidate, "exp", CATALOG / "tasks/exp"))

    def test_whitelisted_torch_factories_pass_static_check(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            candidate.write_text(
                'import torch\nimport asctile\nfrom ._pyasc_runtime import ensure_npu_platform\n'
                '@asctile.jit\ndef kernel(): pass\n'
                'def top_k(x, k, dim=-1, largest=True):\n'
                '    idx = torch.arange(10, device=x.device)\n'
                '    one = torch.ones((2,), device=x.device)\n'
                '    full = torch.full((2,), 0.0, device=x.device)\n'
                '    t = torch.tensor([1, 2], device=x.device)\n'
                '    return x\n')
            problems = driver.static_check(candidate, "top_k", CATALOG / "tasks/top_k")
            self.assertFalse([p for p in problems if 'banned torch' in p], problems)
            candidate.write_text(candidate.read_text().replace(
                'torch.arange(10, device=x.device)', 'torch.sort(x)'))
            problems = driver.static_check(candidate, "top_k", CATALOG / "tasks/top_k")
            self.assertTrue(any('banned torch' in p for p in problems), problems)

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

    def test_pinned_implementation_rejects_reviewer_edits_and_never_rotates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root / "run"; run.mkdir()
            calls = []
            def phase(scratch, name, iteration, models, index, attempts, **kw):
                calls.append((name, models, index))
                if name == "design":
                    (scratch / "design.md").write_text("design")
                elif name == "implement":
                    (scratch / "candidate.py").write_text("GLM implementation")
                else:
                    self.assertTrue(kw.get('readonly_review'))
                    (scratch / "candidate.py").write_text("unauthorized reviewer repair")
                return driver.WorkerResult(0,name,driver.PHASE_SKILLS[name],(),name.upper()+"_DONE"),models[index]
            def compile_candidate(op,candidate,iteration,task):
                (iteration / "local_compile.json").write_text('{"status":"passed"}')
                return {"hard_failure":False,"status":"passed","passed":20,"total":20}
            with patch.object(driver,"SCRATCH_ROOT",root/"scratch"), patch.object(driver,"run_phase",side_effect=phase), patch.object(driver,"local_evaluate",side_effect=compile_candidate) as compiler, patch.object(driver,"static_check",return_value=[]), patch.object(driver,"incumbent_score",return_value=None), patch.object(driver.subprocess,"run",return_value=SimpleNamespace(stdout="1.18.33")):
                result=driver.process_item(driver.WorkItem("exp","generate",task_dir=CATALOG/"tasks/exp"),None,run,2,False,"local",("glm","qwen"),2,implementation_model="glm")
            self.assertEqual(compiler.call_count,2)  # pre-review only; no final acceptance
            self.assertFalse((run/"locally_qualified/cann_bench/exp.py").exists())
            for phase_name,models,index in calls:
                self.assertEqual(models,("qwen",) if phase_name=="review" else ("glm",))
                self.assertEqual(index,0)
            for iteration in (1,2):
                provenance=json.loads((run/f"exp-generate/iter{iteration}/provenance.json").read_text())
                self.assertEqual(provenance['required_implementation_model'],'glm')
                self.assertNotEqual(provenance['implementation_sha256'],provenance['candidate_sha256'])

    def test_pinned_mode_requires_positive_readonly_review(self):
        for verdict, accepted in [('REVIEW_PASS', True), ('REVIEW_REPAIR', False),
                                  ('REVIEW_DONE', False), ('REVIEW_PASS\nREVIEW_REPAIR', False)]:
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); run=root/'run'; run.mkdir()
                def phase(scratch,name,iteration,models,index,attempts,**kw):
                    if name=='design': (scratch/'design.md').write_text('design')
                    if name=='implement': (scratch/'candidate.py').write_text('GLM source')
                    output=verdict+'\nREVIEW_DONE' if name=='review' else name.upper()+'_DONE'
                    return driver.WorkerResult(0,name,driver.PHASE_SKILLS[name],(),output),models[index]
                def compile_candidate(op,candidate,iteration,task):
                    (iteration/'local_compile.json').write_text('{"status":"passed"}')
                    return {'hard_failure':False,'status':'passed','passed':20,'total':20}
                with patch.object(driver,'SCRATCH_ROOT',root/'scratch'), patch.object(driver,'run_phase',side_effect=phase), patch.object(driver,'local_evaluate',side_effect=compile_candidate), patch.object(driver,'static_check',side_effect=lambda *a:[]), patch.object(driver,'incumbent_score',return_value=None), patch.object(driver.subprocess,'run',return_value=SimpleNamespace(stdout='1.18.33')):
                    driver.process_item(driver.WorkItem('exp','generate',task_dir=CATALOG/'tasks/exp'),None,run,1,False,'local',('glm',),1,implementation_model='glm')
                self.assertEqual((run/'locally_qualified/cann_bench/exp.py').exists(),accepted)

    def _repair_phase_factory(self, calls, verdicts, sources, task_checks=None):
        def phase(scratch, name, iteration, models, index, attempts, **kw):
            calls.append((name, models, kw))
            if name == 'design':
                (scratch/'design.md').write_text('design')
                output = 'DESIGN_DONE'
            elif name == 'implement':
                source = sources[min(len([c for c in calls if c[0] == 'implement']) - 1, len(sources) - 1)]
                if task_checks is not None and len([c for c in calls if c[0] == 'implement']) > 1:
                    task_checks((scratch/'task.md').read_text())
                (scratch/'candidate.py').write_text(source)
                output = 'IMPLEMENT_DONE'
            else:
                n = len([c for c in calls if c[0] == 'review'])
                output = verdicts[min(n - 1, len(verdicts) - 1)] + '\nREVIEW_DONE'
            return driver.WorkerResult(0, name, driver.PHASE_SKILLS[name], (), output), models[index]
        return phase

    def _compile_factory(self, compiled, good_source):
        def compile_candidate(op, candidate, iteration, task):
            src = candidate.read_text()
            compiled.append(src)
            passed = src == good_source
            (iteration/'local_compile.json').write_text(json.dumps(
                {'status': 'passed' if passed else 'failed',
                 'compile_passed': 20 if passed else 0}))
            return {'hard_failure': False,
                    'status': 'passed' if passed else 'failed',
                    'passed': 20 if passed else 0, 'total': 20}
        return compile_candidate

    def test_targeted_repair_fixes_candidate_within_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'run'; run.mkdir()
            calls = []; compiled = []
            phase = self._repair_phase_factory(
                calls, ['REVIEW_REPAIR', 'REVIEW_PASS'],
                ['initial defective', 'repaired source'],
                task_checks=lambda text: (
                    self.assertIn('Targeted repair', text),
                    self.assertIn('initial defective', text),
                    self.assertIn('Do not remove or shrink any case', text)))
            compile_candidate = self._compile_factory(compiled, 'repaired source')
            with patch.object(driver,'SCRATCH_ROOT',root/'scratch'), patch.object(driver,'run_phase',side_effect=phase), patch.object(driver,'local_evaluate',side_effect=compile_candidate), patch.object(driver,'static_check',side_effect=lambda *a: []), patch.object(driver,'incumbent_score',return_value=None), patch.object(driver.subprocess,'run',return_value=SimpleNamespace(stdout='1.18.33')):
                result = driver.process_item(driver.WorkItem('exp','generate',task_dir=CATALOG/'tasks/exp'),None,run,1,False,'local',('glm',),1,implementation_model='glm',repair_attempts=1)
            self.assertTrue(result.status.startswith('locally qualified'))
            self.assertEqual(compiled, ['initial defective', 'repaired source', 'repaired source'])
            names = [c[0] for c in calls]
            self.assertEqual(names.count('design'), 1)
            self.assertEqual(names.count('implement'), 2)
            self.assertEqual(names.count('review'), 2)
            for name, models, kw in calls:
                self.assertEqual(models, ('glm',))
                if name == 'review':
                    self.assertTrue(kw.get('readonly_review'))
            accepted = (run/'locally_qualified/cann_bench/exp.py').read_text()
            self.assertEqual(accepted, 'repaired source')
            provenance = json.loads((run/'exp-generate/iter1/provenance.json').read_text())
            self.assertEqual(len(provenance['repair_attempts']), 1)
            self.assertEqual(provenance['repair_attempts'][0]['attempt'], 1)
            self.assertEqual(provenance['repair_attempts'][0]['implement_model'], 'glm')
            self.assertEqual(provenance['repair_attempts'][0]['review_model'], 'glm')
            repair_dir = run/'exp-generate/iter1/repair1'
            self.assertEqual((repair_dir/'candidate.py').read_text(), 'repaired source')
            self.assertTrue((repair_dir/'pre_review_compile.json').is_file())
            self.assertEqual((run/'exp-generate/iter1/candidate.py').read_text(), 'repaired source')
            self.assertTrue(result.history[0]['result'] == 'repair-requested')

    def test_repair_attempts_exhausted_regenerates_next_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'run'; run.mkdir()
            calls = []; compiled = []
            phase = self._repair_phase_factory(calls, ['REVIEW_REPAIR'], ['src'])
            compile_candidate = self._compile_factory(compiled, 'never good')
            with patch.object(driver,'SCRATCH_ROOT',root/'scratch'), patch.object(driver,'run_phase',side_effect=phase), patch.object(driver,'local_evaluate',side_effect=compile_candidate), patch.object(driver,'static_check',side_effect=lambda *a: []), patch.object(driver,'incumbent_score',return_value=None), patch.object(driver.subprocess,'run',return_value=SimpleNamespace(stdout='1.18.33')):
                result = driver.process_item(driver.WorkItem('exp','generate',task_dir=CATALOG/'tasks/exp'),None,run,2,False,'local',('glm',),1,implementation_model='glm',repair_attempts=1)
            self.assertFalse((run/'locally_qualified/cann_bench/exp.py').exists())
            names = [c[0] for c in calls]
            self.assertEqual(names.count('design'), 2)
            self.assertEqual(names.count('implement'), 4)
            self.assertEqual(names.count('review'), 4)
            self.assertTrue((run/'exp-generate/iter1/repair1/candidate.py').is_file())
            self.assertTrue((run/'exp-generate/iter2/repair1/candidate.py').is_file())
            self.assertEqual(result.history[-1]['result'], 'static-fail')
            self.assertEqual(result.history[-1]['repair_attempts_used'], 1)

    def test_repair_attempts_default_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'run'; run.mkdir()
            calls = []; compiled = []
            phase = self._repair_phase_factory(calls, ['REVIEW_REPAIR'], ['src'])
            compile_candidate = self._compile_factory(compiled, 'never good')
            with patch.object(driver,'SCRATCH_ROOT',root/'scratch'), patch.object(driver,'run_phase',side_effect=phase), patch.object(driver,'local_evaluate',side_effect=compile_candidate), patch.object(driver,'static_check',side_effect=lambda *a: []), patch.object(driver,'incumbent_score',return_value=None), patch.object(driver.subprocess,'run',return_value=SimpleNamespace(stdout='1.18.33')):
                result = driver.process_item(driver.WorkItem('exp','generate',task_dir=CATALOG/'tasks/exp'),None,run,1,False,'local',('glm',),1,implementation_model='glm')
            self.assertFalse((run/'locally_qualified/cann_bench/exp.py').exists())
            names = [c[0] for c in calls]
            self.assertEqual(names.count('implement'), 1)
            self.assertEqual(names.count('review'), 1)
            self.assertFalse((run/'exp-generate/iter1/repair1').exists())
            self.assertEqual(result.history[-1]['repair_attempts_used'], 0)

    def test_repair_review_uses_non_implementer_in_mixed_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'run'; run.mkdir()
            calls = []; compiled = []
            phase = self._repair_phase_factory(calls, ['REVIEW_REPAIR'], ['src'])
            compile_candidate = self._compile_factory(compiled, 'never good')
            with patch.object(driver,'SCRATCH_ROOT',root/'scratch'), patch.object(driver,'run_phase',side_effect=phase), patch.object(driver,'local_evaluate',side_effect=compile_candidate), patch.object(driver,'static_check',side_effect=lambda *a: []), patch.object(driver,'incumbent_score',return_value=None), patch.object(driver.subprocess,'run',return_value=SimpleNamespace(stdout='1.18.33')):
                driver.process_item(driver.WorkItem('exp','generate',task_dir=CATALOG/'tasks/exp'),None,run,1,False,'local',('glm','qwen'),1,implementation_model='glm',repair_attempts=1)
            review_calls = [c for c in calls if c[0] == 'review']
            self.assertEqual(len(review_calls), 2)
            for _, models, kw in review_calls:
                self.assertEqual(models, ('qwen',))
                self.assertTrue(kw.get('readonly_review'))

    def test_repair_attempts_negative_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'run'; run.mkdir()
            with self.assertRaises(ValueError):
                driver.process_item(driver.WorkItem('exp','generate',task_dir=CATALOG/'tasks/exp'),None,run,1,False,'local',('glm',),1,implementation_model='glm',repair_attempts=-1)

    def test_finished_on_length_parser(self):
        length_event = json.dumps({"type": "step_finish", "part": {"reason": "length"}})
        ok_event = json.dumps({"type": "step_finish", "part": {"reason": "tool-calls"}})
        self.assertTrue(driver._finished_on_length(ok_event + "\n" + length_event))
        self.assertFalse(driver._finished_on_length(length_event + "\n" + ok_event))
        self.assertFalse(driver._finished_on_length(ok_event))
        self.assertFalse(driver._finished_on_length(""))
        self.assertFalse(driver._finished_on_length("not json\n" + length_event.replace("length", "stop")))

    def test_length_exhaustion_adapts_retry_message(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'run'; run.mkdir()
            scratch = root/'scratch'; scratch.mkdir()
            messages = []
            def worker(scratch_dir, message, log_path, model, **kw):
                messages.append(message)
                if len(messages) == 1:
                    return driver.WorkerResult(0, 's1', driver.PHASE_SKILLS['implement'], (), 'analysis but no artifact', True)
                (scratch_dir/'candidate.py').write_text('repaired')
                return driver.WorkerResult(0, 's2', driver.PHASE_SKILLS['implement'], (), 'IMPLEMENT_DONE', False)
            with patch.object(driver, 'run_worker', side_effect=worker), patch.object(driver.time, 'sleep', return_value=None):
                outcome = driver.run_phase(scratch, 'implement', run, ('glm',), 0, 2)
            self.assertIsNotNone(outcome)
            self.assertEqual(len(messages), 2)
            self.assertNotIn('OUTPUT BUDGET DISCIPLINE', messages[0])
            self.assertIn('OUTPUT BUDGET DISCIPLINE', messages[1])
            self.assertIn('Write the COMPLETE candidate.py file FIRST', messages[1])
            self.assertIn('IMPLEMENT_DONE', messages[1])
            trace1 = json.loads((run/'implement.skill-trace.attempt1.json').read_text())
            self.assertTrue(trace1['finish_length'])
            trace2 = json.loads((run/'implement.skill-trace.attempt2.json').read_text())
            self.assertFalse(trace2['finish_length'])
            self.assertTrue(trace2['skill_gate_passed'])

    def test_implement_prompt_enforces_budget_discipline(self):
        prompt = driver.PHASE_PROMPTS['implement']
        self.assertIn('Budget discipline', prompt)
        self.assertIn('FIRST artifact-producing action', prompt)
        self.assertNotIn('Budget discipline', driver.PHASE_PROMPTS['design'])

    def test_worker_result_finish_length_defaults_false(self):
        result = driver.WorkerResult(0, 's', (), (), 'text')
        self.assertFalse(result.finish_length)


class _HangingProc:
    returncode = 124

    def communicate(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired(cmd="gate", timeout=timeout)
        return "", "gate hung"

    def poll(self):
        return None


class _GarbageProc:
    returncode = 255

    def communicate(self, timeout=None):
        return ("Waiting for shared execution lock: screen.lock\n"
                "exec /usr/local/bin/python: exec format error\n"), ""


class GateInfrastructureTests(unittest.TestCase):
    CANDIDATE = ROOT / "integrations/cannbench/submission/cann_bench/exp.py"

    def test_gate_infra_failure_detection(self):
        cases = [
            ({"fatal_error": "local compile gate emitted non-JSON output"}, "emitted non-JSON output"),
            ({"fatal_error": "x", "stderr_tail": "exec /usr/local/bin/python: exec format error"}, "exec format error"),
            ({"fatal_error": "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"}, "Cannot connect to the Docker daemon"),
            ({"fatal_error": "compile gate image build failed"}, "compile gate image build failed"),
            ({"fatal_error": "compile gate timeout (driver backstop)"}, None),
            ({"status": "failed", "case_results": []}, None),
            ({"status": "passed"}, None),
            ({"error": "ValueError: too many values to unpack (expected 4)"}, None),
        ]
        for report, expected in cases:
            with self.subTest(report=report):
                self.assertEqual(driver._gate_infra_failure(report), expected)

    def test_local_evaluate_raises_infra_abort_and_preserves_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1"
            iter_dir.mkdir()
            with patch.object(driver.subprocess, "Popen", return_value=_GarbageProc()), \
                 patch.object(driver, "_reconcile_container", return_value="absent"):
                with self.assertRaises(driver.GateInfrastructureError) as ctx:
                    driver.local_evaluate("exp", self.CANDIDATE, iter_dir)
            self.assertIn("infrastructure failure", str(ctx.exception))
            record = json.loads((iter_dir / "local_compile.json").read_text())
            self.assertEqual(record["fatal_error"], "local compile gate emitted non-JSON output")

    def test_gate_preflight(self):
        ok_inspect = SimpleNamespace(returncode=0)
        ok_probe = SimpleNamespace(returncode=0, stdout="gate-exec-ok\n", stderr="")
        with patch.object(driver.subprocess, "run", side_effect=[ok_inspect, ok_probe]) as runs:
            self.assertIsNone(driver._gate_preflight())
            self.assertEqual(runs.call_count, 2)
            self.assertIn("linux/amd64", runs.call_args_list[1].args[0])
            self.assertIn("--entrypoint", runs.call_args_list[1].args[0])
        with patch.object(driver.subprocess, "run", side_effect=[SimpleNamespace(returncode=1)]):
            self.assertIn("not inspectable", driver._gate_preflight())
        bad_probe = SimpleNamespace(returncode=255, stdout="", stderr="exec format error")
        with patch.object(driver.subprocess, "run", side_effect=[ok_inspect, bad_probe]):
            self.assertIn("exec failed", driver._gate_preflight())

    def test_infra_failure_does_not_consume_iteration_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'run'; run.mkdir()
            def phase(scratch, name, iteration, models, index, attempts, **kw):
                if name == 'design':
                    (scratch/'design.md').write_text('design')
                elif name == 'implement':
                    (scratch/'candidate.py').write_text('candidate')
                return driver.WorkerResult(0, name, driver.PHASE_SKILLS[name], (), name.upper() + '_DONE'), models[index]
            with patch.object(driver,'SCRATCH_ROOT',root/'scratch'), patch.object(driver,'run_phase',side_effect=phase), patch.object(driver,'local_evaluate',side_effect=driver.GateInfrastructureError('gate infra broken')), patch.object(driver,'static_check',side_effect=lambda *a: []), patch.object(driver,'incumbent_score',return_value=None), patch.object(driver.subprocess,'run',return_value=SimpleNamespace(stdout='1.18.33')):
                item = driver.WorkItem('exp','generate',task_dir=CATALOG/'tasks/exp')
                with self.assertRaises(driver.GateInfrastructureError):
                    driver.process_item(item,None,run,2,False,'local',('glm',),1,implementation_model='glm',repair_attempts=2)
            # no iteration/repair budget consumed: nothing recorded as a
            # candidate failure outcome
            self.assertEqual(item.history, [])
            self.assertFalse((run/'locally_qualified/cann_bench/exp.py').exists())


class BoundedGateTests(unittest.TestCase):
    CANDIDATE = ROOT / "integrations/cannbench/submission/cann_bench/exp.py"

    def test_execution_budget_starts_after_queue(self):
        clock = [0.0]
        marker = SimpleNamespace(exists=lambda: clock[0] >= 10.0)
        calls = []
        def communicate(timeout):
            calls.append(timeout)
            clock[0] += timeout
            if clock[0] >= 11.0:
                return "passed", ""
            raise subprocess.TimeoutExpired("gate", timeout)
        proc = SimpleNamespace(args=["gate"], communicate=communicate)
        with patch.object(driver.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertEqual(driver._communicate_gate(proc, marker, 2), ("passed", ""))
        self.assertEqual(len(calls), 11)

    def test_execution_budget_expires_after_acquisition(self):
        clock = [0.0]
        marker = SimpleNamespace(exists=lambda: clock[0] >= 10.0)
        def communicate(timeout):
            clock[0] += timeout
            raise subprocess.TimeoutExpired("gate", timeout)
        proc = SimpleNamespace(args=["gate"], communicate=communicate)
        with patch.object(driver.time, "monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(subprocess.TimeoutExpired):
                driver._communicate_gate(proc, marker, 2)
        self.assertEqual(clock[0], 12)

    def test_backstop_reconciled_kills_group_and_records_timeout(self):
        killed = []
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1"
            iter_dir.mkdir()
            with patch.object(driver.subprocess, "Popen",
                              return_value=_HangingProc()), \
                 patch.object(driver, "_communicate_gate",
                              side_effect=subprocess.TimeoutExpired("gate", 1)), \
                 patch.object(driver, "_terminate_process_group",
                              side_effect=lambda proc, grace_seconds=15.0:
                              killed.append(proc)), \
                 patch.object(driver, "_reconcile_container",
                              return_value="absent"), \
                 patch.dict(os.environ,
                            {"PYASC_CANN_GATE_TIMEOUT_SECONDS": "1"}):
                digest = driver.local_evaluate(
                    "exp", self.CANDIDATE, iter_dir)
            self.assertEqual(len(killed), 1)
            self.assertTrue(digest["hard_failure"])
            record = json.loads(
                (iter_dir / "local_compile.json").read_text(encoding="utf-8"))
            self.assertTrue(record["gate_timed_out"])
            self.assertEqual(record["gate_timeout_seconds"], 1.0)
            self.assertEqual(record["returncode"], 124)
            self.assertEqual(record["container_reconcile"], "absent")
            self.assertEqual(record["cases"], 20)  # 20 planned retained
            self.assertEqual(record["planned_cases"], 20)

    def test_backstop_unreconciled_retains_lock_and_does_not_kill(self):
        killed = []
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1"
            iter_dir.mkdir()
            with patch.object(driver.subprocess, "Popen",
                              return_value=_HangingProc()), \
                 patch.object(driver, "_communicate_gate",
                              side_effect=subprocess.TimeoutExpired("gate", 1)), \
                 patch.object(driver, "_terminate_process_group",
                              side_effect=lambda proc, grace_seconds=15.0:
                              killed.append(proc)), \
                 patch.object(driver, "_reconcile_container",
                              return_value="unreconciled"), \
                 patch.dict(os.environ,
                            {"PYASC_CANN_GATE_TIMEOUT_SECONDS": "1"}):
                digest = driver.local_evaluate(
                    "exp", self.CANDIDATE, iter_dir)
            self.assertEqual(killed, [])  # fail closed: lock retained
            self.assertTrue(digest["lock_retained"])
            self.assertEqual(digest["total"], 20)
            record = json.loads(
                (iter_dir / "local_compile.json").read_text(encoding="utf-8"))
            self.assertTrue(record["lock_retained"])
            self.assertEqual(record["container_reconcile"], "unreconciled")
            self.assertEqual(record["cases"], 20)

    def test_invalid_timeout_is_validated_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1"
            iter_dir.mkdir()
            with patch.object(driver.subprocess, "Popen",
                              return_value=_HangingProc()), \
                 patch.object(driver, "_communicate_gate",
                              side_effect=subprocess.TimeoutExpired("gate", 1)), \
                 patch.object(driver, "_terminate_process_group"), \
                 patch.object(driver, "_reconcile_container",
                              return_value="absent"), \
                 patch.dict(os.environ,
                            {"PYASC_CANN_GATE_TIMEOUT_SECONDS": "-5"}):
                driver.local_evaluate("exp", self.CANDIDATE, iter_dir)
            record = json.loads(
                (iter_dir / "local_compile.json").read_text(encoding="utf-8"))
            self.assertIn("gate_timeout_warning", record)
            self.assertEqual(record["gate_timeout_seconds"], 900.0)

    def test_docker_state_semantics_never_assume_absence(self):
        def fake_run(command, **kwargs):
            if "nonexistent" in command:
                return SimpleNamespace(returncode=1,
                                       stdout="", stderr="No such object: x")
            return SimpleNamespace(returncode=1, stdout="",
                                   stderr="Cannot connect to the Docker daemon")

        with patch.object(driver.subprocess, "run", side_effect=fake_run):
            self.assertEqual(driver._docker_container_state("nonexistent"),
                             "absent")
            self.assertEqual(driver._docker_container_state("other"),
                             "daemon_unavailable")

    def test_unreachable_daemon_does_not_confirm_container_stopped(self):
        with patch.object(driver, "_docker_container_state",
                          return_value="daemon_unavailable"):
            self.assertEqual(driver._reconcile_container("test", 0),
                             "unreconciled")

    def test_gate_container_name_is_deterministic(self):
        first = driver._gate_container_name("exp", "cand.py")
        second = driver._gate_container_name("exp", "cand.py")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("pyasc-cannbench-gate-exp-"))

    def test_terminate_process_group_kills_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "grandchild.pid"
            child = subprocess.Popen(
                [sys.executable, "-c",
                 "import subprocess, time, pathlib; "
                 f"g = subprocess.Popen(['sleep', '60']); "
                 f"pathlib.Path({str(pid_file)!r}).write_text(str(g.pid)); "
                 "time.sleep(60)"],
                start_new_session=True)
            deadline = 30.0
            while not pid_file.exists():
                time.sleep(0.05)
                deadline -= 0.05
                if deadline <= 0:
                    child.kill()
                    self.fail("grandchild did not start")
            grandchild_pid = int(pid_file.read_text())
            driver._terminate_process_group(child, grace_seconds=1.0)
            child.wait(timeout=10)
            self.assertIsNotNone(child.returncode)
            with self.assertRaises(ProcessLookupError):
                os.kill(grandchild_pid, 0)


class GateReportSequenceTests(unittest.TestCase):
    def reports(self, passed=False):
        completed = {"operator": "grid_sampler_3d", "status": "passed" if passed else "failed",
                     "compile_passed": 20 if passed else 0, "dispatch_passed": 20 if passed else 0,
                     "case_results": [{"case_id": 1, "compile": "passed" if passed else "failed",
                                       "dispatch": "passed" if passed else "failed",
                                       "compile_errors": [] if passed else ["PlainValue type mismatch"]}]}
        lifecycle = {"operator": "grid_sampler_3d", "status": "failed", "gate_timed_out": True,
                     "fatal_error": "compile gate timeout", "cases": 20}
        return completed, lifecycle

    def test_completed_failure_survives_container_timeout(self):
        completed, lifecycle = self.reports()
        result = driver.parse_gate_report(json.dumps(completed)+'\n'+json.dumps(lifecycle))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['case_results'][0]['compile_errors'], ['PlainValue type mismatch'])
        self.assertEqual(result['completed_compile_report'], completed)

    def test_completed_pass_cannot_override_timeout(self):
        completed, lifecycle = self.reports(True)
        result = driver.parse_gate_report(json.dumps(completed)+'\n'+json.dumps(lifecycle))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['compile_passed'], 0)
        self.assertEqual(result['dispatch_passed'], 0)
        self.assertTrue(result['gate_timed_out'])

    def test_unbound_or_non_json_sequence_rejected(self):
        completed, lifecycle = self.reports()
        lifecycle['operator'] = 'other'
        for text in [json.dumps(completed)+'\n'+json.dumps(lifecycle),
                     json.dumps(completed)+'\ngarbage', '[]']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                driver.parse_gate_report(text)

    def test_malformed_case_entry_rejected(self):
        completed, lifecycle = self.reports()
        completed["case_results"] = [None]
        with self.assertRaises(ValueError):
            driver.parse_gate_report(json.dumps(completed) + "\n" + json.dumps(lifecycle))


class WorkerBudgetTests(unittest.TestCase):
    def test_scoped_worker_budget_and_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(driver.worker_timeout_seconds(), 900)
        with patch.dict(os.environ, {"CANNBENCH_WORKER_TIMEOUT_SECONDS": "1800"}):
            self.assertEqual(driver.worker_timeout_seconds(), 1800)

    def test_invalid_or_unbounded_worker_budget_rejected(self):
        for value in ["0", "-1", "7201", "1.5", "", "inf", "９００"]:
            with self.subTest(value=value), patch.dict(os.environ, {"CANNBENCH_WORKER_TIMEOUT_SECONDS": value}):
                with self.assertRaises(ValueError):
                    driver.worker_timeout_seconds()


if __name__ == "__main__":
    unittest.main()
