"""Submission safety and packaging checks; no service calls or credits."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import copy
import zipfile

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('cannbench_smoke', ROOT/'integrations/cannbench/smoke.py')
smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)


class Smoke(unittest.TestCase):
    def test_bundle_is_reproducible_and_excludes_other_operators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runtime = root/'runtime-cp312-cp312-linux_x86_64.whl'
            with zipfile.ZipFile(runtime, 'w') as z:
                for n in ('asc/__init__.py','asctile/__init__.py','pybind11/__init__.py',
                          'asc/_C/libpyasc.so','old.dist-info/licenses/PYASC-LICENSE', 'cann_bench/other.py'):
                    z.writestr(n, b'pass\n')
            for name in ('swi_glu.py','_pyasc_runtime.py'):
                (root/name).write_text('pass\n')
            for name in ('first.zip','second.zip'):
                smoke.make_bundle(runtime, smoke.sha(runtime), root, '2.0.0', root/name)
            self.assertEqual(smoke.sha(root/'first.zip'), smoke.sha(root/'second.zip'))
            with zipfile.ZipFile(root/'first.zip') as z:
                self.assertTrue(z.getinfo('build.sh').external_attr >> 16 & 0o111)
                self.assertIn('setup.py', z.namelist())
                self.assertIn('cann_bench/swi_glu.py', z.namelist())
                data = z.read('vendor/cann_bench-2.0.0-cp312-cp312-linux_x86_64.whl')
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                self.assertNotIn('cann_bench/other.py', z.namelist())
                self.assertIn('cann_bench-2.0.0.dist-info/RECORD', z.namelist())
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                smoke.make_bundle(runtime, '0'*64, root, '2.0.0', root/'wrong.zip')

    def prepared(self, root):
        run=root/'run'; run.mkdir(); ledger=root/'ledger'; ledger.mkdir()
        (run/'submission.zip').write_bytes(b'archive')
        manifest={'files':{'submission.zip':smoke.sha(run/'submission.zip')}, 'catalog':{'benchmark_slug':'official-tasks'}, 'target_hardware':'950pr'}
        smoke.save(run/'manifest.json',manifest)
        state={'phase':'upload-intent', 'run_dir':str(run), 'manifest_sha256':smoke.sha(run/'manifest.json'),
               'archive_sha256':manifest['files']['submission.zip'], 'job_tag':'pyasc-smoke-test',
               'job_id':None,'benchmark_slug':'official-tasks'}
        smoke.save(ledger/(state['archive_sha256']+'.json'),state)
        return run,ledger,state

    def test_unknown_delivery_never_reposts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,ledger,state=self.prepared(Path(tmp)); queue=Mock()
            self.assertEqual(smoke.submit_once(queue,run,ledger),state)
            queue._submit_streaming.assert_not_called()
            queue._check_credit.assert_not_called()
            queue._client.list_jobs.return_value={'jobs':[]}
            with self.assertRaisesRegex(RuntimeError,'reconciliation'):
                smoke.collect(queue,run,ledger)
            queue._submit_streaming.assert_not_called()

    def test_mutated_candidate_archive_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,ledger,_=self.prepared(Path(tmp)); queue=Mock()
            (run/'submission.zip').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                smoke.submit_once(queue,run,ledger)
            self.assertEqual(queue.mock_calls,[])

    def test_wrong_job_tag_cannot_supply_hardware_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,ledger,state=self.prepared(Path(tmp)); queue=Mock()
            state['job_id']='job_test'; smoke.save(ledger/(state['archive_sha256']+'.json'),state)
            queue._client.get_job.return_value={'job':{'id':'job_test','job_tag':'other','status':'succeeded'}}
            with self.assertRaisesRegex(ValueError,'identity'):
                smoke.collect(queue,run,ledger)
            self.assertFalse((run/'report.json').exists())

    def test_upload_rejection_and_uncertain_server_failure_differ(self):
        from evalqueue import EvalQueue, SubmissionRejected, RemoteError
        queue = EvalQueue.__new__(EvalQueue)
        queue._client = SimpleNamespace(token='test-secret', base_url='https://example.invalid')
        queue.upload_timeout=30; queue.target_hardware='950pr'; queue.benchmark_slug='official-tasks'
        for http, error in [('400', SubmissionRejected), ('500', RemoteError), ('408', RemoteError)]:
            with self.subTest(http=http), patch('evalqueue.subprocess.run', return_value=SimpleNamespace(
                    returncode=22, stdout='{"detail":"rejected"}\n'+http, stderr='')) as call:
                with self.assertRaises(error):
                    queue._submit_streaming('/not-uploaded.zip', ['swi_glu'], 'tag', aggregation_token='agg_test')
                command=call.call_args.args[0]
                self.assertIn('aggregation_token=agg_test', command)
                self.assertIn('is_private=true', command)
                self.assertNotIn('test-secret', ' '.join(command))
                self.assertEqual(command[command.index('--retry')+1], '0')
        with patch('evalqueue.subprocess.run', return_value=SimpleNamespace(
                returncode=0, stdout='{"job_id":"job_test"}\n200', stderr='')):
            self.assertEqual(queue._submit_streaming('/unused', ['swi_glu'], 'tag')['job_id'], 'job_test')

    def test_hardware_gate_requires_exact_cases_hardware_and_timings(self):
        catalog={'benchmark_slug':'official-tasks', 'benchmark_version':'2.0.0', 'operators':[
            {'operator':'SwiGlu','function_name':'swi_glu','rel_path':'level1/swi_glu',
             'case_count':1,'case_ids':['level1/swi_glu_0']}]}
        job={'id':'job_test','status':'succeeded','benchmark_slug':'official-tasks','benchmark_version':'2.0.0',
             'target_hardware':'950pr','runner_hardware':'950pr','results':{
             'result_stage':'final','contains_performance':True,'anti_cheat_failed_cases':0,
             'operators':[{'operator':'SwiGlu','rel_path':'level1/swi_glu','total_cases':1,'passed_cases':1,
             'cases':[{'case_id':'level1/swi_glu_0','status':'success','elapsed_us':2.0,'baseline_perf_us':3.0}]}]}}
        self.assertTrue(smoke.evaluate(catalog,[job])['gate_passed'])
        for kind in ('missing','duplicate','wrong-hardware','cheat','timing','failure'):
            bad=copy.deepcopy(job)
            op=bad['results']['operators'][0]
            if kind=='missing': op['cases']=[]
            if kind=='duplicate': op['cases']*=2
            if kind=='wrong-hardware': bad['runner_hardware']='910b'
            if kind=='cheat': bad['results']['anti_cheat_failed_cases']=1
            if kind=='timing': op['cases'][0]['elapsed_us']=float('nan')
            if kind=='failure': bad['status']='correctness_failed'
            with self.subTest(kind=kind): self.assertFalse(smoke.evaluate(catalog,[bad])['gate_passed'])

    def test_intent_precedes_upload_and_error_diagnostics_are_redacted(self):
        from evalqueue import SubmissionRejected
        for rejected in (False, True):
            with self.subTest(rejected=rejected), tempfile.TemporaryDirectory() as tmp:
                run,ledger,state=self.prepared(Path(tmp))
                record=ledger/(state['archive_sha256']+'.json'); record.unlink()
                manifest=json.loads((run/'manifest.json').read_text())
                manifest['aggregation_token']='agg_test'; smoke.save(run/'manifest.json',manifest)
                smoke.save(run/'task.json',{})
                queue=Mock(target_hardware='950pr',benchmark_slug='official-tasks')
                queue._client.token='private-test-token'
                queue._client.list_jobs.return_value={'jobs':[]}
                def upload(*args, **kwargs):
                    self.assertEqual(json.loads(record.read_text())['phase'], 'upload-intent')
                    self.assertEqual(kwargs['aggregation_token'], 'agg_test')
                    error = SubmissionRejected if rejected else TimeoutError
                    raise error('private-test-token at https://example.invalid/?credential=secret')
                queue._submit_streaming.side_effect=upload
                with patch.object(smoke,'task_snapshot',return_value=(manifest['catalog'],{})):
                    with self.assertRaises((SubmissionRejected,TimeoutError)):
                        smoke.submit_once(queue,run,ledger)
                self.assertEqual(json.loads(record.read_text())['phase'], 'rejected' if rejected else 'ambiguous')
                diagnostic=(run/'upload-error.json').read_text()
                self.assertNotIn('private-test-token',diagnostic)
                self.assertNotIn('credential=secret',diagnostic)
                if rejected:
                    with self.assertRaisesRegex(ValueError,'corrected inputs'):
                        smoke.submit_once(queue,run,ledger)
                else:
                    smoke.submit_once(queue,run,ledger)
                self.assertEqual(queue._submit_streaming.call_count,1)

    def test_repository_output_is_rejected(self):
        with self.assertRaises(ValueError):
            smoke.external(ROOT/'results')


if __name__ == '__main__': unittest.main()
