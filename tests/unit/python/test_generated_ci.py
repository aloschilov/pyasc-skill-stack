"""Fail-closed attribution and continuity tests; no model or hardware calls."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('single_operator', ROOT/'integrations/cannbench/ci/single_operator.py')
ci = importlib.util.module_from_spec(spec); spec.loader.exec_module(ci)


class GeneratedCI(unittest.TestCase):
    def test_transient_get_retry_is_bounded_and_does_not_wrap_post(self):
        class HTTPError(RuntimeError):
            def __init__(self,status): self.status=status
        queue=Mock(); get=Mock(side_effect=[HTTPError(503),{'ok':True}]); post=Mock(side_effect=HTTPError(503))
        queue._client._get_json=get;queue._client._post_json=post
        with patch.object(ci.smoke,'client',return_value=queue), patch.object(ci.time,'sleep') as sleep:
            client=ci.ci_client()._client
            self.assertEqual(client._get_json('/api/catalog'),{'ok':True})
            sleep.assert_called_once_with(15)
            self.assertIs(client._post_json,post)
            with self.assertRaises(HTTPError): client._post_json('/api/submit')
            self.assertEqual(post.call_count,1)
        request=Mock(side_effect=HTTPError(503))
        with patch.object(ci.time,'sleep') as sleep, self.assertRaises(HTTPError): ci.retry_get(request)
        self.assertEqual(request.call_count,5);self.assertEqual(sleep.call_count,4)
        request=Mock(side_effect=HTTPError(400))
        with patch.object(ci.time,'sleep') as sleep, self.assertRaises(HTTPError): ci.retry_get(request)
        self.assertEqual(request.call_count,1);sleep.assert_not_called()

    def test_repair_imports_exact_failed_candidate_and_preserves_parent_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('task.json','catalog.json'): ci.save(root/name,{'task':'same'})
            def restore(parent, origin):
                attempt=parent/'generation/attempt-1';attempt.mkdir(parents=True)
                for name in ('task.json','catalog.json'): ci.save(parent/name,{'task':'same'})
                ci.save(parent/'run.json',{'skill_stack_commit':'original-commit'})
                candidate=attempt/'original-candidate.py';candidate.write_text('model output')
                ci.save(attempt/'static.json',{'passed':True})
                ci.save(attempt/'lowering.json',{'status':'passed','candidate_sha256':ci.smoke.sha(candidate)})
                ci.save(attempt/'failure.json',{'error':'numerics failed'})
                (attempt/'simulator.log').write_text('retained noisy stdout\n=== STDERR ===\nactual mismatch')
                ci.save(attempt/'simulator.log.meta.json',{'exit':1})
                (attempt/'session.json').write_text('native session record')
                return {'status':'completed'}
            with patch.object(ci,'restore_checkpoint',side_effect=restore):
                candidate,feedback=ci.repair_seed(root,'123')
            lineage=ci.load(root/'repair-lineage.json')
            self.assertEqual(lineage['candidate_sha256'],ci.smoke.sha(candidate))
            self.assertEqual(lineage['parent_commit'],'original-commit')
            self.assertIn('actual mismatch',feedback)
            self.assertNotIn('noisy stdout',feedback)
            ci.check_hashes(root/'parent-ci',lineage['parent_files'])
            (candidate.parent/'session.json').write_text('replaced')
            with self.assertRaises(ValueError): ci.check_hashes(root/'parent-ci',lineage['parent_files'])

    def test_repair_refuses_active_or_possibly_submitted_parent(self):
        for marker in ('active','submission-intent.json','submission.json','package/state.json','generation.json'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for name in ('task.json','catalog.json'): ci.save(root/name,{})
                def restore(parent, origin):
                    parent.mkdir()
                    for name in ('task.json','catalog.json'): ci.save(parent/name,{})
                    if marker!='active':
                        p=parent/marker;p.parent.mkdir(exist_ok=True);ci.save(p,{})
                    return {'status':'in_progress' if marker=='active' else 'completed'}
                with patch.object(ci,'restore_checkpoint',side_effect=restore), self.assertRaises(ValueError):
                    ci.repair_seed(root,'123')

    def test_interrupted_compile_seed_requires_exact_candidate_and_log(self):
        for tamper in (None, 'candidate/swi_glu.py', 'lowering.log'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for name in ('task.json','catalog.json'): ci.save(root/name,{})
                def restore(parent, origin):
                    attempt=parent/'generation/attempt-1';attempt.mkdir(parents=True)
                    for name in ('task.json','catalog.json'): ci.save(parent/name,{})
                    ci.save(parent/'run.json',{'skill_stack_commit':'original'})
                    for name in ('original-candidate.py','project/candidate.py','candidate/swi_glu.py'):
                        f=attempt/name;f.parent.mkdir(exist_ok=True);f.write_text('model output')
                    ci.save(attempt/'static.json',{'passed':True})
                    ci.save(attempt/'failure.json',{'error':'compiler timeout'})
                    log=attempt/'lowering.log';log.write_text('=== STDERR ===\ncompiler trace')
                    ci.save(attempt/'lowering.log.meta.json',{'exit':124,'log_sha256':ci.smoke.sha(log)})
                    if tamper: (attempt/tamper).write_text('replacement')
                    return {'status':'completed'}
                with patch.object(ci,'restore_checkpoint',side_effect=restore):
                    if tamper:
                        with self.assertRaises(ValueError): ci.repair_seed(root,'123')
                    else:
                        _,feedback=ci.repair_seed(root,'123');self.assertIn('compiler trace',feedback)

    def test_large_native_session_is_complete_scoped_and_model_checked(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); database=root/'session.db'
            c=sqlite3.connect(database)
            c.execute('create table session (id text, model text)')
            for table in ('message','part'):
                c.execute(f'create table {table} (id text, session_id text, time_created integer, data text)')
                c.execute(f'insert into {table} values (?,?,?,?)', ('item','wanted',1,'x'*200000))
                c.execute(f'insert into {table} values (?,?,?,?)', ('private','other',2,'unrelated-private-data'))
            c.execute('insert into session values (?,?)',('wanted',json.dumps({'providerID':'gateway','id':'glm-5.3'})))
            c.commit();c.close()
            ci.capture_session(database,'wanted',root/'session.json')
            data=json.loads((root/'session.json').read_text())
            self.assertEqual(len(data['part'][0]['data']),200000)
            self.assertNotIn('unrelated-private-data',(root/'session.json').read_text())
            c=sqlite3.connect(database);c.execute('update session set model=?', (json.dumps({'providerID':'gateway','id':'other'}),));c.commit();c.close()
            with self.assertRaisesRegex(ValueError,'model differs'):
                ci.capture_session(database,'wanted',root/'bad.json')

    def test_compact_api_contains_public_contracts_not_implementations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); api=root/'runtime-api/asctile/language';api.mkdir(parents=True)
            for name in ('global_tensor','memory_ops','prog_model_ops','unary_ops','binary_ops','creation_ops','indexing_ops','shape_ops','range'):
                (api/(name+'.py')).write_text('')
            (api/'memory_ops.py').write_text('def copy_in(src, offsets, shape, *, real_shape=None):\n    """Public copy contract."""\n    secret_implementation()\n')
            (api/'range.py').write_text('class range:\n    """Loop contract."""\n    def __init__(self, start, stop, step=1): pass\n')
            ci.summarize_api(root)
            summary=(root/'API.md').read_text()
            self.assertIn('real_shape=None',summary)
            self.assertIn('Public copy contract.',summary)
            self.assertIn('Loop contract.',summary)
            self.assertNotIn('secret_implementation',summary)

    def test_worker_can_write_root_outputs_without_opening_shell_or_human_prompt(self):
        import fnmatch
        rules=ci.worker_permissions(Path('/run/context'))
        def permits(path):
            return [v for k,v in rules['edit'].items() if fnmatch.fnmatchcase(path,k)][-1]
        self.assertEqual(permits('candidate.py'),'allow')
        self.assertEqual(permits('design.md'),'allow')
        self.assertEqual(permits('opencode.json'),'deny')
        self.assertEqual(permits('_pyasc_runtime.py'),'deny')
        for tool in ('bash','task','websearch','question','doom_loop'):
            self.assertEqual(rules[tool],'deny')

    def test_native_skill_trace_requires_each_actual_loaded_directory(self):
        root = Path('/recorded/skills')
        events = [{'sessionID': 'session-1', 'type': 'tool_use', 'part': {'tool': 'skill', 'state': {
            'status': 'completed', 'input': {'name': s}, 'metadata': {'dir': str(root/s)}}}} for s in ci.SKILLS]
        encode = lambda e: '\n'.join(map(json.dumps,e))
        self.assertEqual(ci.trace(encode(events),root)['session_id'],'session-1')
        with self.assertRaises(ValueError): ci.trace(encode(events[:-1]), root)
        events[-1]['part']['state']['metadata']['dir']='/different/skill'
        with self.assertRaises(ValueError): ci.trace(encode(events),root)
        with self.assertRaises(ValueError): ci.trace('I loaded all the skills',root)

    def test_evidence_paths_and_bytes_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'candidate.py').write_text('original')
            h=ci.hashes(root); ci.check_hashes(root,h)
            (root/'candidate.py').write_text('replacement')
            with self.assertRaises(ValueError): ci.check_hashes(root,h)
            with self.assertRaises(ValueError): ci.check_hashes(root,{'../escape':'x'})

    def test_quota_unknown_and_over_working_budget_refuse_submission(self):
        for c in ({}, {'remaining':10}, {'remaining':0,'used':0}, {'remaining':99,'used':10}, {'remaining':True,'used':0}):
            with self.subTest(c=c), self.assertRaises(ValueError): ci.require_quota({'credits':c})
        ci.require_quota({'credits':{'remaining':9,'used':1}})
        ci.require_quota({'credits':{'remaining':9,'daily_limit':10}})

    def test_submit_cannot_skip_central_checkpoint(self):
        queue=Mock()
        with patch.object(ci,'generated_binding',return_value=({}, {}, {})), patch.object(ci.smoke,'client',return_value=queue), patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'Central prepared checkpoint'):
                ci.submit(Path('/unused'))
        self.assertEqual(queue.mock_calls, [])

    def test_remote_host_cannot_receive_download_credentials(self):
        queue=Mock()
        for url in ('https://attacker.invalid/api/log', '//attacker.invalid/api/log', '/api/../secret'):
            with self.assertRaises(ValueError): ci.download(queue,url)
        self.assertEqual(queue.mock_calls, [])

    def test_archive_resume_rejects_tampered_original_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'evidence'; root.mkdir(); archive=Path(tmp)/'q.tar.gz'
            ci.save(root/'acceptance.json',{'passed':True})
            (root/'candidate.py').write_text('original')
            ci.save(root/'EVIDENCE.json',{'files':ci.hashes(root)})
            with tarfile.open(archive,'w:gz') as z: z.add(root,arcname='evidence')
            ci.verify_archive(archive,root)
            (root/'candidate.py').write_text('tampered')
            with self.assertRaises(ValueError): ci.verify_archive(archive,root)

    def test_qualification_for_a_different_candidate_cannot_promote_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); iteration=root/'generation/attempt-1';iteration.mkdir(parents=True)
            lowering={'status':'passed','compile_passed':20,'cases':20,'pyasc_commit':ci.PIN['pyasc_commit'],
                'runtime_bundle_sha256':ci.PIN['runtime_bundle_sha256'],'runtime_files_verified':114,
                'candidate_sha256':'candidate'}
            numerical={'passed':True,'cases':[{'passed':True} for _ in range(12)],'candidate_sha256':'candidate'}
            ci.save(iteration/'lowering.json',lowering);ci.save(iteration/'simulator.json',numerical)
            ci.qualified_binding(root,{'attempt':1,'candidate_sha256':'candidate'},{'required_cases':20})
            with self.assertRaises(ValueError):
                ci.qualified_binding(root,{'attempt':1,'candidate_sha256':'replacement'},{'required_cases':20})
            lowering['compile_passed']=19;ci.save(iteration/'lowering.json',lowering)
            with self.assertRaises(ValueError):
                ci.qualified_binding(root,{'attempt':1,'candidate_sha256':'candidate'},{'required_cases':20})

    def test_nested_evaluator_wheel_cannot_substitute_another_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'package/candidate').mkdir(parents=True)
            helper=root/'source/integrations/cannbench/smoke-kernel/_pyasc_runtime.py'; helper.parent.mkdir(parents=True)
            helper.write_bytes(b'helper')
            ci.save(root/'model-config.json',{})
            ci.save(root/'catalog.json',{});ci.save(root/'task.json',{});ci.save(root/'package/task.json',{})
            ci.save(root/'generation.json',{'files':{},'model':ci.MODEL,'provider_model_revision':'unknown',
                'native_trace':{'loaded_skills':list(ci.SKILLS)},'candidate_sha256':'candidate-digest',
                'model_configuration_sha256':ci.smoke.sha(root/'model-config.json')})
            for n,b in [('swi_glu.py',b'original'),('_pyasc_runtime.py',b'helper')]: (root/'package/candidate'/n).write_bytes(b)
            wheel=io.BytesIO()
            with zipfile.ZipFile(wheel,'w') as z:
                z.writestr('cann_bench/swi_glu.py',b'substituted');z.writestr('cann_bench/_pyasc_runtime.py',b'helper')
            with zipfile.ZipFile(root/'package/submission.zip','w') as z:
                z.writestr('vendor/test.whl',wheel.getvalue())
                z.writestr('cann_bench/swi_glu.py',b'original');z.writestr('cann_bench/_pyasc_runtime.py',b'helper')
            prepared={'files':{'candidate/swi_glu.py':'candidate-digest','candidate/_pyasc_runtime.py':ci.smoke.sha(helper)},
                'runtime_bundle_sha256':ci.PIN['runtime_bundle_sha256'],'catalog':{},'wheel_sha256':hashlib.sha256(wheel.getvalue()).hexdigest()}
            with patch.object(ci,'inputs',return_value={}),patch.object(ci,'qualified_binding'),patch.object(ci.smoke,'verify',return_value=prepared):
                with self.assertRaisesRegex(ValueError,'Source/wheel candidate mismatch'): ci.generated_binding(root)

if __name__=='__main__': unittest.main()
