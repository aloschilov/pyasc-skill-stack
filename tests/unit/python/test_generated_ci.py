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
    def test_product_release_tags_do_not_publish_verification_archives(self):
        import fnmatch, yaml
        workflow = yaml.safe_load((ROOT/'.github/workflows/release.yml').read_text())
        patterns = workflow[True]['push']['tags']
        for tag, expected in [('v0.3.0', True), ('v12.0.1-rc1', True),
                              ('verification-integration-c4cf8f5', False),
                              ('qualification-generation-123', False),
                              ('archive-campaign-2026', False), ('v', False)]:
            with self.subTest(tag=tag):
                self.assertEqual(any(fnmatch.fnmatchcase(tag, p) for p in patterns), expected)

    def test_publication_credential_is_scoped_to_release_step(self):
        import yaml
        workflow=yaml.safe_load((ROOT/'.github/workflows/cannbench-generated.yml').read_text())
        job=workflow['jobs']['generated-operator']
        self.assertNotIn('QUALIFICATION_PUBLISH_TOKEN',json.dumps(job['env']))
        uses=[step for step in job['steps'] if 'secrets.QUALIFICATION_PUBLISH_TOKEN' in json.dumps(step)]
        self.assertEqual(len(uses),1)
        self.assertEqual(uses[0]['name'],'Publish durable qualification and verify downloaded archive')
        self.assertEqual(uses[0]['env']['GH_TOKEN'],'${{ secrets.QUALIFICATION_PUBLISH_TOKEN || github.token }}')

    def test_checkpoint_recovery_accepts_new_and_historical_names(self):
        for prefix in ('generated-operator', 'generated-swiglu'):
            with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'GITHUB_REPOSITORY':'owner/repo'}):
                root=Path(tmp)/'restored';name=f'{prefix}-evidence-123-2'
                responses=[json.dumps({'event':'workflow_dispatch','head_repository':{'full_name':'owner/repo'},'head_sha':'revision'}),
                    json.dumps({'artifacts':[{'name':name,'expired':False}]})]
                with patch.object(ci.subprocess,'check_output',side_effect=responses), patch.object(ci.subprocess,'run') as download, \
                     patch.object(ci,'inputs',return_value={'origin_run_id':'123','skill_stack_commit':'revision'}):
                    ci.restore_checkpoint(root,'123')
                    self.assertIn(name,download.call_args.args[0])

    def test_recipe_selection_is_validated_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'CANNBENCH_OPERATOR':'not_supported'}), \
             patch.object(ci, 'ci_client') as network, patch.object(ci, 'call') as model:
            root=Path(tmp)/'new'
            with self.assertRaisesRegex(ValueError, 'Unsupported operator'): ci.generate(root)
            self.assertFalse(root.exists());network.assert_not_called();model.assert_not_called()
        for invalid in ('../swi_glu', '/tmp/swi_glu', 'swi_glu;echo'):
            with self.assertRaises(ValueError): ci.load_recipe(invalid)

    def test_recipe_uses_snapshot_and_rejects_task_mismatch(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'GITHUB_REPOSITORY':'owner/repo', 'CANNBENCH_OPERATOR':'ignored'}):
            root=Path(tmp);self.qualified_parent(root)
            directory=root/'source/integrations/cannbench/ci/operators'
            shutil.copytree(ROOT/'integrations/cannbench/ci/operators', directory)
            meta=ci.load(root/'run.json');meta.update(operator_recipe='swi_glu',source_hashes=ci.hashes(root/'source'))
            ci.save(root/'run.json', meta)
            ci.save(root/'catalog.json', {'operators':[{'operator':'SwiGlu','function_name':'swi_glu'}]})
            self.assertEqual(ci.inputs(root)['operator_recipe'],'swi_glu')
            self.assertEqual(ci.run_recipe(root)['numerical_cases'],12)
            ci.save(root/'catalog.json', {'operators':[{'operator':'Other','function_name':'other'}]})
            with self.assertRaisesRegex(ValueError,'differs from its operator'): ci.inputs(root)

    def test_other_recipe_has_its_own_numerical_acceptance_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.qualified_parent(root)
            directory=root/'source/integrations/cannbench/ci/operators';directory.mkdir(parents=True)
            ci.save(directory/'other.json', {'operator':'Other','function':'other','prompt':'prompt.txt',
                'numerical_probe':'probe.py','numerical_cases':2})
            (directory/'prompt.txt').write_text('other contract');(directory/'probe.py').write_text('# offline fixture')
            meta=ci.load(root/'run.json');meta['operator_recipe']='other';ci.save(root/'run.json',meta)
            attempt=root/'generation/attempt-1';digest=ci.smoke.sha(attempt/'original-candidate.py')
            gen={'attempt':1,'candidate_sha256':digest}
            with self.assertRaisesRegex(ValueError,'simulator preflight'): ci.qualified_binding(root,gen,{'required_cases':20})
            ci.save(attempt/'simulator.json', {'passed':True,'cases':[{'passed':True}]*2,'candidate_sha256':digest})
            ci.qualified_binding(root,gen,{'required_cases':20})

    def test_compile_coverage_cannot_pass_blocked_or_empty_modules(self):
        import os, subprocess, yaml
        workflow=yaml.safe_load((ROOT/'.github/workflows/ci.yml').read_text())
        step=next(s for s in workflow['jobs']['merge-gate']['steps'] if s.get('name')=='Compile-check the AscTile target suite')
        script=step['run'].replace('${{ github.run_id }}','offline').replace('${{ env.PYASC_SIM_IMAGE }}','unused')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'reference/asctile/target';target.mkdir(parents=True)
            binary=root/'bin';binary.mkdir();docker=binary/'docker'
            docker.write_text('#!/bin/sh\n[ "$1" = rm ] && exit 0\nexit "$PROBE_EXIT"\n');docker.chmod(0o755)
            env=dict(os.environ, PATH=str(binary)+':'+os.environ['PATH'])
            for code, files in ((0,True),(2,True),(1,True),(0,False)):
                test=target/'test_fixture.py'
                if files: test.touch()
                elif test.exists(): test.unlink()
                result=subprocess.run(['bash','-c',script],cwd=root,env=dict(env,PROBE_EXIT=str(code)),capture_output=True,text=True)
                self.assertEqual(result.returncode==0, code==0 and files, result.stdout+result.stderr)

    def qualified_parent(self, parent, skills=None):
        skills=ci.SKILLS if skills is None else skills
        attempt=parent/'generation/attempt-1';attempt.mkdir(parents=True)
        source=parent/'source';context=parent/'context';source.mkdir();context.mkdir()
        (source/'reference/images').mkdir(parents=True)
        ci.save(source/'reference/images/ENV.json',{'dev_image':{'ref':'simulator'}})
        for skill in skills:
            for base in (source,context):
                folder=base/'skills'/skill;folder.mkdir(parents=True);(folder/'SKILL.md').write_text(skill)
        ci.save(parent/'catalog.json',{'required_cases':20});ci.save(parent/'task.json',{})
        ci.save(parent/'model-config.json',{'skills':{'paths':['/original/context/skills']}})
        for name in ('original-candidate.py','project/candidate.py','candidate/swi_glu.py'):
            p=attempt/name;p.parent.mkdir(exist_ok=True);p.write_text('generated output')
        digest=ci.smoke.sha(attempt/'original-candidate.py')
        ci.save(attempt/'static.json',{'passed':True})
        ci.save(attempt/'lowering.json',{'status':'passed','compile_passed':20,'cases':20,
            'pyasc_commit':ci.PIN['pyasc_commit'],'runtime_bundle_sha256':ci.PIN['runtime_bundle_sha256'],
            'runtime_files_verified':114,'candidate_sha256':digest})
        ci.save(attempt/'simulator.json',{'passed':True,'cases':[{'passed':True} for _ in range(12)],'candidate_sha256':digest})
        events='\n'.join(json.dumps({'sessionID':'session','type':'tool_use','part':{'tool':'skill','state':{
            'status':'completed','input':{'name':s},'metadata':{'dir':'/original/context/skills/'+s}}}}) for s in skills)
        (attempt/'events.log').write_text(events)
        ci.save(attempt/'skill-trace.json',ci.trace(events,Path('/original/context/skills'),skills))
        ci.save(attempt/'session.json',{'session':{'id':'session','model':json.dumps({'providerID':'gateway','id':'glm-5.3'})},
            'message':[{'original':'message'}],'part':[{'original':'part'}]})
        ci.save(parent/'run.json',{'repository':'owner/repo','origin_run_id':'123','origin_run_attempt':'1',
            'ci_url':'https://github.com/owner/repo/actions/runs/123','skill_stack_commit':'model-commit',
            'opencode_version':'1.19.0','source_hashes':ci.hashes(source),'context_hashes':ci.hashes(context)})

    def test_continue_preserves_qualified_output_and_distinguishes_ci_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new'
            def restore(parent, origin):
                self.qualified_parent(parent)
                ci.save(parent/'generation/attempt-1/failure.json',{'error':'HTTP 503'})
                (parent/'package').mkdir()
                return {'status':'completed'}
            archive=io.BytesIO()
            with tarfile.open(fileobj=archive,mode='w'): pass
            identity={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':'owner/repo','GITHUB_RUN_ID':'456',
                'GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':'continuation-commit'}
            with patch.dict('os.environ',identity), patch.object(ci,'restore_checkpoint',side_effect=restore), \
                 patch.object(ci.subprocess,'check_output',return_value=archive.getvalue()), \
                 patch.object(ci,'ci_client'), patch.object(ci,'call') as model, patch.object(ci,'prepare_qualified') as prepare:
                ci.continue_qualified(root,'123')
                model.assert_not_called();prepare.assert_called_once()
                meta=ci.inputs(root)
                self.assertEqual(meta['skill_stack_commit'],'model-commit')
                self.assertEqual(meta['orchestration_commit'],'continuation-commit')
                self.assertEqual(meta['generation_ci']['origin_run_id'],'123')
                self.assertEqual(meta['origin_run_id'],'456')
                ci.check_hashes(root/'qualified-parent',ci.load(root/'continuation-lineage.json')['parent_files'])
                self.assertEqual(prepare.call_args.args[1]['candidate_sha256'],
                    ci.smoke.sha(root/'qualified-parent/generation/attempt-1/original-candidate.py'))
                self.assertFalse((root/'package').exists())
                ci.save(root/'run.json',dict(meta,skill_stack_commit='false-model-commit'))
                with self.assertRaisesRegex(ValueError,'generation attribution'): ci.inputs(root)

    def test_continue_refuses_active_submitted_unqualified_or_substituted_parent(self):
        for marker in ('active','submission-intent.json','submission.json','prepared-artifact.json','package/state.json',
                       'generation.json','bad-numerics','changed-candidate'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'new'
                def restore(parent, origin):
                    self.qualified_parent(parent)
                    if marker=='bad-numerics': ci.save(parent/'generation/attempt-1/simulator.json',{'passed':False})
                    elif marker=='changed-candidate': (parent/'generation/attempt-1/candidate/swi_glu.py').write_text('different')
                    elif marker!='active':
                        p=parent/marker;p.parent.mkdir(parents=True,exist_ok=True);ci.save(p,{})
                    return {'status':'in_progress' if marker=='active' else 'completed'}
                with patch.dict('os.environ',{'GITHUB_REPOSITORY':'owner/repo'}), \
                     patch.object(ci,'restore_checkpoint',side_effect=restore), patch.object(ci,'prepare_qualified') as prepare:
                    with self.assertRaises(ValueError): ci.continue_qualified(root,'123')
                    prepare.assert_not_called()

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

    def test_receipts_keep_the_recorded_namespace_after_skill_rename(self):
        for group in (ci.SKILLS, ci.LEGACY_SKILLS):
            hashes={'skills/'+name+'/SKILL.md':'fixture' for name in group}
            meta={'source_hashes':dict(hashes), 'context_hashes':dict(hashes)}
            self.assertEqual(ci.receipt_skills(meta), group)
            root=Path('/original/context/skills')
            events=[{'sessionID':'legacy-session', 'type':'tool_use', 'part':{'tool':'skill',
                'state':{'status':'completed', 'input':{'name':name}, 'metadata':{'dir':str(root/name)}}}}
                for name in group]
            result=ci.trace('\n'.join(map(json.dumps,events)),root,ci.receipt_skills(meta))
            self.assertEqual(result['loaded_skills'],sorted(group))
            other=ci.LEGACY_SKILLS if group==ci.SKILLS else ci.SKILLS
            meta['context_hashes']['skills/'+other[0]+'/SKILL.md']='mixed'
            with self.assertRaisesRegex(ValueError,'Mixed'): ci.receipt_skills(meta)
            del meta['context_hashes']['skills/'+other[0]+'/SKILL.md']
            del meta['context_hashes']['skills/'+group[0]+'/SKILL.md']
            with self.assertRaisesRegex(ValueError,'Missing'): ci.receipt_skills(meta)

    def test_legacy_qualified_generation_reconstructs_without_relabeling(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'GITHUB_REPOSITORY':'owner/repo'}):
            root=Path(tmp); self.qualified_parent(root,ci.LEGACY_SKILLS)
            before=ci.hashes(root)
            result=ci.qualified_generation(root)
            self.assertEqual(result['native_trace']['loaded_skills'],sorted(ci.LEGACY_SKILLS))
            self.assertEqual(before,ci.hashes(root))

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
            bindings={'skills/'+s+'/SKILL.md':'fixture' for s in ci.SKILLS}
            meta={'source_hashes':bindings,'context_hashes':bindings}
            with patch.object(ci,'inputs',return_value=meta),patch.object(ci,'qualified_binding'),patch.object(ci.smoke,'verify',return_value=prepared):
                with self.assertRaisesRegex(ValueError,'Source/wheel candidate mismatch'): ci.generated_binding(root)

if __name__=='__main__': unittest.main()
