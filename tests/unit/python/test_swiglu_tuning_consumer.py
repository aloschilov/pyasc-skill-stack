"""Offline public-entry safety: no Docker, model, server or network calls."""
from pathlib import Path
import hashlib
import copy
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

RECIPE = Path(__file__).resolve().parents[3]/'examples/tuning/swiglu'
sys.path.insert(0,str(RECIPE))
spec=importlib.util.spec_from_file_location('swiglu_consumer_recipe',RECIPE/'recipe.py')
recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
from consumer_interface import installed_context_permissions, tool_config, commands, validate_bash
from native_preflight import ReplayReservations, permission_proof
from terminal_usage import classify
sys.path.insert(0,str(RECIPE.parents[2]/'tests/tools'))
from generation_provider import response_usage, request_settings
from session_driver import accounted_rounds, procedure_status


class SessionAccountingTests(unittest.TestCase):
    def response(self, root):
        (root/'private').mkdir()
        raw=json.dumps({'model':'glm-5-3-260814','choices':[{'index':0,'finish_reason':'stop'}],
            'usage':{'prompt_tokens':10,'completion_tokens':8,'total_tokens':18}}).encode()
        (root/'private/provider-response-1.bin').write_bytes(raw)
        # Use the real emitter; its schema has max_tokens, not output_limit_fields.
        settings=request_settings({'max_tokens':32768,'reasoning_effort':'low','thinking':{'type':'enabled'}})
        return {'requests':[{'status':'complete','client_delivery':{'status':'written'},'stream':False,
            'request_settings':settings,'request_sha256':'a'*64,'response_sha256':hashlib.sha256(raw).hexdigest(),
            'usage':response_usage(raw,False,'glm-5-3-260814')}]}

    def test_real_provider_settings_schema_reaches_accounted_round(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);usage=self.response(root);original=copy.deepcopy(usage)
            self.assertNotIn('output_limit_fields',usage['requests'][0]['request_settings'])
            rows=accounted_rounds(root,usage,response_usage)
            self.assertEqual(rows,[{'usage':usage['requests'][0]['usage'],'wire_output_limit_fields':{'max_tokens':32768},
                'request_sha256':'a'*64,'response_sha256':usage['requests'][0]['response_sha256']}])
            self.assertEqual(usage,original)

    def test_missing_wrong_or_untyped_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);usage=self.response(root)
            for settings in ({}, {'schema_version':1,'output_limit_fields':{'max_tokens':32768}},
                    {'schema_version':True,'max_tokens':32768}, {'schema_version':2,'max_tokens':32768},
                    {'schema_version':1,'max_tokens':32768.0}, {'schema_version':1,'max_tokens':True},
                    {'schema_version':1,'max_tokens':65536}):
                with self.subTest(settings=settings),self.assertRaises(AssertionError):
                    bad=copy.deepcopy(usage);bad['requests'][0]['request_settings']=settings
                    accounted_rounds(root,bad,response_usage)

    def test_raw_usage_and_delivery_checks_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);usage=self.response(root)
            for field,value in [('status','unknown'),('client_delivery',{'status':'pending'}),
                    ('response_sha256','0'*64),('usage',{})]:
                with self.subTest(field=field),self.assertRaises(AssertionError):
                    bad=copy.deepcopy(usage);bad['requests'][0][field]=value
                    accounted_rounds(root,bad,response_usage)
            (root/'private/provider-response-1.bin').write_bytes(b'changed')
            with self.assertRaises(AssertionError):accounted_rounds(root,usage,response_usage)

    def test_complete_accounting_and_comparison_do_not_replace_decision_tool(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);usage=self.response(root)
            self.assertEqual(len(accounted_rounds(root,usage,response_usage)),1)
            # A written model decision is not the trusted decision-tool receipt.
            (root/'decision.json').write_text(json.dumps({'selected_candidate_sha256':'a'*64}))
            receipts=[{'action':a,'status':'passed'}for a in ('baseline','variant','compare-1','compare-2','compare-3')]
            status=procedure_status(receipts)
            self.assertTrue(status['comparison_completed']);self.assertFalse(status['model_decision_bound'])
            self.assertFalse(status['completed'])
            for bad in ('failed','unknown'):
                self.assertFalse(procedure_status(receipts+[{'action':'decision','status':bad}])['completed'])
            self.assertTrue(procedure_status(receipts+[{'action':'decision','status':'passed'}])['completed'])

class NativePreflightTests(unittest.TestCase):
    def event(self,tool,data,status='completed',error=None):
        return {'part':{'tool':tool,'state':{'status':status,'input':data,'error':error}}}
    def test_real_native_permission_required_before_model_spend(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'native.stdout'
            events=[self.event('read',{'filePath':'/project/baseline.py'}),self.event('read',{'filePath':'/context/ref.md'})]
            p.write_text('\n'.join(map(json.dumps,events)))
            with self.assertRaises(ValueError):permission_proof(p,'/project/baseline.py','/context/ref.md','printf blocked')
            events.append(self.event('bash',{'command':'printf blocked'},'error','The user has specified a rule which prevents you from using this specific tool call.'))
            p.write_text('\n'.join(map(json.dumps,events)))
            self.assertEqual(len(permission_proof(p,'/project/baseline.py','/context/ref.md','printf blocked')),3)
            events[-1]['part']['state']['status']='completed'
            p.write_text('\n'.join(map(json.dumps,events)))
            with self.assertRaises(ValueError):permission_proof(p,'/project/baseline.py','/context/ref.md','printf blocked')
    def test_duplicate_reservation_blocks_without_redelivery(self):
        gate=ReplayReservations();gate.reserve({'messages':['same']});gate.release()
        with self.assertRaisesRegex(ValueError,'Duplicate'):gate.reserve({'messages':['same']})
        with self.assertRaisesRegex(ValueError,'stopped'):gate.reserve({'messages':['new']})
    def test_overlapping_handler_stops_sequence(self):
        gate=ReplayReservations();gate.reserve({'id':1});errors=[]
        def second():
            try:gate.reserve({'id':2})
            except ValueError as exc:errors.append(str(exc))
        thread=threading.Thread(target=second);thread.start();thread.join(1)
        self.assertEqual(len(errors),1);self.assertTrue(gate.failed);gate.release()
        with self.assertRaises(ValueError):gate.reserve({'id':3})
    def test_documented_timeouts_exact(self):
        for action,timeout in [('baseline',300000),('variant',300000),('compare-1',600000),('compare-2',600000),('compare-3',600000),('decision',300000)]:
            command=commands('/recipe')[action]
            self.assertEqual(validate_bash({'command':command,'timeout':timeout},'/recipe','/project'),action)
            for wrong in (None,1000,120000,timeout-1,timeout+1,570000):
                if wrong==timeout:continue
                with self.subTest(action=action,timeout=wrong),self.assertRaises(ValueError):
                    validate_bash({'command':command,'timeout':wrong},'/recipe','/project')

class TerminalUsageTests(unittest.TestCase):
    def setup_response(self,root,finish='length',reasoning=5):
        (root/'private').mkdir()
        raw=json.dumps({'model':'glm-5-3-260814','choices':[{'index':0,'finish_reason':finish}],
                        'usage':{'prompt_tokens':10,'completion_tokens':8,'total_tokens':18,'completion_tokens_details':{'reasoning_tokens':reasoning}}}).encode()
        (root/'private/provider-response-1.bin').write_bytes(raw)
        parsed=response_usage(raw,False,'glm-5-3-260814')
        row={'status':'rejected','stream':False,'response_sha256':hashlib.sha256(raw).hexdigest(),'usage':parsed,
             'validation_error':'Model completion truncated at output-token cap','client_delivery':{'status':'pending'}}
        usage={'usage_complete':True,'error':'BudgetError: Model completion truncated at output-token cap','requests':[row]}
        (root/'usage.json').write_text(json.dumps(usage));return usage
    def test_complete_length_rejection_known_no_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();self.setup_response(root);result=classify(root,response_usage)
            self.assertTrue(result['outcome_known']);self.assertEqual(result['completion_tokens'],8)
    def test_ambiguous_client_delivery_preserves_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();usage=self.setup_response(root,'stop');row=usage['requests'][0]
            row.update(status='complete',client_delivery={'status':'failed'});usage['error']='Client response delivery failed: ValueError'
            (root/'usage.json').write_text(json.dumps(usage))
            self.assertFalse(classify(root,response_usage)['outcome_known'])
            row['tool_preexecution']={'status':'rejected_before_client_delivery','model_tool_executed':False}
            (root/'usage.json').write_text(json.dumps(usage))
            self.assertTrue(classify(root,response_usage)['outcome_known'])
    def test_bad_hash_and_invalid_usage_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();usage=self.setup_response(root)
            raw=root/'private/provider-response-1.bin';raw.write_bytes(raw.read_bytes()+b' ')
            self.assertFalse(classify(root,response_usage)['outcome_known'])
            usage['requests'][0]['response_sha256']=recipe.sha(raw)
            body=json.loads(raw.read_text());body['usage']['completion_tokens_details']['reasoning_tokens']=9
            raw.write_text(json.dumps(body));usage['requests'][0]['response_sha256']=recipe.sha(raw)
            (root/'usage.json').write_text(json.dumps(usage))
            self.assertFalse(classify(root,response_usage)['outcome_known'])
    def test_no_usage_after_client_started_is_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();self.assertTrue(classify(root,response_usage)['outcome_known'])
            (root/'generation.command.json').write_text('{}')
            self.assertFalse(classify(root,response_usage)['outcome_known'])

    def test_stale_usage_and_extra_responses_preserve_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();usage=self.setup_response(root)
            (root/'usage.json').write_text(json.dumps({'requests':[],'error':None,'usage_complete':False}))
            self.assertEqual(classify(root,response_usage)['kind'],'raw-response-inventory-mismatch')
            (root/'usage.json').write_text(json.dumps(usage))
            (root/'private/provider-response-2.bin').write_bytes(b'original extra response')
            self.assertFalse(classify(root,response_usage)['outcome_known'])
    def test_zero_usage_after_client_started_is_not_before_provider(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'usage.json').write_text(json.dumps({'requests':[],'error':None,'usage_complete':False}))
            (root/'generation.command.json').write_text(json.dumps({'status':'blocked','exit_code':-9}))
            self.assertFalse(classify(root,response_usage)['outcome_known'])

    def test_retire_only_known_owned_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();attempt=root/'attempt';attempt.mkdir();self.setup_response(attempt)
            (attempt/'generation.command.json').write_text(json.dumps({'status':'blocked','exit_code':-9,'pid':987654}))
            lease=root/'active.json';lease.write_text(json.dumps({'run_dir':str(attempt),'status':'active'}))
            with patch.object(recipe.os,'killpg',side_effect=ProcessLookupError):
                recipe.retire_owned(root,{},attempt,lease,provider=True)
            self.assertFalse(lease.exists());self.assertTrue((root/'owner-cleanup.json').exists())
    def test_unknown_cleanup_never_retires(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();attempt=root/'attempt';attempt.mkdir();usage=self.setup_response(attempt)
            usage['usage_complete']=False;(attempt/'usage.json').write_text(json.dumps(usage))
            (attempt/'generation.command.json').write_text(json.dumps({'status':'blocked','exit_code':-9,'pid':987654}))
            lease=root/'active.json';lease.write_text(json.dumps({'run_dir':str(attempt),'status':'active'}))
            with patch.object(recipe.os,'killpg',side_effect=ProcessLookupError),self.assertRaisesRegex(ValueError,'unresolved'):
                recipe.retire_owned(root,{},attempt,lease,provider=True)
            self.assertTrue(lease.exists());self.assertFalse((root/'owner-cleanup.json').exists())
    def test_run_cli_fails_on_known_incomplete_procedure(self):
        import argparse
        args=argparse.Namespace(action='run',directory=Path('/prepared'),preflight=Path('/preflight'))
        for status,code in [('failed',1),('unresolved',1),('blocked',2)]:
            with patch.object(recipe,'parser') as parser,patch.object(recipe,'run_consumer',return_value={'status':status}),patch('builtins.print'):
                parser.return_value.parse_args.return_value=args
                with self.assertRaises(SystemExit) as caught:recipe.main()
                self.assertEqual(caught.exception.code,code)
        with patch.object(recipe,'parser') as parser,patch.object(recipe,'run_consumer',return_value={'status':'completed'}),patch('builtins.print'):
            parser.return_value.parse_args.return_value=args;recipe.main()


class SourcePreparationTests(unittest.TestCase):
    def test_all_code_includes_native_fixture_and_accounting(self):
        self.assertIn('native_preflight.py',recipe.CODE);self.assertIn('terminal_usage.py',recipe.CODE)
    def test_installer_real_links_all_context_files_readable(self):
        # Same real installer as prepare; no network and no mocked filesystem.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();revision=recipe.git('rev-parse','HEAD').decode().strip()
            context=recipe.install_context(root,revision);project=root/'attempt/project'
            pins=recipe.load(root/'context-hashes.json');self.assertEqual(len(pins),29)
            config={'permission':{'read':{'*':'deny'},'external_directory':{'*':'deny'}}}
            config=installed_context_permissions(config,context,project,recipe.SKILLS)
            config=tool_config(config,root/'recipe',project)
            for name in pins:
                alias=context/name;canonical=alias.resolve()
                for spelling in (str(alias),str(canonical),os.path.relpath(alias,project),os.path.relpath(canonical,project)):
                    self.assertEqual(config['permission']['read'][spelling],'allow')
            self.assertEqual(config['permission']['read']['*'],'deny')
            self.assertNotIn(str(root/'*'),config['permission']['external_directory'])
            self.assertNotIn(str(root/'private/*'),config['permission']['external_directory'])
    def test_entry_revision_and_recipe_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'recipe').mkdir();(root/'context').mkdir()
            (root/'recipe/payload.py').write_text('unchanged')
            recipe.save(root/'recipe/recipe-input-hashes.json',{'payload.py':recipe.sha(root/'recipe/payload.py')})
            recipe.save(root/'context-hashes.json',{})
            data={'checkout':str(recipe.CHECKOUT),'directory':str(root),'entry_sha256':recipe.sha(RECIPE/'recipe.py'),
                  'checkout_revision':recipe.git('rev-parse','HEAD').decode().strip(),
                  'recipe_index_sha256':recipe.sha(root/'recipe/recipe-input-hashes.json'),
                  'common_checkout_sources':{},'context_index_sha256':recipe.sha(root/'context-hashes.json')}
            (root/'prepared.json').write_text(json.dumps(data));recipe.verify(root)
            data['entry_sha256']='0'*64;(root/'prepared.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'entry'):recipe.verify(root)
            data['entry_sha256']=recipe.sha(RECIPE/'recipe.py');data['checkout_revision']='0'*40
            (root/'prepared.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'revision'):recipe.verify(root)
            data['checkout_revision']=recipe.git('rev-parse','HEAD').decode().strip();(root/'prepared.json').write_text(json.dumps(data))
            (root/'recipe/payload.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'source changed'):recipe.verify(root)
    def test_engine_inventory_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();archive=root/'engine.tar.gz';index={'engine/payload.py':'a'*64}
            with tarfile.open(archive,'w:gz') as out:
                for name,content in [('source-index.json',json.dumps(index).encode()),('README.md',b'doc')]:
                    member=tarfile.TarInfo(name);member.size=len(content);out.addfile(member,io.BytesIO(content))
                member=tarfile.TarInfo('engine/payload.py');member.type=tarfile.SYMTYPE;member.linkname='/etc/passwd';out.addfile(member)
            lock={'sha256':recipe.sha(archive),'bytes':archive.stat().st_size,'source_files':index}
            with self.assertRaisesRegex(ValueError,'Unsafe'):recipe.extract_sources(archive,root/'out',lock)


if __name__=='__main__':unittest.main()
