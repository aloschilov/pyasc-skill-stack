"""Fake-clock regression of the existing command/watchdog integration."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import threading
import unittest
RECIPE=Path(__file__).resolve().parents[3]/'examples/tuning/swiglu'
sys.path.insert(0,str(RECIPE))
from non_tool_budget import CAPS,SessionWatchdog,watchdog_command,read_intervals,gate_type
from consumer_interface import commands

class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.work=Path(self.tmp.name);self.now=100.0;self.reports=[]
        self.owner={'attempt':'/owned/attempt','recipe_index_sha256':'a'*64,'commands':commands('/owned/recipe')}
        self.identities={123:'456'}
    def watch(self,start=100,whole=2800):
        return SessionWatchdog(self.work,self.owner,start,whole,self.reports.append,
                               identity=self.identities.get,clock=lambda:self.now)
    def live(self,action='baseline',start=850):
        row={'status':'active','action':action,'attempt':self.owner['attempt'],
             'recipe_index_sha256':self.owner['recipe_index_sha256'],'native_command':self.owner['commands'][action],
             'pid':123,'startticks':'456','seconds_limit':CAPS[action],'tool_monotonic_start':start}
        path=self.work/(action+'-clock.json');path.write_text(json.dumps(row))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def completed(self,action='baseline',start=850,end=1000,status='passed'):
        digest=self.live(action,start)
        scope=self.work/action;scope.mkdir()
        row={'action':action,'status':status,'native_command':self.owner['commands'][action],
             'tool_monotonic_start':start,'tool_monotonic_end':end,'tool_clock_sha256':digest}
        (scope/'public-receipt.json').write_text(json.dumps(row));return row
    def test_idle_before_first_request_stops_at_exact_900(self):
        watch=self.watch();self.now=999.999;self.assertIsNone(watch())
        self.now=1000;self.assertIn('900-second',watch());self.assertEqual(len(self.reports),1)
        self.now=1001;watch();self.assertEqual(len(self.reports),1)
    def test_idle_after_final_delivery_stops_same_boundary(self):
        # Prior provider responses do not reset or pause the session clock.
        watch=self.watch();self.now=500;self.assertIsNone(watch())
        self.now=999;self.assertIsNone(watch());self.now=1000;self.assertTrue(watch())
    def test_live_compare_gets_only_bounded_actual_elapsed_credit(self):
        self.live('compare-1');watch=self.watch();self.now=1200
        self.assertIsNone(watch());self.assertFalse(self.reports)
        self.now=1250;self.assertIsNone(watch())  # exactly 400 seconds credited
        self.now=1250.001;self.assertIn('invalid',watch())
    def test_active_pid_mismatch_stops(self):
        self.live();self.identities[123]='different';self.now=900
        self.assertIn('invalid',self.watch()())
    def test_vanished_owner_without_terminal_is_unknown(self):
        self.live();self.identities.clear();self.now=900
        self.assertIn('invalid',self.watch()())
    def test_completed_transition_stops_credit_at_receipt_end(self):
        self.completed();self.identities.clear();watch=self.watch();self.now=1149.999
        self.assertIsNone(watch());self.now=1150;self.assertIn('900-second',watch())
        self.assertEqual(self.reports[-1]['completed_tool_seconds'],150)
        self.assertEqual(self.reports[-1]['active_tool_seconds'],0)
    def test_terminal_publication_during_snapshot_uses_fresh_time(self):
        digest=self.live();self.now=900
        def identity(pid):
            self.now=1050;scope=self.work/'baseline';scope.mkdir()
            (scope/'public-receipt.json').write_text(json.dumps({'status':'passed','action':'baseline',
                'tool_clock_sha256':digest,'native_command':self.owner['commands']['baseline'],
                'tool_monotonic_start':850,'tool_monotonic_end':1050}))
            return None  # Owner exited after publishing the receipt.
        watch=SessionWatchdog(self.work,self.owner,100,2800,self.reports.append,identity=identity,clock=lambda:self.now)
        self.assertIsNone(watch());self.assertFalse(self.reports)

    def test_completed_clock_mismatch_is_not_credited(self):
        self.completed();p=self.work/'baseline/public-receipt.json';row=json.loads(p.read_text())
        row['tool_monotonic_start']=849;p.write_text(json.dumps(row));self.now=900
        self.assertIn('invalid',self.watch()())
    def test_absent_to_completed_publication_between_existence_checks(self):
        from unittest.mock import patch
        clock_path=self.work/'baseline-clock.json';original_exists=Path.exists;published=[]
        def exists(path):
            if path==clock_path and not published:
                published.append(True)
                self.completed(start=850,end=900)
                return False  # Result observed immediately before publication.
            return original_exists(path)
        self.now=950
        with patch.object(Path,'exists',exists):
            self.assertIsNone(self.watch()())
        self.assertEqual(published,[True]);self.assertFalse(self.reports)
    def test_unknown_terminal_and_missing_clock_are_rejected(self):
        self.completed(status='unknown');self.now=900;self.assertIn('invalid',self.watch()())
        (self.work/'baseline-clock.json').unlink()
        with self.assertRaises(ValueError):read_intervals(self.work,self.owner,allow_live=True,identity=self.identities.get)
    def test_whole_deadline_never_extended_by_live_tool(self):
        self.live('compare-1',900);watch=self.watch(100,1200);self.now=1200
        self.assertIn('Whole session',watch())
    def test_two_active_or_overlapping_completed_tools_rejected(self):
        self.live('baseline',800);self.live('compare-1',850);self.now=900
        self.assertIn('invalid',self.watch()())
    def test_foreign_clock_not_credited(self):
        self.live();path=self.work/'baseline-clock.json';row=json.loads(path.read_text());row['attempt']='/other'
        path.write_text(json.dumps(row));self.now=900;self.assertIn('invalid',self.watch()())
    def test_watchdog_does_not_wait_on_provider_dispatch_lock(self):
        entered=threading.Event();release=threading.Event();returned=threading.Event()
        class Base:
            def __init__(self):
                self.error=None;self.closed=False;self.deadline=2800;self.records=[]
                self.limits={'output_tokens_per_response':32768,'max_output_tokens':200000}
            def checkpoint(self):pass
            def dispatch(self,body,deliver=None):
                entered.set();release.wait(2)
        watch=self.watch()
        from unittest.mock import patch
        with patch('non_tool_budget.time.monotonic',return_value=200):
            gate=gate_type(Base,self.work,lambda _:None,watchdog=watch)()
            worker=threading.Thread(target=lambda:gate.dispatch({}));worker.start();self.assertTrue(entered.wait(1))
            self.now=1000
            def poll():watch();returned.set()
            reader=threading.Thread(target=poll);reader.start()
            try:self.assertTrue(returned.wait(.2));self.assertTrue(watch.stopped)
            finally:release.set();worker.join(1);reader.join(1)
            with self.assertRaises(ValueError):gate.dispatch({})
    def test_existing_command_path_uses_idle_watchdog(self):
        watch=self.watch();self.now=1000;seen=[]
        def original(argv,directory,name,seconds,env,abort,cwd):
            seen.append(abort());raise RuntimeError('existing kill/reap path')
        with self.assertRaisesRegex(RuntimeError,'kill/reap'):
            watchdog_command(original,watch)([],self.work,'generation',2700)
        self.assertTrue(seen[0]);self.assertEqual(len(self.reports),1)
    def test_client_exit_after_budget_is_not_success(self):
        watch=self.watch()
        def original(argv,directory,name,seconds,env,abort,cwd):
            self.now=1000;return {'status':'completed','exit_code':0}
        with self.assertRaisesRegex(ValueError,'beyond'):
            watchdog_command(original,watch)([],self.work,'generation',2700)
    def test_non_generation_commands_are_not_replaced(self):
        seen=[]
        def original(*args):seen.append(args);return 'unchanged'
        watch=self.watch();self.now=1000
        self.assertEqual(watchdog_command(original,watch)([],self.work,'project-init',20),'unchanged')
        self.assertFalse(self.reports);self.assertIsNone(seen[0][5])

if __name__=='__main__':unittest.main()
