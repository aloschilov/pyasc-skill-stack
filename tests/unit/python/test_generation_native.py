"""Offline native DB receipts; these fixtures are not live model evidence."""
from copy import deepcopy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
from generation_native import capture_empty_session, early_output_cap, validate_empty_session
from generation_plan import digest


def database_fixture(private, project, text):
    database = private/'data/opencode/opencode.db'; database.parent.mkdir(parents=True)
    with closing(sqlite3.connect(database)) as db, db:
        db.executescript('''
            CREATE TABLE project (id TEXT, worktree TEXT);
            CREATE TABLE session (id TEXT, project_id TEXT, parent_id TEXT, directory TEXT, version TEXT);
            CREATE TABLE message (id TEXT, session_id TEXT, data TEXT);
            CREATE TABLE part (session_id TEXT, message_id TEXT, data TEXT);
        ''')
        db.execute('INSERT INTO project VALUES (?,?)', ('global', str(project)))
        db.execute('INSERT INTO session VALUES (?,?,?,?,?)', ('ses_fixture', 'global', None, str(project), '1.19.0'))
        db.execute('INSERT INTO message VALUES (?,?,?)', ('msg_user', 'ses_fixture', json.dumps({
            'role':'user', 'agent':'build', 'summary':{'diffs':[]}, 'model':{'providerID':'gateway','modelID':'glm-5.3'}})))
        db.execute('INSERT INTO message VALUES (?,?,?)', ('msg_assistant', 'ses_fixture', json.dumps({
            'role':'assistant','parentID':'msg_user','agent':'build','mode':'build',
            'modelID':'glm-5.3','providerID':'gateway','path':{'cwd':str(project),'root':str(project)}})))
        db.execute('INSERT INTO part VALUES (?,?,?)', ('ses_fixture','msg_user',json.dumps({'type':'text','text':text})))
    return database


class EmptyNativeSession(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.private=Path(self.temp.name).resolve()/'private'; self.project=Path(self.temp.name).resolve()/'project'; self.project.mkdir()
        self.text='"a task with \\"quotes\\""'
        self.database=database_fixture(self.private,self.project,self.text)

    def capture(self):
        return capture_empty_session(self.private,self.project,'gateway/glm-5.3',self.text)

    def test_readonly_backup_binds_exact_empty_session_and_stays_private(self):
        before=digest(self.database.read_bytes()); value=self.capture()
        self.assertEqual(digest(self.database.read_bytes()),before)
        self.assertEqual(value['session_id'],'ses_fixture')
        self.assertEqual(value['prompt_sha256'],digest(self.text.encode()))
        self.assertEqual(value['snapshot_sha256'],digest((self.private/'empty-session.db').read_bytes()))
        self.assertEqual((self.private/'empty-session.db').stat().st_mode & 0o777,0o600)
        self.assertNotIn(self.text,json.dumps(value))
        with self.assertRaisesRegex(ValueError,'snapshot already exists'): self.capture()

    def test_additional_or_different_native_activity_is_rejected(self):
        changes=[
            "UPDATE session SET parent_id='ses_prior'",
            "UPDATE session SET version='1.20.0'",
            "UPDATE project SET worktree='/elsewhere'",
            "UPDATE message SET session_id='ses_other'",
            "UPDATE message SET data=json_set(data,'$.modelID','other') WHERE id='msg_assistant'",
            "UPDATE message SET data=json_set(data,'$.parentID','msg_other') WHERE id='msg_assistant'",
            "UPDATE message SET data=json_set(data,'$.summary.diffs',json('[{}]')) WHERE id='msg_user'",
            "UPDATE part SET data='{}'",
            "INSERT INTO part VALUES ('ses_fixture','msg_assistant','{}')",
        ]
        original=self.database.read_bytes()
        for change in changes:
            with self.subTest(change=change):
                self.database.write_bytes(original)
                with closing(sqlite3.connect(self.database)) as db, db: db.execute(change)
                with self.assertRaises(ValueError): self.capture()
                (self.private/'empty-session.db').unlink(missing_ok=True)

    def test_candidate_and_receipt_drift_cannot_be_empty_identity(self):
        (self.project/'candidate.py').write_text('unexpected')
        with self.assertRaisesRegex(ValueError,'Unexpected candidate'): self.capture()
        (self.project/'candidate.py').unlink(); value=self.capture()
        for key,new in [('project','/elsewhere'),('model','gateway/other'),('prompt_sha256','a'*64),
                ('session_id','invalid'),('snapshot_sha256','not-a-hash'),('table_counts',{'project':True,'session':1,'message':2,'part':1})]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_empty_session(dict(value,**{key:new}),self.project,'gateway/glm-5.3',self.text)

    def test_only_known_first_tool_response_cap_can_use_native_receipt(self):
        limits={'rounds_per_attempt':8}
        usage={'usage_complete':True,'error':'BudgetError: Model completion truncated at output-token cap',
            'requests':[{'status':'rejected','usage':{'finish_reason':'length'},
                'audit':{'tool_schema_sha256':'a'*64,'tool_names':['edit','read','skill','write']}}]}
        self.assertTrue(early_output_cap(usage,limits))
        auxiliary={'status':'complete','usage':{'finish_reason':'stop'},'audit':{'tool_schema_sha256':None,'tool_names':[]}}
        usage['requests'].insert(0,auxiliary);self.assertTrue(early_output_cap(usage,limits))
        for change in (lambda u:u.update(usage_complete=False),lambda u:u['requests'][-1].update(status='unknown'),
                lambda u:u['requests'][0]['audit'].update(tool_names=['read']),
                lambda u:u['requests'][0]['usage'].update(finish_reason='tool_calls'),
                lambda u:u['requests'][-1]['audit'].update(tool_names=['bash'])):
            altered=deepcopy(usage);change(altered);self.assertFalse(early_output_cap(altered,limits))
        self.assertFalse(early_output_cap(None,limits));self.assertFalse(early_output_cap({},limits))


if __name__=='__main__': unittest.main()
