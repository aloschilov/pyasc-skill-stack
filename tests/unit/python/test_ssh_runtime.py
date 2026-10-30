import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
import ssh_runtime as ssh
import ssh_worker as worker
import device_identity


class SSHRuntime(unittest.TestCase):
    def args(self, kernel):
        return argparse.Namespace(kernel_py=str(kernel), ssh_target='npu',
            ssh_remote_dir='/tmp/pyasc-test', ssh_python='python3', ssh_env='/opt/CANN env.sh',
            backend='NPU', mode='execute', platform=None, timeout=5,
            jit_scalar_args='{}', no_prove_numerics=False, project_dir=None)

    def test_payload_excludes_credentials_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); project=root/'source'; project.mkdir()
            kernel=project/'kernel.py'; kernel.write_text('import helper\n')
            (project/'helper.py').write_text('VALUE=1\n')
            (project/'.secrets.py').write_text('secret')
            (project/'.env').write_text('secret')
            (project/'results.json').write_text('private output')
            relative, hashes = ssh.stage(root/'payload', kernel)
            self.assertEqual(relative, 'project/kernel.py')
            self.assertEqual(set(p for p in hashes if p.startswith('project/')),
                             {'project/kernel.py','project/helper.py'})
            self.assertEqual(kernel.read_text(), 'import helper\n')
            (project/'escape.py').symlink_to(kernel)
            with self.assertRaisesRegex(ValueError, 'Symlinks'):
                ssh.stage(root/'rejected', kernel)

    def test_unreachable_is_skip_without_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            kernel=Path(tmp)/'kernel.py';kernel.write_text('pass')
            with patch.object(ssh.subprocess,'run',return_value=subprocess.CompletedProcess([],255,b'',b'Permission denied')) as call:
                code, report=ssh.run(self.args(kernel))
            self.assertEqual(code,2); self.assertEqual(report['status'],'SKIP')
            self.assertIn('Permission denied',report['detail']);call.assert_called_once()

    def test_transport_forwarding_tar_and_rsync(self):
        for rsync in (True,False):
            with self.subTest(rsync=rsync), tempfile.TemporaryDirectory() as tmp:
                kernel=Path(tmp)/'kernel.py';kernel.write_text('pass')
                expected={'status':'PASS','verification_level':'numerically_verified','launcher_calls':1,
                          'detail':'original detail','device_identity':{'status':'ready'},
                          'evidence':{'ssh':{'payload_sha256':{'project/kernel.py':'hash'},'inputs_unchanged':True}}}
                responses=[subprocess.CompletedProcess([],0,b'/usr/bin/rsync\n' if rsync else b'',b''),
                           subprocess.CompletedProcess([],0,b'',b''),subprocess.CompletedProcess([],0,b'banner',b''),
                           subprocess.CompletedProcess([],0,json.dumps(expected).encode(),b'')]
                with patch.object(ssh,'stage',return_value=('project/kernel.py',{'project/kernel.py':'hash'})), \
                     patch.object(ssh.shutil,'which',return_value='/usr/bin/rsync' if rsync else None), \
                     patch.object(ssh.subprocess,'run',side_effect=responses) as call:
                    code,report=ssh.run(self.args(kernel))
                self.assertEqual(code,0);self.assertEqual(report['detail'],'original detail')
                self.assertEqual(report['launcher_calls'],1)
                self.assertEqual(report['evidence']['ssh']['transport'],'rsync' if rsync else 'tar')
                self.assertIn('cat ',call.call_args_list[-1].args[0][-1])

    def test_timeouts_distinguish_transport_and_execution(self):
        for fail_at in (0,2):
            with tempfile.TemporaryDirectory() as tmp:
                kernel=Path(tmp)/'kernel.py';kernel.write_text('pass')
                responses=[subprocess.CompletedProcess([],0,b'',b'')]*fail_at
                responses += [subprocess.TimeoutExpired(['ssh'],10)]
                with patch.object(ssh.subprocess,'run',side_effect=responses), patch.object(ssh.shutil,'which',return_value=None):
                    code,report=ssh.run(self.args(kernel))
                self.assertEqual(code,1 if fail_at==2 else 2)
                self.assertIsNone(report['verification_level'])

    def request(self, root):
        (root/'project').mkdir(); (root/'project/kernel.py').write_text('pass')
        return {'kernel':'project/kernel.py','sha256':worker.hashes(root,['project/kernel.py']),
                'platform':None,'mode':'execute','timeout':5,'scalar_args':'{}','no_prove_numerics':False}

    def test_worker_uses_detected_soc_and_keeps_control_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);request=self.request(root)
            original={'status':'PASS','verification_level':'numerically_verified','launcher_calls':1,
                      'evidence':{'control':{'exit_code':1}},'detail':'verified'}
            with patch.object(worker,'probe',return_value={'status':'ready','soc_version':'Ascend950PR_9579'}), \
                 patch.object(worker,'run_host',return_value=(0,json.dumps(original),'')) as run:
                code,report=worker.execute(root,request)
            self.assertEqual(code,0)
            self.assertEqual(report['evidence']['control'],{'exit_code':1})
            command=run.call_args.args[0]
            self.assertEqual(command[command.index('--platform')+1],'Ascend950PR_9579')
            self.assertEqual(command[command.index('--backend')+1],'NPU')

    def test_worker_rejects_mismatch_corruption_and_mutation(self):
        for failure in ('platform','payload','mutation','malformed'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);request=self.request(root)
                if failure=='platform':request['platform']='Ascend950PR_9599'
                if failure=='payload':request['sha256']['project/kernel.py']='wrong'
                def run(*args):
                    if failure=='mutation':(root/'project/kernel.py').write_text('modified')
                    return 0, ('no report' if failure=='malformed' else json.dumps(
                        {'status':'PASS','verification_level':'executed'})), ''
                with patch.object(worker,'probe',return_value={'status':'ready','soc_version':'Ascend950PR_9579'}), \
                     patch.object(worker,'run_host',side_effect=run) as call:
                    code,report=worker.execute(root,request)
                self.assertEqual(code,1);self.assertEqual(report['status'],'FAIL')
                if failure in ('platform','payload'):call.assert_not_called()

    def test_no_hardware_identity_is_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); request=self.request(root)
            with patch.object(worker,'probe',return_value={'status':'unavailable','reason':'no NPU'}), \
                 patch.object(worker,'run_host') as run:
                code,report=worker.execute(root,request)
            self.assertEqual(code,2);self.assertEqual(report['status'],'SKIP');run.assert_not_called()

    def test_toolkit_version_from_selected_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ascend_toolkit_install.info';path.write_text('version=9.2.0\n')
            with patch.dict('os.environ',{'ASCEND_HOME_PATH':tmp}):
                self.assertEqual(device_identity.toolkit_info()['version'],'9.2.0')


if __name__=='__main__':unittest.main()
