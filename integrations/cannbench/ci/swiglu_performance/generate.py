"""CI generation in an empty OpenCode project. Retain the original model-written bytes."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from geometry import validate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import single_operator as ci


def ended_at_limit(events):
    reasons=[]
    for line in events.splitlines():
        try:
            event=json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get('type')=='error':
            return False
        if event.get('type')=='step_finish':
            reasons.append(event['part'].get('reason'))
    return bool(reasons) and reasons[-1]=='length'


def pre_submission_overflow(jobs):
    evaluations = [j for j in jobs if j['name'] == 'swiglu-performance / evaluate']
    if len(evaluations) != 1 or evaluations[0]['conclusion'] != 'failure':
        raise ValueError('Expected a failed evaluation')
    job, = evaluations
    steps = {s['name']:s['conclusion'] for s in job['steps']}
    if (steps.get('Run numerical CaModel controls and lower the same bytes to AscendC') != 'failure'
            or steps.get('Submit once and collect service timings') != 'skipped'):
        raise ValueError('Policy recovery requires failure during CaModel before submission')
    return job['id']


def recover_kernel(root, origin, context, prompt, kernel_profile):
    """Recover unchanged compute after a policy length stop or a confirmed pre-submit UB overflow."""
    if not re.fullmatch(r'[1-9][0-9]*',origin):
        raise ValueError('Invalid kernel origin')
    repo=os.environ['GITHUB_REPOSITORY']
    run=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/actions/runs/{origin}']))
    if not (run['head_repository']['full_name']==repo and run['event']=='workflow_dispatch'
            and run['path']=='.github/workflows/ci.yml' and run['conclusion']=='failure'):
        raise ValueError('Expected a failed in-repository CI policy generation')
    artifact=f"swiglu-generation-{origin}-{run['run_attempt']}"
    prior=root/'original-kernel'
    subprocess.run(['gh','run','download',origin,'--repo',repo,'--name',artifact,'--dir',str(prior)],check=True)
    profile=ci.load(prior/'project/opencode.json')
    prior_skills=Path(profile['skills']['paths'][0])
    ci.trace((prior/'events.log').read_text(),prior_skills)
    prior_prompt=(prior/'prompt.txt').read_text().replace(str(prior_skills.parent/'task'),str(context/'task'))
    if prior_prompt!=prompt or ci.hashes(prior/'context')!=ci.hashes(context):
        raise ValueError('Prior kernel context or prompt differs')
    original_profile_path = prior/'kernel-profile.json'
    if not original_profile_path.exists():
        original_profile_path = prior/'project/opencode.json'
    original_profile = ci.load(original_profile_path)
    if original_profile['model']!=kernel_profile['model'] or original_profile['provider']!=kernel_profile['provider']:
        raise ValueError('Prior kernel model profile differs')
    origin_details = {}
    if (prior/'project/policy.json').exists():
        jobs = json.loads(subprocess.check_output(
            ['gh','api',f"repos/{repo}/actions/runs/{origin}/attempts/{run['run_attempt']}/jobs"]))['jobs']
        job_id = pre_submission_overflow(jobs)
        log = subprocess.check_output(['gh','run','view',origin,'--repo',repo,'--job',str(job_id),
                                       '--log-failed']).decode(errors='replace')
        match = re.search(r'UB overflow: (\d+) bytes are available, (\d+) bytes are used\.', log)
        if not match:
            raise ValueError('The CaModel log does not confirm a UB overflow')
        receipt = ci.load(prior/'generation.json')
        if (str(receipt['run_id'])!=origin or receipt['revision']!=run['head_sha']
                or receipt['policy_sha256']!=ci.smoke.sha(prior/'project/policy.json')
                or receipt['candidate_sha256']!=ci.smoke.sha(prior/'project/candidate.py')):
            raise ValueError('Previous generation receipt differs')
        origin_details = {'failed_policy_sha256':receipt['policy_sha256'],'camodel_error':match.group()}
    elif not ended_at_limit((prior/'policy-continuation.log').read_text()):
        raise ValueError('Missing confirmed policy length stop')
    # Bind the bytes in the retained file to OpenCode's completed write in the native session.
    trace=ci.load(prior/'session.json')
    authored=[]
    for entry in trace['part']:
        part=json.loads(entry['data'])
        state=part.get('state',{})
        args=state.get('input',{})
        if part.get('tool')=='write' and state.get('status')=='completed' and Path(args.get('filePath','')).name=='candidate.py':
            authored.append(args['content'].encode())
    candidate=(prior/'project/candidate.py').read_bytes()
    if candidate not in authored:
        raise ValueError('Retained kernel differs from its model write')
    (root/'project/candidate.py').write_bytes(candidate)
    return {'run':origin,'attempt':run['run_attempt'],'revision':run['head_sha'],
            'candidate_sha256':ci.smoke.sha(root/'project/candidate.py'),
            'artifact':artifact,'profile_sha256':ci.smoke.sha(original_profile_path),**origin_details}


def main(root):
    identity = ci.ci_identity()
    root.mkdir(parents=True, exist_ok=False)
    context = root / 'context'
    project = root / 'project'
    project.mkdir()
    skill_root = Path(os.environ.get('SWIGLU_SKILL_SOURCE') or ci.ROOT)
    skill_revision = subprocess.check_output(['git','-C',str(skill_root),'rev-parse','HEAD'],text=True).strip()
    for name in ci.SKILLS:
        shutil.copytree(skill_root/'skills'/name, context/'skills'/name)
    catalog, task = ci.smoke.task_snapshot(ci.ci_client())
    ci.save(root/'catalog.json', catalog)
    ci.save(root/'task.json', task)
    task_dir = context/'task'
    task_dir.mkdir()
    (task_dir/'golden.py').write_text(task['golden_source'])
    ci.save(task_dir/'cases.yaml', task['cases_yaml'])
    ci.save(task_dir/'proto.json', task['proto'])
    prompt = (HERE/'prompt.txt').read_text().format(task=task_dir)
    (root/'prompt.txt').write_text(prompt)
    profile = ci.load(ci.ROOT/'integrations/opencode/profiles/cloud-glm-5.3.json')
    profile['provider']['gateway']['options']['baseURL'] = os.environ['OPENCODE_BASE_URL']
    profile['skills'] = {'paths':[str(context/'skills')]}
    profile['permission'] = ci.worker_permissions(context)
    for path in ('policy.json','**/policy.json'):
        profile['permission']['edit'][path] = 'allow'
    profile['agent'] = {'build':{'steps':32}}
    ci.save(project/'opencode.json', profile)
    private = root.parent/(root.name+'-private')
    private.mkdir(mode=0o700)
    env = {k:v for k,v in os.environ.items() if k not in (
        'GH_TOKEN','GITHUB_TOKEN','BENCHSITE_API_TOKEN','CANNBENCH_AGGREGATION_TOKEN','QUALIFICATION_PUBLISH_TOKEN')}
    env.update(XDG_DATA_HOME=str(private), XDG_CONFIG_HOME=str(private/'config'),
               OPENCODE_CONFIG_CONTENT=json.dumps(profile), OPENCODE_DISABLE_EXTERNAL_SKILLS='1',
               OPENCODE_DISABLE_CLAUDE_CODE_SKILLS='1')
    version = ci.call(['opencode','--version'],root/'version.log',env=env).strip()
    command=['opencode','run','--pure','--format','json','--dir',str(project),'-m',ci.MODEL]
    database=private/'opencode/opencode.db'

    def finish_file(path,log,previous):
        if not path.is_file() and ended_at_limit(previous):
            reminder=f'The preceding response reached its output limit. Use the write tool now to create {path.name} from the instructions already loaded.'
            ci.call([*command,reminder],root/log,cwd=project,env=env,timeout=1200)
            ci.capture_session(database,session,root/'session.json')
        if not path.is_file():
            raise ValueError(f'Missing {path.name}; inspect the retained session and finish reason')

    origin=os.environ.get('KERNEL_ORIGIN')
    candidate=project/'candidate.py'
    lineage={}
    if origin:
        lineage['kernel_origin']=recover_kernel(root,origin,context,prompt,profile)
        session=None
    else:
        events=ci.call([*command,prompt],root/'events.log',cwd=project,env=env,timeout=1200)
        trace=ci.trace(events,context/'skills')
        session=trace['session_id']
        ci.capture_session(database,session,root/'session.json')
        command.extend(['--session',session])
        finish_file(candidate,'kernel-continuation.log',events)
    before=ci.smoke.sha(candidate)
    policy_prompt=(HERE/'policy-prompt.txt').read_text()
    if origin:
        policy_prompt='Load the four native skills '+', '.join(ci.SKILLS)+'. Read candidate.py. '+policy_prompt
        if 'camodel_error' in lineage['kernel_origin']:
            old_policy = (root/'original-kernel/project/policy.json').read_text()
            policy_prompt += ('\nRepair the previous policy. Its first CaModel compiler error was '
                              + lineage['kernel_origin']['camodel_error'] + '\nPrevious table:\n' + old_policy)
    (root/'policy-prompt.txt').write_text(policy_prompt)
    policy_profile=json.loads(json.dumps(profile))
    low=ci.load(ci.ROOT/'integrations/opencode/profiles/cloud-glm-5.3-low.json')
    policy_profile['provider']['gateway']['models']=low['provider']['gateway']['models']
    ci.save(root/'policy-profile.json',policy_profile)
    # Project config is merged after other sources by the CLI. Keep both identical.
    ci.save(root/'kernel-profile.json',profile)
    ci.save(project/'opencode.json',policy_profile)
    env['OPENCODE_CONFIG_CONTENT']=json.dumps(policy_profile)
    response=ci.call([*command,policy_prompt],root/'policy-events.log',cwd=project,env=env,timeout=1200)
    if origin:
        trace=ci.trace(response,context/'skills')
        session=trace['session_id']
        command.extend(['--session',session])
    ci.capture_session(database,session,root/'session.json')
    policy=project/'policy.json'
    finish_file(policy,'policy-continuation.log',response)
    if ci.smoke.sha(candidate)!=before:
        raise ValueError('Policy generation changed the frozen compute kernel')
    validate(ci.load(policy))
    ci.save(root/'generation.json', {'run_id':identity['GITHUB_RUN_ID'], 'run_attempt':identity['GITHUB_RUN_ATTEMPT'],
            'revision':identity['GITHUB_SHA'], 'candidate_sha256':ci.smoke.sha(candidate),
            **lineage, 'kernel_profile_sha256':ci.smoke.sha(root/'kernel-profile.json'),
            'policy_profile':'cloud-glm-5.3-low', 'policy_prompt_sha256':ci.smoke.sha(root/'policy-prompt.txt'),
            'skills_revision':skill_revision, 'policy_sha256':ci.smoke.sha(policy),
            'prompt_sha256':ci.smoke.sha(root/'prompt.txt'), 'model':ci.MODEL, 'opencode':version,
            'profile_sha256':ci.smoke.sha(project/'opencode.json'), 'session_id':trace['session_id'],
            'loaded_skills':trace['loaded_skills'], 'context_hashes':ci.hashes(context)})
    print('Generated candidate', ci.smoke.sha(candidate))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
