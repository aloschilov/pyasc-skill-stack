"""CI generation in an empty OpenCode project. Retain the original model-written bytes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from geometry import validate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import single_operator as ci


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
    out = ci.call(['opencode','run','--pure','--format','json','--dir',str(project),'-m',ci.MODEL,prompt],
                  root/'events.log',cwd=project,env=env,timeout=1200)
    trace = ci.trace(out, context/'skills')
    ci.capture_session(private/'opencode/opencode.db', trace['session_id'], root/'session.json')
    candidate = project/'candidate.py'
    if not candidate.is_file():
        raise ValueError('OpenCode completed without writing candidate.py; see its retained session')
    policy = project/'policy.json'
    validate(ci.load(policy))
    ci.save(root/'generation.json', {'run_id':identity['GITHUB_RUN_ID'], 'run_attempt':identity['GITHUB_RUN_ATTEMPT'],
            'revision':identity['GITHUB_SHA'], 'candidate_sha256':ci.smoke.sha(candidate),
            'skills_revision':skill_revision, 'policy_sha256':ci.smoke.sha(policy),
            'prompt_sha256':ci.smoke.sha(root/'prompt.txt'), 'model':ci.MODEL, 'opencode':version,
            'profile_sha256':ci.smoke.sha(project/'opencode.json'), 'session_id':trace['session_id'],
            'loaded_skills':trace['loaded_skills'], 'context_hashes':ci.hashes(context)})
    print('Generated candidate', ci.smoke.sha(candidate))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
