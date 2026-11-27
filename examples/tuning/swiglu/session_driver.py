"""One native OpenCode tuning session, owned by the public recipe entry.

Reuses the checkout's shared executor lock, provider accounting and client
cleanup. The entry derives admission metadata and retires only known outcomes.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
from non_tool_budget import gate_type, SessionWatchdog, watchdog_command
from tool_delivery import audit as delivery_audit
from consumer_interface import LIMITS, tool_config, request_audit, audit_events, native_source_binding, installed_context_permissions, commands


if not __debug__:raise RuntimeError('Assertions must remain enabled')

def run_once(checkout, recipe, attempt, context, ledger, plan, profile_config):
    checkout,recipe,attempt,context,ledger=map(lambda p:Path(p).resolve(),(checkout,recipe,attempt,context,ledger))
    # Caller pins this checkout and full recipe in the immutable admission.
    sys.path.insert(0,str(checkout/'tests/tools'))
    from generation_executor import configure,run_session,exclusive,native_trace,save
    from generation_provider import response_usage
    from generation_protocols import skill_body
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    admission=json.loads((recipe/'admission.json').read_text())
    local=json.loads((recipe/'session-config.json').read_text())
    assert admission['status']=='ADMITTED' and sha(recipe/'session-config.json')==admission['session_config_sha256']
    assert str(attempt)==local['attempt'] and str(ledger)==local['ledger']
    assert not attempt.exists(),'New session only; unresolved request is never replayed'
    assert plan['model']['profile']=='cloud-glm-5.3-low' and plan['model']['expected_response']=='glm-5-3-260814'
    provider,model=profile_config['model'].split('/')
    assert profile_config['provider'][provider]['models'][model]['limit']['output']==32768
    plan['limits'].update(LIMITS)
    assert time.time()+2700<admission['deadline_epoch'],'Insufficient frozen session window'
    with exclusive(ledger,attempt) as lease:
        attempt.mkdir(mode=0o700);project=Path(local['project']);assert project==attempt/'project';project.mkdir()
        shutil.copyfile(recipe/'baseline.py',project/'baseline.py')
        shutil.copyfile(recipe/'TUNING_TASK.md',project/'TUNING_TASK.md')
        (project/'baseline.py').chmod(0o444);(project/'TUNING_TASK.md').chmod(0o444)
        config=configure(profile_config,plan,context,project)
        config['agent'].update(title={'disable':True},summary={'disable':True})
        config['permission'].update(glob='deny',grep='deny',read={'*':'deny'})
        skillmap={name:context/'skills'/name for name in plan['skills']}
        config=installed_context_permissions(config,context,project,plan['skills'])
        config=tool_config(config,recipe,project)
        prompt=(recipe/'prompt.txt').read_text().replace('@RECIPE@',str(recipe))
        save(attempt/'tuning-plan.json',{'limits':plan['limits'],'recipe_index_sha256':sha(recipe/'recipe-input-hashes.json'),'source_profile_sha256':plan['model']['profile_sha256'],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'tool_access_equal_between_arms':True})
        def observe(number,body,payload):
            # This callback runs before each provider POST under BudgetGate.
            # A known recipe failure remains evidence; unknown cleanup blocks
            # another provider request and preserves the existing shared latch.
            for p in Path(local['work']).glob('*/public-receipt.json'):
                assert json.loads(p.read_text())['status']!='unknown','Unresolved owned evaluator outcome'
            audit=request_audit(body)
            def strings(v):
                if isinstance(v,str):yield v
                elif isinstance(v,list):
                    for x in v:yield from strings(x)
                elif isinstance(v,dict):
                    for x in v.values():yield from strings(x)
            content='\n'.join(strings(body['messages']))
            audit['visible_skills']=[n for n,p in skillmap.items()if skill_body((p/'SKILL.md').read_text())in content]
            return audit
        started=time.monotonic();whole=started+2700
        save(attempt/'session-clock.json',{'run_dir':str(attempt),'started_monotonic':started,'deadline_monotonic':whole,'whole_session_seconds':2700})
        ownership={'attempt':str(attempt),'recipe_index_sha256':sha(recipe/'recipe-input-hashes.json'),'commands':commands(recipe)}
        watchdog=SessionWatchdog(local['work'],ownership,started,whole,lambda report:save(attempt/'non-tool-watchdog.json',report))
        import generation_executor
        original_gate=generation_executor.BudgetGate;original_command=generation_executor.command
        generation_executor.BudgetGate=gate_type(original_gate,local['work'],lambda report:save(attempt/'non-tool-budget.json',report),
                lambda raw,body:delivery_audit(raw,bool(body.get('stream')),{t['function']['name']for t in body.get('tools',[])},recipe,project),
                ownership=ownership,session_clock=(started,whole),watchdog=watchdog)
        generation_executor.command=watchdog_command(original_command,watchdog)
        try:
            result,usage=run_session(attempt,plan,project,config,prompt,observe_request=observe)
        finally:
            generation_executor.BudgetGate=original_gate
            generation_executor.command=original_command
        # No synthetic transcript: inspect native terminal events and link each
        # actual executed command to the receipt returned to the model.
        assert result['status']=='completed' and result['exit_code']==0
        assert usage['error']is None and usage['usage_complete']is True
        events=[json.loads(x)for x in (attempt/'generation.stdout').read_text().splitlines()]
        trace=native_trace((attempt/'generation.stdout').read_text(),skillmap,allow_client_error=False)
        receipts=audit_events(events,recipe,project,local['work'])
        source_binding=native_source_binding(events,project,local['work'])
        rounds=[]
        for i,row in enumerate(usage['requests'],1):
            raw=attempt/'private'/f'provider-response-{i}.bin'
            assert row['status']=='complete' and row['client_delivery']['status']=='written' and sha(raw)==row['response_sha256']
            assert response_usage(raw.read_bytes(),row['stream'],'glm-5-3-260814')==row['usage']
            assert row['request_settings']['output_limit_fields']=={'max_tokens':32768}
            rounds.append({'usage':row['usage'],'wire_output_limit_fields':row['request_settings']['output_limit_fields'],'request_sha256':row['request_sha256'],'response_sha256':row['response_sha256']})
        performed={r['action']:r['status']for r in receipts if 'action'in r}
        procedure={'baseline_qualified':performed.get('baseline')=='passed','variant_attempt_recorded':'variant'in performed,'comparison_completed':performed.get('compare')=='passed','model_decision_bound':performed.get('decision')=='passed'}
        procedure['completed']=procedure['baseline_qualified'] and procedure['variant_attempt_recorded'] and procedure['model_decision_bound'] and (procedure['comparison_completed'] or performed.get('variant')=='failed')
        save(attempt/'tuning-accounting.json',{'rounds':rounds,'procedure':procedure,'native_trace':trace,'tool_receipts':receipts,'native_source_binding':source_binding,'native_events_sha256':sha(attempt/'generation.stdout'),'usage_sha256':sha(attempt/'usage.json'),'latch_retained_for_original_dispatcher_cleanup':True})
        assert json.loads(lease.read_text())['run_dir']==str(attempt)
        # Intentionally no latch unlink. Existing dispatcher owns original
        # process/container absence and accounting retirement after return.
        return result,usage
