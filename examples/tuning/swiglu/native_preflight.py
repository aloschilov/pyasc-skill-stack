"""A fixed zero-model OpenCode fixture, not a general mock-provider service.

Only this preflight bypasses production pre-delivery tool validation, so a
harmless forbidden command tests native denial. Real evaluator commands and
receipts are unchanged. No endpoint or credential for a real provider is read.
"""
from pathlib import Path
import hashlib
import json
import os
import secrets
import time
import threading


def permission_proof(path, baseline, reference, denied_command):
    """Require actual completed native events before spending any Model budget."""
    native=[json.loads(line) for line in Path(path).read_text().splitlines()]
    denied=[];reads=[]
    for event in native:
        part=event.get("part") or {};state=part.get("state") or {};inputs=state.get("input") or {}
        if part.get("tool")=="bash" and inputs.get("command")==denied_command:
            if state.get("status")!="error" or not str(state.get("error", "")).startswith("The user has specified a rule which prevents you from using this specific tool call."):
                raise ValueError("Native denied command was not rejected")
            denied.append(event)
        if part.get("tool")=="read" and state.get("status")=="completed":
            reads.append(inputs.get("filePath"))
    if len(denied)!=1 or str(baseline) not in reads or str(reference) not in reads:
        raise ValueError("Native denial and both allowed reads must precede baseline evaluation")
    return native


class ReplayReservations:
    """One fixed synthetic sequence: reject concurrent handlers and SDK retries."""
    def __init__(self):
        self.lock=threading.Lock();self.seen=set();self.failed=False
    def reserve(self, body):
        if not self.lock.acquire(blocking=False):
            self.failed=True
            raise ValueError("Overlapping synthetic request; no delivery")
        try:
            if self.failed:raise ValueError("Original synthetic sequence already stopped")
            digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()
            if digest in self.seen:raise ValueError("Duplicate synthetic request; no replay")
            self.seen.add(digest)
            return digest
        except Exception:
            self.failed=True;self.lock.release();raise
    def release(self):self.lock.release()


def run(checkout, recipe, project, context, work, attempt, deadline):
    import sys
    sys.path.insert(0,str(checkout/'tests/tools'))
    from generation_executor import command,native_trace,save
    from generation_provider import budget_server
    from consumer_interface import tool_config,installed_context_permissions,commands,audit_events,native_source_binding,request_audit
    from selection import select
    from tool import collected_comparison
    from parameter_guard import BASELINE_SHA256
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    skills=('asctile-cannbench-kernel','asctile-syntax-constraints','asctile-api-patterns','asctile-build-run-verify')
    skillmap={name:context/'skills'/name for name in skills}
    reference=context/'skills/asctile-api-patterns/references/strided-row-tiles.md'
    denied_command='printf native-preflight-denial-sentinel'
    calls=commands(recipe)
    fixtures=json.loads((recipe/'engine/probe/fixtures.json').read_text())['small-tail']
    ids=[f['id']for f in fixtures]
    hypothesis={'hypothesis':'Synthetic preflight uses the retained parameter derivative to exercise the evaluator; not a new model hypothesis.',
                'parameter_change':'Retained scalar height cap; no source repair.'}
    variant=(attempt/'retained-variant.py').read_text()
    config={'provider':{'replay':{'npm':'@ai-sdk/openai-compatible','models':{'native-fixture':{'name':'Synthetic native preflight; no model','limit':{'context':256000,'output':32768}}}}},
            'model':'replay/native-fixture','small_model':'replay/native-fixture','enabled_providers':['replay'],
            'skills':{'paths':[str(context/'skills')]},'agent':{'build':{'steps':12},'title':{'disable':True},'summary':{'disable':True}},
            'compaction':{'auto':False},'permission':{'*':'deny','read':{'*':'deny'},'skill':'allow','glob':'deny','grep':'deny'}}
    config=installed_context_permissions(config,context,project,skills)
    config=tool_config(config,recipe,project)
    env={k:os.environ[k]for k in ('PATH','LANG','LC_ALL','TMPDIR')if k in os.environ}
    private=attempt/'native-private';private.mkdir()
    env.update(HOME=str(private),XDG_DATA_HOME=str(private/'data'),XDG_CONFIG_HOME=str(private/'config'),XDG_CACHE_HOME=str(private/'cache'),
               OPENCODE_DISABLE_EXTERNAL_SKILLS='1',OPENCODE_DISABLE_CLAUDE_CODE_SKILLS='1',OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX='32768')
    events=[]

    class Replay:
        token=secrets.token_urlsafe(32)
        error=None
        number=0
        def __init__(self):
            self.deadline=deadline;self.reservations=ReplayReservations()
        def dispatch(self,body,deliver=None):
            try:
                digest=self.reservations.reserve(body)
            except Exception:
                self.error='Duplicate or overlapping native preflight request; stopped before delivery'
                raise
            try:
                if self.error or self.number>=12 or time.monotonic()>=self.deadline:
                    raise ValueError('Fixed preflight sequence exhausted or stopped')
                if body.get('model')!='native-fixture':raise ValueError('Only the synthetic fixture identity is allowed')
                request_audit(body)
                names={t['function']['name']for t in body.get('tools',[])}
                if not {'read','write','skill','bash'}<=names:raise ValueError('Native tool declaration missing')
                n=self.number
                reservation={'number':n,'request_sha256':digest,'status':'reserved-before-delivery','synthetic':True}
                events.append(reservation)
                save(attempt/'native-replay.json',{'status':'running','events':events,'external_model_calls':0,'billed_tokens':0})
                if n==0:requested=[('skill',{'name':name})for name in skills]
                elif n==1:requested=[('read',{'filePath':str(project/'baseline.py')}),('read',{'filePath':str(reference)})]
                elif n==2:requested=[('bash',{'command':denied_command,'timeout':1000,'description':'Test native denial of a harmless unapproved command'})]
                elif n==3:
                    permission_proof(attempt/'native-preflight.stdout',project/'baseline.py',reference,denied_command)
                    requested=[('bash',{'command':calls['baseline'],'timeout':300000,'description':'Qualify retained baseline'})]
                elif n==4:
                    if json.loads((work/'baseline/public-receipt.json').read_text())['status']!='passed':raise ValueError('Original baseline qualification failed')
                    hypothesis['baseline_receipt_sha256']=sha(work/'baseline/public-receipt.json')
                    requested=[('write',{'filePath':str(project/'candidate.py'),'content':variant}),('write',{'filePath':str(project/'decision.json'),'content':json.dumps(hypothesis,indent=2)+'\n'})]
                elif n==5:requested=[('bash',{'command':calls['variant'],'timeout':300000,'description':'Qualify retained parameter derivative'})]
                elif n in (6,7,8):
                    prior='variant'if n==6 else f'compare-{n-6}'
                    if json.loads((work/prior/'public-receipt.json').read_text())['status']!='passed':raise ValueError('Original preceding evaluator failed')
                    requested=[('bash',{'command':calls[f'compare-{n-5}'],'timeout':600000,'description':'Compare one frozen shape of retained snapshots'})]
                elif n==9:
                    result,measured=collected_comparison(work,ids)
                    if result['status']!='passed':raise ValueError('Original three-shape comparison incomplete')
                    chosen=select(result,ids)
                    decision=dict(hypothesis,selected_candidate_sha256=sha(work/chosen['selected']/'candidate/candidate.py'),baseline_sha256=BASELINE_SHA256,
                                  variant_sha256=sha(work/'variant/candidate/candidate.py'),comparison_receipt_sha256s=measured,
                                  rationale='Synthetic preflight deterministically follows the frozen selection rule; not GLM reasoning.')
                    requested=[('write',{'filePath':str(project/'decision.json'),'content':json.dumps(decision,indent=2)+'\n'})]
                elif n==10:requested=[('bash',{'command':calls['decision'],'timeout':300000,'description':'Bind preflight selection'})]
                else:requested=[]
                toolcalls=[{'index':i,'id':f'preflight_{n}_{i}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}for i,(name,args)in enumerate(requested)]
                message={'role':'assistant','content':None,'tool_calls':toolcalls}if requested else{'role':'assistant','content':'Synthetic native preflight completed. No model was called.'}
                base={'id':f'native-preflight-{n}','object':'chat.completion','created':0,'model':'native-fixture'}
                finish='tool_calls'if requested else'stop'
                if body.get('stream'):
                    parts=[dict(base,object='chat.completion.chunk',choices=[{'index':0,'delta':message,'finish_reason':None}]),
                           dict(base,object='chat.completion.chunk',choices=[{'index':0,'delta':{},'finish_reason':finish}]),
                           dict(base,object='chat.completion.chunk',choices=[],usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0})]
                    raw=(''.join('data: '+json.dumps(p)+'\n\n'for p in parts)+'data: [DONE]\n\n').encode();kind='text/event-stream'
                else:
                    raw=json.dumps(dict(base,choices=[{'index':0,'message':message,'finish_reason':finish}],usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0})).encode();kind='application/json'
                reservation.update(tool_names=[x[0]for x in requested],response_sha256=hashlib.sha256(raw).hexdigest())
                save(attempt/'native-replay.json',{'status':'running','events':events,'external_model_calls':0,'billed_tokens':0})
                if self.reservations.failed:raise ValueError('Concurrent request stopped sequence before delivery')
                deliver(raw,kind)
                reservation['status']='delivered';self.number+=1
                save(attempt/'native-replay.json',{'status':'running','events':events,'external_model_calls':0,'billed_tokens':0})
            except Exception:
                self.error='Fixed native preflight stopped; inspect original receipts, no retry'
                self.reservations.failed=True
                raise
            finally:
                self.reservations.release()

    replay=Replay()
    init=command(['git','-c','init.templateDir=','init','--quiet',str(project)],attempt,'preflight-project-init',20,cwd=project)
    if init['exit_code']:raise ValueError('Preflight project initialization failed')
    with budget_server(replay)as endpoint:
        config['provider']['replay']['options']={'baseURL':endpoint,'apiKey':'{env:NATIVE_PREFLIGHT_TOKEN}'}
        env.update(OPENCODE_CONFIG_CONTENT=json.dumps(config),NATIVE_PREFLIGHT_TOKEN=replay.token)
        result=command(['opencode','run','--pure','--format','json','--dir',str(project),'-m','replay/native-fixture','Execute the fixed native preflight tool sequence. This is synthetic; no model or tuning inference is requested.'],
                       attempt,'native-preflight',max(.01,deadline-time.monotonic()),env,abort=lambda:replay.error,cwd=project)
    if result['status']!='completed'or result['exit_code']!=0 or replay.number!=12:
        raise ValueError('Native preflight did not complete the original fixed sequence')
    native=permission_proof(attempt/'native-preflight.stdout',project/'baseline.py',reference,denied_command)
    trace=native_trace((attempt/'native-preflight.stdout').read_text(),skillmap)
    receipts=audit_events(native,recipe,project,work)
    bindings=native_source_binding(native,project,work)
    report={'status':'passed','synthetic':True,'model_calls':0,'billed_tokens':0,'native_permission_denials':1,'baseline_and_reference_read':True,
            'native_trace':trace,'tool_receipts':receipts,'source_binding':bindings,'native_events_sha256':sha(attempt/'native-preflight.stdout'),'events':events}
    save(attempt/'native-preflight-proof.json',report)
    return report
