"""Preserve the 900-second non-tool budget within a 2700-second tool session."""
from pathlib import Path
import hashlib
import json
import math
import time
import threading

CAPS={'baseline':270,'variant':270,'compare':595,'decision':10}

def remaining(start,whole,now,receipts,active=()):
    intervals=[];pins=[];live_pins=[]
    if len(active)>1:raise ValueError('Overlapping live evaluators')
    for live,rows in ((False,receipts),(True,active)):
        for action,row,digest in rows:
            if action not in CAPS or row['action']!=action or row['status'] not in (('active',) if live else ('passed','failed')):
                raise ValueError('Unowned or unresolved evaluator interval')
            begin,end=row['tool_monotonic_start'],now if live else row['tool_monotonic_end']
            if any(type(v)not in(int,float)or not math.isfinite(v)for v in(begin,end)) or not start<=begin<=end<=now or end-begin>CAPS[action]:
                raise ValueError('Evaluator interval outside admitted bound')
            intervals.append((begin,end))
            (live_pins if live else pins).append({'action':action,'clock_sha256'if live else'receipt_sha256':digest,'seconds':end-begin})
    intervals.sort()
    if any(a[1]>b[0]for a,b in zip(intervals,intervals[1:])):
        raise ValueError('Overlapping evaluator intervals')
    completed=sum(p['seconds']for p in pins);live_seconds=sum(p['seconds']for p in live_pins)
    spent=now-start-completed-live_seconds
    left=min(900-spent,whole-now)
    return left,{'non_tool_seconds':spent,'completed_tool_seconds':completed,'active_tool_seconds':live_seconds,
                 'remaining_seconds':max(0,left),'tool_receipts':pins,'active_tools':live_pins,
                 'definition':'elapsed session time minus bounded nonoverlapping owned evaluator intervals; live credit requires exact PID/startticks and expires at its bound; not provider billed time'}


def process_startticks(pid):
    if type(pid)is not int or pid<=0:raise ValueError('Invalid live evaluator PID')
    try:
        fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0] in ('Z','X') else fields[19]
    except FileNotFoundError:return None


def read_intervals(work,ownership,*,allow_live=False,identity=process_startticks):
    """Read atomic immutable clocks/receipts, without the provider dispatch lock."""
    rows=[];active=[];work=Path(work)
    for action in CAPS:
        path=work/action/'public-receipt.json';clock_path=work/(action+'-clock.json')
        if not clock_path.exists():
            if not path.exists():continue
            # A fast action may publish both immutable files between checks.
            if not clock_path.exists():raise ValueError('Completed evaluator lacks its owned clock')
        data=clock_path.read_bytes();clock=json.loads(data);digest=hashlib.sha256(data).hexdigest()
        if (clock.get('status')!='active' or clock.get('action')!=action or clock.get('attempt')!=ownership['attempt']
                or clock.get('recipe_index_sha256')!=ownership['recipe_index_sha256']
                or clock.get('native_command')!=ownership['commands'][action]
                or clock.get('seconds_limit')!=CAPS[action]
                or type(clock.get('pid'))is not int or clock['pid']<=0
                or not isinstance(clock.get('startticks'),str) or not clock['startticks'].isdigit()):
            raise ValueError('Unbound evaluator clock')
        live=False
        if not path.exists():
            live=identity(clock['pid'])==clock['startticks']
            # The owner may have published its terminal receipt between the
            # first read and the identity check. Re-read that transition once.
        if path.exists():
            raw=path.read_bytes();row=json.loads(raw)
            if (row.get('tool_clock_sha256')!=digest or row.get('native_command')!=clock['native_command']
                    or row.get('tool_monotonic_start')!=clock.get('tool_monotonic_start')):
                raise ValueError('Completed evaluator changed its original clock')
            rows.append((action,row,hashlib.sha256(raw).hexdigest()))
        elif live and allow_live:
            active.append((action,clock,digest))
        else:
            raise ValueError('Evaluator has no terminal receipt or confirmed live owner')
    return rows,active


class SessionWatchdog:
    """Predicate for the existing command loop; no thread or provider lock."""
    def __init__(self,work,ownership,start,whole,checkpoint,identity=process_startticks,clock=time.monotonic):
        self.work,self.ownership,self.start,self.whole=work,ownership,start,whole
        self.checkpoint,self.identity,self.clock=checkpoint,identity,clock
        self.stopped=None
    def __call__(self):
        if self.stopped:return self.stopped['reason']
        try:
            rows,active=read_intervals(self.work,self.ownership,allow_live=True,identity=self.identity)
            now=self.clock()  # A receipt may have been atomically published during the reads.
            left,report=remaining(self.start,self.whole,now,rows,active)
            if left>0:return None
            reason='Whole session deadline exhausted'if now>=self.whole else'900-second non-tool budget exhausted'
        except Exception as exc:
            now=self.clock()
            reason='Evaluator timing evidence invalid: '+type(exc).__name__
            report={'remaining_seconds':None}
        self.stopped=dict(report,status='stopped',reason=reason,observed_monotonic=now)
        self.checkpoint(self.stopped)
        return reason


def watchdog_command(original,watchdog):
    """Use the existing owned client kill/reap path, including terminal idle."""
    def wrapped(argv,directory,name,seconds,env=None,abort=None,cwd=None):
        if name!='generation':return original(argv,directory,name,seconds,env,abort,cwd)
        def stop():return watchdog()or(abort()if abort else None)
        result=original(argv,directory,name,seconds,env,stop,cwd)
        # A process may exit between the last loop check and poll(). A late
        # successful client exit must not turn an exhausted budget into PASS.
        if watchdog():raise ValueError('Generation ended beyond its non-tool budget')
        return result
    return wrapped


def gate_type(base,work,checkpoint,delivery_audit=None,ownership=None,session_clock=None,watchdog=None):
    class NonToolBudgetGate(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.lock=threading.RLock()
            self.whole_deadline=min(self.deadline,session_clock[1]) if session_clock else self.deadline
            self.session_started=session_clock[0] if session_clock else self.whole_deadline-2700
        def dispatch(self,body,deliver=None):
            # Same reentrant lock protects this update, base dispatch and finish.
            # A queued request cannot extend an in-flight read or write after seal.
            with self.lock:
                if self.closed or self.error or (watchdog and watchdog.stopped):
                    raise ValueError('Closed or failed original gate; no mutation or dispatch')
                # Reserve one full native response without reducing its wire cap.
                # run_session alone does not apply the outer campaign token cap.
                used=sum(row['usage']['completion_tokens'] for row in self.records)
                if used+self.limits['output_tokens_per_response']>self.limits['max_output_tokens']:
                    self.error='Aggregate output budget exhausted'
                    self.checkpoint()
                    raise ValueError('Insufficient remaining output budget for a native response')
                try:
                    rows=[]
                    if ownership is not None:
                        rows,_=read_intervals(work,ownership,allow_live=False)
                    else:
                        for action in CAPS:
                            p=Path(work)/action/'public-receipt.json'
                            if p.exists():
                                data=p.read_bytes();rows.append((action,json.loads(data),hashlib.sha256(data).hexdigest()))
                    now=time.monotonic();left,report=remaining(self.session_started,self.whole_deadline,now,rows)
                    checkpoint(report)
                    if left<=0:raise ValueError('900-second non-tool generation budget exhausted')
                    self.deadline=now+left
                except Exception:
                    self.error='Non-tool budget validation failed; no provider dispatch'
                    self.checkpoint()
                    raise
                def checked_delivery(raw,kind):
                    if delivery_audit is not None:
                        try:
                            record=delivery_audit(raw,body)
                            self.records[-1]['tool_preexecution']=record
                            self.checkpoint()
                        except Exception as exc:
                            self.records[-1]['tool_preexecution']={'status':'rejected_before_client_delivery','error_type':type(exc).__name__,'model_tool_executed':False}
                            self.checkpoint()
                            raise
                    if deliver is not None:return deliver(raw,kind)
                return super().dispatch(body,deliver=checked_delivery if delivery_audit is not None else deliver)
    return NonToolBudgetGate
