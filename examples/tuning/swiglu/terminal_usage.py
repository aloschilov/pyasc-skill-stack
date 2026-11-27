"""Classify retained original model responses; never replay or invent usage."""
from pathlib import Path
import hashlib
import json


def classify(attempt, response_usage):
    attempt=Path(attempt)
    usage_path=attempt/'usage.json'
    if not usage_path.exists():
        before_provider=not (attempt/'generation.command.json').exists() and not list((attempt/'private').glob('provider-response-*.bin'))
        return {'outcome_known':before_provider,'kind':'before-provider'if before_provider else'unknown','requests':0,'completion_tokens':0 if before_provider else None}
    usage=json.loads(usage_path.read_text());rows=usage.get('requests',[])
    expected={f'provider-response-{i}.bin' for i in range(1,len(rows)+1)}
    actual={p.name for p in (attempt/'private').glob('provider-response-*.bin')}
    if actual!=expected:
        return {'outcome_known':False,'kind':'raw-response-inventory-mismatch','requests':len(rows),'completion_tokens':None}
    if not rows and not usage.get('error') and not (attempt/'generation.command.json').exists():
        return {'outcome_known':True,'kind':'before-provider','requests':0,'completion_tokens':0}
    if usage.get('usage_complete') is not True:
        return {'outcome_known':False,'kind':'unknown-response-or-usage','requests':len(rows),'completion_tokens':None}
    try:
        for i,row in enumerate(rows,1):
            raw=attempt/'private'/f'provider-response-{i}.bin'
            if hashlib.sha256(raw.read_bytes()).hexdigest()!=row['response_sha256']:
                raise ValueError('Raw response hash mismatch')
            parsed=response_usage(raw.read_bytes(),row['stream'],'glm-5-3-260814')
            if parsed!=row['usage']:raise ValueError('Usage replay does not match saved response')
            delivery=row.get('client_delivery',{}).get('status')
            if row['status']=='complete' and delivery=='written':continue
            if (row['status']=='rejected' and parsed['finish_reason']=='length'
                    and row.get('validation_error')=='Model completion truncated at output-token cap'
                    and delivery=='pending'):
                continue  # Retained complete response rejected before deliver().
            if (row['status']=='complete' and delivery=='failed'
                    and row.get('tool_preexecution',{}).get('status')=='rejected_before_client_delivery'
                    and row['tool_preexecution'].get('model_tool_executed') is False):
                continue  # Explicit pre-delivery validation; zero bytes forwarded.
            raise ValueError('Delivery or request outcome unresolved')
    except (ValueError,KeyError,TypeError,OSError):
        return {'outcome_known':False,'kind':'unknown-response-or-delivery','requests':len(rows),'completion_tokens':None}
    allowed={None,'Request budget exhausted','Aggregate output budget exhausted',
             'BudgetError: Model completion truncated at output-token cap',
             'Client response delivery failed: ValueError'}
    error=usage.get('error')
    if error=='Non-tool budget validation failed; no provider dispatch':
        p=attempt/'non-tool-budget.json'
        if not p.is_file() or json.loads(p.read_text()).get('remaining_seconds')!=0:
            return {'outcome_known':False,'kind':'unreconciled-budget-validation','requests':len(rows),'completion_tokens':None}
    elif error not in allowed:
        return {'outcome_known':False,'kind':'unreconciled-provider-error','requests':len(rows),'completion_tokens':None}
    return {'outcome_known':True,'kind':'known-rejection'if error else'fully-accounted',
            'requests':len(rows),'completion_tokens':sum(r['usage']['completion_tokens']for r in rows),
            'usage_sha256':hashlib.sha256(usage_path.read_bytes()).hexdigest()}
