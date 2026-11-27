"""Check native tool arguments before forwarding unchanged provider bytes."""
import hashlib
import json
from consumer_interface import validate_bash

def audit(raw,stream,declared,recipe,project):
    if stream:
        lines=[line[5:].strip()for line in raw.decode().splitlines()if line.startswith('data:')]
        if not lines or lines[-1]!='[DONE]':raise ValueError('Incomplete SSE before tool delivery')
        parts=[json.loads(s)for s in lines[:-1]]
    else:parts=[json.loads(raw)]
    calls={}
    for part in parts:
        for choice in part.get('choices',[]):
            if choice.get('index',0)!=0:raise ValueError('Unexpected choice index')
            value=choice.get('delta',{})if stream else choice.get('message',{})
            if value.get('function_call'):raise ValueError('Legacy function calls are unsupported')
            for position,item in enumerate(value.get('tool_calls',[])):
                index=item.get('index',position)
                if type(index)is not int or not 0<=index<16:raise ValueError('Tool index bound')
                row=calls.setdefault(index,{'id':'','name':'','arguments':''})
                if item.get('type','function')!='function':raise ValueError('Tool kind')
                for key,chunk in [('id',item.get('id','')),('name',item.get('function',{}).get('name','')),('arguments',item.get('function',{}).get('arguments',''))]:
                    if not isinstance(chunk,str):raise ValueError('Tool fragment type')
                    row[key]+=chunk
                    if len(row[key])>(16384 if key=='arguments'else 256):raise ValueError('Tool fragment bound')
    rows=[];bash=0
    for index,row in sorted(calls.items()):
        if not row['id']or row['name']not in declared or row['name']not in ('read','write','edit','skill','bash'):
            raise ValueError('Undeclared native tool')
        args=json.loads(row['arguments'])
        if not isinstance(args,dict):raise ValueError('Tool arguments must be an object')
        action=None
        if row['name']=='bash':action=validate_bash(args,recipe,project);bash+=1
        rows.append({'name':row['name'],'action':action,'arguments_sha256':hashlib.sha256(row['arguments'].encode()).hexdigest()})
    if bash and len(rows)!=1:raise ValueError('A measurement command must be the only call in its response')
    return {'status':'passed_before_client_delivery','tool_calls':rows,'response_sha256':hashlib.sha256(raw).hexdigest()}
