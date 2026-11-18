"""Reparse recorded explicit output fields; never rerun inference or change gold."""
import argparse,json
from experiment import ROOT,parse_answer,parse_mediator,write_json

ap=argparse.ArgumentParser();ap.add_argument('--split',required=True);args=ap.parse_args()
changes=[]
for p in (ROOT/f'results/traces/{args.split}').glob('*.json'):
    t=json.loads(p.read_text());before={'e0':t['e0'],'e1':t['e1'],'pipeline_status':t['pipeline_status']}
    t['e0']=parse_answer(t['stages']['E0']['text'])
    t['votes']=[parse_answer(t['stages']['explorer_'+x]['text']) for x in ['causal','causation','causality']]
    t['yes_count']=sum(t['votes']) if all(v is not None for v in t['votes']) else None
    if t['yes_count']==0:t['e1']=0;t['pipeline_status']='ok'
    elif t['yes_count'] is not None and t.get('route') in ['direct','indirect']:
        t['e1']=parse_answer(t['stages'][t['route']]['text'])
        if t['e1'] is not None:t['pipeline_status']='ok'
    after={'e0':t['e0'],'e1':t['e1'],'pipeline_status':t['pipeline_status']}
    if before!=after:
        change={'sample_id':t['sample_id'],'input_hash':t['input_hash'],'before':before,'after':after,'reason':'explicit Answer field recoverable despite unescaped quotation marks in evidence'}
        changes.append(change);t['parse_repair']=change;write_json(p,t)
target=ROOT/f'results/reparse_{args.split}.json'
if target.exists():changes=json.loads(target.read_text())+changes
write_json(target,changes)
print('Recovered explicit output fields:',len(changes))
