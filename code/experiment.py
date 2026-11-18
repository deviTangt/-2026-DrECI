"""Real API evaluation of Vanilla and the complete Dr.ECI pipeline.

API calls never receive gold labels. Every request/response is durably recorded.
Threshold variants replay the same measured component outputs (no new API calls).
"""
from pathlib import Path
import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import threading
import time
from zoneinfo import ZoneInfo
import requests

ROOT = Path(__file__).resolve().parents[1]
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def sha(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def module(n):
    path=ROOT/f'prompts/official_agent{n}.py';spec=importlib.util.spec_from_file_location(f'official_agent{n}',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
A1,A2,A3=module(1),module(2),module(3)

def parse_json(text):
    decoder=json.JSONDecoder();objects=[]
    for match in re.finditer(r'[\[{]',text):
        try: obj,end=decoder.raw_decode(text[match.start():]);objects.append((match.start(),obj))
        except (ValueError,TypeError):pass
    return objects
def parse_answer(text):
    for _,obj in reversed(parse_json(text)):
        if isinstance(obj,dict):
            val=next((v for k,v in obj.items() if k.lower()=='answer'),None)
            if isinstance(val,str) and val.strip().lower() in ['yes','no']:
                return int(val.strip().lower()=='yes')
    # A model may quote evidence without escaping quotation marks. Recover only
    # an explicit Answer field at the start of a JSON-like object; never infer
    # the answer from prose or the gold label. Keep the original text on disk.
    fields=re.findall(r'\{\s*"Answer"\s*:\s*"(Yes|No)"',text,re.I)
    if fields:return int(fields[-1].lower()=='yes')
    return None
def parse_mediator(text):
    for _,obj in reversed(parse_json(text)):
        if isinstance(obj,dict) and isinstance(obj.get('mediators'),list):
            if all(isinstance(x,str) for x in obj['mediators']):return obj['mediators']
    return None

def build_messages(row,stage,mediators=None):
    # Explicit allowlist: gold labels, Answer sentence and provenance are absent.
    source,target,text=row['source_event'],row['target_event'],row['context']
    final='Return only a valid JSON object. '
    if stage=='E0':
        system=A1.system
        user=A1.causal(source,target,text)+final+'Use exactly {"Answer":"Yes"} or {"Answer":"No"}.'
    elif stage.startswith('explorer_'):
        expression=stage.split('_',1)[1]
        system=A1.system
        user=getattr(A1,expression)(source,target,text)+final+'Use exactly {"Answer":"Yes"} or {"Answer":"No"}.'
    elif stage=='mediator':
        system=A2.mediator_system
        user=A2.mediator(source,target,text)+'''\nOutput the results of the four subquestions in ONE valid JSON object, without markdown:
{"source_related":["word or phrase"],"target_related":["word or phrase"],"mediators":["shared word or phrase"]}.
Use an empty list for mediators if there is no shared entity or event. Do not include source and target themselves in mediators.'''
    elif stage in ['direct','indirect']:
        system=A3.reasoner_system
        # Official implementation uses one conditional prompt covering all eight
        # rules. Keep its rule semantics intact; route names describe Z's state.
        user=A3.reasoner(source,target,text,json.dumps(mediators or [],ensure_ascii=False))+'''\nReturn ONE valid JSON object without markdown:
{"Answer":"Yes or No","matched_rules":[1],"evidence":"brief quotation or explanation, at most 35 words"}.
Use an empty matched_rules list if none applies. Evaluate the supplied rules; do not reproduce their full text.'''
    else:raise ValueError(stage)
    return [{'role':'system','content':system.strip()},{'role':'user','content':user.strip()}]

def rates_now():
    now=dt.datetime.now(ZoneInfo('Asia/Shanghai'))
    peak=now.weekday()<5 and (9<=now.hour<12 or 14<=now.hour<18)
    # Current run is November 2026, not a PRC public holiday. Peak upper rates
    # outside this period are conservative if the script runs on a holiday.
    return {'input_hit':.04 if peak else .02,'input_miss':2.0 if peak else 1.0,'output':8.0 if peak else 4.0,'unit':'CNY per million tokens','peak':peak}
def cost(usage,rates):
    total=usage.get('prompt_tokens',0);hit=usage.get('prompt_cache_hit_tokens',0)
    miss=usage.get('prompt_cache_miss_tokens',max(0,total-hit))
    return (hit*rates['input_hit']+miss*rates['input_miss']+usage.get('completion_tokens',0)*rates['output'])/1e6

class BudgetExceeded(RuntimeError):pass
class Client:
    def __init__(self,config):
        self.config=config;self.dir=ROOT/'results/api_calls';self.dir.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.key_locks={};self.reserved=0.;self.spent=0.;self.calls=0
        self.key=os.environ.get('DEEPSEEK_API_KEY')
        path=Path(config['secret_file'])
        if path.exists():
            for line in path.read_text().splitlines():
                if '=' in line and not line.lstrip().startswith('#'):
                    k,v=line.split('=',1)
                    if k.strip() in ['DEEPSEEK_API_KEY','OPENAI_API_KEY','API_KEY']:
                        self.key=v.strip().strip('"').strip("'")
        if not self.key:raise RuntimeError('Missing DEEPSEEK_API_KEY')
        for path in self.dir.glob('*.json'):
            record=json.loads(path.read_text());self.spent+=record['charged_upper_cny'];self.calls+=len(record['attempts'])
    def call(self,messages,stage):
        config=self.config
        payload={'model':config['model'],'messages':messages,'temperature':config['temperature'],
                 'thinking':{'type':'disabled'},'max_tokens':config['max_tokens'][stage if stage in ['E0','mediator','direct','indirect'] else 'explorer'],
                 'response_format':{'type':'json_object'},'stream':False}
        call_id=sha(payload);path=self.dir/(call_id+'.json')
        with self.lock:lock=self.key_locks.setdefault(call_id,threading.Lock())
        with lock:
            if path.exists():return json.loads(path.read_text())
            attempts=[];total_charge=0.;successful=None;start=time.time()
            for attempt in range(config['max_attempts']):
                rates=rates_now()
                # A byte-count bound deliberately exceeds normal tokenizer length.
                bound=(len(json.dumps(messages,ensure_ascii=False).encode())+64)*2/1e6+payload['max_tokens']*8/1e6
                with self.lock:
                    if self.spent+self.reserved+bound>config['spend_limit_cny']:
                        raise BudgetExceeded(f'Budget guard: spent upper {self.spent:.4f}, reserved {self.reserved:.4f}')
                    self.reserved+=bound
                attempt_start=time.time();response=None;error=None;charge=0.
                try:
                    response=requests.post(config['base_url'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},json=payload,timeout=(15,150))
                    if response.status_code==200:
                        body=response.json();endrates=rates_now()
                        billing_rates={k:max(rates[k],endrates[k]) for k in ['input_hit','input_miss','output']}
                        charge=cost(body.get('usage',{}),billing_rates)
                        if not body.get('usage'):charge=bound
                        successful=body
                    else:
                        body={'http_status':response.status_code,'body':response.text[:3000]}
                        error=f'HTTP {response.status_code}'
                except Exception as exc:
                    body=None;error=type(exc).__name__;charge=bound
                finally:
                    with self.lock:self.reserved-=bound;self.spent+=charge;self.calls+=1
                attempts.append({'time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'elapsed_s':time.time()-attempt_start,'rates':rates,'charge_upper_cny':charge,'error':error,'response':body})
                total_charge+=charge
                # Persist each attempt, including unknown-charge network failures.
                rec={'call_id':call_id,'stage':stage,'request':payload,'attempts':attempts,'response':successful,'charged_upper_cny':total_charge,'elapsed_s':time.time()-start,'status':'ok' if successful else 'api_error'}
                write_json(path,rec)
                if successful:break
                if response is not None and response.status_code not in [429,500,502,503,504]:break
                time.sleep(min(2**attempt,8))
            return rec
    def summary(self):
        with self.lock:return {'api_attempts':self.calls,'charged_upper_cny':round(self.spent,6),'reserved_cny':round(self.reserved,6),'limit_cny':self.config['spend_limit_cny']}

def response_text(rec):
    if rec['status']!='ok':return ''
    return rec['response'].get('choices',[{}])[0].get('message',{}).get('content') or ''

def infer(row,client,methods):
    stages={};ids=[];start=time.time()
    def call(stage,mediators=None):
        rec=client.call(build_messages(row,stage,mediators),stage);ids.append(rec['call_id']);stages[stage]={'call_id':rec['call_id'],'status':rec['status'],'text':response_text(rec)};return stages[stage]['text']
    e0=None
    if 'E0' in methods:e0=parse_answer(call('E0'))
    votes=[];med=None;answer=None;route=None;status='ok'
    if 'E1' in methods:
        for exp in ['causal','causation','causality']:votes.append(parse_answer(call('explorer_'+exp)))
        if any(v is None for v in votes):status='explorer_error'
        elif sum(votes)==0:answer=0;route='filtered'
        else:
            med=parse_mediator(call('mediator'))
            if med is None:status='mediator_error'
            else:
                route='indirect' if med else 'direct';answer=parse_answer(call(route,med))
                if answer is None:status='reasoner_error'
    return {'input_hash':row['input_hash'],'sample_id':row['sample_id'],'scope':row['scope'],'split':row['split'],
            'e0':e0,'votes':votes,'yes_count':sum(votes) if votes and all(v is not None for v in votes) else None,
            'mediators':med,'route':route,'e1':answer,'pipeline_status':status,'stages':stages,
            'call_ids':ids,'elapsed_s':time.time()-start}

def prediction(trace,method,q=1):
    if method=='E0':return trace.get('e0')
    yes=trace.get('yes_count')
    if yes is None:return None
    if yes<q:return 0
    return trace.get('e1')

def metrics(rows,traces,method,q=1):
    tp=fp=fn=tn=fail=0
    for row in rows:
        val=prediction(traces.get(row['input_hash'],{}),method,q)
        gold=row['label']
        if val is None:
            fail+=1
            # Abstentions are always errors; do not drop failed cases or silently
            # turn negative failures into correct predictions.
            fn+=int(gold==1);fp+=int(gold==0)
        elif val==1:tp+=gold;fp+=1-gold
        else:fn+=gold;tn+=1-gold
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return {'n':len(rows),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'failures':fail,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'accuracy':(tp+tn)/len(rows) if rows else 0.,'coverage':1-fail/len(rows) if rows else 0.,'failure_policy':'abstention counts as error; positive abstention FN, negative abstention FP'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['train','validation','test'],required=True);parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--config',default=str(ROOT/'configs/runtime.json'));args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    if args.split=='test' and not (ROOT/'configs/frozen.json').exists():raise SystemExit('Test requires frozen validation selection')
    rows=[json.loads(x) for x in (ROOT/f'data_processed/{args.split}.jsonl').read_text().splitlines()]
    unique=list({x['input_hash']:x for x in rows}.values())
    if args.limit:unique=unique[:args.limit]
    output=ROOT/f'results/traces/{args.split}';output.mkdir(parents=True,exist_ok=True)
    client=Client(config);started=time.time();done=0;budget_hit=False
    def task(row):
        path=output/(row['input_hash']+'.json')
        if path.exists():return 'cached'
        trace=infer(row,client,['E0','E1']);write_json(path,trace);return trace['pipeline_status']
    # Bounded submission avoids thousands of queued requests after a budget stop.
    iterator=iter(unique)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        active={pool.submit(task,row):row for row in itertools_islice(iterator,args.workers)}
        while active:
            completed,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in completed:
                row=active.pop(fut)
                try:status=fut.result();done+=1
                except BudgetExceeded as exc:budget_hit=True;print(str(exc),flush=True);status='budget_stop'
                except Exception as exc:print('Task error',row['sample_id'],type(exc).__name__,str(exc),flush=True);status='error'
                if done%25==0 or status not in ['ok','cached']:
                    progress={'split':args.split,'completed':done,'total_unique':len(unique),'last_status':status,'elapsed_s':round(time.time()-started,2),**client.summary()};write_json(ROOT/f'results/progress_{args.split}.json',progress);print(json.dumps(progress),flush=True)
                if not budget_hit:
                    nextrow=next(iterator,None)
                    if nextrow is not None:active[pool.submit(task,nextrow)]=nextrow
    print('FINISHED',json.dumps({'split':args.split,'completed':done,'budget_stop':budget_hit,**client.summary()}),flush=True)
    if not args.limit and not budget_hit:
        traces={p.stem:json.loads(p.read_text()) for p in output.glob('*.json')}
        result={'E0':metrics(rows,traces,'E0'),'E1':metrics(rows,traces,'E1')}
        if args.split=='validation':
            result['q2']=metrics(rows,traces,'E1',2);result['q3']=metrics(rows,traces,'E1',3)
        else:
            frozen=json.loads((ROOT/'configs/frozen.json').read_text()) if args.split=='test' else {'selected_q':2}
            result['E2']=metrics(rows,traces,'E1',frozen['selected_q'])
        write_json(ROOT/f'results/{args.split}_metrics.json',result);print(json.dumps(result),flush=True)

def itertools_islice(iterator,n):
    import itertools
    return itertools.islice(iterator,n)

if __name__=='__main__':main()
