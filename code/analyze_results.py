"""Export auditable per-row results, costs, traces and figures after frozen test."""
import collections
import csv
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT,metrics,prediction,write_json,parse_json

def read_rows(split):return [json.loads(s) for s in (ROOT/f'data_processed/{split}.jsonl').read_text().splitlines()]
def read_traces(split):return {p.stem:json.loads(p.read_text()) for p in (ROOT/f'results/traces/{split}').glob('*.json')}
def stage_ids(trace,method,q):
    stages=trace['stages']
    if method=='E0':return [stages['E0']['call_id']]
    result=[v['call_id'] for k,v in stages.items() if k.startswith('explorer_')]
    if trace['yes_count'] is not None and trace['yes_count']>=q:
        result.extend(v['call_id'] for k,v in stages.items() if k in ['mediator','direct','indirect'])
    return result
def matched_rules(trace):
    stage=trace['stages'].get(trace.get('route'),{})
    for _,obj in reversed(parse_json(stage.get('text',''))):
        if isinstance(obj,dict) and 'matched_rules' in obj:return obj['matched_rules']
    return []

def main():
    frozen=json.loads((ROOT/'configs/frozen.json').read_text());q=frozen['selected_q']
    calls={p.stem:json.loads(p.read_text()) for p in (ROOT/'results/api_calls').glob('*.json')}
    tables=[];full={};all_ids=set()
    for split in ['validation','test']:
        rows=read_rows(split);traces=read_traces(split)
        assert {r['input_hash'] for r in rows}==set(traces),'Do not summarize incomplete evaluation'
        full[split]={}
        for name,threshold in [('E0',None),('E1',1),('E2',q)]:
            out=ROOT/f'results/{name}';out.mkdir(exist_ok=True)
            records=[];ids=set();per_row_lat=[];routecounts=collections.Counter()
            for row in rows:
                trace=traces[row['input_hash']];pred=prediction(trace,name,threshold)
                use=stage_ids(trace,name,threshold);ids.update(use);all_ids.update(use)
                duration=sum(calls[i]['elapsed_s'] for i in use);per_row_lat.append(duration)
                route='single' if name=='E0' else ('filtered' if trace['yes_count'] is not None and trace['yes_count']<threshold else trace['route'])
                routecounts[route]+=1
                records.append({'sample_id':row['sample_id'],'input_hash':row['input_hash'],'true_label':row['label'],'predicted_label':pred,
                                'scope':row['scope'],'status':'ok' if pred is not None else 'failed','route':route,'q':threshold,
                                'yes_count':trace['yes_count'],'document_candidates':[m['doc'] for m in row['document_candidates']],
                                'call_ids':use,'sequential_api_seconds':duration,'probability':None})
            (out/f'{split}_predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
            with (out/f'{split}_predictions.csv').open('w',newline='') as f:
                fields=['sample_id','true_label','predicted_label','scope','status','route','q','yes_count','sequential_api_seconds']
                writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
            stats={scope:metrics([r for r in rows if scope=='overall' or r['scope']==scope],traces,name,threshold) for scope in ['overall','intra','inter']}
            charge=sum(calls[i]['charged_upper_cny'] for i in ids)
            tokens=collections.Counter()
            for i in ids:
                if calls[i]['response']:tokens.update({k:v for k,v in calls[i]['response'].get('usage',{}).items() if isinstance(v,(float,int))})
            stats['cost']={'unique_request_count':len(ids),'component_measured_seconds':sum(calls[i]['elapsed_s'] for i in ids),
                           'mean_serial_api_seconds_per_row':statistics.mean(per_row_lat),
                           'p50_serial_api_seconds_per_row':statistics.median(per_row_lat),
                           'equivalent_cost_cny':charge,'token_usage':dict(tokens),
                           'note':'Logical method cost obtained from its actual component request records. Calls shared across E0/E1/E2 are billed once in the joint run; method costs must not be added.'}
            stats['routes']=dict(routecounts);write_json(out/f'{split}_metrics.json',stats);full[split][name]=stats
            for scope in ['overall','intra','inter']:
                tables.append({'split':split,'experiment':name,'q':threshold,'scope':scope,**stats[scope],
                               'unique_request_count':len(ids),'equivalent_cost_cny':charge,'mean_serial_api_seconds':statistics.mean(per_row_lat)})
        # q2/q3 selection on validation, additional replay only (no test selection).
    with (ROOT/'results/comparison.csv').open('w',newline='') as f:
        fields=list(tables[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(tables)
    billing={'all_calls':len(calls),'all_attempts':sum(len(x['attempts']) for x in calls.values()),'total_charge_upper_cny':sum(x['charged_upper_cny'] for x in calls.values()),
             'evaluation_joint_charge_cny':sum(calls[i]['charged_upper_cny'] for i in all_ids),
             'api_errors':sum(x['status']!='ok' for x in calls.values()),'actual_model_ids':sorted({x['response']['model'] for x in calls.values() if x['response']}),
             'cost_rule':'Actual returned usage × applicable published CNY rates. Network timeouts, if any, reserved at conservative byte-count maximum.',
             'authorized_budget_cny':20}
    write_json(ROOT/'results/billing.json',billing);write_json(ROOT/'results/summary.json',full)
    rows=read_rows('test');traces=read_traces('test')
    candidates=[]
    for row in rows:
        trace=traces[row['input_hash']];pred=prediction(trace,'E1',1)
        if pred is None:kind='failure'
        elif pred==row['label']:kind='TP' if pred else 'TN'
        else:kind='FP' if pred else 'FN'
        candidates.append({**row,'prediction_E0':trace['e0'],'prediction_E1':pred,'prediction_E2':prediction(trace,'E2',q),
                           'error_type':kind,'trace':trace,'matched_rules':matched_rules(trace)})
    transitions={'e0_wrong_e1_correct':sum(x['prediction_E0']!=x['label'] and x['prediction_E1']==x['label'] for x in candidates),
                 'e0_correct_e1_wrong':sum(x['prediction_E0']==x['label'] and x['prediction_E1']!=x['label'] for x in candidates),
                 'e1_false_negative_filtered':sum(x['error_type']=='FN' and x['trace']['route']=='filtered' for x in candidates),
                 'e1_false_negative_reasoner':sum(x['error_type']=='FN' and x['trace']['route']!='filtered' for x in candidates),
                 'e1_false_positive_rules':dict(collections.Counter(str(r) for x in candidates if x['error_type']=='FP' for r in x['matched_rules'])),
                 'note':'Counts on all original test rows, including duplicate inputs; diagnostic analysis only, no changes to frozen settings.'}
    write_json(ROOT/'results/error_transitions.json',transitions)
    chosen=[];seen=set()
    for kind,scope in [('TP','inter'),('TN','intra'),('FN','intra'),('FN','inter'),('FP','intra'),('FP','inter')]:
        options=[x for x in candidates if x['error_type']==kind and x['scope']==scope and x['input_hash'] not in seen]
        options.sort(key=lambda x:(len(x['context'])>600, x['trace']['route']=='filtered' if kind=='TP' else False, len(x['context'])))
        if options:chosen.append(options[0]);seen.add(options[0]['input_hash'])
    while sum(x['error_type'] in ['FP','FN','failure'] for x in chosen)<5:
        options=[x for x in candidates if x['error_type'] in ['FP','FN','failure'] and x['input_hash'] not in seen]
        if not options:break
        options.sort(key=lambda x:len(x['context']));chosen.append(options[0]);seen.add(options[0]['input_hash'])
    selection=ROOT/'configs/case_selection.json'
    if selection.exists():
        ids=json.loads(selection.read_text())['sample_ids'];lookup={x['sample_id']:x for x in candidates}
        chosen=[lookup[i] for i in ids]
    write_json(ROOT/'results/cases_selected.json',chosen)
    # A separate set makes it easy to choose a case by hand without rereading all calls.
    write_json(ROOT/'results/cases_candidates.json',[x for x in candidates if x['error_type'] in ['FP','FN']][:60])
    for split in ['validation','test']:
        trace=read_traces(split);hist=collections.Counter(x['yes_count'] for x in trace.values());print(split,'votes',dict(hist))
    figs=ROOT/'results/figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    palette=['#8497A8','#087F8C','#BC7B29']
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for j,name in enumerate(['E0','E1','E2']):
        ys=[full['test'][name][s]['f1']*100 for s in ['intra','inter','overall']];xs=[i+(j-1)*.24 for i in range(3)]
        bars=ax.bar(xs,ys,width=.22,label=name,color=palette[j]);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
    ax.set_xticks(range(3),['Intra-sentence','Inter-sentence','Overall']);ax.set_ylim(0,100);ax.set_ylabel('Positive-class F1 (%)');ax.legend(ncols=3,loc='upper right');fig.savefig(figs/'test_f1.png',dpi=220);plt.close(fig)
    validation=json.loads((ROOT/'results/validation_metrics.json').read_text())
    fig,ax=plt.subplots(figsize=(6,3.4),layout='constrained');vals=[validation[n]['f1']*100 for n in ['E1','q2','q3']]
    ax.plot([1,2,3],vals,'o-',color='#087F8C');ax.set_xticks([1,2,3]);ax.set_xlabel('Explorer threshold q');ax.set_ylabel('Validation F1 (%)')
    for x,y in zip([1,2,3],vals):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center')
    ax.margins(y=.3);fig.savefig(figs/'validation_q.png',dpi=220);plt.close(fig)
    m=full['test']['E1']['overall'];fig,ax=plt.subplots(figsize=(4.4,3.6),layout='constrained');matrix=[[m['tn'],m['fp']],[m['fn'],m['tp']]]
    ax.imshow(matrix,cmap='Blues');ax.set_xticks([0,1],['Non-causal','Causal']);ax.set_yticks([0,1],['Non-causal','Causal']);ax.set_xlabel('Predicted');ax.set_ylabel('Gold')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(matrix[i][j]),ha='center',va='center',fontsize=18,color='white' if matrix[i][j]>max(sum(matrix,[]))*.55 else '#17324D')
    fig.savefig(figs/'e1_confusion.png',dpi=220);plt.close(fig)
    print(json.dumps(billing,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
