"""Independently recompute headline metrics and check record integrity."""
from pathlib import Path
import csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'results/summary.json').read_text())
frozen=json.loads((ROOT/'configs/frozen.json').read_text())
assert hashlib.sha256((ROOT/'code/experiment.py').read_bytes()).hexdigest()==frozen['experiment_code_sha256']
for filename,expected in frozen['prompt_sha256'].items():assert hashlib.sha256((ROOT/'prompts'/filename).read_bytes()).hexdigest()==expected
counts={}
for split in ['validation','test']:
    counts[split]={}
    for method in ['E0','E1','E2']:
        rows=[json.loads(s) for s in (ROOT/f'results/{method}/{split}_predictions.jsonl').read_text().splitlines()]
        tp=sum(x['true_label']==1 and x['predicted_label']==1 for x in rows)
        tn=sum(x['true_label']==0 and x['predicted_label']==0 for x in rows)
        fn=sum(x['true_label']==1 and x['predicted_label']!=1 for x in rows)
        fp=sum(x['true_label']==0 and x['predicted_label']!=0 for x in rows)
        assert tp+tn+fn+fp==len(rows)
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        assert abs(f1-summary[split][method]['overall']['f1'])<1e-12
        assert len({x['sample_id'] for x in rows})==len(rows)
        for r in rows:
            for callid in r['call_ids']:assert (ROOT/f'results/api_calls/{callid}.json').exists()
        counts[split][method]={'n':len(rows),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':f1}
bill=json.loads((ROOT/'results/billing.json').read_text());assert bill['total_charge_upper_cny']<=20
calls=list((ROOT/'results/api_calls').glob('*.json'))
for path in calls:
    rec=json.loads(path.read_text());key=hashlib.sha256(json.dumps(rec['request'],ensure_ascii=False,sort_keys=True).encode()).hexdigest();assert key==path.stem
    assert all('authorization' not in k.lower() for k in rec)
    assert all(m['role'] in ['system','user'] for m in rec['request']['messages'])
audit=json.loads((ROOT/'data_info/audit.json').read_text());assert all(not any(p.values()) for p in audit['overlap'].values())
print(json.dumps({'status':'PASS','metrics_recomputed':counts,'api_calls_verified':len(calls),'total_charge_upper_cny':bill['total_charge_upper_cny'],'frozen_code_and_prompts':'match','cross_split_overlap':0},ensure_ascii=False,indent=2))
