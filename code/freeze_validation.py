from pathlib import Path
import datetime
import json
from experiment import ROOT, sha, write_json

def main():
    target=ROOT/'configs/frozen.json'
    if target.exists():raise SystemExit('Configuration already frozen; do not overwrite after test')
    result=json.loads((ROOT/'results/validation_metrics.json').read_text())
    rows=[json.loads(x) for x in (ROOT/'data_processed/validation.jsonl').read_text().splitlines()]
    paths=list((ROOT/'results/traces/validation').glob('*.json'))
    assert len(paths)==len({r['input_hash'] for r in rows})
    assert result['E1']['failures']==0,'Resolve validation infrastructure errors before freeze'
    selected=max([2,3],key=lambda q:(result[f'q{q}']['f1'],-q))
    config=json.loads((ROOT/'configs/runtime.json').read_text())
    config.pop('secret_file',None)
    freeze={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'selected_q':selected,'selection':'Compare q=2 and q=3 on validation F1 only; tie goes to lower q.',
            'validation_metrics':result,'runtime_without_secrets':config,
            'data_audit_hash':sha(json.loads((ROOT/'data_info/audit.json').read_text())),
            'experiment_code_sha256':__import__('hashlib').sha256((ROOT/'code/experiment.py').read_bytes()).hexdigest(),
            'prompt_sha256':{p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in (ROOT/'prompts').glob('*.py')},
            'test_policy':'No test inference has been run before this freeze. Do not adjust model, prompts, q or labels using test results.'}
    write_json(target,freeze)
    for name,q,method in [('E0',None,'vanilla'),('E1',1,'Dr.ECI'),('E2',selected,'Dr.ECI')]:
        write_json(ROOT/f'configs/{name}.json',{'experiment':name,'method':method,'q':q,'model':config['model'],'temperature':config['temperature'],'seed':json.loads((ROOT/'configs/split.json').read_text())['seed'],
                   'only_changed_factor_vs_E1':'q' if name=='E2' else None,'split_file':'configs/split.json','runtime_file':'configs/runtime.json','cache_policy':'identical deterministic stage requests reused; raw call IDs preserved'})
    print('Frozen E2 q =',selected)

if __name__=='__main__':main()
