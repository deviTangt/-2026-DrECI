"""Replay measured agent responses; optionally run a new real API inference."""
import argparse
import json
from pathlib import Path
from experiment import ROOT,Client,infer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sample-id');ap.add_argument('--list',action='store_true');ap.add_argument('--live',action='store_true');ap.add_argument('--interactive',action='store_true');ap.add_argument('--config',default=str(ROOT/'configs/runtime.json'));args=ap.parse_args()
    casepath=ROOT/'results/cases_selected.json'
    cases=json.loads(casepath.read_text()) if casepath.exists() else []
    if args.interactive:
        if not cases:raise SystemExit('No recorded cases available')
        while True:
            print('\n=== Dr.ECI 真实案例交互演示（记录重放，不调用API） ===')
            for i,x in enumerate(cases,1):print(f'{i}. {x["sample_id"]} [{x["error_type"]}] {x["source_event"]} / {x["target_event"]}')
            print('0. 退出')
            try:choice=input('输入案例编号：').strip()
            except EOFError:return
            if choice=='0':return
            if not choice.isdigit() or not 1<=int(choice)<=len(cases):continue
            display(cases[int(choice)-1],False)
        return
    if args.list:
        for x in cases:print(x['sample_id'],x['scope'],x['error_type'],x['source_event'],'/',x['target_event'])
        return
    case=next((x for x in cases if x['sample_id']==args.sample_id),None) if args.sample_id else (cases[0] if cases else None)
    if args.live:
        # Live demonstration is on training-only input; does not alter frozen test.
        full=ROOT/'data_processed/train.jsonl'
        rows=[json.loads(s) for s in full.read_text().splitlines()] if full.exists() else json.loads((ROOT/'data_info/demo_train.json').read_text())
        if args.sample_id:
            case=next((x for x in rows if x['sample_id']==args.sample_id),None)
            if case is None:raise SystemExit('Requested training sample is not available in this copy')
        else:
            case=rows[-1] if not full.exists() else json.loads((ROOT/'data_info/demo_train.json').read_text())[-1]
        case['trace']=infer(case,Client(json.loads(Path(args.config).read_text())),['E0','E1'])
    if not case:raise SystemExit('No case found; use --list')
    display(case,args.live)

def display(case,live):
    print('=== '+('实时API/缓存调用' if live else '真实实验记录重放（无新增费用）')+' ===')
    print('样本:',case['sample_id']);print('原文:',case['context']);print('事件:',case['source_event'],'/',case['target_event'])
    trace=case['trace']
    for key,step in trace['stages'].items():
        print('\n---',key,'---');print('记录ID:',step['call_id']);print(step['text'])
    print('\n探索票数:',trace['yes_count'],'路径:',trace['route'],'最终预测:',trace['e1'],'标注:',case['label'])
    if 'prediction_E2' in case:print('E0 / E1 / E2:',case['prediction_E0'],case['prediction_E1'],case['prediction_E2'])
    print('注：Yes/No为因果存在性；本实验不输出因果方向或概率。')

if __name__=='__main__':main()
