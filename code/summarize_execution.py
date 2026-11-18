from pathlib import Path
import datetime as dt,json
from experiment import ROOT,write_json

calls={p.stem:json.loads(p.read_text()) for p in (ROOT/'results/api_calls').glob('*.json')}
result={}
for split in ['train','validation','test']:
    traces=[json.loads(p.read_text()) for p in (ROOT/f'results/traces/{split}').glob('*.json')]
    ids={i for t in traces for i in t['call_ids']}
    intervals=[]
    for i in ids:
        for a in calls[i]['attempts']:
            end=dt.datetime.fromisoformat(a['time_utc']).timestamp();intervals.append((end-a['elapsed_s'],end))
    result[split]={'unique_input_traces':len(traces),'unique_api_requests':len(ids),'all_attempts':sum(len(calls[i]['attempts']) for i in ids),
                   'api_first_start_utc':dt.datetime.fromtimestamp(min(x[0] for x in intervals),dt.timezone.utc).isoformat(),
                   'api_last_end_utc':dt.datetime.fromtimestamp(max(x[1] for x in intervals),dt.timezone.utc).isoformat(),
                   'api_schedule_wall_seconds':max(x[1] for x in intervals)-min(x[0] for x in intervals),
                   'component_elapsed_seconds':sum(calls[i]['elapsed_s'] for i in ids),
                   'charged_upper_cny':sum(calls[i]['charged_upper_cny'] for i in ids)}
result['timing_note']='API schedule wall time from first request start to last response end; concurrent batch wall time differs from sum of measured component latencies. Trace replay requires no new model call.'
frozen=json.loads((ROOT/'configs/frozen.json').read_text());assert result['test']['api_first_start_utc']>frozen['created_utc']
write_json(ROOT/'results/run_summary.json',result)
print(json.dumps(result,indent=2))
