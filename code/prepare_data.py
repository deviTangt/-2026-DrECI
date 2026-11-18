"""Recover provenance for the official Dr.ECI ESC CSV and freeze topic splits.

No answer sentence or label is used for matching or selecting split membership.
Repeated text can map to multiple source documents; all candidates are retained.
"""
from pathlib import Path
import collections
import csv
import hashlib
import itertools
import json
import random
import re
import secrets
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def norm(text):
    return ' '.join(text.lower().split())

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def read_docs():
    docs = {}
    for path in sorted((ROOT / 'data_raw/EventStoryLine/annotated_data/v0.9').glob('*/*.xml')):
        tree = ET.parse(path).getroot()
        sentences = collections.defaultdict(list)
        tokens = {}
        for tok in tree.findall('token'):
            sid = int(tok.attrib['sentence'])
            sentences[sid].append(tok.text or '')
            tokens[tok.attrib['t_id']] = {'text': tok.text, 'sentence': sid}
        events = {}
        for mark in tree.find('Markables'):
            if mark.tag.startswith('ACTION') or mark.tag.startswith('NEG_ACTION'):
                anchors = [x.attrib['t_id'] for x in mark.findall('token_anchor')]
                if anchors:
                    events[mark.attrib['m_id']] = {'tokens': anchors, 'text': ' '.join(tokens[a]['text'] for a in anchors), 'sentence': tokens[anchors[0]]['sentence']}
        links = []
        for link in tree.findall('.//PLOT_LINK'):
            s, t = link.find('source'), link.find('target')
            if s is not None and t is not None:
                links.append({'source': s.attrib['m_id'], 'target': t.attrib['m_id'], 'attributes': dict(link.attrib)})
        docs[path.name] = {'topic': path.parent.name, 'sentences': {str(k): ' '.join(v) for k, v in sentences.items()}, 'events': events, 'plot_links': links}
    return docs

def make_rows(docs):
    index = collections.defaultdict(set)
    for name, doc in docs.items():
        for s in doc['sentences'].values():
            words = norm(s).split()
            for j in range(len(words) - 4):
                index[tuple(words[j:j+5])].add(name)
    source = ROOT / 'upstream/data/zhang et/full_eventstoryline.csv'
    raw = list(csv.DictReader(source.open()))
    rows, unmatched = [], []
    for idx, item in enumerate(raw):
        context = item['Source sentence'].split('</s></s>')[-1].strip()
        normalized = norm(context)
        words = normalized.split()
        votes = collections.Counter()
        for j in range(len(words) - 4):
            votes.update(index.get(tuple(words[j:j+5]), set()))
        shortlist = [n for n, v in votes.most_common()]
        if not shortlist:
            shortlist = list(docs)
        matches = []
        for name in shortlist:
            ss = docs[name]['sentences']
            ns = {i: norm(t) for i, t in ss.items()}
            found = [[i] for i, t in ns.items() if t == normalized]
            if not found:
                found = [[i, j] for i, a in ns.items() for j, b in ns.items() if i != j and a+' '+b == normalized]
            if found:
                matches.append({'doc': name, 'sentence_ids': found})
        if not matches:
            unmatched.append({'row': idx, 'context': context})
        topics = sorted({docs[m['doc']]['topic'] for m in matches}, key=int)
        src, tgt = item['event1'].strip(), item['event2'].strip()
        signature = hashlib.sha256(json.dumps([normalized,norm(src),norm(tgt),item['type']], ensure_ascii=False).encode()).hexdigest()
        rows.append({'sample_id': f'ESC-CSV-{idx:05d}', 'original_row': idx, 'context': context, 'source_event': src, 'target_event': tgt,
                     'label': int(item['label']=='Cause-Effect'), 'original_label': item['label'], 'scope': item['type'],
                     'document_candidates': matches, 'topics': topics, 'input_hash': signature})
    if unmatched:
        write_json(ROOT/'data_info/unmatched.json', unmatched)
        raise ValueError(f'{len(unmatched)} unmatched contexts; stop before freezing splits')
    return rows

def split_rows(rows):
    path = ROOT / 'configs/split.json'
    if path.exists():
        config = json.loads(path.read_text())
    else:
        topics = sorted({t for row in rows for t in row['topics']}, key=int)
        parent = {t:t for t in topics}
        def find(t):
            while parent[t]!=t:
                parent[t]=parent[parent[t]];t=parent[t]
            return t
        def union(a,b): parent[find(b)]=find(a)
        # Keep ambiguous provenance AND exact duplicate input text in one split.
        by_signature = collections.defaultdict(set)
        for row in rows:
            for t in row['topics'][1:]: union(row['topics'][0],t)
            by_signature[row['input_hash']].update(row['topics'])
        for ts in by_signature.values():
            ts=sorted(ts)
            for t in ts[1:]: union(ts[0],t)
        groups=collections.defaultdict(list)
        for t in topics:groups[find(t)].append(t)
        groups=sorted(groups.values(),key=lambda g:int(g[0]))
        seed=secrets.randbelow(900000)+100000
        random.Random(seed).shuffle(groups)
        n=len(groups); ndev=max(1,round(n*.2)); ntest=max(1,round(n*.2))
        config={'seed':seed,'unit':'topic-connected-components','ratios_requested':[.6,.2,.2],
                'train_topics':sorted(sum(groups[:n-ndev-ntest],[]),key=int),
                'validation_topics':sorted(sum(groups[n-ndev-ntest:n-ntest],[]),key=int),
                'test_topics':sorted(sum(groups[n-ntest:],[]),key=int),
                'protocol':'Custom fixed topic-disjoint 60/20/20 split of complete official Dr.ECI ESC CSV; not the original paper test protocol.'}
        write_json(path,config)
    split={t:name for name in ['train','validation','test'] for t in config[name+'_topics']}
    for row in rows:
        memberships={split[t] for t in row['topics']}
        assert len(memberships)==1
        row['split']=memberships.pop()
    return config

def main():
    docs=read_docs();rows=make_rows(docs);config=split_rows(rows)
    data=ROOT/'data_processed';data.mkdir(exist_ok=True)
    write_json(data/'documents.json',docs)
    stats={}
    for split in ['train','validation','test']:
        subset=[x for x in rows if x['split']==split]
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in subset))
        stats[split]={'rows':len(subset),'unique_inputs':len({x['input_hash'] for x in subset}),
                      'documents':len({m['doc'] for x in subset for m in x['document_candidates']}),
                      'topics':config[split+'_topics'], 'positive':sum(x['label'] for x in subset),
                      'negative':sum(1-x['label'] for x in subset),
                      'scopes':{sc:{'n':sum(x['scope']==sc for x in subset),'positive':sum(x['label'] for x in subset if x['scope']==sc)} for sc in ['intra','inter']}}
    seen=collections.defaultdict(list)
    for row in rows:seen[row['input_hash']].append(row)
    conflicts=[{'input_hash':k,'sample_ids':[x['sample_id'] for x in v],'labels':[x['label'] for x in v]} for k,v in seen.items() if len({x['label'] for x in v})>1]
    overlap={}
    for a,b in itertools.combinations(['train','validation','test'],2):
        ra=[x for x in rows if x['split']==a];rb=[x for x in rows if x['split']==b]
        overlap[f'{a}/{b}']={'documents':len({m['doc'] for x in ra for m in x['document_candidates']} & {m['doc'] for x in rb for m in x['document_candidates']}),
                              'input_hashes':len({x['input_hash'] for x in ra}&{x['input_hash'] for x in rb})}
    assert all(not any(v.values()) for v in overlap.values())
    source=ROOT/'upstream/data/zhang et/full_eventstoryline.csv'
    audit={'source_rows':len(rows),'raw_documents':len(docs),'source_sha256':digest(source),'stats':stats,
           'overlap':overlap,'ambiguous_document_rows':sum(len(x['document_candidates'])>1 for x in rows),
           'duplicate_input_groups':sum(len(v)>1 for v in seen.values()),'duplicate_extra_rows':sum(len(v)-1 for v in seen.values()),
           'label_conflicts':conflicts,'label_policy':'Preserve every original label and every source row. Reuse identical model inputs, score all original rows.',
           'input_policy':'Remove constant causal/unrelated prefix; never feed Answer sentence or labels to models. CSV has event strings but no occurrence offsets.',
           'split_config':config,
           'upstream_commit':subprocess.check_output(['git','-C',str(ROOT/'upstream'),'rev-parse','HEAD'],text=True).strip(),
           'esc_commit':subprocess.check_output(['git','-C',str(ROOT/'data_raw/EventStoryLine'),'rev-parse','HEAD'],text=True).strip(),
           'files':{s:digest(data/f'{s}.jsonl') for s in stats}}
    write_json(ROOT/'data_info/audit.json',audit)
    write_json(ROOT/'data_info/input_example.json',{'sample':rows[0], 'model_input_fields':['context','source_event','target_event']})
    print(json.dumps({k:audit[k] for k in ['source_rows','stats','overlap','ambiguous_document_rows','duplicate_extra_rows']},ensure_ascii=False,indent=2))
    print('Label conflict groups:',len(conflicts))

if __name__=='__main__':main()
