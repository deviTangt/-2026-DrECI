import collections
import json
import networkx as nx
from experiment import ROOT,write_json

docs=json.loads((ROOT/'data_processed/documents.json').read_text())
issues=[];types=collections.Counter();total=0
for name,doc in docs.items():
    graph=nx.DiGraph();graph.add_nodes_from(doc['events']);unknown=[];missing=[]
    for link in doc['plot_links']:
        rel=link['attributes'].get('relType');types[str(rel)]+=1
        a,b=link['source'],link['target']
        if a not in graph or b not in graph:missing.append(link);continue
        if rel=='PRECONDITION':graph.add_edge(a,b)
        elif rel=='FALLING_ACTION':graph.add_edge(b,a)
        else:unknown.append(link)
    total+=graph.number_of_edges()
    cyclic=[sorted(c) for c in nx.strongly_connected_components(graph) if len(c)>1]
    selfloops=list(nx.selfloop_edges(graph))
    if cyclic or unknown or missing or selfloops:issues.append({'doc':name,'cyclic_components':cyclic,'self_loops':selfloops,'unknown_relation_count':len(unknown),'non_event_endpoint_count':len(missing)})
write_json(ROOT/'data_info/graph_audit.json',{'raw_version':'ESC XML v0.9','documents':len(docs),'direction_mapping':'PRECONDITION: source→target; FALLING_ACTION: target→source','raw_relation_counts':dict(types),'mapped_unique_edges':total,'documents_with_cycles':sum(bool(x['cyclic_components'] or x['self_loops']) for x in issues),'issues':issues,'scope_note':'Raw original annotation sanity check only. Official Dr.ECI CSV binary labels are preserved; no inferred/transitive edges are added to evaluation gold.'})
print('Raw graphs checked:',len(docs),'cyclic docs:',sum(bool(x['cyclic_components'] or x['self_loops']) for x in issues))
