import copy
import json
import unittest
from pathlib import Path
from experiment import ROOT, build_messages, parse_answer, parse_mediator, prediction, metrics

class IntegrityTests(unittest.TestCase):
    def test_prompt_independent_of_gold(self):
        row=json.loads((ROOT/'data_processed/train.jsonl').read_text().splitlines()[0])
        changed=copy.deepcopy(row);changed['label']=1-row['label'];changed['original_label']='MUST_NOT_REACH_LLM'
        for stage in ['E0','explorer_causal','explorer_causation','explorer_causality','mediator','direct','indirect']:
            self.assertEqual(build_messages(row,stage,['rain']),build_messages(changed,stage,['rain']))
            self.assertNotIn('MUST_NOT_REACH_LLM',str(build_messages(changed,stage)))
    def test_final_answer_not_rule_intermediate(self):
        self.assertEqual(parse_answer('Some analysis {"Answer":"Yes"}\nFinal: { "Answer" : "No" }'),0)
        self.assertIsNone(parse_answer('No clear answer; maybe yes.'))
        self.assertIsNone(parse_answer('{"Answer":"Yes or No"}'))
        self.assertEqual(parse_answer('{"Answer":"Yes","evidence":""quoted text""}'),1)
    def test_missing_mediator_does_not_silently_become_direct(self):
        self.assertEqual(parse_mediator('{"mediators":[]}'),[])
        self.assertIsNone(parse_mediator('No parseable response'))
    def test_threshold_gates_before_failed_reasoner(self):
        row={'yes_count':1,'e1':None}
        self.assertIsNone(prediction(row,'E1',1))
        self.assertEqual(prediction(row,'E1',2),0)
    def test_failure_included_and_penalized(self):
        rows=[{'input_hash':str(i),'label':gold} for i,gold in enumerate([1,0,1,0])]
        traces={'0':{'e0':1},'1':{'e0':0},'2':{'e0':None},'3':{'e0':None}}
        result=metrics(rows,traces,'E0')
        self.assertEqual(result['n'],4);self.assertEqual(result['failures'],2)
        self.assertEqual(result['accuracy'],.5);self.assertEqual(result['f1'],.5)
    def test_topic_and_document_isolation(self):
        audit=json.loads((ROOT/'data_info/audit.json').read_text())
        self.assertEqual(audit['source_rows'],9978)
        self.assertTrue(all(all(v==0 for v in pair.values()) for pair in audit['overlap'].values()))
        self.assertEqual(sum(x['rows'] for x in audit['stats'].values()),9978)

if __name__=='__main__':unittest.main(verbosity=2)
