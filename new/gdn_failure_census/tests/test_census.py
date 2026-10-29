"""Dependency-free regression tests; no weights, tokenizer download, or GPU."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import MockTokenizer, last_logit_kwargs, single_token_choices, validate_loading_info
from spec import CONDITIONS, make_probe
from summarize import summarize_records
import run as census_run


class CharTokenizer:
    def encode(self,text,add_special_tokens=False):
        return [ord(c)+10 for c in text]


def rec(pair,condition,correct=1,length=512):
    return {'status':'ok','probe_id':f'{pair}-{condition}','pair_id':pair,'condition':condition,
            'actual_length':length,'needle_fraction':.15,'num_events':8,'correct':correct,
            'margin':.2,'gold_conditional_nll':.3,'latency_ms':10.,'is_context_extrapolated':False}


class CensusTests(unittest.TestCase):
    def test_all_conditions_reproducible_and_exact(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                kwargs=dict(condition=condition,length=4096,seed=17,needle_fraction=.3,num_events=8)
                p,q=make_probe(CharTokenizer(),**kwargs),make_probe(CharTokenizer(),**kwargs)
                self.assertEqual(p,q)
                self.assertEqual(len(p.token_ids),4096)
                self.assertNotIn('token_ids',p.to_meta())
                self.assertTrue(p.question_suffix.endswith('Answer:'))

    def test_complete_suite_has_identical_absolute_target_offset(self):
        a=census_run.parse_args(['--backend','mock','--out','unused','--lengths','4096','--seeds','1','--positions','.3','--events','8'])
        probes=list(census_run._cases(a,MockTokenizer()))
        self.assertEqual(len(probes),len(CONDITIONS))
        self.assertEqual(len({p.needle_token for p in probes}),1)

    def test_paired_controls_share_fact(self):
        ps=[make_probe(CharTokenizer(),condition=c,length=4096,seed=42,needle_fraction=.17,num_events=16)
            for c in ('plain','unrelated','collision')]
        for attr in ('pair_id','answer','needle_token','key'):
            self.assertEqual(len({getattr(p,attr) for p in ps}),1)
        self.assertEqual(len({p.prompt_hash for p in ps}),3)

    def test_event_placement(self):
        kw=dict(condition='collision',length=4096,seed=123,num_events=8)
        a,b=(make_probe(CharTokenizer(),event_placement=x,**kw) for x in ('early','late'))
        self.assertEqual(a.needle_token,b.needle_token)
        self.assertLess(a.event_token,b.event_token)
        self.assertEqual(len(a.token_ids),len(b.token_ids))

    def test_revision_gold_semantics(self):
        for seed in range(8):
            kw=dict(length=4096,seed=seed,num_events=8)
            newest=make_probe(CharTokenizer(),condition='revision_latest',**kw)
            oldest=make_probe(CharTokenizer(),condition='revision_earliest',**kw)
            self.assertEqual(newest.pair_id,oldest.pair_id)
            self.assertEqual(newest.answer,oldest.latest_answer)
            self.assertEqual(oldest.answer,oldest.initial_answer)

    def test_refuses_support_truncation(self):
        with self.assertRaisesRegex(ValueError,'required text'):
            make_probe(CharTokenizer(),condition='collision',length=128,seed=17,num_events=100)

    def test_next_token_validation(self):
        ids=single_token_choices(CharTokenizer(),'\nQuestion: test?\nAnswer:')
        self.assertEqual(ids['A'],ord('A')+10)
        self.assertEqual(len(set(ids.values())),4)
        class Bad:
            def encode(self,text,**kwargs):
                return [1]
        with self.assertRaises(ValueError):
            single_token_choices(Bad(),'Answer:')

    def test_last_logit_memory_optimization_is_signature_checked(self):
        class New:
            def forward(self,input_ids,logits_to_keep=0): pass
        class Old:
            def forward(self,input_ids,num_logits_to_keep=0): pass
        class Other:
            def forward(self,input_ids,**kwargs): pass
        self.assertEqual(last_logit_kwargs(New()),{'logits_to_keep':1})
        self.assertEqual(last_logit_kwargs(Old()),{'num_logits_to_keep':1})
        self.assertEqual(last_logit_kwargs(Other()),{})

    def test_checkpoint_with_missing_weights_is_rejected(self):
        validate_loading_info({'missing_keys':[],'unexpected_keys':[]})
        for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError,'Checkpoint'):
                validate_loading_info({key:['bad.weight']})

    def test_paired_summary_small_n_is_provisional(self):
        rows=[rec('x','plain'),rec('x','unrelated'),rec('x','collision',0),
              rec('y','plain'),rec('y','unrelated'),rec('y','collision')]
        s=summarize_records(rows)
        self.assertEqual(s['short_control']['competence_gate'],'PROVISIONAL_SHORT_CONTROL')
        self.assertEqual(s['paired'][0]['n_triplets'],2)
        self.assertAlmostEqual(s['paired'][0]['unrelated_vs_collision_mean'],-.5)

    def test_short_failure_blocks_memory_claim(self):
        s=summarize_records([rec('x',c,0) for c in ('plain','unrelated','collision')])
        self.assertEqual(s['short_control']['competence_gate'],'INSUFFICIENT_SHORT_CONTROL')

    def test_dry_run_loads_no_weights(self):
        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls,*args,**kwargs): return CharTokenizer()
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules,{'transformers':SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)}), patch.object(census_run,'PretrainedCausalLM',side_effect=AssertionError('weights loaded')):
            rc=census_run.main(['--dry-run','--out',temp,'--lengths','2048','--seeds','1','--events','8','--positions','.2','--conditions','plain,unrelated,collision'])
            manifest=json.loads((Path(temp)/'manifest.json').read_text())
            self.assertEqual(rc,0)
            self.assertEqual(manifest['planned_cases'],3)
            self.assertFalse((Path(temp)/'results.jsonl').exists())


if __name__=='__main__': unittest.main()
