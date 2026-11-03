import importlib
import sys
from collections import Counter
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load(name):
    assert (ROOT / (name + '.py')).exists(), f'{name} implementation is missing'
    return importlib.import_module(name)

class CharTokenizer:
    """Token-length contract fixture, never model-performance evidence."""
    def encode(self, text, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [ord(c) + 10 for c in text]


def test_transition_balance_and_disjoint_splits():
    d = load('data')
    train = [d.make_group(i, 'train', 7) for i in range(24)]
    assert Counter((g.old, g.new) for g in train) == Counter({(a,b):2 for a in 'ABCD' for b in 'ABCD' if a!=b})
    assert Counter(g.new for g in train) == Counter({x:6 for x in 'ABCD'})
    assert len({g.entity for g in train}) == 24
    test = [d.make_group(i, 'test', 7) for i in range(24)]
    assert not ({g.entity for g in train} & {g.entity for g in test})
    assert not ({g.template for g in train} & {g.template for g in test})
    assert d.make_group(8, 'train', 7) == d.make_group(8, 'train', 7)


def test_group_semantics_not_state_equality():
    d = load('data'); g = d.make_group(0, 'train', 7)
    assert (g.old, g.new) == ('A','B')
    assert g.other_old not in (g.old, g.new)
    rows = d.render_group(g, CharTokenizer(), length=2048, gap=100)
    assert len(rows) == 8
    by = {r.kind:r for r in rows}
    assert by['update_a'].gold == by['update_c'].gold == 'B'
    assert by['earliest_a'].gold == 'A'
    assert by['earliest_c'].gold == g.other_old
    assert by['protect_a'].gold == by['protect_c'].gold == g.protected
    assert by['new_only'].gold == by['repeat_new'].gold == 'B'
    assert all(len(r.token_ids)==2048 for r in rows)
    assert len({r.update_position for r in rows}) == 1
    for r in rows:
        assert r.token_ids[r.update_position] == ord('B')+10
        assert set(r.choice_ids) == set('ABCD')
    # Equal prefix to and including the update for alternate questions.
    stop = by['update_a'].update_position + 1
    assert by['update_a'].token_ids[:stop] == by['protect_a'].token_ids[:stop] == by['earliest_a'].token_ids[:stop]
    assert by['update_a'].group_id == by['update_c'].group_id


def test_no_truncation_and_invalid_budget():
    d = load('data'); g=d.make_group(0,'val',0)
    with pytest.raises(ValueError): d.render_group(g,CharTokenizer(),length=128,gap=100)
    with pytest.raises(ValueError): d.make_group(0,'unknown',0)
    with pytest.raises(ValueError): d.render_group(g,CharTokenizer(),length=1024,gap=-1)


def test_data_fingerprint_and_order():
    d=load('data')
    a=d.build_groups(CharTokenizer(),split='train',seed=1,count=12,length=2048,gap=100)
    b=d.build_groups(CharTokenizer(),split='train',seed=1,count=12,length=2048,gap=100)
    assert d.dataset_hash(a)==d.dataset_hash(b)
    with pytest.raises(ValueError): d.build_groups(CharTokenizer(),split='train',seed=1,count=13,length=2048,gap=100)


def test_js_zero_symmetric_and_gradients():
    o=load('objectives')
    a=torch.tensor([1.,2.,3.,4.],requires_grad=True)
    b=torch.tensor([3.,1.,2.,4.],requires_grad=True)
    assert o.js_logits(a,a).item() == pytest.approx(0.,abs=1e-6)
    v=o.js_logits(a,b)
    assert v.item() == pytest.approx(o.js_logits(b,a).item(),abs=1e-6)
    assert 0 < v < .693148
    v.backward(); assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    assert torch.isfinite(o.js_logits(torch.tensor([1e4,-1e4,0.,2.]),b))


def test_objective_ce_and_consistency_only_current_pair():
    o=load('objectives')
    logits=torch.tensor([[3.,0.,0.,0.],[0.,3.,0.,0.],[1.,2.,0.,0.]],requires_grad=True)
    gold=torch.tensor([0,0,1])
    ce=torch.nn.functional.cross_entropy(logits,gold)
    loss,terms=o.group_objective(logits,gold,[(0,1)],js_weight=.5)
    assert loss.item()==pytest.approx((ce+.5*o.js_logits(logits[0],logits[1])).item())
    loss.backward(); assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError): o.group_objective(logits,gold,[(0,2)],js_weight=.5)
    with pytest.raises(ValueError): o.group_objective(logits,gold,[(0,1)],js_weight=-1.)


def test_counterfactual_old_values_are_balanced_too():
    d=load('data'); groups=[d.make_group(i,'train',42) for i in range(12)]
    assert Counter((g.other_old,g.new) for g in groups)==Counter({(a,b):1 for a in 'ABCD' for b in 'ABCD' if a!=b})
