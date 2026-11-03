"""CPU contracts only. No test result from this module is pretrained evidence."""
from collections import Counter
from pathlib import Path
import sys
import pytest
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

class Tok:
    def encode(self,text,add_special_tokens=False):
        return ([1] if add_special_tokens else [])+[ord(c)+10 for c in text]

def test_balanced_binding_split():
    from rwdiag.panel import scenario
    cases=[scenario('train',i,71) for i in range(24)]
    for field in ('b','c'):
        assert set(Counter((s.a,getattr(s,field)) for s in cases).values())=={2}
        assert len(set((s.a,getattr(s,field)) for s in cases))==12
    assert all(s.x!=s.y and len({s.a,s.b,s.c,s.d})==4 for s in cases)
    assert not {s.x for s in cases}&{scenario('test',i,71).x for i in range(24)}

def test_swapped_worlds_and_donors():
    from rwdiag.panel import scenario,render
    s=scenario('test',2,71); f=lambda w,**opts:render(Tok(),s,'swap',w,'other',0,'x_now',**opts)
    a,b=f(0),f(1)
    assert a.question==b.question and a.gold!=b.gold and a.old==b.old
    assert Counter(a.prefix)==Counter(b.prefix)
    for donor in ['clean','wrong','self']:
        r=f(0,donor=donor); assert r.write==a.write and len(r.prefix)==len(a.prefix)
        if donor=='self':assert r.prefix==a.prefix
    assert f(0,donor='clean').prefix!=a.prefix
    assert f(0,donor='wrong').prefix[a.write]!=a.prefix[a.write]

def test_phases_and_crosstalk():
    from rwdiag.panel import scenario,render
    s=scenario('dev',0,71)
    def f(w,t,d=0):return render(Tok(),s,'cross',w,t,d,'x_now')
    assert f(0,'other').gold==f(1,'other').gold
    assert f(0,'other').distractor!=f(1,'other').distractor
    assert len(f(0,'neutral',128).prefix)==len(f(0,'records',128).prefix)
    assert f(0,'neutral',128).prefix[:len(f(0,'other').prefix)]==f(0,'other').prefix
    assert f(0,'write').gap==0 and f(0,'records',128).gap==f(0,'other').gap+128
    assert f(0,'pre').gold==s.a

def kernel(*,q,k,v,beta,g,output_final_state=False,**kw):
    q,k=F.normalize(q,dim=-1),F.normalize(k,dim=-1)
    state=q.new_zeros(1,1,k.shape[-1],v.shape[-1]); out=[]
    for t in range(q.shape[1]):
        ki=k[:,t]; vi=v[:,t]; pred=torch.einsum('bhk,bhkv->bhv',ki,state)
        state=state+beta[:,t,:,None,None]*ki.unsqueeze(-1)*(vi-pred).unsqueeze(-2)
        out.append(torch.einsum('bhk,bhkv->bhv',q[:,t],state)*q.shape[-1]**-.5)
    return torch.stack(out,1), state if output_final_state else None
chunk_gated_delta_rule=kernel
class Layer(nn.Module):
    num_heads=num_v_heads=1;allow_neg_eigval=False
    def forward(self,**kw):return chunk_gated_delta_rule(**kw)[0]

def fixture():
    torch.manual_seed(19)
    return dict(q=torch.randn(1,4,1,3),k=torch.randn(1,4,1,3),v=torch.randn(1,4,1,2),g=torch.zeros(1,4,1),beta=torch.full((1,4,1),.6),initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)

def test_q_only_leaves_state_and_other_tokens_unchanged():
    from rwdiag.boundary import observe
    layer=Layer(); kw=fixture()
    with observe(layer,0,(),True) as a: base=layer(**kw)
    qnew=torch.tensor([.2,.6,.1],requires_grad=True)
    with observe(layer,0,(('q',3,qnew),),True) as b: output=layer(**kw)
    assert torch.equal(a['state'],b['state']) and torch.equal(base[:,:-1],output[:,:-1])
    assert not torch.allclose(base[:,-1],output[:,-1]);output[:,-1].sum().backward()
    assert qnew.grad is not None and qnew.grad.norm()>0
    assert torch.allclose(base[0,-1,0], F.normalize(kw['q'][0,-1,0],dim=0)@a['state']/3**.5,atol=1e-6)
    assert chunk_gated_delta_rule is kernel

@pytest.mark.parametrize('field',['k','v','beta'])
def test_write_only_changes_selected_field(field):
    from rwdiag.boundary import observe
    layer=Layer(); kw=fixture(); base=layer(**kw)
    replacement=torch.zeros_like(kw[field][0,1,0])+.2
    with observe(layer,0,((field,1,replacement),),True): out=layer(**kw)
    assert torch.equal(base[:,0],out[:,0]);assert not torch.allclose(base[:,1:],out[:,1:])
    assert chunk_gated_delta_rule is kernel

def test_hook_failure_restores_original():
    from rwdiag.boundary import observe
    layer=Layer()
    with pytest.raises(ValueError):
        with observe(layer,0,(('v',100,torch.ones(2)),),False):layer(**fixture())
    assert chunk_gated_delta_rule is kernel and not layer._forward_hooks

def test_statistics_pairs_not_independent_rows():
    from rwdiag.stats import summarize
    rows=[]
    for sid in range(4):
        for arm,ok in [('native',False),('q',True),('k',False),('q+k',True)]:
            for w in (0,1):
                rows.append(dict(scene=sid,panel='swap',stage='other',distance=0,question='x_now',arm=arm,world=w,gold='AB'[w],pred='AB'[w] if ok else 'D',old='D',distractor='C',margin=1. if ok else -1.))
    r=summarize(rows,rounds=100)
    f=next(x for x in r['factorial'] if x['write']=='k')
    assert f['n_scenes']==4 and f['q_effect'][0]==1 and f['interaction'][0]==0
    with pytest.raises(ValueError):summarize(rows[:-1],rounds=100)
