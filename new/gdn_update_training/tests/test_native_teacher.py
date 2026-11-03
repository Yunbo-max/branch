import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def load(name):
    assert (ROOT/(name+'.py')).exists(), f'{name} implementation is missing'
    return importlib.import_module(name)

# Independent differentiable CPU reference with the same keyword boundary.
# It is not FLA/CUDA, and is never used by the pretrained CLI.
def chunk_gated_delta_rule(*,q,k,v,g,beta,initial_state=None,output_final_state=False,
                           use_qk_l2norm_in_kernel=True,**kwargs):
    if use_qk_l2norm_in_kernel:
        q,k=F.normalize(q,dim=-1),F.normalize(k,dim=-1)
    b,t,h,d=k.shape
    state=k.new_zeros(b,h,v.shape[-1],d) if initial_state is None else initial_state
    out=[]
    for i in range(t):
        state=state*g[:,i].exp()[...,None,None]
        pred=torch.einsum('bhvd,bhd->bhv',state,k[:,i])
        state=state+beta[:,i][...,None,None]*(v[:,i]-pred)[...,None]*k[:,i][...,None,:]
        out.append(torch.einsum('bhvd,bhd->bhv',state,q[:,i]))
    return torch.stack(out,dim=1),state if output_final_state else None

fused_recurrent_gated_delta_rule=chunk_gated_delta_rule

class ReferenceLayer(nn.Module):
    allow_neg_eigval=False
    num_heads=1
    num_v_heads=1
    def forward(self,q,k,v,g,beta,fail=False):
        result=chunk_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,
                                     use_qk_l2norm_in_kernel=True)
        if fail: raise RuntimeError('deliberate downstream failure')
        return result[0]


def inputs():
    q=torch.tensor([[[[1.,0.]],[[1.,0.]]]])
    k=q.clone().requires_grad_()
    v=torch.tensor([[[[1.,0.]],[[0.,1.]]]])
    g=torch.zeros(1,2,1)
    beta=torch.tensor([[[1.],[.1]]],requires_grad=True)
    return q,k,v,g,beta


def test_native_tap_noop_and_gradient_at_single_site():
    n=load('native'); layer=ReferenceLayer(); xs=inputs()
    baseline=layer(*xs)
    fn=chunk_gated_delta_rule
    with n.NativeTap(layer,position=1,head=0) as tap:
        got=layer(*xs)
        assert tap.capture is not None
        assert tap.calls==1
        assert torch.allclose(got,baseline)
        assert torch.allclose(tap.capture.key,torch.tensor([1.,0.]))
        (tap.capture.key.sum()+tap.capture.beta).backward()
    assert chunk_gated_delta_rule is fn
    assert xs[1].grad is not None and xs[-1].grad is not None
    assert len(layer._forward_hooks)==len(layer._forward_pre_hooks)==0


def test_native_tap_changes_only_selected_write_and_exports_nothing():
    n=load('native'); layer=ReferenceLayer(); xs=inputs()
    orig=layer(*xs)
    state_keys=tuple(layer.state_dict())
    key=torch.tensor([1.,0.],requires_grad=True)
    beta=torch.tensor(.9,requires_grad=True)
    with n.NativeTap(layer,position=1,head=0,override=n.Write(key,beta)):
        got=layer(*xs)
    assert torch.equal(got[:,0],orig[:,0])
    assert got[0,1,0,1]>orig[0,1,0,1]
    got[0,1,0,1].backward()
    assert beta.grad.item()>0
    assert tuple(layer.state_dict())==state_keys
    assert torch.equal(layer(*xs),orig)


def test_native_tap_restores_on_error_and_rejects_unsupported_shape():
    n=load('native'); layer=ReferenceLayer(); fn=chunk_gated_delta_rule
    with pytest.raises(RuntimeError,match='deliberate'):
        with n.NativeTap(layer,position=1,head=0): layer(*inputs(),fail=True)
    assert chunk_gated_delta_rule is fn
    assert not layer._forward_pre_hooks and not layer._forward_hooks
    with pytest.raises(ValueError):
        with n.NativeTap(layer,position=999,head=0): layer(*inputs())
    assert chunk_gated_delta_rule is fn
    layer.allow_neg_eigval=True
    with pytest.raises(ValueError):
        with n.NativeTap(layer,position=1,head=0): pass


def test_teacher_improves_actual_replay_and_returns_detached_native_target():
    n=load('native'); t=load('teacher')
    base=n.Write(torch.tensor([1.,0.]),torch.tensor(.1))
    def evaluate(w):
        # Exact two-write reference: old coefficient 1-beta, new beta.
        loss=(1-w.beta).square()
        protected=torch.stack((w.key[1].square(), w.key[1].square()))
        return loss,protected
    result=t.search_write(base,evaluate,steps=12,lr=.3,penalty=.01,protect_tolerance=.001)
    assert result.accepted
    assert result.after < result.before
    assert result.target.beta>base.beta
    assert not result.target.key.requires_grad and not result.target.beta.requires_grad
    assert torch.linalg.norm(result.target.key).item()==pytest.approx(1.,abs=1e-6)
    assert 0<=result.target.beta<=1
    assert result.replays>=2


def test_teacher_safeguard_and_failure_never_accept_bad_candidate():
    n=load('native'); t=load('teacher'); base=n.Write(torch.tensor([1.,0.]),torch.tensor(.2))
    def unsafe(w): return -w.beta,torch.stack((w.beta,))
    result=t.search_write(base,unsafe,steps=5,lr=.3,penalty=0.,protect_tolerance=0.)
    assert not result.accepted
    assert torch.equal(result.target.beta,base.beta)
    def nonfinite(w): return w.beta*float('nan'),torch.zeros(1)
    with pytest.raises(FloatingPointError): t.search_write(base,nonfinite,steps=2)


def test_teacher_input_validation():
    n=load('native');t=load('teacher');b=n.Write(torch.tensor([1.,0.]),torch.tensor(.1))
    for kwargs in ({'steps':0},{'lr':-1},{'penalty':float('nan')},{'protect_tolerance':-1}):
        with pytest.raises(ValueError): t.search_write(b,lambda w:(w.beta,torch.zeros(1)),**kwargs)
