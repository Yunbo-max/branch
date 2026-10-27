import pytest
import torch
from gated_woodbury_geometry import (QueryProtectionGate,gated_recent_basis,
    dense_gated_direction,woodbury_gated_direction,gated_tail_bound,selected_rank)


def samples(dtype=torch.float64, T=25,d=8):
    torch.manual_seed(26)
    q = torch.nn.functional.normalize(torch.randn(2,T,3,d,dtype=dtype),dim=-1)
    k = torch.nn.functional.normalize(torch.randn(2,7,3,d,dtype=dtype),dim=-1)
    w = 1.+torch.rand(2,T,3,dtype=dtype)
    rho = 0.4+0.59*torch.rand(2,T,3,dtype=dtype)
    return q,k,w,rho

@pytest.mark.parametrize('r',[0,1,4,10,50])
def test_dual_exact_for_retained_covariance(r):
    q,k,w,rho=samples()
    r = min(q.shape[1],r)
    full=dense_gated_direction(q[:, -r:] if r else q[:,:0],w[:,-r:] if r else w[:,:0],rho[:,-r:] if r else rho[:,:0],k)
    dual=woodbury_gated_direction(q,w,rho,k,rank=r)
    torch.testing.assert_close(full,dual,atol=1e-12,rtol=1e-12)


def test_full_history_gradients_match():
    q,k,w,rho=samples(T=8,d=10)
    q,w,rho,k=[a.clone().requires_grad_(True) for a in (q,w,rho,k)]
    out0=dense_gated_direction(q,w,rho,k)
    out1=woodbury_gated_direction(q,w,rho,k,rank=8)
    torch.testing.assert_close(out0,out1,atol=1e-12,rtol=1e-12)
    t=(q,w,rho,k)
    grads0=torch.autograd.grad(out0.square().sum(),t,retain_graph=True)
    grads1=torch.autograd.grad(out1.square().sum(),t)
    for a,b in zip(grads0,grads1):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)


def test_truncation_bound():
    q,k,w,rho=samples(T=32,d=12)
    full=gated_recent_basis(q,w,rho,rank=32,mu=1.)
    for r in [0,1,4,16,31,32]:
        rec=gated_recent_basis(q,w,rho,rank=r,mu=1.)
        cov=full.transpose(-1,-2)@full - rec.transpose(-1,-2)@rec
        eig=torch.linalg.eigvalsh(cov)
        assert eig[...,0].min()>-1e-10
        assert eig[...,-1].max() <= gated_tail_bound(rho,r,w_max=2.).max()+1e-10


def test_gating_selects_short_history_or_dense_fallback():
    rho=torch.full((2,64,3),0.5)
    assert selected_rank(rho,tolerance=.01,max_rank=16)==8
    assert selected_rank(rho,tolerance=.01,max_rank=7) is None
    long_rho=torch.full((2,4096,3),0.999831)
    assert selected_rank(long_rho,tolerance=.01,max_rank=128) is None


def test_gate_init_and_gradient():
    g=QueryProtectionGate(16,4,initial_retention=0.95).double()
    x=torch.randn(2,11,16,dtype=torch.float64)
    out=g(x)
    torch.testing.assert_close(out,torch.full_like(out,.95),atol=1e-8,rtol=1e-8)
    (out.square().mean()).backward()
    assert g.linear.weight.grad.abs().sum()>0


def test_future_only_affects_later_chunks():
    q,k,w,rho=samples()
    a=woodbury_gated_direction(q[:,:12],w[:,:12],rho[:,:12],k,rank=8)
    q2=q.clone();q2[:,12:]=torch.randn_like(q2[:,12:])
    rho2=rho.clone();rho2[:,12:]=.999
    b=woodbury_gated_direction(q2[:,:12],w[:,:12],rho2[:,:12],k,rank=8)
    assert torch.equal(a,b)
