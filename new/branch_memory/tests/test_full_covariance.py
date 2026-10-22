"""Full second-moment geometry; these checks are not native benchmark results."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
import torch.nn.functional as F
from branch_memory import BranchMemoryMixer, MemoryConfig, TinyMemoryLM, LMConfig
import branch_memory.math as bm

ROOT = Path(__file__).resolve().parents[1]

def full_solver():
    fn = getattr(bm, 'correction_full', None)
    assert callable(fn), 'full-matrix correction is not implemented'
    return fn

def moment_update():
    from branch_memory import geometry
    return geometry.update_protection

def cfg(**kw):
    args = dict(d_model=12, n_heads=2, d_key=3, d_value=3,
                d_route=3, max_states=4, covariance='full')
    args.update(kw)
    return MemoryConfig(**args)

def test_full_is_default():
    assert getattr(MemoryConfig(), 'covariance', None) == 'full'

def test_dense_normal_equations_and_objective():
    solve = full_solver(); torch.manual_seed(93)
    S=torch.randn(2,3,4,5,dtype=torch.double)
    k=torch.randn(2,3,5,dtype=torch.double)
    v=torch.randn(2,3,4,dtype=torch.double)
    R=torch.randn(2,3,5,5,dtype=torch.double)
    H=R@R.mT+torch.eye(5,dtype=torch.double)*.7; tau=.4
    d,c=solve(S,k,v,H,tau)
    e=v-torch.einsum('...vk,...k->...v',S,k)
    lhs=H+k[...,None]*k[...,None,:]/tau
    rhs=e[...,None]*k[...,None,:]/tau
    expected=torch.linalg.solve(lhs,rhs.mT).mT
    torch.testing.assert_close(d,expected,atol=1e-11,rtol=1e-10)
    obj=(torch.einsum('...vk,...k->...v',d,k)-e).square().sum(-1)/(2*tau)
    obj=obj+torch.einsum('...vk,...kl,...vl->...',d,H,d)/2
    torch.testing.assert_close(c,obj)

def test_full_broadcast_over_bank():
    solve=full_solver();torch.manual_seed(8)
    S=torch.randn(2,3,4,2,3,dtype=torch.double)
    k=torch.randn(2,3,1,3,dtype=torch.double);v=torch.randn(2,3,1,2,dtype=torch.double)
    H=torch.eye(3,dtype=torch.double).expand(2,3,4,3,3)*2
    d,c=solve(S,k,v,H,.7)
    assert d.shape==S.shape and c.shape==(2,3,4)
    rd,rc=bm.correction(S,k,v,H.diagonal(dim1=-2,dim2=-1),.7)
    torch.testing.assert_close(d,rd);torch.testing.assert_close(c,rc)

def test_cross_terms_change_update():
    solve=full_solver()
    # Old queries live on (1,1); new key (1,-1) is invisible to that direction.
    q=torch.tensor([1.,1.],dtype=torch.double)/math.sqrt(2)
    k=torch.tensor([1.,-1.],dtype=torch.double)/math.sqrt(2)
    C=20*torch.outer(q,q);H=torch.eye(2,dtype=torch.double)+C
    S=torch.zeros(1,2,dtype=torch.double);v=torch.ones(1,dtype=torch.double)
    d,c=solve(S,k,v,H,1.)
    dd,cc=bm.correction(S,k,v,H.diagonal(),1.)
    torch.testing.assert_close(d@q,torch.zeros(1,dtype=torch.double),atol=1e-12,rtol=0)
    torch.testing.assert_close(d@k,torch.tensor([.5],dtype=torch.double))
    assert (d@k).item()>(dd@k).item()*4 and c.item()<cc.item()

def test_rotation_equivariance():
    solve=full_solver();torch.manual_seed(18)
    S=torch.randn(2,4,dtype=torch.double);k=torch.randn(4,dtype=torch.double)
    v=torch.randn(2,dtype=torch.double);R=torch.randn(4,4,dtype=torch.double)
    H=R@R.T+torch.eye(4,dtype=torch.double)
    O,_=torch.linalg.qr(torch.randn(4,4,dtype=torch.double))
    d,c=solve(S,k,v,H,.4)
    dr,cr=solve(S@O.T,O@k,v,O@H@O.T,.4)
    torch.testing.assert_close(dr,d@O.T);torch.testing.assert_close(cr,c)

def test_dense_solver_gradcheck_including_spd_geometry():
    solve=full_solver();torch.manual_seed(7)
    S=torch.randn(2,3,dtype=torch.double,requires_grad=True)
    k=torch.randn(3,dtype=torch.double,requires_grad=True)
    v=torch.randn(2,dtype=torch.double,requires_grad=True)
    R=torch.randn(3,3,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda s,k,v,r:solve(s,k,v,r@r.T+torch.eye(3,dtype=torch.double),.8),
                                   (S,k,v,R),atol=1e-5,eps=1e-6)

@pytest.mark.parametrize('bad', ['asymmetric','indefinite','nan'])
def test_dense_bad_geometry_fails_not_diagonal_fallback(bad):
    solve=full_solver();H=torch.eye(2)
    if bad=='asymmetric':H[0,1]=1.
    elif bad=='indefinite':H[1,1]=-1.
    else:H[0,0]=float('nan')
    with pytest.raises((ValueError,RuntimeError)):
        solve(torch.zeros(1,2),torch.ones(2),torch.ones(1),H,1.)

def test_dense_zero_key():
    solve=full_solver()
    d,c=solve(torch.ones(2,3),torch.zeros(3),torch.ones(2),torch.eye(3),.5)
    torch.testing.assert_close(d,torch.zeros_like(d));torch.testing.assert_close(c,torch.tensor(2.))

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_low_precision_solver_promotes(dtype):
    solve=full_solver()
    S=torch.zeros(2,3,dtype=dtype);k=torch.ones(3,dtype=dtype);v=torch.ones(2,dtype=dtype)
    H=torch.eye(3,dtype=dtype)
    d,c=solve(S,k,v,H,.5)
    assert d.dtype==torch.float32 and c.dtype==torch.float32
    torch.testing.assert_close(d,torch.full((2,3),1/3.5))

def test_full_selector_uses_offdiagonal_for_parent_choice():
    full_solver()
    S=torch.zeros(1,1,2,1,2,dtype=torch.double)
    v=torch.ones(1,1,1,dtype=torch.double)
    k=torch.tensor([[[1.,-1.]]],dtype=torch.double)/math.sqrt(2)
    q=torch.tensor([1.,1.],dtype=torch.double)/math.sqrt(2)
    # Same diagonals, opposite orientation. Full geometry favors slot 1.
    C=torch.stack([20*torch.outer(k.flatten(),k.flatten()),20*torch.outer(q,q)]).reshape(1,1,2,2,2)
    a=torch.ones(1,1,2,dtype=torch.bool)
    r=bm.choose_write(S,C,a,k,v,tau=1.,ridge=1.,protect=1.,fork_cost=.1)
    assert r.parent.item()==1 and not r.fork.item()
    rd=bm.choose_write(S,C.diagonal(dim1=-2,dim2=-1),a,k,v,tau=1.,ridge=1.,protect=1.,fork_cost=.1)
    assert rd.parent.item()==0  # exact tie of the two diagonal summaries

def test_full_fork_keeps_parent_and_resets_child_statistics():
    full_solver()
    S=torch.zeros(1,1,3,1,2,dtype=torch.double)
    C=torch.zeros(1,1,3,2,2,dtype=torch.double)
    C[...,0,:,:]=torch.tensor([[10.,9.],[9.,10.]],dtype=torch.double)
    a=torch.tensor([[[True,False,False]]])
    k=torch.tensor([[[1.,1.]]],dtype=torch.double)/math.sqrt(2)
    oldS,oldC=S.clone(),C.clone()
    r=bm.choose_write(S,C,a,k,torch.ones(1,1,1,dtype=torch.double),tau=1.,ridge=1.,protect=10.,fork_cost=.001)
    assert r.fork.item()
    torch.testing.assert_close(S,oldS);torch.testing.assert_close(C,oldC)
    torch.testing.assert_close(r.states[...,0,:,:],S[...,0,:,:])
    torch.testing.assert_close(r.protection[...,0,:,:],C[...,0,:,:])
    assert not r.protection[...,1,:,:].any()

def test_full_ema_exact_outer_and_psd():
    fn=moment_update(); C=torch.zeros(1,1,2,2,2,dtype=torch.double)
    q=torch.tensor([[[1.,-1.]]],dtype=torch.double)/math.sqrt(2)
    use=torch.tensor([[[1.,.25]]],dtype=torch.double);a=torch.tensor([[[True,True]]])
    actual=fn(C,q,use,a,decay=.9,covariance='full',detach=True)
    expected=.1*use[...,None,None]*torch.einsum('bhk,bhl->bhkl',q,q).unsqueeze(2)
    torch.testing.assert_close(actual,expected)
    assert actual[...,0,1].min()<0
    assert torch.linalg.eigvalsh(actual).min()>-1e-14
    assert not C.any()

def test_moment_gradient_policy_explicit():
    fn=moment_update()
    q=torch.tensor([[[.3,.7]]],dtype=torch.double,requires_grad=True)
    use=torch.tensor([[[.8]]],dtype=torch.double,requires_grad=True)
    C=torch.zeros(1,1,1,2,2,dtype=torch.double);a=torch.ones(1,1,1,dtype=torch.bool)
    result=fn(C,q,use,a,decay=.9,covariance='full',detach=False)
    gq,gu=torch.autograd.grad(result.sum(),(q,use))
    assert gq.abs().sum()>0 and gu.abs().sum()>0
    stopped=fn(C,q,use,a,decay=.9,covariance='full',detach=True)
    assert not stopped.requires_grad

@pytest.mark.parametrize('differentiate',[False,True])
def test_full_end_to_end_causality_streaming_gradients(differentiate):
    torch.manual_seed(31)
    m=BranchMemoryMixer(cfg(differentiate_covariance=differentiate,protect=20.,fork_cost=1e-5)).double()
    x=torch.randn(2,7,12,dtype=torch.double,requires_grad=True)
    y,cache,stats=m(x)
    assert cache.protection.shape==(2,2,4,3,3)
    assert cache.protection.requires_grad==differentiate
    assert cache.protection.diagonal(dim1=-2,dim2=-1).sum()>0
    off=cache.protection-torch.diag_embed(cache.protection.diagonal(dim1=-2,dim2=-1))
    assert off.abs().sum()>0
    torch.testing.assert_close(cache.protection,cache.protection.mT)
    assert torch.linalg.eigvalsh(cache.protection).min()>-1e-12
    g=torch.autograd.grad(y[:,:3].square().sum(),x,retain_graph=True)[0]
    assert not g[:,3:].any()
    y.square().mean().backward()
    for name in ['q_proj.weight','k_proj.weight','v_proj.weight','route_q.weight','value_probes','content_proj']:
        p=dict(m.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    with torch.no_grad():
        xx=x.detach();full,fc,_=m(xx);st=None;parts=[]
        for s,e in [(0,2),(2,5),(5,7)]:
            old=None if st is None else st.protection.clone()
            z,new,_=m(xx[:,s:e],st)
            if st is not None:torch.testing.assert_close(st.protection,old)
            st=new;parts.append(z)
        torch.testing.assert_close(full,torch.cat(parts,1),atol=1e-8,rtol=1e-7)
        torch.testing.assert_close(fc.protection,st.protection,atol=1e-9,rtol=1e-8)

def test_cache_mode_mismatch_rejected():
    m=BranchMemoryMixer(cfg());d=BranchMemoryMixer(cfg(covariance='diagonal'))
    x=torch.randn(1,3,12);_,dc,_=d(x)
    with pytest.raises(ValueError,match='(covariance|protection)'):
        m(x,dc)

def test_full_cache_memory_count_and_serialization(tmp_path):
    m=BranchMemoryMixer(cfg());x=torch.randn(2,3,12)
    _,c,_=m(x)
    assert c.protection.numel()*c.protection.element_size()==2*2*4*3*3*4
    torch.save({'cache':c.to_dict()},tmp_path/'c.pt')
    from branch_memory.cache import MemoryCache
    cc=MemoryCache.from_dict(torch.load(tmp_path/'c.pt',weights_only=True)['cache'])
    torch.testing.assert_close(cc.protection,c.protection)
    torch.testing.assert_close(m(x,c)[0],m(x,cc)[0])
    assert cc.allocated_tensor_bytes()==c.allocated_tensor_bytes()

@pytest.mark.parametrize('bad', [dict(covariance='lowrank'),dict(ridge=float('nan')),dict(protect=float('inf'))])
def test_new_config_checks(bad):
    with pytest.raises(ValueError):cfg(**bad)

def test_legacy_config_migration_is_explicit_diagonal():
    # Older checkpoints omitted covariance; migration must not silently change the model.
    assert callable(getattr(LMConfig,'from_dict',None)), 'config migration missing'
    old=dict(vocab_size=20,d_model=12,n_layers=1,n_heads=2,d_key=3,d_value=3,d_route=3,max_states=4)
    with pytest.warns(UserWarning,match='diagonal'):
        converted=LMConfig.from_dict(old)
    assert converted.covariance=='diagonal'

def test_cpu_autocast_preserves_full_covariance_precision():
    m=BranchMemoryMixer(cfg(protect=10.,fork_cost=.001))
    x=torch.randn(1,6,12)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        y,c,s=m(x)
        loss=y.float().square().mean()
    assert c.states.dtype==torch.float32 and c.protection.dtype==torch.float32
    assert torch.isfinite(y).all()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

def test_full_cli_and_checkpoint_roundtrip(tmp_path):
    cmd=[sys.executable,str(ROOT/'train.py'),'--covariance','full','--differentiate-covariance',
         '--steps','2','--seq-len','10','--batch-size','2','--d-model','12','--heads','1','--layers','1',
         '--d-key','3','--d-value','3','--d-route','3','--threads','1','--output',str(tmp_path/'run')]
    run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=30,
                       env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    assert run.returncode==0,run.stdout+run.stderr
    report=json.loads((tmp_path/'run'/'report.json').read_text())
    assert report['config']['covariance']=='full'
    assert report['config']['differentiate_covariance']
    assert report['final_heldout_batch']['covariance_tensor_bytes']>0
    ckpt=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=True,map_location='cpu')
    model=TinyMemoryLM(LMConfig.from_dict(ckpt['config']));model.load_state_dict(ckpt['model'])
    y,_=model(ckpt['heldout_tokens']);assert torch.isfinite(y).all()

@pytest.mark.parametrize('differentiate',[False,True])
def test_selected_write_respects_covariance_gradient_switch(differentiate):
    full_solver()
    S=torch.zeros(1,1,1,1,2,dtype=torch.double,requires_grad=True)
    C=torch.eye(2,dtype=torch.double).reshape(1,1,1,2,2).requires_grad_()
    r=bm.choose_write(S,C,torch.ones(1,1,1,dtype=torch.bool),
                      torch.tensor([[[.6,.8]]],dtype=torch.double),torch.ones(1,1,1,dtype=torch.double),
                      tau=1.,ridge=1.,protect=2.,fork_cost=.1,detach_protection=not differentiate)
    g=torch.autograd.grad(r.current.square().sum(),C,allow_unused=True)[0]
    if differentiate:assert g is not None and g.abs().sum()>0
    else:assert g is None

def test_dense_ridge_handles_rank_deficient_second_moment():
    solve=full_solver();K=16
    k=torch.ones(K,dtype=torch.double)/math.sqrt(K)
    C=torch.outer(k,k)
    H=torch.eye(K,dtype=torch.double)*.01+100*C
    S=torch.zeros(3,K,dtype=torch.double);v=torch.ones(3,dtype=torch.double)
    d,c=solve(S,k,v,H,.5)
    assert torch.isfinite(d).all() and torch.isfinite(c).all()
    residual=torch.ones(3,dtype=torch.double)-d@k
    expected=.5/(.5+1/100.01)
    torch.testing.assert_close(residual,torch.full((3,),expected,dtype=torch.double))

def test_full_second_moment_gradcheck():
    fn=moment_update()
    torch.manual_seed(8)
    C=torch.eye(2,dtype=torch.double).reshape(1,1,1,2,2).requires_grad_()
    q=torch.randn(1,1,2,dtype=torch.double,requires_grad=True)
    w=torch.tensor([[[.7]]],dtype=torch.double,requires_grad=True)
    a=torch.ones(1,1,1,dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda c,q,w:fn(c,q,w,a,decay=.9,covariance='full',detach=False),
                                   (C,q,w),atol=1e-5,eps=1e-6)

def test_causal_ema_trace_and_condition_bound():
    torch.manual_seed(45)
    m=BranchMemoryMixer(cfg(fork_cost=.001)).double().eval()
    with torch.no_grad():
        _,c,_=m(torch.randn(1,48,12,dtype=torch.double))
    # ||q|| <= 1 and each slot's use <= 1 => trace(C) <= 1.
    trace=c.protection.diagonal(dim1=-2,dim2=-1).sum(-1)
    assert trace.max()<=1+1e-12
    H=m.config.ridge*torch.eye(3,dtype=torch.double)+m.config.protect*c.protection
    eig=torch.linalg.eigvalsh(H)
    assert eig.min()>=m.config.ridge-1e-12
    assert (eig[...,-1]/eig[...,0]).max()<=1+m.config.protect/m.config.ridge+1e-10
