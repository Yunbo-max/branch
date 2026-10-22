from pathlib import Path
import math
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]

def test_implementation_present():
    assert (ROOT/'branch_memory'/'math.py').exists(), 'closed-form solver has not been implemented'

def api():
    from branch_memory.math import correction, choose_write
    from branch_memory.mixer import MemoryConfig, BranchMemoryMixer
    from branch_memory.model import TinyMemoryLM, LMConfig, next_token_loss
    return correction, choose_write, MemoryConfig, BranchMemoryMixer, TinyMemoryLM, LMConfig, next_token_loss

def test_closed_form_matches_normal_equations():
    correction, *_ = api()
    torch.manual_seed(0)
    S=torch.randn(2,3,4,5,dtype=torch.double)
    k=torch.randn(2,3,5,dtype=torch.double)
    v=torch.randn(2,3,4,dtype=torch.double)
    h=torch.rand(2,3,5,dtype=torch.double)+.5
    tau=.7
    d,c=correction(S,k,v,h,tau)
    e=v-torch.einsum('...vk,...k->...v',S,k)
    lhs=torch.diag_embed(h)+k.unsqueeze(-1)*k.unsqueeze(-2)/tau
    direct=torch.linalg.solve(lhs,(e.unsqueeze(-1)*k.unsqueeze(-2)/tau).transpose(-1,-2)).transpose(-1,-2)
    torch.testing.assert_close(d,direct)
    objective=(torch.einsum('...vk,...k->...v',d,k)-e).square().sum(-1)/(2*tau)+(d.square()*h.unsqueeze(-2)).sum((-1,-2))/2
    torch.testing.assert_close(c,objective)

def test_solver_gradcheck():
    correction,*_=api()
    torch.manual_seed(2)
    S=torch.randn(2,3,dtype=torch.double,requires_grad=True)
    k=torch.randn(3,dtype=torch.double,requires_grad=True)
    v=torch.randn(2,dtype=torch.double,requires_grad=True)
    h=(torch.rand(3,dtype=torch.double)+1).requires_grad_()
    assert torch.autograd.gradcheck(lambda a,b,c,d:correction(a,b,c,d,.6),(S,k,v,h),eps=1e-6,atol=1e-5)

def test_isotropic_reduces_to_delta():
    correction,*_=api()
    S=torch.randn(2,3); k=torch.nn.functional.normalize(torch.randn(3),dim=-1); v=torch.randn(2)
    d,_=correction(S,k,v,torch.ones(3),1.)
    torch.testing.assert_close(d,.5*(v-S@k)[:,None]*k[None,:])

def test_fork_preserves_parent_and_uses_unprotected_update():
    _,choose,*_=api()
    S=torch.tensor([[[[[1.,0.]],[[0.,0.]]]]]) # B H M V K
    C=torch.tensor([[[[100.,0.],[0.,0.]]]])
    active=torch.tensor([[[True,False]]])
    k=torch.tensor([[[1.,0.]]]);v=torch.tensor([[[-1.]]])
    old=S.clone()
    result=choose(S,C,active,k,v,tau=1.,ridge=1.,protect=10.,fork_cost=.01)
    assert result.fork.item(); assert result.parent.item()==0; assert result.destination.item()==1
    torch.testing.assert_close(result.states[...,0,:,:],old[...,0,:,:])
    assert torch.equal(S,old)
    torch.testing.assert_close(result.current,torch.zeros(1,1,1,2))

def test_full_capacity_forces_replace():
    _,choose,*_=api()
    S=torch.zeros(1,1,1,2,2); C=torch.ones(1,1,1,2)*100
    a=torch.ones(1,1,1,dtype=torch.bool)
    result=choose(S,C,a,torch.tensor([[[1.,0.]]]),torch.ones(1,1,2),tau=1.,ridge=1.,protect=10.,fork_cost=0.)
    assert not result.fork.item();assert result.active.sum()==1

def test_zero_residual_never_forks():
    _,choose,*_=api()
    S=torch.zeros(1,1,3,2,2);C=torch.ones(1,1,3,2);a=torch.tensor([[[True,False,False]]])
    r=choose(S,C,a,torch.tensor([[[1.,0.]]]),torch.zeros(1,1,2),tau=1.,ridge=1.,protect=10.,fork_cost=.01)
    assert not r.fork.item();assert r.active.sum()==1

@pytest.mark.parametrize('mode',['delta','snapshot','branch'])
def test_causality_streaming_and_cache_immutability(mode):
    _,_,Cfg,Mixer,*_=api()
    torch.manual_seed(4)
    m=Mixer(Cfg(d_model=16,n_heads=2,d_key=4,d_value=4,d_route=4,max_states=4,mode=mode,snapshot_interval=3)).double().eval()
    x=torch.randn(2,9,16,dtype=torch.double)
    y,cache,_=m(x)
    altered=x.clone();altered[:,5:]+=5
    y2,_,_=m(altered)
    torch.testing.assert_close(y[:,:5],y2[:,:5],atol=1e-9,rtol=1e-8)
    ys=[];c=None
    for t in range(9):
        if c is not None:old=c.states.clone()
        z,new,_=m(x[:,t:t+1],c)
        if c is not None:torch.testing.assert_close(c.states,old)
        c=new;ys.append(z)
    torch.testing.assert_close(y,torch.cat(ys,1),atol=1e-8,rtol=1e-7)
    torch.testing.assert_close(cache.states,c.states,atol=1e-8,rtol=1e-7)
    assert int(c.active.sum(-1).max())<=4

def test_no_future_gradient():
    _,_,Cfg,Mixer,*_=api()
    m=Mixer(Cfg(d_model=16,n_heads=2,d_key=4,d_value=4,d_route=4,max_states=3))
    x=torch.randn(2,8,16,requires_grad=True)
    y,_,_=m(x);g=torch.autograd.grad(y[:,:4].square().sum(),x)[0]
    assert torch.count_nonzero(g[:,4:])==0

def test_snapshot_strict_boundary():
    _,_,Cfg,Mixer,*_=api()
    m=Mixer(Cfg(d_model=12,n_heads=1,d_key=4,d_value=4,d_route=4,max_states=3,mode='snapshot',snapshot_interval=2))
    _,_,s=m(torch.randn(1,5,12))
    assert s['historical_candidates'][0,:,0].tolist()==[0,0,1,1,2]

def test_branch_gradient_paths():
    _,_,Cfg,Mixer,*_=api()
    torch.manual_seed(8)
    m=Mixer(Cfg(d_model=16,n_heads=2,d_key=4,d_value=4,d_route=4,max_states=6,protect=20.,fork_cost=1e-6))
    x=torch.randn(2,12,16,requires_grad=True)
    y,cache,stats=m(x);(y.square().mean()+.05*y.mean()).backward()
    assert stats['fork'].any()
    for name in ['q_proj.weight','k_proj.weight','v_proj.weight','route_q.weight','content_proj','value_probes']:
        p=dict(m.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    assert cache.states.grad_fn is not None
    assert cache.detach().states.grad_fn is None

def test_lm_shift_matches_explicit_ce():
    *_,LM,LCfg,loss_fn=api()
    m=LM(LCfg(vocab_size=20,d_model=16,n_layers=1,n_heads=2,d_key=4,d_value=4,d_route=4,max_states=3))
    ids=torch.randint(20,(2,7));logits,_=m(ids)
    l=loss_fn(logits,ids)
    ref=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,20),ids[:,1:].reshape(-1))
    torch.testing.assert_close(l,ref)
    l.backward();assert m.embedding.weight.grad is not None

def test_config_rejects_bad_values():
    _,_,Cfg,*_=api()
    for kwargs in [{'max_states':0},{'tau':0},{'ridge':0},{'protect':-1},{'conv_kernel':0},{'mode':'nope'}]:
        with pytest.raises(ValueError):Cfg(**kwargs)

def test_snapshot_capacity_is_explicit_not_silent_eviction():
    _,_,Cfg,Mixer,*_=api()
    m=Mixer(Cfg(d_model=8,n_heads=1,d_key=2,d_value=2,d_route=2,mode='snapshot',max_states=1,snapshot_interval=2))
    with pytest.raises(RuntimeError,match='capacity exhausted'):
        m(torch.randn(1,4,8))

def test_zero_key_and_invalid_geometry():
    correction,*_=api()
    S=torch.ones(2,3);k=torch.zeros(3);v=torch.ones(2)
    d,c=correction(S,k,v,torch.ones(3),1.)
    torch.testing.assert_close(d,torch.zeros_like(S));torch.testing.assert_close(c,torch.tensor(1.))
    with pytest.raises(ValueError):correction(S,k,v,torch.zeros(3),1.)

def test_two_layer_lm_streaming():
    *_,LM,LCfg,_loss=api()
    torch.manual_seed(23)
    m=LM(LCfg(vocab_size=20,d_model=16,n_layers=2,n_heads=2,d_key=4,d_value=4,d_route=4,max_states=3)).eval()
    ids=torch.randint(20,(2,6));whole,_=m(ids);cache=None;pieces=[]
    for t in range(ids.shape[1]):
        part,_,cache=m(ids[:,t:t+1],cache,True);pieces.append(part)
    torch.testing.assert_close(whole,torch.cat(pieces,1),atol=3e-6,rtol=1e-5)

def test_training_entry_present():
    assert (ROOT/'train.py').exists(), 'training entry is not implemented'

def test_older_parent_can_win_without_mixing():
    _,choose,*_=api()
    S=torch.tensor([[[[[1.,0.]],[[0.,1.]],[[0.,0.]]]]])
    C=torch.zeros(1,1,3,2);a=torch.tensor([[[True,True,False]]])
    r=choose(S,C,a,torch.tensor([[[1.,0.]]]),torch.tensor([[[1.]]]),tau=1.,ridge=1.,protect=1.,fork_cost=.1)
    assert r.parent.item()==0 and not r.fork.item()
    torch.testing.assert_close(r.current,S[...,0,:,:])
    torch.testing.assert_close(r.states[...,1,:,:],S[...,1,:,:])

def test_null_only_read_is_zero_and_inactive_slots_do_not_leak():
    _,_,Cfg,Mixer,*_=api()
    m=Mixer(Cfg(d_model=8,n_heads=1,d_key=2,d_value=2,d_route=2,max_states=3))
    S=torch.randn(1,1,3,2,2)*1000
    a=torch.randn(1,1,3,2);q=torch.randn(1,1,2);rq=torch.randn(1,1,2);x=torch.randn(1,8)
    y,w=m.historical_read(q,rq,x,S,a,torch.zeros(1,1,3,dtype=torch.bool))
    assert not y.any() and not w.any()

def test_live_cache_does_not_survive_implicit_reset():
    _,_,Cfg,Mixer,*_=api()
    m=Mixer(Cfg(d_model=8,n_heads=1,d_key=2,d_value=2,d_route=2,max_states=3)).eval()
    x=torch.randn(1,5,8)
    first,_,_=m(x)
    m(torch.randn(1,20,8))
    second,_,_=m(x)
    torch.testing.assert_close(first,second)

def test_bytes_training_entry_runs_and_writes_checkpoint(tmp_path):
    import subprocess,sys,json,os
    (tmp_path/'train.txt').write_text('alpha beta gamma.\n'*5,encoding='utf-8')
    (tmp_path/'valid.txt').write_text('gamma delta alpha.\n'*5,encoding='utf-8')
    cmd=[sys.executable,str(ROOT/'train.py'),'--task','bytes','--train-text',str(tmp_path/'train.txt'),
         '--valid-text',str(tmp_path/'valid.txt'),'--steps','2','--batch-size','2','--seq-len','10',
         '--d-model','16','--heads','1','--layers','1','--d-key','4','--d-value','4','--d-route','4',
         '--max-states','2','--threads','1','--output',str(tmp_path/'run')]
    out=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=25,
                       env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    assert out.returncode==0,out.stderr
    report=json.loads((tmp_path/'run'/'report.json').read_text())
    assert report['config']['vocab_size']==256 and report['steps']==2
    ckpt=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=True,map_location='cpu')
    from branch_memory import TinyMemoryLM,LMConfig
    model=TinyMemoryLM(LMConfig(**ckpt['config']));model.load_state_dict(ckpt['model'])
    model.eval();logits,_=model(ckpt['heldout_tokens'])
    assert torch.isfinite(logits).all()
