import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def load():
    assert (ROOT/'train.py').exists(), 'training entry point is missing'
    return importlib.import_module('train')


def test_parameter_scope_and_no_new_parameters():
    t=load()
    class Attn(nn.Module):
        def __init__(self):
            super().__init__(); self.k_proj=nn.Linear(3,3); self.b_proj=nn.Linear(3,1)
            self.o_proj=nn.Linear(3,3)
    model=nn.Module(); model.attn=Attn(); model.other=nn.Linear(3,3)
    before={k:tuple(v.shape) for k,v in model.state_dict().items()}
    names=t.select_trainable(model,'projections')
    assert set(names)=={'attn.k_proj.weight','attn.k_proj.bias','attn.b_proj.weight','attn.b_proj.bias'}
    assert not model.attn.o_proj.weight.requires_grad
    assert {k:tuple(v.shape) for k,v in model.state_dict().items()}==before
    with pytest.raises(ValueError): t.select_trainable(nn.Linear(3,3),'projections')


def test_frozen_teacher_restores_flags_even_on_exception():
    t=load(); m=nn.Sequential(nn.Linear(2,2),nn.Linear(2,1))
    m[1].weight.requires_grad_(False)
    flags=[p.requires_grad for p in m.parameters()]
    with pytest.raises(RuntimeError):
        with t.frozen_parameters(m):
            assert not any(p.requires_grad for p in m.parameters())
            raise RuntimeError('exit')
    assert flags==[p.requires_grad for p in m.parameters()]


def test_plan_cli_does_not_load_or_overwrite(tmp_path,capsys):
    t=load(); out=tmp_path/'plan'
    assert t.main(['plan','--out',str(out),'--mode','write_distill','--steps','2','--train-groups','12','--eval-groups','12'])==0
    doc=json.loads((out/'plan.json').read_text())
    assert doc['status']=='plan_only_no_weights_loaded'
    assert doc['train_queries']==96
    assert doc['teacher_replays_per_selected_query_upper_bound']==9
    with pytest.raises(FileExistsError): t.main(['plan','--out',str(out)])
    assert json.loads((out/'plan.json').read_text())==doc


def test_training_step_modes_real_autograd():
    t=load(); from data import make_group,render_group
    from test_learning import CharTokenizer
    rows=render_group(make_group(0,'train',0),CharTokenizer(),length=2048,gap=100)
    class SimpleRunner:
        def __init__(self): self.weights=nn.Parameter(torch.zeros(8,300)); self.model=nn.Linear(1,1)
        def forward(self,row,**kw): return self.weights[rows.index(row)],None
    losses=[]
    for mode in ('sft','consistency'):
        runner=SimpleRunner();opt=torch.optim.SGD([runner.weights],lr=.1)
        loss,metrics=t.student_loss(runner,rows,mode=mode,js_weight=.2)
        assert loss.requires_grad
        loss.backward();opt.step()
        assert runner.weights.abs().sum()>0
        assert metrics['ce']>0
        losses.append(float(loss.detach()))
    assert losses[0]==pytest.approx(losses[1]) # identical uniform distributions => JS zero


def test_native_evaluation_has_metrics_no_teacher(tmp_path):
    t=load(); from data import make_group,render_group
    from test_learning import CharTokenizer
    rows=render_group(make_group(0,'test',0),CharTokenizer(),length=2048,gap=100)
    class Fixture:
        def forward(self,row,**kwargs):
            assert not kwargs
            x=torch.zeros(300);x[row.choice_ids[row.gold]]=4
            return x,None
    report=t.evaluate(Fixture(),[rows],tmp_path/'eval.jsonl')
    assert report['n']==8 and report['accuracy']==1
    assert report['by_kind']['update_a']['n']==1
    assert report['stale_rate']==0
    assert len((tmp_path/'eval.jsonl').read_text().splitlines())==8
    with pytest.raises(FileExistsError): t.evaluate(Fixture(),[rows],tmp_path/'eval.jsonl')


def test_invalid_hyperparameters_fail_before_outputs(tmp_path):
    t=load()
    with pytest.raises(ValueError): t.main(['plan','--out',str(tmp_path/'bad'),'--teacher-steps','0'])
    assert not (tmp_path/'bad').exists()


def test_trainable_master_precision_and_restore():
    t=load(); m=nn.Linear(3,3).to(torch.bfloat16)
    m.bias.requires_grad_(False)
    assert hasattr(t,'promote_trainables'), 'BF16 trainable weights need FP32 optimizer storage'
    saved=t.promote_trainables(m)
    assert m.weight.dtype==torch.float32 and m.bias.dtype==torch.bfloat16
    t.restore_storage(m,saved)
    assert m.weight.dtype==torch.bfloat16


def test_write_distillation_is_not_an_alias_for_sft():
    t=load(); from data import make_group,render_group
    from test_learning import CharTokenizer
    from native import Write
    rows=render_group(make_group(0,'train',0),CharTokenizer(),length=2048,gap=100)
    class Fixture:
        def __init__(self):
            self.logits=nn.Parameter(torch.zeros(8,300))
            self.key=nn.Parameter(torch.tensor([1.,0.]))
            self.beta=nn.Parameter(torch.tensor(.2))
        def forward(self,row,*,tap=False):
            return self.logits[rows.index(row)],Write(self.key,self.beta) if tap else None
    runner=Fixture()
    target=SimpleNamespace(accepted=True,target=Write(torch.tensor([.98,.2]),torch.tensor(.8)))
    loss,metrics=t.student_loss(runner,rows,mode='write_distill',teacher=target)
    loss.backward()
    assert metrics['write_loss']>0 and metrics['teacher_accepted']
    assert runner.key.grad.abs().sum()>0 and runner.beta.grad.abs()>0
    assert runner.logits.grad.abs().sum()>0
