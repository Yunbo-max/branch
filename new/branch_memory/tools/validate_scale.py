#!/usr/bin/env python3
"""Bounded CPU forward/backward validation, NOT a speed or task benchmark."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import platform
import resource
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from branch_memory import BranchMemoryMixer, MemoryConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d-key',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--heads',type=int,default=2)
    p.add_argument('--states',type=int,default=8)
    p.add_argument('--seq-len',type=int,default=16)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--differentiate-covariance',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results/scale128.json'))
    a=p.parse_args()
    for n in ['d_key','batch_size','heads','states','seq_len','repeats']:
        if getattr(a,n)<1:p.error(f'{n} must be positive')
    torch.set_num_threads(1);torch.manual_seed(2026)
    records=[]
    for covariance in ['diagonal','full']:
        torch.manual_seed(2026)
        cfg=MemoryConfig(d_model=a.heads*a.d_key,n_heads=a.heads,d_key=a.d_key,d_value=a.d_key,
                         d_route=16,max_states=a.states,covariance=covariance,
                         differentiate_covariance=a.differentiate_covariance)
        model=BranchMemoryMixer(cfg)
        x=torch.randn(a.batch_size,a.seq_len,cfg.d_model,requires_grad=True)
        elapsed=[]
        for rep in range(a.repeats):
            model.zero_grad(set_to_none=True);x.grad=None
            start=time.perf_counter()
            y,cache,stats=model(x)
            loss=y.square().mean();loss.backward()
            elapsed.append(time.perf_counter()-start)
            assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if covariance=='full':
                eig=torch.linalg.eigvalsh(cache.protection.detach())
                minimum_eigenvalue=float(eig.min())
        record={'config':asdict(cfg),'batch_size':a.batch_size,'seq_len':a.seq_len,
                'forward_backward_seconds':elapsed,'cache_bytes_by_field':cache.tensor_bytes_by_field(),
                'cache_tensor_bytes':cache.allocated_tensor_bytes(),
                'max_active_states':int(stats['active_states'].max()),
                'loss':float(loss.detach()),'finite_backward':True,
                'minimum_second_moment_eigenvalue':minimum_eigenvalue if covariance=='full' else None}
        records.append(record);print(json.dumps(record),flush=True)
        del model,x,y,cache,stats,loss;gc.collect()
    report={'scope':'CPU engineering dimension check; no matched-quality/native task or GPU speed claim',
            'torch':str(torch.__version__),'python':platform.python_version(),'threads':1,'records':records,
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'process_peak_rss_bytes':report['process_peak_rss_bytes']}))

if __name__=='__main__':main()
