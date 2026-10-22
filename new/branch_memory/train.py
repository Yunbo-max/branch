#!/usr/bin/env python3
"""Native training entry; no downloads, APIs, CUDA extensions or hidden jobs.

The built-in recall fixture is a DEVELOPMENT DIAGNOSTIC, not official MQAR.
--fixed-batch tests optimization/overfitting only. Report held-out results separately.
User text is UTF-8 bytes with a fixed byte vocabulary; this is not a pretrained LM.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import platform
from pathlib import Path
import resource
import time
import torch
from branch_memory import TinyMemoryLM, LMConfig, next_token_loss


def recall_batch(batch_size: int, seq_len: int, generator: torch.Generator) -> torch.Tensor:
    if seq_len<10 or seq_len>38 or seq_len%2:
        raise ValueError('recall diagnostic requires an even seq_len in [10,38]')
    pairs=(seq_len-6)//2
    rows=[]
    for _ in range(batch_size):
        keys=torch.randperm(16,generator=generator)[:pairs]
        values=torch.randint(16,(pairs,),generator=generator)+16
        chosen=int(torch.randint(pairs,(1,),generator=generator))
        row=[32]
        for k,v in zip(keys.tolist(),values.tolist()):row.extend([k,v])
        row.extend([33,int(keys[chosen]),34,int(values[chosen]),35])
        rows.append(row)
    return torch.tensor(rows,dtype=torch.long)


def text_batch(data: torch.Tensor,batch_size:int,seq_len:int,generator:torch.Generator):
    if data.numel()<seq_len:raise ValueError('text too short for requested seq_len')
    starts=torch.randint(data.numel()-seq_len+1,(batch_size,),generator=generator)
    return torch.stack([data[int(s):int(s)+seq_len] for s in starts])


def gradient_summary(model):
    out={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():raise FloatingPointError(f'non-finite gradient: {name}')
            out[name]=float(p.grad.detach().norm())
    return out


def evaluate(model,tokens,recall=False):
    was_training=model.training;model.eval()
    with torch.no_grad():
        logits,stats,caches=model(tokens,return_cache=True)
        result={'nll':float(next_token_loss(logits,tokens)),
                'forks':int(sum(s['fork'].sum().item() for s in stats)),
                'max_active_states':int(max(s['active_states'].max().item() for s in stats)),
                'cache_tensor_bytes':sum(c.allocated_tensor_bytes() for c in caches),
                'covariance_tensor_bytes':sum(c.protection.numel()*c.protection.element_size() for c in caches),
                'covariance_offdiag_l2':float(sum((c.protection-torch.diag_embed(c.protection.diagonal(dim1=-2,dim2=-1))).square().sum() for c in caches).sqrt()) if model.config.covariance=='full' else 0.0}
        if recall:
            result['answer_accuracy']=float((logits[:,-3].argmax(-1)==tokens[:,-2]).float().mean())
            result['answer_nll']=float(torch.nn.functional.cross_entropy(logits[:,-3],tokens[:,-2]))
    model.train(was_training)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode',choices=['branch','snapshot','delta'],default='branch')
    p.add_argument('--covariance',choices=['full','diagonal'],default='full')
    p.add_argument('--differentiate-covariance',action='store_true',help='backpropagate through causal query/routing second moments; hard decisions remain discrete')
    p.add_argument('--protection-decay',type=float,default=.95)
    p.add_argument('--task',choices=['recall-diagnostic','bytes'],default='recall-diagnostic')
    p.add_argument('--fixed-batch',action='store_true',help='overfit one development batch, NOT a benchmark result')
    p.add_argument('--train-text',type=Path);p.add_argument('--valid-text',type=Path)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--steps',type=int,default=100);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--seq-len',type=int,default=24);p.add_argument('--d-model',type=int,default=48)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--layers',type=int,default=1)
    p.add_argument('--d-key',type=int,default=8);p.add_argument('--d-value',type=int,default=8)
    p.add_argument('--d-route',type=int,default=8);p.add_argument('--max-states',type=int,default=8)
    p.add_argument('--snapshot-interval',type=int,default=4)
    p.add_argument('--tau',type=float,default=1.);p.add_argument('--ridge',type=float,default=1.)
    p.add_argument('--protect',type=float,default=8.);p.add_argument('--fork-cost',type=float,default=.02)
    p.add_argument('--lr',type=float,default=.003);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--output',type=Path,default=Path('results/run'))
    a=p.parse_args()
    for name in ['steps','batch_size','seq_len','threads','log_every']:
        if getattr(a,name)<1:p.error(f'{name} must be positive')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA requested but unavailable')
    if a.mode=='snapshot' and a.seq_len//a.snapshot_interval>a.max_states:
        p.error('snapshot needs max_states >= floor(seq_len / snapshot_interval)')
    if a.task=='bytes' and (not a.train_text or not a.valid_text):
        p.error('bytes training requires separate --train-text and --valid-text')
    if a.task=='bytes' and a.train_text.resolve()==a.valid_text.resolve():
        p.error('train and validation paths must differ')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    gen=torch.Generator().manual_seed(a.seed+100)
    vgen=torch.Generator().manual_seed(a.seed+10000)
    device=torch.device(a.device)
    if a.task=='bytes':
        # UTF-8 decoding verifies text inputs; use a bounded file to respect the CPU profile.
        for path in [a.train_text,a.valid_text]:
            if path.stat().st_size>64*1024**2:p.error('reference text loader supports files <=64 MiB; shard larger corpora')
        tr=torch.tensor(list(a.train_text.read_text(encoding='utf-8').encode('utf-8')),dtype=torch.long)
        va=torch.tensor(list(a.valid_text.read_text(encoding='utf-8').encode('utf-8')),dtype=torch.long)
        sample=lambda:text_batch(tr,a.batch_size,a.seq_len,gen).to(device)
        held=text_batch(va,a.batch_size,a.seq_len,vgen).to(device);vocab=256
    else:
        sample=lambda:recall_batch(a.batch_size,a.seq_len,gen).to(device)
        held=recall_batch(a.batch_size,a.seq_len,vgen).to(device);vocab=36
    fixed=sample()
    cfg=LMConfig(vocab_size=vocab,d_model=a.d_model,n_layers=a.layers,n_heads=a.heads,
                 d_key=a.d_key,d_value=a.d_value,d_route=a.d_route,max_states=a.max_states,
                 mode=a.mode,snapshot_interval=a.snapshot_interval,tau=a.tau,ridge=a.ridge,
                 protect=a.protect,fork_cost=a.fork_cost,covariance=a.covariance,
                 differentiate_covariance=a.differentiate_covariance,protection_decay=a.protection_decay)
    model=TinyMemoryLM(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    a.output.mkdir(parents=True,exist_ok=True)
    initial=evaluate(model,fixed,a.task=='recall-diagnostic')
    held_initial=evaluate(model,held,a.task=='recall-diagnostic')
    start=time.perf_counter();cpu_start=time.process_time();history=[];first_grads={}
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    with (a.output/'metrics.jsonl').open('w',encoding='utf-8') as log:
        for step in range(1,a.steps+1):
            model.train();tokens=fixed if a.fixed_batch else sample()
            optimizer.zero_grad(set_to_none=True)
            # Each independent sequence starts with an empty/reset cache.
            logits,stats=model(tokens)
            loss=next_token_loss(logits,tokens)
            if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
            loss.backward()
            if step==1:first_grads=gradient_summary(model)
            total_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            record={'step':step,'nll':float(loss.detach()),'grad_norm':float(total_norm),
                    'forks':int(sum(s['fork'].sum().item() for s in stats)),
                    'max_active_states':int(max(s['active_states'].max().item() for s in stats)),
                    'elapsed_seconds':time.perf_counter()-start}
            history.append(record);log.write(json.dumps(record)+'\n');log.flush()
            if step==1 or step%a.log_every==0 or step==a.steps:print(json.dumps(record),flush=True)
    if a.device=='cuda':torch.cuda.synchronize()
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu_start
    final=evaluate(model,fixed,a.task=='recall-diagnostic')
    held_final=evaluate(model,held,a.task=='recall-diagnostic')
    report={'scope':'fixed-batch optimization diagnostic, NOT native benchmark' if a.fixed_batch else 'small research training run, NOT native benchmark',
            'official_march_reproduction':False,'config':asdict(cfg),
            'seed':a.seed,'task':a.task,'fixed_batch':a.fixed_batch,'steps':a.steps,
            'batch_size':a.batch_size,'seq_len':a.seq_len,'parameters':sum(p.numel() for p in model.parameters()),
            'training_wall_seconds':wall,'training_process_cpu_seconds':cpu,
            'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if platform.system()=='Darwin' else 1),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
            'initial_fixed_batch':initial,'final_fixed_batch':final,
            'initial_heldout_batch':held_initial,'final_heldout_batch':held_final,
            'first_step_gradient_norms':first_grads,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':str(device),'threads':a.threads},
            'limitations':['hard write decisions have no gradient across switches','past-read second moment is a proxy for future utility, even with a full matrix',
                           'no fused kernels; sequential loop; full geometry refactors H for every candidate/token, cubic in d_key','fixed physical bank cap; no deletion, CED or global KV',
                           'snapshot content descriptors differ from MARCH cross-layer anchor tokens']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    torch.save({'format_version':2,'config':asdict(cfg),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'step':a.steps,'seed':a.seed,'torch_rng':torch.get_rng_state(),'data_rng':gen.get_state(),
                'heldout_tokens':held.cpu(),'fixed_tokens':fixed.cpu()},a.output/'checkpoint.pt')
    print(json.dumps({'initial':initial,'final':final,'heldout':held_final,'training_wall_seconds':wall}),flush=True)

if __name__=='__main__':main()
