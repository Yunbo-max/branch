#!/usr/bin/env python3
"""Native GDN update learning: matched SFT / consistency / write distillation.

CPU tests exercise mathematics and runner contracts only. The CLI never creates
random weights or falls back to a CPU/proxy model for a real experiment.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F

from data import LABELS, build_groups, dataset_hash
from native import NativeTap
from objectives import group_objective, write_distillation
from teacher import search_write

BASE = '6bbba481a56b254df965238f91e427fc9fce030e'
MODES = ('sft','consistency','write_distill')


def save_json(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def select_trainable(model, scope):
    allowed={'q_proj','k_proj','v_proj','a_proj','b_proj'}
    names=[]
    for name,p in model.named_parameters():
        enable=scope=='all' or (scope=='projections' and len(name.split('.'))>=2 and name.split('.')[-2] in allowed)
        p.requires_grad_(enable)
        if enable: names.append(name)
    if scope not in ('all','projections') or not names:
        raise ValueError('scope selects no native parameters or is invalid')
    return names


def promote_trainables(model):
    """FP32 optimizer storage; native forward uses BF16 autocast when requested."""
    saved={}
    for name,p in model.named_parameters():
        if p.requires_grad:
            saved[name]=p.dtype
            p.data=p.data.float()
    return saved


def restore_storage(model,saved):
    for name,p in model.named_parameters():
        if name in saved: p.data=p.data.to(saved[name])


@contextlib.contextmanager
def frozen_parameters(model):
    flags=[p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters(): p.requires_grad_(False)
        yield
    finally:
        for p,flag in zip(model.parameters(),flags): p.requires_grad_(flag)


class Runner:
    def __init__(self, model, *, device, layer_index=10, head=0, allow_extrapolation=False, autocast_dtype=None):
        self.model,self.device=model,torch.device(device)
        self.layer_index,self.head=layer_index,head
        self.allow_extrapolation=allow_extrapolation
        self.autocast_dtype=autocast_dtype
        self.calls=0; self.input_tokens=0
        params=inspect.signature(model.forward).parameters
        self.last_kwargs={}
        for key in ('logits_to_keep','num_logits_to_keep'):
            if key in params:
                self.last_kwargs={key:1};break

    def layer(self):
        try: layer=self.model.model.layers[self.layer_index].attn
        except (AttributeError,IndexError) as exc:
            raise ValueError('unsupported native GDN layer layout/index') from exc
        return layer

    def forward(self,row,*,write=None,tap=False):
        context=getattr(self.model.config,'max_position_embeddings',None)
        if context and len(row.token_ids)>context and not self.allow_extrapolation:
            raise ValueError('context exceeds configured window; explicitly allow extrapolation')
        tokens=torch.tensor([row.token_ids],dtype=torch.long,device=self.device)
        capture=None
        amp=lambda: torch.autocast(device_type=self.device.type,dtype=self.autocast_dtype,enabled=self.autocast_dtype is not None)
        if tap or write is not None:
            with amp(), NativeTap(self.layer(),position=row.update_position,head=self.head,override=write) as observer:
                out=self.model(input_ids=tokens,use_cache=False,**self.last_kwargs)
            capture=observer.capture
        else:
            with amp():
                out=self.model(input_ids=tokens,use_cache=False,**self.last_kwargs)
        self.calls+=1;self.input_tokens+=tokens.numel()
        logits=out.logits[0,-1].float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('nonfinite real model logits')
        if max(row.choice_ids.values())>=logits.numel():
            raise ValueError('answer token outside model vocabulary')
        return logits,capture


def teacher_for_group(runner,rows,index,args):
    # Corresponding current/protected/history questions have identical prefixes
    # through the intervention site. No future query is fed to the student.
    protect_index=4+index; historical_index=6+index
    row=rows[index]
    with frozen_parameters(runner.model):
        with torch.no_grad(): _,base=runner.forward(row,tap=True)
        def replay(write):
            logits,_=runner.forward(row,write=write)
            loss=F.softplus(args.teacher_margin+logits[row.choice_ids[row.old]]-logits[row.choice_ids[row.new]])
            checks=[]
            for i in (protect_index,historical_index):
                guard=rows[i]
                output,_=runner.forward(guard,write=write)
                checks.append(-output.log_softmax(-1)[guard.choice_ids[guard.gold]])
            return loss,torch.stack(checks)
        return search_write(base,replay,steps=args.teacher_steps,lr=args.teacher_lr,
                            penalty=args.teacher_penalty,protect_tolerance=args.protect_tolerance)


def student_loss(runner,rows,*,mode,js_weight=.1,teacher=None,teacher_index=0,distill_weight=1.):
    if mode not in MODES: raise ValueError('unknown training mode')
    outputs=[];student=None
    accepted=mode=='write_distill' and teacher is not None and teacher.accepted
    for i,row in enumerate(rows):
        logits,capture=runner.forward(row,tap=True) if accepted and i==teacher_index else runner.forward(row)
        outputs.append(logits)
        if capture is not None: student=capture
    stacked=torch.stack(outputs)
    gold=torch.tensor([r.choice_ids[r.gold] for r in rows],device=stacked.device)
    ids=torch.tensor([[r.choice_ids[x] for x in LABELS] for r in rows],device=stacked.device)
    loss,metrics=group_objective(stacked,gold,[(0,1)],
        js_weight=js_weight if mode=='consistency' else 0.,choice_indices=ids)
    aux=stacked.new_zeros(())
    if accepted:
        if student is None: raise RuntimeError('accepted teacher has no native student capture')
        aux=write_distillation(student.key,student.beta,teacher.target.key,teacher.target.beta)
        loss=loss+distill_weight*aux
    metrics.update(write_loss=float(aux.detach()),teacher_accepted=bool(accepted))
    return loss,metrics


def evaluate(runner,groups,path):
    counts=defaultdict(lambda:[0,0]);n=correct=stale=changed=0
    # Exclusive open prevents accidental overwrites, even before first forward.
    with Path(path).open('x',encoding='utf-8') as stream,torch.no_grad():
        for rows in groups:
            for row in rows:
                logits,_=runner.forward(row)
                selected=torch.stack([logits[row.choice_ids[c]] for c in LABELS])
                pred=LABELS[int(selected.argmax())]
                ok=pred==row.gold
                n+=1;correct+=ok;counts[row.kind][0]+=ok;counts[row.kind][1]+=1
                is_update=row.kind in ('update_a','update_c') and row.old!=row.new
                if is_update: changed+=1;stale+=(pred==row.old)
                result={'group_id':row.group_id,'condition':row.kind,'gold':row.gold,
                        'old':row.old,'new':row.new,'prediction':pred,'correct':int(ok),
                        'gold_full_nll':float(-logits.log_softmax(-1)[row.choice_ids[row.gold]]),
                        'gold_choice_nll':float(-selected.log_softmax(-1)[LABELS.index(row.gold)]),
                        'prompt_hash':row.prompt_hash,'tokens':len(row.token_ids),
                        'update_position':row.update_position,'gap':row.actual_gap}
                stream.write(json.dumps(result,allow_nan=False)+'\n')
    return {'n':n,'accuracy':correct/n if n else None,'stale_rate':stale/changed if changed else None,
            'n_changed_update_queries':changed,'n_groups':len(groups),
            'by_kind':{k:{'correct':v[0],'n':v[1],'accuracy':v[0]/v[1]} for k,v in counts.items()},
            'uncertainty':'descriptive only; repeated queries per group are not independent training replicas'}


def code_identity():
    root=Path(__file__).resolve().parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def load_real(args):
    if not torch.cuda.is_available():
        raise RuntimeError('real training/evaluation requires CUDA; no random/CPU fallback')
    census=Path(__file__).resolve().parents[1]/'gdn_failure_census'
    sys.path.insert(0,str(census))
    from provenance import resolve_inputs,hub_revision,runtime_versions
    from backend import PretrainedCausalLM
    resolved=resolve_inputs(argparse.Namespace(backend='hf',model=args.model,tokenizer=None,
        revision=args.revision,tokenizer_revision=None))
    evaluator=PretrainedCausalLM(args.model,revision=hub_revision(resolved['model_revision']),
        tokenizer_revision=hub_revision(resolved['tokenizer_revision']),device='cuda',dtype=args.dtype,
        allow_extrapolation=args.allow_extrapolation)
    model=evaluator.model
    if getattr(model.config,'model_type',None)!='gated_deltanet' or getattr(model.config,'attn',None) is not None:
        raise ValueError('this experiment requires a pure, strictly loaded native GDN')
    if args.mode=='write_distill':
        version=importlib.metadata.version('flash-linear-attention')
        if version!='0.5.0':
            raise ValueError('write teacher reviewed only for FLA 0.5.0; validate another boundary explicitly')
    # Gradients remain enabled, while stochastic dropout is disabled for paired
    # comparisons and exact teacher replay. Long inputs still use chunk mode.
    model.eval()
    if hasattr(model,'gradient_checkpointing_disable'): model.gradient_checkpointing_disable()
    return model,evaluator.tok,{'resolved':resolved,'model_info':asdict(evaluator.info),'runtime':runtime_versions()}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','train','eval'))
    p.add_argument('--mode',choices=MODES,default='sft')
    p.add_argument('--model',default='puigde/gated-deltanet-360M-15B-slimpajama')
    p.add_argument('--revision')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dtype',choices=('bf16','fp32'),default='bf16')
    p.add_argument('--scope',choices=('projections','all'),default='projections')
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--train-groups',type=int,default=120)
    p.add_argument('--eval-groups',type=int,default=24)
    p.add_argument('--length',type=int,default=1024)
    p.add_argument('--gap',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data-seed',type=int,default=1701)
    p.add_argument('--eval-split',choices=('val','test'),default='val')
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--js-weight',type=float,default=.1)
    p.add_argument('--distill-weight',type=float,default=1.)
    p.add_argument('--layer',type=int,default=10)
    p.add_argument('--head',type=int,default=0)
    p.add_argument('--teacher-steps',type=int,default=4)
    p.add_argument('--teacher-lr',type=float,default=.15)
    p.add_argument('--teacher-penalty',type=float,default=.05)
    p.add_argument('--teacher-margin',type=float,default=.5)
    p.add_argument('--protect-tolerance',type=float,default=.02)
    p.add_argument('--teacher-every',type=int,default=1)
    p.add_argument('--allow-extrapolation',action='store_true')
    return p


def validate_args(a):
    if a.steps<=0 or a.length<=64 or a.gap<0 or a.layer<0 or a.head<0 or a.teacher_steps<=0 or a.teacher_every<=0:
        raise ValueError('invalid step/token/site budget')
    for count in (a.train_groups,a.eval_groups):
        if count<=0 or count%12: raise ValueError('group counts must be positive multiples of 12')
    for key in ('lr','js_weight','distill_weight','teacher_lr','teacher_penalty','teacher_margin','protect_tolerance'):
        value=getattr(a,key)
        if not math.isfinite(value) or value<0: raise ValueError('invalid '+key)
    if a.lr==0 or a.teacher_lr==0: raise ValueError('learning rates must be positive')


def main(argv=None):
    args=parser().parse_args(argv);validate_args(args)
    args.out.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.action=='plan':
        plan={'status':'plan_only_no_weights_loaded','config':config,'base_commit':BASE,
              'train_queries':8*args.train_groups,'eval_queries':8*args.eval_groups,
              'teacher_replays_per_selected_query_upper_bound':1+2*args.teacher_steps,
              'forwards_per_teacher_replay':3,'code':code_identity(),
              'native_gpu_validation':'not performed by plan','inference_graph_changes':False}
        save_json(args.out/'plan.json',plan);print(json.dumps(plan,indent=2));return 0
    save_json(args.out/'config.json',config)
    try:
        torch.manual_seed(args.seed);random.seed(args.seed)
        model,tok,provenance=load_real(args)
        runner=Runner(model,device='cuda',layer_index=args.layer,head=args.head,
                      allow_extrapolation=args.allow_extrapolation,
                      autocast_dtype=torch.bfloat16 if args.dtype=='bf16' else None)
        evaluation=build_groups(tok,split=args.eval_split,seed=args.data_seed,count=args.eval_groups,length=args.length,gap=args.gap)
        if args.action=='eval':
            report=evaluate(runner,evaluation,args.out/'eval.jsonl')
            save_json(args.out/'report.json',dict(report,provenance=provenance,dataset_hash=dataset_hash(evaluation),code=code_identity()))
            save_json(args.out/'status.json',{'status':'completed','evidence_type':'pretrained_lm'})
            return 0
        groups=build_groups(tok,split='train',seed=args.data_seed,count=args.train_groups,length=args.length,gap=args.gap)
        names=select_trainable(model,args.scope)
        schema={k:tuple(v.shape) for k,v in model.state_dict().items()}
        manifest={'base_commit':BASE,'provenance':provenance,'code':code_identity(),
                  'train_hash':dataset_hash(groups),'eval_hash':dataset_hash(evaluation),
                  'trainable_names':names,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'teacher_variant':'direct key/beta gradient plus replay; not closed-form state-gradient oracle',
                  'evidence_type':'pretrained_training','stochastic_layers':'eval mode, autograd enabled',
                  'general_lm_retention':'not tested; no general-text LM mixture in v1'}
        save_json(args.out/'manifest.json',manifest)
        save_json(args.out/'before.json',evaluate(runner,evaluation,args.out/'before_rows.jsonl'))
        original_storage=promote_trainables(model)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
        order=list(range(len(groups)));rng=random.Random(args.seed);rng.shuffle(order)
        with (args.out/'train.jsonl').open('x',encoding='utf-8') as log:
            for step in range(args.steps):
                if step and step%len(order)==0:rng.shuffle(order)
                rows=groups[order[step%len(order)]]
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize();tic=time.perf_counter()
                calls0,tokens0=runner.calls,runner.input_tokens
                target=None;idx=step%2;teacher_seconds=0.
                if args.mode=='write_distill' and step%args.teacher_every==0:
                    tt=time.perf_counter()
                    target=teacher_for_group(runner,rows,idx,args)
                    torch.cuda.synchronize();teacher_seconds=time.perf_counter()-tt
                loss,metrics=student_loss(runner,rows,mode=args.mode,js_weight=args.js_weight,
                                          teacher=target,teacher_index=idx,distill_weight=args.distill_weight)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite training loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize()
                metrics.update(step=step+1,group_id=rows[0].group_id,loss=float(loss.detach()),grad_norm=float(norm),
                               seconds=time.perf_counter()-tic,teacher_seconds=teacher_seconds,
                               forward_calls=runner.calls-calls0,forward_input_tokens=runner.input_tokens-tokens0,
                               teacher_replays=0 if target is None else target.replays,
                               teacher_before=None if target is None else target.before,
                               teacher_after=None if target is None else target.after)
                log.write(json.dumps(metrics,allow_nan=False)+'\n');log.flush()
                if (step+1)%10==0:print(json.dumps(metrics),flush=True)
        restore_storage(model,original_storage)
        save_json(args.out/'after.json',evaluate(runner,evaluation,args.out/'after_rows.jsonl'))
        if schema!={k:tuple(v.shape) for k,v in model.state_dict().items()}:
            raise RuntimeError('native state_dict schema changed; refusing export')
        # FLA 0.5.0 declares _tied_weights_keys as a list; transformers 5.x
        # save_pretrained calls .keys() on it. The config here is untied, so {}.
        if isinstance(getattr(model, "_tied_weights_keys", None), (list, tuple)):
            model._tied_weights_keys = {}
        model.save_pretrained(args.out/'model',safe_serialization=True)
        tok.save_pretrained(args.out/'model')
        save_json(args.out/'status.json',{'status':'completed','native_export':'model',
            'extra_inference_parameters':0,'extra_history_states':0,
            'forward_calls':runner.calls,'forward_input_tokens':runner.input_tokens})
        return 0
    except BaseException as exc:
        save_json(args.out/'failure.json',{'status':'failed','type':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':
    raise SystemExit(main())
