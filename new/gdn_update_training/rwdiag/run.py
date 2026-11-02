"""python -m rwdiag.run {plan,check,fit,eval} ... (from new/gdn_update_training)."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import torch
from torch.nn import functional as F
from .panel import ALPHA,PANELS,QUESTIONS,STAGES,scenario,render,enc
from .model import Engine,Readers,like,sham
from .stats import summarize


def save(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)

def code_hash():
    files=sorted(Path(__file__).parent.glob('*.py'))
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in files)).hexdigest()

def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','check','fit','eval'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',default='puigde/gated-deltanet-360M-15B-slimpajama');p.add_argument('--revision');p.add_argument('--dtype',default='bf16',choices=('bf16','fp32'));p.add_argument('--allow-extrapolation',action='store_true')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--data-seed',type=int,default=9001);p.add_argument('--layer',type=int,default=10);p.add_argument('--head',type=int,default=0);p.add_argument('--rank',type=int,default=16)
    p.add_argument('--scenes',type=int,default=24);p.add_argument('--steps',type=int,default=192);p.add_argument('--lr',type=float,default=.001);p.add_argument('--split',choices=('train','dev','test'));p.add_argument('--reader',type=Path)
    p.add_argument('--panels',nargs='+',choices=PANELS,default=['swap','cross']);p.add_argument('--stages',nargs='+',choices=STAGES,default=['write','other','neutral','records']);p.add_argument('--distances',type=int,nargs='+',default=[128,512]);p.add_argument('--questions',nargs='+',choices=QUESTIONS,default=list(QUESTIONS));p.add_argument('--beta',type=float,default=.9);p.add_argument('--span',choices=('value','terminal','record'),default='value');p.add_argument('--rounds',type=int,default=1000)
    p.add_argument('--route-only',action='store_true',help='only native/q/q_sham/q_wrong_entity arms (route decision); skip probes, write interventions and donor captures')
    a=p.parse_args(argv);a.split=a.split or ('train' if a.action=='fit' else 'dev')
    if a.action=='fit' and a.split!='train' or a.action=='eval' and (a.split=='train' or not a.reader):p.error('fit=train only; eval=dev/test with --reader FIT_DIRECTORY')
    if min(a.scenes,a.rank,a.steps)<=0 or a.scenes%24 or min(a.layer,a.head,*a.distances)<0 or a.rounds<100 or not 0<a.lr<1 or not 0<=a.beta<=1:p.error('invalid budget/site; --scenes must be a positive multiple of 24')
    if any(len(v)!=len(set(v)) for v in (a.panels,a.stages,a.distances,a.questions)):p.error('duplicate panel/stage/gap/question')
    return a

def cells(a):return [(p,t,d) for p in a.panels for t in a.stages for d in (a.distances if t in ('neutral','records') else [0])]

def load(a):
    # Reuse the census strict loader, immutable Hub revision or local byte hash.
    from train import load_real
    a.mode='write_distill'  # selects FLA 0.5.0 ABI validation, does not invoke a teacher
    m,t,identity=load_real(a);eng=Engine(m,t,a.layer,a.head,a.allow_extrapolation)
    diagnostic=Readers(m.get_input_embeddings().weight.shape[1],eng.layer.head_k_dim,eng.layer.head_v_dim,a.rank).to(eng.device)
    return eng,diagnostic,identity

def fit(a,e,r):
    rng=random.Random(a.seed);grid=cells(a)
    if set(a.panels)&{'absent','new_only'} and 'pre' in a.stages:raise ValueError('do not train on unanswerable pre-edit controls')
    if 'absent' in a.panels:raise ValueError('absent-target control is never a training label')
    order=list(range(a.scenes));todo=[]
    for i in range(a.steps):
        if i%a.scenes==0:rng.shuffle(order)
        s=scenario('train',order[i%a.scenes],a.data_seed);p,t,d=rng.choice(grid);w=rng.randrange(2);q=rng.choice(a.questions)
        todo.append(render(e.tok,s,p,w,t,d,q))
    perm=[ALPHA.index(x.gold) for x in todo];rng.shuffle(perm)
    opt=torch.optim.AdamW(r.parameters(),lr=a.lr)
    with (a.out/'fit.jsonl').open('x') as f:
        for i,row in enumerate(todo):
            with torch.no_grad():s=e.prefix_state(row);feat=e.features(row)
            target=torch.tensor([ALPHA.index(row.gold)],device=e.device);shuf=torch.tensor([perm[i]],device=e.device)
            seq=row.prefix+row.question
            # Native kernel performs L2 normalization; this direction only sees the question.
            alt=F.normalize(r.q(feat),dim=0)
            z,_=e.call(seq,(('q',len(seq)-1,alt),));scores=z[list(row.ids)]
            ce=lambda z,y:F.cross_entropy(z[None],y)
            terms=[ce(scores,target),ce(r.state_logits(s,feat),target),ce(r.state_logits(s,feat,True),shuf),ce(r.question_only(feat),target)]
            loss=sum(terms);opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(r.parameters(),1.,error_if_nonfinite=True)
            if i==0 and (r.q[-1].weight.grad is None or not bool(r.q[-1].weight.grad.abs().any())):raise RuntimeError('no native q gradient on first fit example; inspect site before proceeding')
            opt.step();f.write(json.dumps(dict(step=i,losses=[float(x.detach()) for x in terms],grad_norm=float(norm)))+'\n')
    torch.save(r.state_dict(),a.out/'readers.pt')

def record(row,s,p,w,t,d,q,arm,logits):
    if logits.shape!=(4,) or not bool(torch.isfinite(logits).all()):raise ValueError('invalid choice logits')
    i=ALPHA.index(row.gold);o=ALPHA.index(row.old)
    return dict(scene=s.id,panel=p,world=w,stage=t,distance=d,actual_gap=row.gap,question=q,arm=arm,gold=row.gold,pred=ALPHA[int(logits.argmax())],old=row.old,distractor=row.distractor,margin=float(logits[i]-logits[o]),choice_logits=logits.float().cpu().tolist(),prefix_hash=hashlib.sha256(str(row.prefix).encode()).hexdigest(),update_position=row.write,input_length=len(row.prefix)+len(row.question))

@torch.no_grad()
def assess(a,e,r):
    import itertools
    rows=[];r.eval()
    with (a.out/'rows.jsonl').open('x',encoding='utf8') as f:
        for i in range(a.scenes):
            s=scenario(a.split,i,a.data_seed)
            # Canonical address excludes every record/value/answer; identical in both worlds.
            address_ids=tuple(e.tok.encode('Record identifier: '+s.x,add_special_tokens=True));address=e.call(address_ids)[1]['k'][-1]
            prev_state={}
            for p,t,d in cells(a):
                print(f'[rwdiag eval scene {i}/{a.scenes}] {p} {t} d={d}',flush=True)
                states=[]
                if not a.route_only:
                    for w in (0,1):
                        rr=render(e.tok,s,p,w,t,d,'x_now');states.append(e.prefix_state(rr))
                for w,q in itertools.product((0,1),a.questions):
                    row=render(e.tok,s,p,w,t,d,q);seq=row.prefix+row.question;last=len(seq)-1
                    z,base=e.call(seq);feat=e.features(row);direction=like(r.q(feat),base['q'][-1]);qd=('q',last,direction);lock=('q',last,base['q'][-1]);labels=list(row.ids)
                    def emit(arm,scores):
                        rr=record(row,s,p,w,t,d,q,arm,scores);rows.append(rr);f.write(json.dumps(rr,allow_nan=False)+'\n');f.flush()
                    def native(arm,patches):emit(arm,e.call(seq,patches)[0][labels])
                    emit('native',z[labels]);native('q',[qd]);native('q_sham',[('q',last,sham(base['q'][-1],direction,s.id+q))]);
                    wrongq='y_now' if q.startswith('x') else 'x_now';wf=e.features(render(e.tok,s,p,w,t,d,wrongq));native('q_wrong_entity',[('q',last,like(r.q(wf),base['q'][-1]))])
                    if not a.route_only:
                        emit('probe',r.state_logits(states[w],feat));emit('probe_state_swap',r.state_logits(states[1-w],feat));emit('probe_zero',r.state_logits(torch.zeros_like(states[w]),feat));emit('probe_permuted_labels',r.state_logits(states[w],feat,True));emit('question_only',r.question_only(feat))
                    # Baseline accuracy for the swapped probe is scored against RECIPIENT gold;
                    # to test donor-following, use donor gold in the opposite-world raw row.
                    if t=='pre' or p=='absent' or a.route_only:continue
                    sites={'value':[row.write],'terminal':[row.end-1],'record':range(row.start,row.end)}[a.span]
                    clean=render(e.tok,s,p,w,t,d,q,'clean');wrong=render(e.tok,s,p,w,t,d,q,'wrong')
                    if (row.write,row.end,len(row.prefix))!=(clean.write,clean.end,len(clean.prefix)) or (row.write,row.end,len(row.prefix))!=(wrong.write,wrong.end,len(wrong.prefix)):raise ValueError('unaligned donor spans')
                    # Capture from prefixes only, never from questions. V donor sees the explicit INPUT value.
                    vc=e.call(clean.prefix)[1]['v'];vw=e.call(wrong.prefix)[1]['v']
                    for field in ('k','beta','v'):
                        changes=[];neg=[]
                        for pos in sites:
                            raw=base[field][pos]
                            value=like(address,raw) if field=='k' else torch.full_like(raw,a.beta) if field=='beta' else vc[pos]
                            changes.append((field,pos,value))
                            if field=='v':neg.append((field,pos,vw[pos]))
                            elif field=='k':neg.append((field,pos,sham(raw,value,s.id+str(pos))))
                        native(field,changes+[lock]);native('q+'+field,changes+[qd]);native(field+'_free_q',changes)
                        if neg:native(field+('_wrong_value' if field=='v' else '_sham'),neg+[lock])
                        if a.span=='value':native(field+'_wrong_site',[(field,row.oldpos,changes[0][2]),lock])
                    f.flush()
    summary=summarize(rows,a.rounds);save(a.out/'summary.json',summary)

def main(argv=None):
    a=parse(argv);a.out.mkdir(parents=True,exist_ok=False)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};save(a.out/'config.json',cfg)
    if a.action=='plan':
        save(a.out/'plan.json',dict(evidence='plan_only',scenes=a.scenes,cells=cells(a),worlds=2,native_question_rows=2*a.scenes*len(cells(a))*len(a.questions),fit_models='diagnostic readers only',note='multiple interventions per row; no GPU job submitted'));return
    try:
        torch.manual_seed(a.seed);e,r,identity=load(a)
        fingerprint=dict(resolved=identity['resolved'],layer=a.layer,head=a.head,rank=a.rank,dtype=a.dtype,code=code_hash(),data_seed=a.data_seed)
        save(a.out/'manifest.json',dict(protocol='gdn-rwdiag.v1',identity=fingerprint,runtime=identity['runtime'],model=identity['model_info'],frozen=not any(p.requires_grad for p in e.model.parameters()),heads_tested=1))
        if a.action=='eval':
            prev=json.loads((a.reader/'manifest.json').read_text());status=json.loads((a.reader/'status.json').read_text())
            if prev['identity']!=fingerprint or status.get('action')!='fit' or status.get('status')!='completed':raise ValueError('reader identity mismatch or fit incomplete')
            path=a.reader/'readers.pt';hash_=hashlib.sha256(path.read_bytes()).hexdigest()
            if status['readers_sha256']!=hash_:raise ValueError('reader checkpoint checksum mismatch')
            r.load_state_dict(torch.load(path,map_location=e.device,weights_only=True),strict=True)
        initial=render(e.tok,scenario('train' if a.action=='fit' else 'dev',0,a.data_seed),'swap',0,'other',0,'x_now')
        save(a.out/'qualification.json',e.qualify(initial))
        if a.action=='fit':fit(a,e,r)
        elif a.action=='eval':assess(a,e,r)
        output=dict(action=a.action,status='completed',native_forwards=e.calls)
        if a.action=='fit':output['readers_sha256']=hashlib.sha256((a.out/'readers.pt').read_bytes()).hexdigest()
        if a.action=='eval':output['readers_sha256']=hash_
        save(a.out/'status.json',output)
    except BaseException as exc:
        save(a.out/'failure.json',dict(error=type(exc).__name__,message=str(exc)));raise

if __name__=='__main__':main()
