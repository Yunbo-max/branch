"""Postprocess a completed rwdiag eval. Run: python -m rwdiag.report RUN --out report.json"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

QUESTIONS=('x_now','y_now','x_first','z_now')
ARMS={'native','q','k','beta','v','q+k','q+beta','q+v','probe','probe_state_swap','question_only','probe_zero','probe_permuted_labels'}

def interval(x,n):
    x=np.asarray(x,dtype=float)
    if len(x)<2:return [float(x.mean()),None,None]
    rng=np.random.default_rng(199)
    samples=[]
    for j in range(0,n,100):
        samples.extend(x[rng.integers(len(x),size=(min(100,n-j),len(x)))].mean(axis=1))
    return [float(x.mean()),*map(float,np.quantile(samples,[.025,.975]))]

def analyze(rows,rounds=1000):
    if rounds<100:raise ValueError('at least 100 cluster draws')
    table=defaultdict(dict);hidden=0
    for x in rows:
        p,t,d,q,a,i,w=(x[k] for k in ('panel','stage','distance','question','arm','scene','world'))
        if q.startswith('x_') and (p=='absent' or (p=='new_only' and t=='pre')):
            hidden+=1;continue
        if a not in ARMS:continue
        cell=table[p,t,d,a,q];gold=x['gold'];pred=x['pred']
        if type(w)!=int or w not in (0,1) or (i,w) in cell or gold not in ('A','B','C','D') or pred not in ('A','B','C','D'):raise ValueError('bad or duplicate paired row')
        cell[i,w]=(gold,pred,x['actual_gap'])
    if not table and not hidden:raise ValueError('empty evaluation')
    def ids(cell):
        scenes=sorted({i for i,w in cell})
        if len(cell)!=2*len(scenes):raise ValueError('missing world')
        return scenes
    def ok(cell,i):return all(cell[i,w][0]==cell[i,w][1] for w in (0,1))
    joint=[];donors=[];gaps=[]
    for (p,t,d,a,q),cell in sorted(table.items()):
        scenes=ids(cell);kw=dict(panel=p,stage=t,distance=d,arm=a)
        if a=='probe_state_swap' and p in ('swap','swap_yfirst','order'):
            eligible=[i for i in scenes if cell[i,0][0]!=cell[i,1][0]]
            if eligible:
                vals=[all(cell[i,w][1]==cell[i,1-w][0] for w in (0,1)) for i in eligible]
                donors.append(dict(**kw,question=q,n_scenes=len(eligible),donor_pair_rate=interval(vals,rounds)))
        if q=='x_now' and all((p,t,d,a,j) in table for j in QUESTIONS):
            arms=[table[p,t,d,a,j] for j in QUESTIONS]
            if any(c.keys()!=cell.keys() for c in arms):raise ValueError('four-question coverage mismatch')
            joint.append(dict(**kw,n_scenes=len(scenes),rate=interval([all(ok(c,i) for c in arms) for i in scenes],rounds)))
        if t=='records' and (p,'neutral',d,a,q) in table:
            control=table[p,'neutral',d,a,q]
            if cell.keys()!=control.keys() or any(v[0]!=control[k][0] or v[2]!=control[k][2] for k,v in cell.items()):raise ValueError('nonmatched histories, gold or actual token distance')
            gaps.append(dict(**kw,question=q,n_scenes=len(scenes),delta=interval([int(ok(cell,i))-int(ok(control,i)) for i in scenes],rounds)))
    return dict(all_eight=joint,donor_following=donors,records_minus_neutral=gaps,hidden_label_rows=hidden,unit='one scene = two worlds; within-cell paired bootstrap; null X queries excluded')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--rounds',type=int,default=1000);a=p.parse_args(argv)
    status=json.loads((a.run/'status.json').read_text())
    if (status.get('action'),status.get('status'))!=('eval','completed'):raise ValueError('requires completed native eval')
    meta=json.loads((a.run/'manifest.json').read_text());cfg=json.loads((a.run/'config.json').read_text())
    if not meta.get('frozen') or meta.get('protocol')!='gdn-rwdiag.v1':raise ValueError('wrong protocol or unfrozen baseline')
    digest=hashlib.sha256();native_counts=defaultdict(int)
    def records():
        with (a.run/'rows.jsonl').open('rb') as f:
            for line in f:
                digest.update(line);r=json.loads(line)
                if r['arm']=='native':native_counts[r['panel'],r['stage'],r['distance'],r['question']]+=1
                yield r
    result=analyze(records(),a.rounds)
    expected={(panel,t,d,q):2*cfg['scenes'] for panel in cfg['panels'] for t in cfg['stages'] for d in (cfg['distances'] if t in ('neutral','records') else [0]) for q in cfg['questions']}
    if dict(native_counts)!=expected:raise ValueError('native rows do not cover configured grid')
    result.update(input_rows_sha256=digest.hexdigest(),reader_sha256=status['readers_sha256'],identity=meta['identity'])
    with a.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
