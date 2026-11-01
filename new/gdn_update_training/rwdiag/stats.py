"""Paired world metrics and scene-cluster percentile intervals (points in [−1,1])."""
from collections import defaultdict,Counter
import numpy as np

def ci(values,rounds=1000):
    x=np.asarray(values,dtype=float)
    if len(x)<2:return [float(x.mean()) if len(x) else None,None,None]
    rng=np.random.default_rng(199);b=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(rounds)]
    return [float(x.mean()),*map(float,np.quantile(b,[.025,.975]))]

def summarize(rows,rounds=1000):
    cells=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    if rounds<100:raise ValueError('at least 100 bootstrap draws')
    for r in rows:
        key=(r['panel'],r['stage'],r['distance'],r['question']);d=cells[key][r['arm']][r['scene']];w=r['world']
        if w not in (0,1) or w in d:raise ValueError('bad/duplicate world')
        d[w]=r
    if not cells:raise ValueError('empty run')
    metrics=[];factorial=[]
    for key,arms in cells.items():
        if 'native' not in arms:raise ValueError('missing native baseline')
        ids=sorted(arms['native']);common=dict(zip(('panel','stage','distance','question'),key));pairs={}
        for arm,scenes in arms.items():
            if set(scenes)!=set(ids) or any(set(d)!={0,1} for d in scenes.values()):raise ValueError('incomplete paired scene')
            raw=[scenes[i][w] for i in ids for w in (0,1)];base=[arms['native'][i][w] for i in ids for w in (0,1)]
            if any(x['gold']!=y['gold'] for x,y in zip(raw,base)):raise ValueError('gold mismatch')
            ok=np.array([r['pred']==r['gold'] for r in raw]);bo=np.array([r['pred']==r['gold'] for r in base]);pairs[arm]=ok.reshape(-1,2).all(1).astype(float)
            ch=[r for r in raw if key[3]=='x_now' and r['old']!=r['gold']];distinct=[r for r in ch if len({r['old'],r['gold'],r['distractor']})==3]
            cross=[scenes[i] for i in ids if key[0]=='cross' and key[3]=='x_now' and scenes[i][0]['gold']==scenes[i][1]['gold'] and scenes[i][0]['distractor']!=scenes[i][1]['distractor'] and all(v['distractor']!=v['gold'] for v in scenes[i].values())]
            metrics.append(dict(**common,arm=arm,n_scenes=len(ids),n_rows=len(raw),pair_accuracy=ci(pairs[arm],rounds),accuracy=float(ok.mean()),rescued=int((ok&~bo).sum()),broken=int((~ok&bo).sum()),stale_n=len(ch),stale_rate=sum(x['pred']==x['old'] for x in ch)/len(ch) if ch else None,other_n=len(distinct),other_rate=sum(x['pred']==x['distractor'] for x in distinct)/len(distinct) if distinct else None,cross_follow_n=len(cross),cross_follow_both=sum(all(r['pred']==r['distractor'] for r in g.values()) for g in cross)/len(cross) if cross else None,label_histogram=dict(Counter(x['gold'] for x in raw)),mean_new_old_margin=float(np.mean([x['margin'] for x in ch])) if ch else None))
        for write in ('k','beta','v'):
            if write not in pairs:continue
            if 'q' not in pairs or 'q+'+write not in pairs:raise ValueError('incomplete factorial')
            a,b,c,d=(pairs[x] for x in ('native','q',write,'q+'+write));factorial.append(dict(**common,write=write,n_scenes=len(ids),q_effect=ci(b-a,rounds),write_effect=ci(c-a,rounds),joint_effect=ci(d-a,rounds),interaction=ci(d-b-c+a,rounds)))
    return {'metrics':metrics,'factorial':factorial,'interval':'[estimate,2.5%,97.5%], resample entire paired scenes within each prespecified cell; exploratory, no multiplicity correction'}
