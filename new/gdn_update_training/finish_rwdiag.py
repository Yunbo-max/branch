"""One held-out decision for prototype selection, not a proof of a GDN limitation.
From new/gdn_update_training: python finish_rwdiag.py [--pilot RUN_DIR] [--plan].
This wrapper is outside rwdiag/ so previously fitted binding-v2 readers remain valid.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = [f'bind_{f}_{o}' for f in ('x', 'y', 'swap') for o in ('xy', 'yx')]
Q = ['x_now', 'y_now', 'x_first', 'z_now']
CONTROLS = {'q':['q_sham'], 'k':['k_sham'], 'beta':['beta_wrong_site'], 'v':['v_wrong_value'],
            'q+k':['q','k'], 'q+beta':['q','beta'], 'q+v':['q','v']}
N = 48

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def write(path, obj):
    with path.open('x', encoding='utf8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')

def discover(root):
    pilots = sorted(p.parent.parent for p in root.glob('pilot_*/dev/decision.json'))
    if len(pilots) != 1:
        raise ValueError(f'Found {len(pilots)} pilot directories. Pass explicit --pilot RUN_DIR.')
    return pilots[0]

def validate(cfg, manifest, decision, status, fit):
    good = (cfg.get('action')=='eval' and cfg.get('split')=='dev' and set(cfg['panels'])==set(P)
            and set(Q[:2])<=set(cfg['questions']) and not cfg.get('allow_extrapolation')
            and cfg.get('span','value')=='value' and manifest.get('frozen') is True
            and manifest.get('protocol')=='gdn-rwdiag.v1' and decision.get('protocol')=='rwdiag.binding-v2'
            and decision.get('identity')==manifest['identity'] and status.get('status')=='completed'
            and status.get('action')=='eval' and fit.get('action')=='fit' and fit.get('status')=='completed'
            and bool(decision.get('reader_sha256'))
            and decision['reader_sha256']==status.get('readers_sha256')==fit.get('readers_sha256'))
    if not good: raise ValueError('Not a matched, completed, frozen binding-v2 dev pilot with value-span controls.')

def commands(cfg, reader, root):
    def cmd(module,*args): return [sys.executable,'-m',module,*map(str,args)]
    dest=root/'test'; score='rwdiag.binding_score'
    ev=cmd('rwdiag.run','eval','--reader',reader,'--out',dest,'--split','test','--scenes',N,
           '--panels',*P,'--questions',*Q,'--stages','write','other','neutral','records','--distances',128,1024)
    for key in ('model','layer','head','rank','dtype','data_seed','beta','span','revision','seed'):
        if cfg.get(key) is not None: ev.extend(['--'+key.replace('_','-'),str(cfg[key])])
    return [cmd(score,'check-reader','--reader',reader),
            cmd(score,'audit','--split','test','--scenes',N,'--seed',cfg['data_seed'],'--out',root/'audit.json'),
            ev,cmd('rwdiag.report',dest,'--out',dest/'base_report.json'),cmd(score,'report',dest,'--out',dest/'decision.json')]

def route(contrasts, candidate):
    gain=contrasts['primary']
    passed=(gain[0]>=.05 and gain[1]>0 and all(contrasts['against_'+c][1]>0 for c in CONTROLS[candidate])
            and all(contrasts['protected_'+q][1]>-.05 for q in Q[1:]))
    if not passed: return 'behavioral_prototype_only'
    if 'v' in candidate: return 'prototype:causal_value_encoding_not_oracle_donor'
    return 'prototype:'+candidate

def secondary(rows, ci):
    # Hypothesis-generating comparisons, not additional confirmatory winners.
    ids=sorted({k[0] for k in rows});follow=[];length=[];interference=[]
    if len(ids)!=N: raise ValueError('incomplete secondary scene set')
    for sid in ids:
        f=[];ld=it=0
        for order in ('xy','yx'):
            for w in (0,1):
                def get(fam,question,stage,d,world=w):
                    return rows[sid,world,'bind_'+fam+'_'+order,question,stage,d]
                x=get('y','x_now','other',0);y=get('y','y_now','other',0);opp=get('y','y_now','other',0,1-w)
                if x[0] not in (y[0],opp[0]) and y[0]!=opp[0]:
                    f.append(int(x[1]==y[0])-int(x[1]==opp[0]))
                short=get('x','x_now','neutral',128);long=get('x','x_now','neutral',1024);rec=get('x','x_now','records',1024)
                if len({short[0],long[0],rec[0]})!=1 or long[2]!=rec[2] or long[2]-short[2]!=896:
                    raise ValueError('wrong gold or unmatched actual retention gaps')
                correct=lambda r:int(r[0]==r[1]);ld+=correct(long)-correct(short);it+=correct(rec)-correct(long)
        if len(f)==4:follow.append(sum(f)/4)
        length.append(ld/4);interference.append(it/4)
    f=ci(follow) if follow else None; l=ci(length); h=ci(interference)
    lead='no_local_mechanism_selected'
    if f and f[0]>=.05 and f[1]>0 and h[0]<=-.05 and h[2]<0: lead='fixed_budget_write_isolation'
    elif l[0]<=-.05 and l[2]<0: lead='long_distance_retention_or_readout_drift'
    return dict(eligible_y_scenes=len(follow),y_follow_aligned_minus_swapped=f,neutral_1024_minus_128=l,
                records_minus_neutral_1024=h,exploratory_design_lead=lead)

def decision_card(dest,candidate):
    import numpy as np
    from rwdiag.report import interval
    info=read(dest/'decision.json'); rows={}; baseline={}; digest=hashlib.sha256()
    arms={'native',candidate,*CONTROLS[candidate]}
    with (dest/'rows.jsonl').open('rb') as source:
        for line in source:
            digest.update(line);r=json.loads(line)
            if r['arm']=='native' and r['panel'] in P[:4] and r['question'] in Q[:2]:
                baseline[r['scene'],r['world'],r['panel'],r['question'],r['stage'],r['distance']]=(r['gold'],r['pred'],r['actual_gap'])
            if r['panel'] not in P[:2] or r['stage']!='records' or r['distance']!=1024 or r['arm'] not in arms: continue
            key=(r['scene'],r['world'],r['panel'],r['question'],r['arm'])
            if key in rows: raise ValueError('duplicate confirmatory row')
            rows[key]=(r['gold'],r['pred'])
    if digest.hexdigest()!=info['input_rows_sha256']: raise ValueError('raw row hash mismatch')
    ids=sorted({k[0] for k in rows});
    if len(ids)!=N: raise ValueError('wrong number of independent test scenes')
    def contrast(q,a,b):
        values=[]
        for scene in ids:
            delta=0
            for w in (0,1):
                for order in P[:2]:
                    u=rows[scene,w,order,q,a];v=rows[scene,w,order,q,b]
                    if u[0]!=v[0]: raise ValueError('mismatched gold')
                    delta+=int(u[0]==u[1])-int(v[0]==v[1])
            values.append(delta/4)
        return interval(np.asarray(values),2000)
    c={'primary':contrast('x_now',candidate,'native')}
    c.update({'against_'+b:contrast('x_now',candidate,b) for b in CONTROLS[candidate]})
    c.update({'protected_'+q:contrast(q,candidate,'native') for q in Q[1:]})
    patterns=secondary(baseline,lambda x:interval(np.asarray(x),2000))
    return dict(route=route(c,candidate),candidate=candidate,scene_count=N,paired_contrasts=c,secondary_patterns=patterns,
                scale='fractions: estimate, lower, upper; scene-level percentile bootstrap',
                scope='one nominated intervention/site/checkpoint; candidate sufficiency, not necessity',
                input_rows_sha256=info['input_rows_sha256'],reader_sha256=info['reader_sha256'],
                secondary='See test/decision.json for state swap, joint binding, cross-talk and matched records-minus-neutral. Negative gate means no mechanism claim, not absence of information.')

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--pilot',type=Path);a.add_argument('--plan',action='store_true')
    a.add_argument('--candidate',choices=CONTROLS,default='q',help='one primary hypothesis, chosen before test; default q')
    args=a.parse_args(); pilot=(args.pilot or discover(ROOT/'runs/binding_v2')).resolve(); dev=pilot/'dev'; cfg=read(dev/'config.json')
    reader=Path(cfg['reader']);reader=(ROOT/reader).resolve() if not reader.is_absolute() else reader
    m=read(dev/'manifest.json');d=read(dev/'decision.json');validate(cfg,m,d,read(dev/'status.json'),read(reader/'status.json'))
    if hashlib.sha256((reader/'readers.pt').read_bytes()).hexdigest()!=d['reader_sha256']:raise ValueError('reader weights changed')
    out=pilot/'design_confirmation'; cmds=commands(cfg,reader,out)
    lock=dict(version=1,candidate=args.candidate,controls=CONTROLS[args.candidate],test_scenes=N,base_question_rows=N*6*2*4*6,
              primary='bind_x_xy/yx, X current, records, 1024 filler, mean across 2 worlds x 2 orders per scene',
              gate='gain >=5pp, paired 95% lower >0 vs native and named controls; each protected lower >-5pp',
              fallback='begin same-data behavioral update prototype; do not select a new primary on these test rows',
              reader_sha256=d['reader_sha256'],dev_rows_sha256=d['input_rows_sha256'],site=m['identity'],commands=cmds)
    if args.plan: print(json.dumps(lock,indent=2));return
    out.mkdir(exist_ok=False);write(out/'lock.json',lock);print('Locked confirmation:',out,flush=True)
    for c in cmds: subprocess.run(c,cwd=ROOT,check=True)
    write(out/'route.json',decision_card(out/'test',args.candidate)); print('Finished:',out/'route.json',flush=True)

if __name__=='__main__':main()
