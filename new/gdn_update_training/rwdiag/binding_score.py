"""Audit binding-v2 stimuli without weights; score both entities in both orders after both updates."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from .panel import ALPHA, FAMILIES, ORDERS, BINDING_PANELS, scenario, render

CURRENT = ('x_now','y_now')
SLOTS = tuple((o,w,q) for o in ORDERS for w in (0,1) for q in CURRENT)
POST = ('other','neutral','records')

class Characters:
    def encode(self,text,add_special_tokens=False):
        return ([1] if add_special_tokens else [])+[ord(c)+10 for c in text]

def audit(split,count,seed):
    if count<=0 or count%24: raise ValueError('use complete blocks of 24 scenes')
    tok=Characters(); n_redacted=n_same=first_wins=last_wins=0
    for i in range(count):
        s=scenario(split,i,seed); first=[];last=[]
        for family in FAMILIES:
            for order in ORDERS:
                for q in CURRENT:
                    texts=[]; labels=[]
                    for w in (0,1):
                        r=render(tok,s,f'{family}_{order}',w,'other',0,q)
                        text=''.join(chr(n-10) for n in r.prefix if n>=10)
                        vals=re.findall(r'Record (\w+): updated to ([ABCD])\.',text)
                        if [v[0] for v in vals]!=([s.x,s.y] if order=='xy' else [s.y,s.x]): raise ValueError('update order drift')
                        if dict(vals)[r.name]!=r.gold:raise ValueError('question/label mismatch')
                        texts.append(text);labels.append(r.gold)
                        n_same+=int(q=='x_now' and vals[0][1]==vals[1][1])
                        if family=='bind_swap':
                            first.append(vals[0][1]==r.gold);last.append(vals[-1][1]==r.gold)
                    if family!='bind_swap':
                        changed='x' if family=='bind_x' else 'y'; name=getattr(s,changed)
                        expr=rf'(Record {name}: updated to )[ABCD]'
                        hidden=[re.subn(expr,r'\1?',t) for t in texts]
                        if hidden[0]!=hidden[1] or hidden[0][1]!=1 or texts[0]==texts[1]:raise ValueError('counterfactual leaked outside updated value')
                        if (labels[0]!=labels[1])!=(q==changed+'_now'):raise ValueError('signed supervision mismatch')
                        n_redacted+=1
        first_wins+=int(all(first));last_wins+=int(all(last))
    if first_wins or last_wins or n_same==0:raise ValueError('shortcut audit failed')
    return dict(evidence='symbolic_stimulus_audit_not_model_accuracy',scene_count=count,
                signed_pairs_redacted=n_redacted,same_value_rows=n_same,
                swap_first_update_all8=first_wins/count,swap_last_update_all8=last_wins/count,
                redacted_changed_pair_ceiling=0,ceiling_scope='deterministic predictor; identical redacted inputs and question but opposite golds; both worlds must be right')

def check_reader(path):
    c=json.loads((path/'config.json').read_text());s=json.loads((path/'status.json').read_text())
    if (s.get('action'),s.get('status'),c.get('split'))!=('fit','completed','train') or set(c.get('panels',()))!=set(BINDING_PANELS) or not set(CURRENT)<=set(c.get('questions',())):
        raise ValueError('requires completed v2 fit on both question roles and all six panels; never reuse legacy readers')

def crossed(rows,rounds=1000):
    from .report import interval
    if rounds<100:raise ValueError('use at least 100 paired bootstrap draws')
    tabs=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    for r in rows:
        p,t,d,q=r['panel'],r['stage'],r['distance'],r['question']
        if p not in BINDING_PANELS or t not in POST or q not in CURRENT:continue
        family,order=p.rsplit('_',1); w=r['world']; sid=r['scene']; arm=r['arm']
        cell=tabs[family,t,d][arm][sid];slot=(order,w,q)
        if w not in (0,1) or slot in cell:raise ValueError('invalid or duplicated cell')
        if r['gold'] not in tuple(ALPHA) or r['pred'] not in tuple(ALPHA):raise ValueError('invalid letter')
        cell[slot]=(r['gold'],r['pred'])
    if not tabs:raise ValueError('need at least one post-update stage with both orders and questions')
    output=[]
    for (fam,stage,d),arms in sorted(tabs.items()):
        if 'native' not in arms:raise ValueError('missing native')
        native=arms['native'];ids=sorted(native)
        for arm,scenes in sorted(arms.items()):
            if set(scenes)!=set(ids):raise ValueError('incomplete arm scene set')
            joint=[];raw_delta=[];both_delta=[];role_ok={q:[] for q in CURRENT}; rescued=broken=0; donor=[];cross_follow=[]
            for sid in ids:
                cell=scenes[sid];b=native[sid]
                if set(cell)!=set(SLOTS) or set(b)!=set(SLOTS):raise ValueError('missing world, queried entity, or update order')
                if any(cell[k][0]!=b[k][0] for k in SLOTS):raise ValueError('arm changed ground truth')
                for w in (0,1):
                    for q in CURRENT:
                        if b['xy',w,q][0]!=b['yx',w,q][0]:raise ValueError('order changed final labels')
                changing=[q for q in CURRENT if b['xy',0,q][0]!=b['xy',1,q][0]]
                expected={'bind_x':['x_now'],'bind_y':['y_now'],'bind_swap':list(CURRENT)}[fam]
                if changing!=expected:raise ValueError('incorrect counterfactual labels')
                if fam=='bind_swap' and any(b['xy',w,'x_now'][0]!=b['xy',1-w,'y_now'][0] for w in (0,1)):raise ValueError('not a matched assignment swap')
                ok=[cell[k][0]==cell[k][1] for k in SLOTS];bo=[b[k][0]==b[k][1] for k in SLOTS]
                joint.append(int(all(ok)));both_delta.append(int(all(ok))-int(all(bo)))
                rescued+=sum(x and not y for x,y in zip(ok,bo));broken+=sum(y and not x for x,y in zip(ok,bo));raw_delta.append((sum(ok)-sum(bo))/len(SLOTS))
                for q in CURRENT:role_ok[q].append(int(all(cell[o,w,q][0]==cell[o,w,q][1] for o in ORDERS for w in (0,1))))
                if arm=='probe_state_swap':
                    donor.append(int(all(cell[o,w,q][1]==b[o,1-w,q][0] for o in ORDERS for w in (0,1) for q in changing)))
                if fam=='bind_y' and all(b['xy',w,'x_now'][0]!=b['xy',w,'y_now'][0] for w in (0,1)):
                    cross_follow.append(int(all(cell[o,w,'x_now'][1]==b[o,w,'y_now'][0] for o in ORDERS for w in (0,1))))
            stat=lambda x: interval(x,rounds) if x else None
            output.append(dict(family=fam,stage=stage,distance=d,arm=arm,n_scenes=len(ids),n_rows=8*len(ids),
                 all8=stat(joint),all8_delta=stat(both_delta),role_all4={q:stat(v) for q,v in role_ok.items()},
                 rescued_rows=rescued,broken_rows=broken,net_row_delta=stat(raw_delta),
                 donor_follow_all_changed=stat(donor),xy_crosstalk_eligible=len(cross_follow),xy_crosstalk_all4=stat(cross_follow)))
    return output

def make_report(path):
    # base_report is made by rwdiag.report only after checking status, native grid,
    # frozen manifest and paired token gaps. Verify that both read the same rows.
    cfg=json.loads((path/'config.json').read_text());base=json.loads((path/'base_report.json').read_text());h=hashlib.sha256()
    if set(cfg['panels'])!=set(BINDING_PANELS) or not set(CURRENT)<=set(cfg['questions']):raise ValueError('incomplete binding-v2 evaluation schedule')
    def read():
        with (path/'rows.jsonl').open('rb') as f:
            for line in f:h.update(line);yield json.loads(line)
    metrics=crossed(read(),cfg.get('rounds',1000))
    if h.hexdigest()!=base['input_rows_sha256']:raise ValueError('base report does not match raw rows')
    stages=[(t,d) for t in cfg['stages'] if t in POST for d in (cfg['distances'] if t in ('neutral','records') else [0])]
    expected={(f,t,d) for f in FAMILIES for t,d in stages}
    for arm in ('native','q','k','beta','v','q+k','q+beta','q+v','probe','probe_state_swap','q_sham','question_only'):
        got={(r['family'],r['stage'],r['distance']) for r in metrics if r['arm']==arm and r['n_scenes']==cfg['scenes']}
        if got!=expected:raise ValueError(f'missing v2 scene coverage for {arm}')
    return dict(protocol='rwdiag.binding-v2',input_rows_sha256=h.hexdigest(),reader_sha256=base['reader_sha256'],
                identity=base['identity'],crossed_binding=metrics,records_minus_neutral=base['records_minus_neutral'],
                historical_protected_joint=base['all_eight'],note='crossed all8 = two entities x two worlds x two orders, after both updates; role_all4 uses one entity. Intervals resample scenes. Immediate stage and within-order Q/write factorial are in summary.json.')

def main():
    p=argparse.ArgumentParser(description=__doc__);cmd=p.add_subparsers(dest='action',required=True)
    a=cmd.add_parser('audit');a.add_argument('--split',choices=('train','dev','test'),default='dev');a.add_argument('--scenes',type=int,default=24);a.add_argument('--seed',type=int,default=9001);a.add_argument('--out',type=Path,required=True)
    r=cmd.add_parser('report');r.add_argument('run',type=Path);r.add_argument('--out',type=Path,required=True)
    v=cmd.add_parser('check-reader');v.add_argument('--reader',type=Path,required=True)
    args=p.parse_args()
    if args.action=='check-reader':check_reader(args.reader);return
    obj=audit(args.split,args.scenes,args.seed) if args.action=='audit' else make_report(args.run)
    with args.out.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
    if args.action=='audit':print(json.dumps(obj))
    else:print('Wrote',args.out)

if __name__=='__main__':main()
