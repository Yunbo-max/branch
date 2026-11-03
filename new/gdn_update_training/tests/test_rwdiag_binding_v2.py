"""CPU-only, audits rendered histories. These are not GDN accuracy tests."""
from collections import Counter
from itertools import product
from pathlib import Path
import re
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rwdiag import panel

class CharTok:
    def encode(self, text, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [ord(c)+10 for c in text]
TOK=CharTok()
KINDS=('bind_x','bind_y','bind_swap')
ORDERS=('xy','yx')
PANELS=tuple(f'{k}_{o}' for k in KINDS for o in ORDERS)

def decoded(row):
    return ''.join(chr(n-10) for n in row.prefix if n>=10)

def test_new_panels_exist_without_removing_legacy():
    assert set(PANELS).issubset(panel.PANELS), 'missing crossed-order single-entity panels'
    assert {'swap','cross','swap_yfirst'}.issubset(panel.PANELS)

@pytest.mark.parametrize('split',['train','dev','test'])
def test_counterfactual_semantics_and_balance_in_actual_prompts(split):
    assert set(PANELS).issubset(panel.PANELS), 'missing corrected panels'
    hist=Counter(); collisions=0
    for i in range(24):
        s=panel.scenario(split,i,9001)
        for family in KINDS:
            by_order={}
            for order in ORDERS:
                pname=f'{family}_{order}'; worlds=[]
                for w in (0,1):
                    rx,ry=(panel.render(TOK,s,pname,w,'other',0,q) for q in ('x_now','y_now'))
                    assert rx.prefix==ry.prefix and rx.question!=ry.question
                    txt=decoded(rx); updates=re.findall(r'Record (\w+): updated to ([ABCD])\.',txt)
                    assert [n for n,_ in updates]==([s.x,s.y] if order=='xy' else [s.y,s.x])
                    vals=dict(updates); assert (rx.gold,ry.gold)==(vals[s.x],vals[s.y])
                    assert txt.rstrip().endswith(f'Record {s.z}: {s.zval}.')
                    assert rx.prefix[rx.write]==ord(rx.gold)+10
                    worlds.append((txt,rx.gold,ry.gold)); hist[pname,w,'x',rx.gold]+=1;hist[pname,w,'y',ry.gold]+=1
                    collisions+=int(rx.gold==ry.gold)
                (t0,x0,y0),(t1,x1,y1)=worlds
                assert (x0!=x1)==(family!='bind_y') and (y0!=y1)==(family!='bind_x')
                if family!='bind_swap':
                    changed=s.x if family=='bind_x' else s.y
                    def redact(t):return re.sub(rf'(Record {changed}: updated to )[ABCD]',r'\1?',t)
                    assert t0!=t1 and redact(t0)==redact(t1)  # no-target transcript is identical
                else:
                    assert (x0,y0)==(y1,x1) and x0!=y0
                by_order[order]=[(x,y) for _,x,y in worlds]
            assert by_order['xy']==by_order['yx']
    assert collisions>0, 'different entities must sometimes share a value'
    assert len(hist)==len(PANELS)*2*2*4 and set(hist.values())=={6}

def test_donor_sites_and_neutral_record_gap_match():
    assert set(PANELS).issubset(panel.PANELS), 'missing corrected panels'
    s=panel.scenario('dev',3,90)
    for name,w in product(PANELS,(0,1)):
        def row(stage,donor='self'):
            return panel.render(TOK,s,name,w,stage,128 if stage in ('neutral','records') else 0,'x_now',donor)
        r=row('other'); clean=row('other','clean');wrong=row('other','wrong')
        assert (r.start,r.end,r.write)==(clean.start,clean.end,clean.write)==(wrong.start,wrong.end,wrong.write)
        assert r.prefix[r.write]!=wrong.prefix[r.write] and r.prefix[r.write]==clean.prefix[r.write]
        assert r.prefix!=clean.prefix and row('write').gap==0
        assert row('records').gap==row('neutral').gap==r.gap+128

def test_blind_references_cannot_earn_joint_binding_success():
    assert set(PANELS).issubset(panel.PANELS), 'missing corrected panels'
    from rwdiag.binding_score import audit
    r=audit('dev',24,9001)
    assert r['scene_count']==24 and r['signed_pairs_redacted']==192
    assert r['swap_first_update_all8']==r['swap_last_update_all8']==0
    assert r['redacted_changed_pair_ceiling']==0 and r['same_value_rows']>0
