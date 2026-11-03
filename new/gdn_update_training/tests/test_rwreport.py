import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def analysis(rows):
    assert importlib.util.find_spec('rwdiag.report') is not None,'decision report is missing'
    from rwdiag.report import analyze
    return analyze(iter(rows),rounds=100)

def example():
    for scene in ['s1','s2','s3']:
        for t,d in [('other',0),('neutral',128),('records',128)]:
            for w in (0,1):
                for q in ['x_now','y_now','x_first','z_now']:
                    gold={'x_now':'AB'[w],'y_now':'BA'[w],'x_first':'C','z_now':'D'}[q]
                    for arm in ['native','probe_state_swap']:
                        pred=gold if arm=='native' else {'x_now':'BA'[w],'y_now':'AB'[w]}.get(q,gold)
                        if t=='records' and q=='x_now' and arm=='native':pred='C'
                        yield dict(scene=scene,panel='swap',stage=t,distance=d,actual_gap=d+9,world=w,question=q,arm=arm,gold=gold,pred=pred)

def test_donor_joint_and_matched_distance():
    r=analysis(list(example()))
    assert all(x['donor_pair_rate'][0]==1 for x in r['donor_following'])
    assert {x['question'] for x in r['donor_following']}=={'x_now','y_now'}
    joint=next(x for x in r['all_eight'] if x['stage']=='records' and x['arm']=='native');assert joint['n_scenes']==3 and joint['rate'][0]==0
    v=next(x for x in r['records_minus_neutral'] if x['arm']=='native' and x['question']=='x_now');assert v['delta']==[-1.,-1.,-1.]

def test_hidden_gold_not_ability():
    a=[dict(x,panel='absent') for x in example()];r=analysis(a)
    assert r['hidden_label_rows'] and not r['all_eight'] and not r['donor_following']
    assert all(x['question'] in ('y_now','z_now') for x in r['records_minus_neutral'])

def test_missing_world_duplicate_or_unequal_distance_fails():
    rows=list(example())
    for bad in [rows[:-1],rows+[rows[0]], [dict(x,actual_gap=900) if x['stage']=='records' else x for x in rows]]:
        with pytest.raises(ValueError):analysis(bad)
