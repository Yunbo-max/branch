import pathlib, sys
import pytest
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))

def script():
    assert (root/'finish_rwdiag.py').is_file(), 'missing finite held-out launcher'
    import finish_rwdiag as m
    return m

def test_fixed_grid_no_refit():
    m=script(); d={'model':'local','rank':16,'head':0,'layer':10,'dtype':'bf16','data_seed':777,'beta':0.9,'span':'value'}
    cmds=m.commands(d,pathlib.Path('/x/fit'),pathlib.Path('/y/run')); ev=next(c for c in cmds if 'eval' in c)
    assert ev[ev.index('--split')+1]=='test' and ev[ev.index('--scenes')+1]=='48'
    assert set(m.Q)=={'x_now','y_now','x_first','z_now'} and ev[ev.index('--distances')+1:ev.index('--distances')+3]==['128','1024']
    assert ev[ev.index('--data-seed')+1]=='777' and not any(c[3:4]==['fit'] for c in cmds)
    assert '--allow-extrapolation' not in ev and len(m.P)==6

def test_autofind_refuses_ambiguity(tmp_path):
    m=script()
    for name in ['pilot_A','pilot_B']:
        (tmp_path/name/'dev').mkdir(parents=True);(tmp_path/name/'dev'/'decision.json').write_text('{}')
    with pytest.raises(ValueError,match='explicit'):m.discover(tmp_path)

def test_reject_stale_or_wrong_pilot():
    m=script();cfg=dict(action='eval',split='dev',panels=m.P,questions=['x_now','y_now'],stages=['other'],allow_extrapolation=False)
    src={'identity':{'resolved':{'model_id':'m'},'head':0},'frozen':True,'protocol':'gdn-rwdiag.v1'};d={'protocol':'rwdiag.binding-v2','identity':src['identity'],'reader_sha256':'abc'}
    s=dict(status='completed',action='eval',readers_sha256='abc');f=dict(status='completed',action='fit',readers_sha256='abc')
    m.validate(cfg,src,d,s,f)
    for bad in [dict(cfg,split='test'),dict(cfg,panels=['swap']),dict(cfg,allow_extrapolation=True),dict(cfg,questions=['x_now'])]:
        with pytest.raises(ValueError):m.validate(bad,src,d,s,f)
    with pytest.raises(ValueError):m.validate(cfg,src,d,dict(s,readers_sha256='other'),f)

def test_specificity_and_guard_scores_not_optional():
    m=script()
    ci=lambda v: [v,v,v]
    c={'primary':ci(.12),'against_q_sham':ci(.1),'protected_y_now':ci(0),'protected_x_first':ci(0),'protected_z_now':ci(0)}
    assert m.route(c,'q')=='prototype:q'
    assert m.route(dict(c,against_q_sham=ci(0)),'q')=='behavioral_prototype_only'
    assert m.route(dict(c,protected_y_now=ci(-.08)),'q')=='behavioral_prototype_only'
    assert m.route(dict(c,primary=[.12,-.03,.21]),'q')=='behavioral_prototype_only'

def test_secondary_separates_distance_from_crosstalk():
    m=script()
    rates={}
    for i in range(48):
        for w in (0,1):
            for order in ('xy','yx'):
                for t,d in [('other',0),('neutral',128),('neutral',1024),('records',1024)]:
                    for fam in ('x','y'):
                        x='A' if fam=='y' else 'AB'[w];y='BC'[w]
                        for role,gold in [('x_now',x),('y_now',y)]:
                            pred=y if role=='x_now' and ((t=='other' and fam=='y') or t=='records') else gold
                            rates[(i,w,'bind_'+fam+'_'+order,role,t,d)]=(gold,pred,100+d)
    # Deterministic synthetic strings: perfect neutral retention, but a records penalty and Y-following.
    r=m.secondary(rates,lambda values:[sum(values)/len(values)]*3)
    assert r['eligible_y_scenes']==48 and r['y_follow_aligned_minus_swapped'][0]==1
    assert r['neutral_1024_minus_128'][0]==0 and r['records_minus_neutral_1024'][0]<0
    assert r['exploratory_design_lead']=='fixed_budget_write_isolation'
