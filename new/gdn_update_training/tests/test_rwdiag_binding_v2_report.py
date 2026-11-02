"""Scorer and launcher contracts. Hand-labelled fixtures, never pretrained results."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rwdiag.panel import BINDING_PANELS, FAMILIES, ORDERS, scenario, binding_values
from rwdiag.binding_score import crossed, make_report, check_reader
from rwdiag.report import main as base_report

ARMS=('native','q','k','beta','v','q+k','q+beta','q+v','probe','probe_state_swap','q_sham','question_only')

def rows():
    # Native always copies the first updated value. Q is the symbolic oracle.
    for i in range(2):
        s=scenario('dev',i,8)
        for f in FAMILIES:
            vals=[binding_values(s,f,w)[2:] for w in (0,1)]
            for o in ORDERS:
                for w in (0,1):
                    for j,q in enumerate(('x_now','y_now')):
                        gold=vals[w][j]
                        for a in ARMS:
                            pred=vals[w][int(o=='yx')] if a=='native' else vals[1-w][j] if a=='probe_state_swap' else gold
                            yield dict(panel=f+'_'+o,stage='other',distance=0,question=q,arm=a,scene=s.id,world=w,gold=gold,pred=pred,actual_gap=17)

def test_crossed_score_pairs_and_guardrails():
    table=list(rows());m=crossed(table,100)
    pick=lambda arm,f:next(r for r in m if r['arm']==arm and r['family']==f)
    assert pick('native','bind_swap')['all8'][0]==0
    q=pick('q','bind_swap');assert q['all8']==q['all8_delta']==[1.,1.,1.]
    assert q['rescued_rows']==8 and q['broken_rows']==0 and q['n_scenes']==2
    assert q['net_row_delta'][0]==.5  # four of eight native trials correct
    assert all(pick('probe_state_swap',f)['donor_follow_all_changed'][0]==1 for f in FAMILIES)
    for bad in (table[:-1],table+[table[0]]):
        with pytest.raises(ValueError):crossed(bad,100)

def test_actual_report_flow_and_mutation_detection(tmp_path):
    path=tmp_path;raw=path/'rows.jsonl'
    cfg={'panels':list(BINDING_PANELS),'questions':['x_now','y_now'],'stages':['other'],'distances':[128],'scenes':2,'rounds':100}
    for name,obj in [('config',cfg),('status',{'action':'eval','status':'completed','readers_sha256':'fixture'}),('manifest',{'frozen':True,'protocol':'gdn-rwdiag.v1','identity':{'kind':'fixture'}})]:
        (path/(name+'.json')).write_text(json.dumps(obj))
    raw.write_text(''.join(json.dumps(r)+'\n' for r in rows()));base_report([str(path),'--out',str(path/'base_report.json'),'--rounds','100'])
    r=make_report(path);assert r['protocol']=='rwdiag.binding-v2' and r['input_rows_sha256']==hashlib.sha256(raw.read_bytes()).hexdigest()
    raw.write_text(raw.read_text().replace('"actual_gap": 17','"actual_gap": 18',1))
    with pytest.raises(ValueError,match='does not match'):make_report(path)

def test_legacy_reader_rejected(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'split':'train','panels':['swap','cross'],'questions':['x_now','y_now']}))
    (tmp_path/'status.json').write_text(json.dumps({'action':'fit','status':'completed'}))
    with pytest.raises(ValueError,match='v2 fit'):check_reader(tmp_path)

def test_launcher_audit_is_offline_and_preserves_prior_runs(tmp_path):
    launcher=ROOT/'run_rwdiag.sh'; env=dict(os.environ,OUT_ROOT=str(tmp_path),PYTHON=sys.executable,MODEL='/nonexistent/model',TRAIN_SCENES='24',DEV_SCENES='24')
    subprocess.run(['bash','-n',str(launcher)],check=True)
    for _ in range(2):subprocess.run(['bash',str(launcher),'audit'],env=env,cwd=tmp_path,check=True,capture_output=True)
    ds=list(tmp_path.glob('audit_*'));assert len(ds)==2
    for d in ds:
        assert json.loads((d/'audit_dev.json').read_text())['signed_pairs_redacted']==192
        assert (d/'audit_train.json').exists() and not (d/'fit').exists()
