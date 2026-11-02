"""Token-matched counterfactual histories. No filenames, roles or gold in reader inputs."""
from dataclasses import dataclass
from itertools import permutations
from functools import lru_cache
import hashlib
import random

ALPHA='ABCD'
QUESTIONS=('x_now','y_now','x_first','z_now')
PANELS=('swap','swap_yfirst','order','cross','new_only','repeat','absent')
STAGES=('pre','write','other','neutral','records')
FORM={'train':('current entry','first entry'),'dev':('latest entry','original entry'),'test':('present value','earliest value')}

@dataclass(frozen=True)
class Scene:
    id:str; split:str; x:str; y:str; z:str; a:str; b:str; c:str; d:str; zval:str; reverse:bool

def scenario(split,i,seed):
    if split not in FORM or i<0:raise ValueError('invalid scene')
    block=i//24; perms=list(permutations(ALPHA));random.Random(f'{seed}/{split}/{block}').shuffle(perms)
    a,b,c,d=perms[i%24]; sid=f'{seed}/{split}/{i}'
    names=[hashlib.sha256(f'{sid}/{j}'.encode()).hexdigest()[:12] for j in range(3)]
    r=random.Random(sid); return Scene(sid,split,*names,a,b,c,d,r.choice(ALPHA),bool(r.randrange(2)))

@dataclass(frozen=True)
class Item:
    prefix:tuple; question:tuple; text:str; ids:tuple; write:int; oldpos:int; start:int; end:int; gap:int
    gold:str; old:str; distractor:str; name:str; version:str

def enc(tok,t):return list(tok.encode(t,add_special_tokens=False))

def padding(tok,n):
    if n<0:raise ValueError('negative padding')
    unit=enc(tok,' Routine archive entry.');return (unit*(1+n//len(unit)))[:n]

@lru_cache(maxsize=4096)
def choices(tok,text):
    p=enc(tok,text)
    for pre in (' ','','\n'):
        opts=[enc(tok,text+pre+a) for a in ALPHA]
        if all(len(x)==len(p)+1 and x[:-1]==p for x in opts) and len({x[-1] for x in opts})==4:
            return tuple(x[-1] for x in opts)
    raise ValueError('not four distinct single next-token choices')

def render(tok,s,panel,world,stage,distance,question,donor='self'):
    if panel not in PANELS or world not in (0,1) or stage not in STAGES or question not in QUESTIONS or distance<0 or donor not in ('self','clean','wrong'):
        raise ValueError('invalid panel request')
    if stage not in ('neutral','records') and distance:raise ValueError('non-delay stage with delay')
    x0,y0=s.a,s.d; xn,yn=(s.c,s.b) if world else (s.b,s.c)
    if panel=='order':x0,xn=(s.b,s.a) if world else (s.a,s.b);yn=s.c
    if panel=='cross':xn=s.b;yn=(s.d if world else s.c);y0=s.a
    if panel=='repeat':x0=xn
    gold_new=xn
    if donor=='wrong':xn=ALPHA[(ALPHA.index(xn)+1)%4]
    hdr=list(tok.encode('Ledger. Report the requested value of the named record.\n',add_special_tokens=True)); hdr+=padding(tok,max(0,96-len(hdr)))
    def rec(name,value,update=False):
        lead=enc(tok,'\nRecord '+name+(': updated to' if update else ':'))
        values=[enc(tok,' '+c) for c in ALPHA]
        if not values[ALPHA.index(value)] or len({len(v) for v in values})!=1:raise ValueError('unmatched value-token widths')
        val=values[ALPHA.index(value)]; tail=enc(tok,'.\n');return lead+val+tail,len(lead)+len(val)-1
    oldrec,oldoff=rec(s.x,x0); yr,_=rec(s.y,y0);zr,_=rec(s.z,s.zval)
    hidden_old=donor=='clean' or panel in ('new_only','absent');hist=padding(tok,len(oldrec)) if hidden_old else oldrec
    if s.reverse:p=hdr+yr+zr;oldpos=len(p)+oldoff;p+=hist
    else:p=hdr;oldpos=len(p)+oldoff;p=p+hist+yr+zr
    # Duplicate Z at the end on purpose: the latest visible value is not a shortcut.
    yr1,_=rec(s.y,yn,True); xrec,xoff=rec(s.x,xn,True)
    yfirst=panel=='swap_yfirst'; p=p+(yr1 if yfirst else [])
    start=len(p);wp=start+xoff;end=start+len(xrec)
    if stage!='pre':p+=padding(tok,len(xrec)) if panel=='absent' else xrec
    if stage in ('other','neutral','records'):p+=([] if yfirst else yr1)+zr
    if stage=='records':
        remaining=distance;j=0
        while remaining:
            name=hashlib.sha256(f'{s.id}/other/{j}'.encode()).hexdigest()[:12]
            line,_=rec(name,ALPHA[j%4],True)
            if len(line)>remaining:break
            p+=line;remaining-=len(line);j+=1
        p+=padding(tok,remaining)
    if stage=='neutral':p+=padding(tok,distance)
    name={'x_now':s.x,'x_first':s.x,'y_now':s.y,'z_now':s.z}[question]
    version=FORM[s.split][int(question=='x_first')];text=f'\nQuestion: What is the {version} of {name}?\nAnswer:'
    yactive=yfirst or stage in ('other','neutral','records')
    old={'x_now':x0,'x_first':x0,'y_now':y0,'z_now':s.zval}[question]
    gold={'x_now':x0 if stage=='pre' else gold_new,'x_first':gold_new if panel=='new_only' else x0,'y_now':yn if yactive else y0,'z_now':s.zval}[question]
    return Item(tuple(p),tuple(enc(tok,text)),text,choices(tok,text),wp,oldpos,start,end,len(p)-end if stage!='pre' else -1,gold,old,yn if yactive else y0,name,version)
