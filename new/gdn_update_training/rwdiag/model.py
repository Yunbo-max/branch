"""Native model wrapper and two independent diagnostic readers. Model parameters stay frozen."""
from __future__ import annotations
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .boundary import observe

class Readers(nn.Module):
    def __init__(self,emb,k,v,rank=16):
        super().__init__()
        def direction():return nn.Sequential(nn.Linear(2*emb,rank),nn.Tanh(),nn.Linear(rank,k))
        self.q=direction();self.probe=direction();self.decode=nn.Linear(v,4)
        self.perm=direction();self.pdecode=nn.Linear(v,4);self.question_only=nn.Sequential(nn.Linear(2*emb,rank),nn.Tanh(),nn.Linear(rank,4))
    def state_logits(self,state,feat,shuffled=False):
        a=(self.perm if shuffled else self.probe)(feat)
        return (self.pdecode if shuffled else self.decode)(F.normalize(a,dim=0)@state)

class Engine:
    def __init__(self,model,tok,layer,head,allow=False):
        import inspect
        self.model=model.eval().requires_grad_(False);self.tok=tok
        if not 0<=layer<len(model.model.layers):raise ValueError('bad layer')
        self.layer=model.model.layers[layer].attn;self.head=head;self.allow=allow
        self.device=next(model.parameters()).device;self.calls=0
        ps=inspect.signature(model.forward).parameters
        self.last=next(({k:1} for k in ('logits_to_keep','num_logits_to_keep') if k in ps),{})
    def send(self,seq):
        lim=getattr(self.model.config,'max_position_embeddings',None)
        if not seq or (lim and len(seq)>lim and not self.allow):raise ValueError(f'empty or extrapolated prompt: {len(seq)}, configured limit {lim}')
        self.calls+=1
        return self.model(input_ids=torch.tensor([seq],device=self.device),use_cache=False,**self.last).logits[0,-1].float()
    def call(self,seq,edits=(),state=False):
        with observe(self.layer,self.head,edits,state) as cap:logits=self.send(seq)
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite model output')
        return logits,cap
    def features(self,row):
        # Only question tokens and the bare entity string; never prefix, cache, data seed or answer.
        emb=self.model.get_input_embeddings()
        def pool(ids):return emb(torch.tensor(ids,device=self.device)).float().mean(0).detach()
        return torch.cat([pool(self.tok.encode(row.name,add_special_tokens=False)),pool(row.question)])
    def prefix_state(self,row):
        return self.call(row.prefix,state=True)[1]['state']
    def qualify(self,row,atol=.08,rtol=.03):
        seq=row.prefix+row.question
        with torch.no_grad():
            plain=self.send(seq);read,c=self.call(seq,state=True)
            torch.testing.assert_close(read,plain,atol=atol,rtol=rtol)
            reconstructed=F.normalize(c['q'][-1].float(),dim=0)@c['state']*c['scale']
            torch.testing.assert_close(reconstructed,c['last_out'],atol=atol,rtol=rtol)
            edits=[(x,row.write,c[x][row.write]) for x in ('k','v','beta')]+[('q',len(seq)-1,c['q'][-1])]
            same=self.call(seq,edits)[0];torch.testing.assert_close(same,plain,atol=atol,rtol=rtol)
        return {'full_vocab_max_abs':float((plain-read).abs().max()),'self_patch_max_abs':float((same-plain).abs().max()),'state_read_max_abs':float((reconstructed-c['last_out']).abs().max()),'atol':atol,'rtol':rtol}

def like(direction,raw):
    return F.normalize(direction.float(),dim=0)*raw.float().norm().detach()

def sham(raw,target,tag):
    """Same parallel component and perpendicular norm, with answer-independent orientation."""
    a=F.normalize(raw.float(),dim=0);t=target.float();parallel=(t@a)*a
    r=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8],'big')%(2**63-1))
    n=torch.randn(a.shape,generator=r).to(a);n=F.normalize(n-(n@a)*a,dim=0)
    return parallel+n*(t-parallel).norm()
