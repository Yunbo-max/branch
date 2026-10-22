"""Serial PyTorch reference, NOT an optimized attention kernel.

All three modes share projections and a causal short convolution. `snapshot`
reuses the periodic-snapshot/soft-read concept of MARCH but NOT its learned
anchor-token stream across layers. See docs/SOURCE_AUDIT.md for exact differences.
"""
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .cache import MemoryCache
from .math import choose_write
from .geometry import update_protection
import warnings

@dataclass
class MemoryConfig:
    d_model: int = 64
    n_heads: int = 2
    d_key: int = 16
    d_value: int = 16
    d_route: int = 16
    n_probes: int = 2
    max_states: int = 8
    mode: str = 'branch'
    snapshot_interval: int = 4
    conv_kernel: int = 3
    tau: float = 1.0
    ridge: float = 1.0
    protect: float = 8.0
    fork_cost: float = 0.02
    protection_decay: float = 0.95
    covariance: str = 'full'
    differentiate_covariance: bool = False

    def __post_init__(self):
        for name in ['d_model','n_heads','d_key','d_value','d_route','n_probes',
                     'max_states','snapshot_interval','conv_kernel']:
            if getattr(self,name)<1: raise ValueError(f'{name} must be positive')
        if self.covariance not in {'full','diagonal'}:raise ValueError('covariance must be full or diagonal')
        for name in ['tau','ridge','protect','fork_cost','protection_decay']:
            if not math.isfinite(getattr(self,name)):raise ValueError(f'{name} must be finite')
        if self.mode not in {'delta','snapshot','branch'}:raise ValueError('invalid mode')
        if self.tau<=0 or self.ridge<=0:raise ValueError('tau and ridge must be positive')
        if self.protect<0 or self.fork_cost<0:raise ValueError('negative cost weight')
        if not 0<=self.protection_decay<1:raise ValueError('invalid protection decay')


    @classmethod
    def from_dict(cls, data: dict):
        copied = dict(data)
        if 'covariance' not in copied:
            warnings.warn('Legacy config uses diagonal geometry; set covariance explicitly to upgrade.', UserWarning, stacklevel=2)
            copied['covariance'] = 'diagonal'
        return cls(**copied)


class BranchMemoryMixer(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__();self.config=config
        c=config
        self.conv=nn.Conv1d(c.d_model,c.d_model,c.conv_kernel,groups=c.d_model,bias=False)
        self.q_proj=nn.Linear(c.d_model,c.n_heads*c.d_key,bias=False)
        self.k_proj=nn.Linear(c.d_model,c.n_heads*c.d_key,bias=False)
        self.v_proj=nn.Linear(c.d_model,c.n_heads*c.d_value,bias=False)
        self.route_q=nn.Linear(c.d_model,c.n_heads*c.d_route,bias=False)
        self.null_route=nn.Linear(c.d_model,c.n_heads,bias=True)
        # Two-stage content descriptor: value-side probes reveal key-space signatures.
        # This is our explicit substitute for MARCH's cross-layer anchor tokens.
        self.value_probes=nn.Parameter(torch.randn(c.n_heads,c.n_probes,c.d_value)/math.sqrt(c.d_value))
        self.content_proj=nn.Parameter(torch.randn(c.n_heads,c.n_probes*c.d_key,c.d_route)/math.sqrt(c.n_probes*c.d_key))
        self.out_proj=nn.Linear(c.n_heads*c.d_value,c.d_model,bias=False)
        nn.init.constant_(self.null_route.bias,1.0)

    def initial_cache(self,x:Tensor)->MemoryCache:
        c=self.config;B=x.shape[0];H=c.n_heads
        M=1 if c.mode=='delta' else c.max_states
        dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
        z=lambda *s:torch.zeros(*s,device=x.device,dtype=dtype)
        active=torch.zeros(B,H,M,device=x.device,dtype=torch.bool)
        if c.mode=='branch': active[...,0]=True
        C=z(B,H,M,c.d_key,c.d_key) if c.covariance=='full' else z(B,H,M,c.d_key)
        return MemoryCache(z(B,H,M,c.d_value,c.d_key),C,active,
                           z(B,H,M,c.d_route),z(B,H,c.d_value,c.d_key),
                           x.new_zeros(B,c.d_model,c.conv_kernel-1),covariance=c.covariance)

    def descriptor(self,state:Tensor)->Tensor:
        # [...,H,V,K] here is concretely B,H,V,K.
        sig=torch.einsum('bhvk,hpv->bhpk',state,self.value_probes.to(state.dtype))
        sig=sig.flatten(-2)
        # No LayerNorm: zero-state remains a zero descriptor; the null route is separate.
        sig=sig / sig.square().mean(-1,keepdim=True).add(1e-6).sqrt()
        return torch.einsum('bhp,hpr->bhr',sig,self.content_proj.to(state.dtype))

    def historical_read(self,q:Tensor,r:Tensor,x:Tensor,states:Tensor,
                        addresses:Tensor,visible:Tensor)->tuple[Tensor,Tensor]:
        c=self.config
        logits=torch.einsum('bhr,bhmr->bhm',r,addresses)/math.sqrt(c.d_route)
        logits=logits.masked_fill(~visible,float('-inf'))
        null=self.null_route(x).to(q.dtype).unsqueeze(-1)
        weights=F.softmax(torch.cat([logits,null],-1),-1)[...,:-1]
        historical=torch.einsum('bhm,bhmvk,bhk->bhv',weights,states,q)
        return historical,weights

    def forward(self,x:Tensor,cache:MemoryCache|None=None)->tuple[Tensor,MemoryCache,dict[str,Tensor]]:
        if x.ndim!=3 or x.shape[-1]!=self.config.d_model or x.shape[1]<1:
            raise ValueError('x must be [B,T,d_model], with T>=1')
        if not torch.isfinite(x).all():raise ValueError('non-finite input')
        c=self.config;B,T,_=x.shape;H=c.n_heads
        if cache is None:cache=self.initial_cache(x)
        if cache.states.shape[0]!=B or cache.states.device!=x.device:
            raise ValueError('cache batch/device mismatch; reset unrelated streams')
        M=1 if c.mode=='delta' else c.max_states
        expected_C=(B,H,M,c.d_key,c.d_key) if c.covariance=='full' else (B,H,M,c.d_key)
        if cache.covariance!=c.covariance or cache.protection.shape!=expected_C:
            raise ValueError('cache covariance/protection mismatch; reset or explicitly convert geometry')
        if cache.states.shape!=(B,H,M,c.d_value,c.d_key):
            raise ValueError('cache state dimensions mismatch')
        # Causal, streaming-compatible short convolution. Never reads a future token.
        conv_input=torch.cat([cache.conv_context,x.transpose(1,2)],-1)
        local=x+F.silu(self.conv(conv_input).transpose(1,2))
        next_context=conv_input[...,-(c.conv_kernel-1):] if c.conv_kernel>1 else conv_input[...,:0]
        dtype=cache.states.dtype
        q=F.normalize(self.q_proj(local).view(B,T,H,c.d_key).to(dtype),dim=-1,eps=1e-6)
        k=F.normalize(self.k_proj(local).view(B,T,H,c.d_key).to(dtype),dim=-1,eps=1e-6)
        v=torch.tanh(self.v_proj(local).view(B,T,H,c.d_value).to(dtype))
        rq=self.route_q(local).view(B,T,H,c.d_route).to(dtype)
        states,C,active,addresses,current=(cache.states,cache.protection,cache.active,cache.addresses,cache.current)
        ys=[];trace={z:[] for z in ['parent','destination','fork','active_states','historical_candidates','write_cost']}
        for t in range(T):
            qt,kt,vt=q[:,t],k[:,t],v[:,t]
            parent=torch.zeros(B,H,device=x.device,dtype=torch.long)
            destination=parent
            fork=torch.zeros(B,H,device=x.device,dtype=torch.bool)
            cost=torch.zeros(B,H,device=x.device,dtype=dtype)
            if c.mode=='branch':
                w=choose_write(states,C,active,kt,vt,tau=c.tau,ridge=c.ridge,
                               protect=c.protect,fork_cost=c.fork_cost,
                               detach_protection=not c.differentiate_covariance)
                states,C,active,current=w.states,w.protection,w.active,w.current
                parent,destination,fork,cost=w.parent,w.destination,w.fork,w.cost
                dest_mask=torch.arange(states.shape[2],device=x.device)==destination[...,None]
                new_address=self.descriptor(current)
                addresses=torch.where(dest_mask[...,None],new_address.unsqueeze(2),addresses)
                # Current candidate has a direct read; do not double count it in history.
                visible=active & ~dest_mask
            else:
                # Alpha=1. Same isotropic proximal coefficient as branch's unprotected write.
                residual=vt-torch.einsum('bhvk,bhk->bhv',current,kt)
                beta=1/(1+c.ridge*c.tau)
                current=current+beta*residual.unsqueeze(-1)*kt.unsqueeze(-2)
                visible=active if c.mode=='snapshot' else torch.zeros_like(active)
            current_out=torch.einsum('bhvk,bhk->bhv',current,qt)
            if c.mode=='delta':
                hist=torch.zeros_like(current_out);weights=torch.zeros_like(active,dtype=dtype)
            else:
                hist,weights=self.historical_read(qt,rq[:,t],local[:,t],states,addresses,visible)
            ys.append(current_out+hist)
            trace['historical_candidates'].append(visible.sum(-1).detach())
            if c.mode=='branch':
                # Statistics use only observed past/current queries, AFTER this write decision.
                # Optional differentiable statistics; hard action switches remain non-differentiable.
                use=weights+dest_mask.to(dtype)
                C=update_protection(C,qt,use,active,decay=c.protection_decay,
                                    covariance=c.covariance,detach=not c.differentiate_covariance)
            elif c.mode=='snapshot' and (cache.step+t+1)%c.snapshot_interval==0:
                slot=(cache.step+t+1)//c.snapshot_interval-1
                if slot>=c.max_states:
                    raise RuntimeError('snapshot capacity exhausted: increase max_states; no silent eviction is performed')
                mask=torch.arange(c.max_states,device=x.device)==slot
                states=torch.where(mask[None,None,:,None,None],current.unsqueeze(2),states)
                addresses=torch.where(mask[None,None,:,None],self.descriptor(current).unsqueeze(2),addresses)
                active=active|mask[None,None,:]
                # Appended AFTER reading: boundary b can only be read at token t>b.
            trace['parent'].append(parent.detach());trace['destination'].append(destination.detach())
            trace['fork'].append(fork.detach());trace['active_states'].append(active.sum(-1).detach())
            trace['write_cost'].append(cost.detach())
        output=torch.stack(ys,1).reshape(B,T,H*c.d_value)
        output=self.out_proj(output.to(local.dtype))
        new_cache=MemoryCache(states,C,active,addresses,current,next_context,cache.step+T,c.covariance)
        stats={name:torch.stack(values,1) for name,values in trace.items()}
        return output,new_cache,stats
