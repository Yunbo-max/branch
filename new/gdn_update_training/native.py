"""Removable training-only tap at FLA's real post-convolution kernel boundary.

No replacement recurrence. The original kernel still executes. Single process,
no packed inputs, no checkpoint recomputation, no grouped-value/negative-beta
teacher support in v1. Export is ordinary native model state_dict only.
"""
from __future__ import annotations
import inspect
import threading
from dataclasses import dataclass
import torch
from torch.nn import functional as F

@dataclass
class Write:
    key: torch.Tensor
    beta: torch.Tensor

_LOCK = threading.Lock()

class NativeTap:
    def __init__(self, layer, *, position: int, head: int, override: Write | None = None):
        self.layer,self.position,self.head,self.override=layer,position,head,override
        self.capture=None
        self.calls=0
        self.active=False
        self.handles=[]
        self.originals={}
        self.locked=False

    def _kernel(self, original, *args, **kwargs):
        if not self.active:
            return original(*args,**kwargs)
        if threading.get_ident()!=self.thread:
            raise RuntimeError('concurrent forward is unsupported during a native tap')
        if args or kwargs.get('cu_seqlens') is not None or kwargs.get('head_first',False):
            raise ValueError('teacher requires unpacked keyword B,T,H,D native calls')
        if kwargs.get('initial_state') is not None or kwargs.get('output_final_state',False):
            raise ValueError('teacher requires a full no-cache forward')
        if not kwargs.get('use_qk_l2norm_in_kernel',False):
            raise ValueError('native key normalization contract is required')
        k,beta=kwargs['k'],kwargs['beta']
        if k.ndim!=4 or beta.shape!=k.shape[:3] or k.shape[0]!=1:
            raise ValueError('teacher v1 requires batch=1 and equal key/value head counts')
        if not (0<=self.position<k.shape[1] and 0<=self.head<k.shape[2]):
            raise ValueError('write position/head outside native tensor')
        self.calls+=1
        if self.calls!=1:
            raise RuntimeError('multiple target calls: checkpoint recomputation/reused layer is unsupported')
        raw=k[0,self.position,self.head]
        if not bool(torch.isfinite(raw).all()) or float(raw.float().norm().detach())<1e-8:
            raise ValueError('nonfinite or zero native key at write site')
        self.capture=Write(F.normalize(raw.float(),dim=-1),beta[0,self.position,self.head].float())
        if self.override is not None:
            key,value=self.override.key,self.override.beta
            if key.shape!=raw.shape or value.numel()!=1 or not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all()):
                raise ValueError('invalid teacher key/beta')
            if float(key.detach().float().norm())<1e-8 or not 0<=float(value.detach())<=1:
                raise ValueError('teacher key must be nonzero and beta in [0,1]')
            changed_k=k.clone(); changed_beta=beta.clone()
            # Preserve raw norm; native L2 normalization remains in the kernel.
            changed_k[0,self.position,self.head]=(F.normalize(key.float(),dim=-1)*raw.float().norm()).to(k.dtype)
            changed_beta[0,self.position,self.head]=value.to(beta.dtype)
            kwargs=dict(kwargs,k=changed_k,beta=changed_beta)
        return original(**kwargs)

    def __enter__(self):
        if self.position<0 or self.head<0 or getattr(self.layer,'allow_neg_eigval',False):
            raise ValueError('invalid site or negative-eigenvalue GDN is unsupported')
        if getattr(self.layer,'num_v_heads',None)!=getattr(self.layer,'num_heads',None):
            raise ValueError('GVA teacher not supported')
        if not _LOCK.acquire(blocking=False):
            raise RuntimeError('nested/concurrent native taps are unsupported')
        self.locked=True
        self.thread=threading.get_ident()
        self.module=inspect.getmodule(type(self.layer))
        try:
            for name in ('chunk_gated_delta_rule','fused_recurrent_gated_delta_rule'):
                if self.module is not None and hasattr(self.module,name):
                    original=getattr(self.module,name)
                    self.originals[name]=original
                    def wrapper(*args,_original=original,**kwargs):
                        return self._kernel(_original,*args,**kwargs)
                    setattr(self.module,name,wrapper)
            if not self.originals:
                raise ValueError('layer has no supported native GDN kernel boundary')
            def before(module,args): self.active=True
            def after(module,args,output): self.active=False
            self.handles.append(self.layer.register_forward_pre_hook(before))
            self.handles.append(self.layer.register_forward_hook(after,always_call=True))
            return self
        except BaseException:
            self._restore()
            raise

    def _restore(self):
        for h in self.handles: h.remove()
        self.handles=[]
        for name,fn in self.originals.items(): setattr(self.module,name,fn)
        self.originals={}
        self.active=False
        if self.locked:
            self.locked=False
            _LOCK.release()

    def __exit__(self,typ,value,tb):
        self._restore()
        if typ is None and self.calls!=1:
            raise RuntimeError('selected native GDN layer was not called exactly once')
        return False
