"""Observe or edit exactly one frozen FLA layer. Full uncached rerun; no shared live cache."""
from contextlib import ExitStack,contextmanager
from unittest.mock import patch
import inspect
import threading
import torch

_LOCK=threading.Lock()

@contextmanager
def observe(layer,head,edits=(),snapshot=False):
    if head<0 or layer.num_heads!=layer.num_v_heads or head>=layer.num_heads or layer.allow_neg_eigval:
        raise ValueError('requires equal K/V head counts, valid head and ordinary beta')
    if not _LOCK.acquire(blocking=False):raise RuntimeError('concurrent/nested native observation')
    owner=threading.get_ident();mod=inspect.getmodule(type(layer)); active=[False];cap={};calls=[0]
    def wrap(fn):
        def inner(*args,**kw):
            if not active[0]:return fn(*args,**kw)
            if threading.get_ident()!=owner or args or any(kw.get(x) is not None for x in ('cu_seqlens','initial_state','cp_context')) or kw.get('output_final_state') or kw.get('transpose_state_layout') or not kw.get('use_qk_l2norm_in_kernel'):
                raise ValueError('unsupported FLA packed/cache/layout/normalization or caller')
            if kw['k'].ndim!=4 or kw['k'].shape[0]!=1 or kw['beta'].shape!=kw['k'].shape[:3]:raise ValueError('invalid B,T,H,K layout')
            calls[0]+=1
            if calls[0]!=1:raise RuntimeError('target layer was reused')
            # Store only the selected head; no gradients flow via diagnostic captures.
            for name in ('q','k','v','beta'):cap[name]=kw[name][0,:,head].detach().clone()
            touched=set()
            for name,pos,value in edits:
                if name not in ('q','k','v','beta') or not 0<=pos<kw['k'].shape[1] or (name,pos) in touched:raise ValueError('bad/duplicate edit site')
                touched.add((name,pos));src=kw[name];old=src[0,pos,head]
                if value.shape!=old.shape or not bool(torch.isfinite(value).all()):raise ValueError('invalid replacement tensor')
                if name=='beta' and not 0<=float(value.detach())<=1:raise ValueError('beta outside [0,1]')
                if name in ('q','k') and float(value.detach().norm())==0:raise ValueError('zero direction')
                src=src.clone();src[0,pos,head]=value.to(old);kw[name]=src
            out,state=fn(**dict(kw,output_final_state=snapshot))
            if snapshot:
                if state is None or tuple(state.shape)!=(1,layer.num_heads,kw['k'].shape[-1],kw['v'].shape[-1]):raise RuntimeError('native final state is not [B,H,K,V]')
                cap['state']=state[0,head].float().detach().clone();cap['last_out']=out[0,-1,head].float().detach().clone()
                cap['scale']=kw.get('scale') or kw['k'].shape[-1]**-.5
            return out,None
        return inner
    try:
        with ExitStack() as stack:
            count=0
            for name in ('chunk_gated_delta_rule','fused_recurrent_gated_delta_rule'):
                if hasattr(mod,name):stack.enter_context(patch.object(mod,name,wrap(getattr(mod,name))));count+=1
            if not count:raise ValueError('native operator not found')
            stack.callback(layer.register_forward_pre_hook(lambda m,a:active.__setitem__(0,True)).remove)
            stack.callback(layer.register_forward_hook(lambda m,a,o:active.__setitem__(0,False),always_call=True).remove)
            yield cap
            if calls[0]!=1:raise RuntimeError('selected kernel not called once')
    finally:_LOCK.release()
