from dataclasses import dataclass, asdict
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .mixer import MemoryConfig, BranchMemoryMixer

@dataclass
class LMConfig(MemoryConfig):
    vocab_size: int = 256
    n_layers: int = 2
    ff_mult: int = 2
    def __post_init__(self):
        super().__post_init__()
        if self.vocab_size<2 or self.n_layers<1 or self.ff_mult<1:raise ValueError('invalid LM dimensions')

class Block(nn.Module):
    def __init__(self,c:LMConfig):
        super().__init__()
        mixer_cfg=MemoryConfig(**{k:v for k,v in asdict(c).items() if k in MemoryConfig.__dataclass_fields__})
        self.norm1=nn.LayerNorm(c.d_model);self.mixer=BranchMemoryMixer(mixer_cfg)
        self.norm2=nn.LayerNorm(c.d_model)
        self.ff=nn.Sequential(nn.Linear(c.d_model,c.ff_mult*c.d_model),nn.GELU(),nn.Linear(c.ff_mult*c.d_model,c.d_model))
    def forward(self,x,cache=None):
        z,cache,stats=self.mixer(self.norm1(x),cache);x=x+z
        return x+self.ff(self.norm2(x)),cache,stats

class TinyMemoryLM(nn.Module):
    def __init__(self,config:LMConfig):
        super().__init__();self.config=config
        self.embedding=nn.Embedding(config.vocab_size,config.d_model)
        self.blocks=nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm=nn.LayerNorm(config.d_model)
        self.head=nn.Linear(config.d_model,config.vocab_size,bias=False)
    def forward(self,input_ids:Tensor,caches=None,return_cache:bool=False):
        if input_ids.ndim!=2:raise ValueError('input_ids must be [batch,time]')
        if caches is not None and len(caches)!=len(self.blocks):raise ValueError('wrong number of caches')
        x=self.embedding(input_ids);out_caches=[];stats=[]
        for i,block in enumerate(self.blocks):
            x,cache,s=block(x,None if caches is None else caches[i]);out_caches.append(cache);stats.append(s)
        logits=self.head(self.norm(x))
        return (logits,stats,out_caches) if return_cache else (logits,stats)

def next_token_loss(logits:Tensor,tokens:Tensor)->Tensor:
    """Exactly one teacher-forcing shift: input x_t predicts token x_(t+1)."""
    if tokens.shape[1]<2:raise ValueError('at least two tokens are required')
    return F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),tokens[:,1:].reshape(-1))
