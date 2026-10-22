from dataclasses import dataclass, fields
from typing import Any
import torch
from torch import Tensor


@dataclass
class MemoryCache:
    states: Tensor       # B H M V K
    protection: Tensor   # B H M K K (full) or B H M K (explicit diagonal ablation)
    active: Tensor       # B H M
    addresses: Tensor    # B H M R
    current: Tensor      # B H V K, separate current branch/readout state
    conv_context: Tensor # B D (kernel-1)
    step: int = 0
    covariance: str = 'diagonal'  # explicit construction overrides this for new full caches

    def detach(self) -> 'MemoryCache':
        """Explicit truncated-BPTT boundary; never applied implicitly in forward."""
        return MemoryCache(**self.to_dict(detach=True))

    def to_dict(self, *, detach: bool = True) -> dict[str, Any]:
        """Plain tensor/primitive cache serialization, compatible with weights_only.

        Saving/loading a cache is an inference/TBPTT boundary: autograd history
        is not serialized. detach=False only affects this in-memory dictionary.
        """
        return {f.name: (getattr(self, f.name).detach()
                         if detach and isinstance(getattr(self, f.name), Tensor)
                         else getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MemoryCache':
        copied = dict(data)
        if 'covariance' not in copied:
            copied['covariance'] = 'full' if copied['protection'].ndim == 5 else 'diagonal'
        return cls(**copied)

    def tensor_bytes_by_field(self) -> dict[str, int]:
        return {f.name: x.numel()*x.element_size() for f in fields(self)
                if isinstance(x := getattr(self, f.name), Tensor)}

    def allocated_tensor_bytes(self) -> int:
        """Payload including C; excludes activations, allocator and factor temporaries.

        Some tensors may share backing storage. Counting both is conservative;
        this is NOT process RSS or peak training memory. Physical pool is fixed.
        """
        return sum(self.tensor_bytes_by_field().values())
