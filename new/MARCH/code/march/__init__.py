"""MARCH reference implementation (eager PyTorch) + HF/FLAME-compatible wrapper."""

from .anchors import anchor_layout, expand_with_anchors
from .config import MARCHConfig
from .hf import MARCHForCausalLM, register as register_march
from .layer import MARCHLayer
from .model import MARCHModel
from .recurrence import GatedDeltaRuleReference
from .routing import historical_readout, routing_weights

# Make `AutoModelForCausalLM.from_config(MARCHConfig(...))` work as soon as this
# package is imported -- that is the path FLAME uses for every model.
register_march()

__all__ = [
    "MARCHConfig",
    "MARCHForCausalLM",
    "MARCHLayer",
    "MARCHModel",
    "GatedDeltaRuleReference",
    "anchor_layout",
    "expand_with_anchors",
    "routing_weights",
    "historical_readout",
    "register_march",
]
