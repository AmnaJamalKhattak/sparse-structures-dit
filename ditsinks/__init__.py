"""High-norm tokens, attention sinks and massive activation channels in DiTs.

One analysis, three model families (FLUX.1, FLUX.2, PixArt-Sigma).
"""
from .config import SweepConfig, DEFAULT_PROMPTS
from .registry import MODEL_REGISTRY, ModelSpec, registry_table, resolve_model
from .runner import SweepResult, run_sweep

__all__ = [
    "SweepConfig",
    "DEFAULT_PROMPTS",
    "MODEL_REGISTRY",
    "ModelSpec",
    "registry_table",
    "resolve_model",
    "SweepResult",
    "run_sweep",
]
__version__ = "0.1.0"
