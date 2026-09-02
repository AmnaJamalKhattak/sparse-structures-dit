"""Configuration for a layer sweep."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .registry import ModelSpec, resolve_model

DEFAULT_PROMPTS = [
    "A close-up portrait of an astronaut in a reflective helmet, cinematic lighting, ultra detailed",
    "A futuristic city street at night after rain, neon reflections, detailed architecture",
    "A red fox sitting in a snowy forest, shallow depth of field, cinematic",
]


@dataclass
class SweepConfig:
    """Everything needed to reproduce one layer sweep.

    A sweep is: for each (prompt, seed), run the denoiser once, and at every
    transformer layer of every captured denoising step record
      * residual-stream token norms          -> high-norm tokens
      * incoming image->image attention mass -> attention sinks
      * per-channel activation magnitudes    -> massive activation channels
    """

    model: str = "flux1-schnell"

    prompts: List[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS[:1]))
    seeds: List[int] = field(default_factory=lambda: [0])

    num_inference_steps: Optional[int] = None       # None -> model default
    guidance_scale: Optional[float] = None          # None -> model default
    height: Optional[int] = None                    # None -> model default
    width: Optional[int] = None

    # Which denoising steps to capture. None -> only the final step.
    capture_steps: Optional[List[int]] = None

    # Which layers to sweep. None -> every layer (that is the point of the atlas).
    layers: Optional[List[int]] = None

    # Layers for which the FULL per-head attention matrix is kept (for the
    # per-head attention-map grid). Full maps are O(H*N^2) so keep this short.
    focus_layers: List[int] = field(default_factory=list)
    # If focus_layers is empty these fractions of network depth are used instead,
    # so the same config works for a 28-block PixArt and a 57-block FLUX.
    focus_layer_fractions: List[float] = field(default_factory=lambda: [0.05, 0.35, 0.65, 0.95])
    focus_map_max_tokens: int = 128                 # downsample target for stored maps

    # High-norm token definition.
    highnorm_ratio: float = 3.0                     # norm > ratio * median
    highnorm_percentile: float = 99.0               # secondary, percentile-based
    register_topk: int = 8                          # cap on stored register vectors / layer

    # Attention-sink definition.
    sink_topk: int = 10
    # A layer "has sinks" when the top-1 image key absorbs at least this many
    # times the uniform share (1/N) of the image->image attention mass.
    sink_ratio_threshold: float = 10.0

    # Massive-activation-channel definition (Sun et al. style): a channel is
    # massive when its max |activation| over tokens exceeds this multiple of the
    # median over channels of that same statistic.
    massive_channel_ratio: float = 10.0
    channel_topk: int = 16

    dtype: str = "bfloat16"
    device: Optional[str] = None                    # None -> cuda if available
    quantize_4bit: bool = False                     # for FLUX.2-dev on one GPU
    attn_chunk: int = 256                           # query chunk for stat computation

    output_dir: str = "outputs/ditsinks"
    save_images: bool = True
    verify_capture: bool = True                     # assert hooks do not alter the model output

    def __post_init__(self):
        spec = self.spec
        if self.num_inference_steps is None:
            self.num_inference_steps = spec.default_steps
        if self.guidance_scale is None:
            self.guidance_scale = spec.default_guidance
        if self.height is None:
            self.height = spec.default_size
        if self.width is None:
            self.width = self.height
        if self.capture_steps is None:
            self.capture_steps = [self.num_inference_steps - 1]
        self.capture_steps = sorted({int(s) % max(self.num_inference_steps, 1) for s in self.capture_steps})

    # ------------------------------------------------------------------ views
    @property
    def spec(self) -> ModelSpec:
        return resolve_model(self.model)

    @property
    def family(self) -> str:
        return self.spec.family

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.spec.key

    def wants_step(self, step: int) -> bool:
        return step in set(self.capture_steps)

    def wants_layer(self, layer: int) -> bool:
        return self.layers is None or layer in set(self.layers)

    # ------------------------------------------------------------------- I/O
    def to_json(self) -> str:
        d = asdict(self)
        d["resolved_repo_id"] = self.spec.repo_id
        d["resolved_family"] = self.spec.family
        return json.dumps(d, indent=2)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path is not None else self.run_dir / "sweep_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path) -> "SweepConfig":
        d = json.loads(Path(path).read_text())
        d.pop("resolved_repo_id", None)
        d.pop("resolved_family", None)
        return cls(**d)
