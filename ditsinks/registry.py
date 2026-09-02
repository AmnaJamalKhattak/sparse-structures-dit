"""Model registry: one entry per checkpoint we can run the sink analysis on.

Everything downstream (adapters, capture, figures) is driven by the `family`
field, so adding a new checkpoint of a known family is a one-line change here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    family: str                       # "flux1" | "flux2" | "pixart"
    pipeline_cls: str                 # diffusers pipeline class name
    default_steps: int
    default_guidance: float
    default_size: int
    # Sequence-length knob passed to the pipeline (None = pipeline default).
    max_sequence_length: Optional[int] = 512
    # True when the pipeline runs classifier-free guidance by duplicating the
    # batch. The analysis then reads the conditional half.
    cfg_batched: bool = False
    approx_params_b: Optional[float] = None
    notes: str = ""
    aliases: Tuple[str, ...] = ()
    extra_call_kwargs: Dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.key} ({self.repo_id})"


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # ---------------------------------------------------------------- FLUX.1
    "flux1-schnell": ModelSpec(
        key="flux1-schnell",
        repo_id="black-forest-labs/FLUX.1-schnell",
        family="flux1",
        pipeline_cls="FluxPipeline",
        default_steps=4,
        default_guidance=0.0,
        default_size=512,
        max_sequence_length=512,
        approx_params_b=12.0,
        notes="19 dual + 38 single blocks, d_model 3072, 24 heads. Timestep-distilled: guidance_scale must stay 0.",
    ),
    "flux1-dev": ModelSpec(
        key="flux1-dev",
        repo_id="black-forest-labs/FLUX.1-dev",
        family="flux1",
        pipeline_cls="FluxPipeline",
        default_steps=20,
        default_guidance=3.5,
        default_size=512,
        max_sequence_length=512,
        approx_params_b=12.0,
        notes="Same topology as schnell. Guidance is embedded (no CFG batch duplication). Gated repo: accept the licence first.",
    ),
    # ---------------------------------------------------------------- FLUX.2
    "flux2-dev": ModelSpec(
        key="flux2-dev",
        repo_id="black-forest-labs/FLUX.2-dev",
        family="flux2",
        pipeline_cls="Flux2Pipeline",
        default_steps=28,
        default_guidance=4.0,
        default_size=512,
        max_sequence_length=512,
        approx_params_b=32.0,
        notes=(
            "8 dual + 48 single blocks, parallel single-stream blocks (attention and MLP fused into one "
            "projection). ~32B params: needs >=64GB in bf16, so use quantize_4bit=True on a single 40/48GB GPU."
        ),
    ),
    "flux2-klein": ModelSpec(
        key="flux2-klein",
        repo_id="black-forest-labs/FLUX.2-klein-base-9B",
        family="flux2",
        pipeline_cls="Flux2KleinPipeline",
        default_steps=28,
        default_guidance=4.0,
        default_size=512,
        max_sequence_length=512,
        approx_params_b=9.0,
        aliases=("flux2-schnell", "flux2-9b"),
        notes=(
            "FLUX.2 has no model called 'schnell'. The small, fast, openly released FLUX.2 model is "
            "'klein' (9B), which is what 'flux2-schnell' resolves to here."
        ),
    ),
    # ------------------------------------------------------------ PixArt-Sigma
    "pixart-sigma-512": ModelSpec(
        key="pixart-sigma-512",
        repo_id="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        family="pixart",
        pipeline_cls="PixArtSigmaPipeline",
        default_steps=20,
        default_guidance=4.5,
        default_size=512,
        max_sequence_length=300,
        cfg_batched=True,
        approx_params_b=0.6,
        notes=(
            "28 single-stream blocks, d_model 1152, 16 heads. Image self-attention (attn1) is pure "
            "image-to-image, so sinks are not diluted by text keys; text enters through cross-attention (attn2). "
            "Real CFG duplicates the batch, so the analysis reads the conditional half."
        ),
        extra_call_kwargs={"use_resolution_binning": False},
    ),
    "pixart-sigma-1024": ModelSpec(
        key="pixart-sigma-1024",
        repo_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        family="pixart",
        pipeline_cls="PixArtSigmaPipeline",
        default_steps=20,
        default_guidance=4.5,
        default_size=1024,
        max_sequence_length=300,
        cfg_batched=True,
        approx_params_b=0.6,
        notes="Same topology as the 512 checkpoint; 4096 image tokens at 1024x1024.",
        extra_call_kwargs={"use_resolution_binning": False},
    ),
}


# ------------------------------------------------ synthetic smoke-test models
# Real diffusers block classes at toy width, random weights, no download.
# Used by the notebook's smoke test and by the unit tests: same code path,
# same figures, no GPU.
for _fam, _blocks in (("flux1", "2 dual + 3 single"), ("flux2", "2 dual + 3 single"), ("pixart", "5")):
    MODEL_REGISTRY[f"tiny-{_fam}"] = ModelSpec(
        key=f"tiny-{_fam}",
        repo_id="synthetic",
        family=_fam,
        pipeline_cls="synthetic",
        default_steps=2,
        default_guidance=0.0,
        default_size=128,
        max_sequence_length=None,
        cfg_batched=(_fam == "pixart"),
        notes=f"Synthetic {_fam} stand-in ({_blocks} blocks, random weights) for offline smoke tests.",
    )


_ALIASES: Dict[str, str] = {}
for _k, _spec in MODEL_REGISTRY.items():
    _ALIASES[_k.lower()] = _k
    _ALIASES[_spec.repo_id.lower()] = _k
    for _a in _spec.aliases:
        _ALIASES[_a.lower()] = _k


def resolve_model(name: str) -> ModelSpec:
    """Look a model up by key, alias, or full Hugging Face repo id."""
    key = _ALIASES.get(str(name).strip().lower())
    if key is None:
        raise KeyError(
            f"Unknown model {name!r}. Known: {sorted(MODEL_REGISTRY)} "
            f"(aliases: {sorted(a for a in _ALIASES if a not in MODEL_REGISTRY)})"
        )
    return MODEL_REGISTRY[key]


def registry_table() -> "List[Dict[str, object]]":
    """Rows for a human-readable summary of what can be run."""
    rows = []
    for spec in MODEL_REGISTRY.values():
        rows.append(
            dict(
                key=spec.key,
                family=spec.family,
                repo_id=spec.repo_id,
                params_B=spec.approx_params_b,
                steps=spec.default_steps,
                guidance=spec.default_guidance,
                cfg_batched=spec.cfg_batched,
                aliases=", ".join(spec.aliases),
            )
        )
    return rows
