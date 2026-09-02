"""Per-family adapters.

Everything the sweep needs to know about an architecture lives here:
which modules are "layers", where the image tokens sit inside a sequence,
how to load and call the pipeline. The capture code itself is family-agnostic.

Layout facts this file encodes (diffusers >= 0.36):

FLUX.1  19 dual + 38 single blocks. Dual blocks carry image tokens in
        `hidden_states` and text in `encoder_hidden_states`; attention runs on
        the concatenated [text, image] sequence. Single blocks either receive
        the concatenated stream (<=0.35) or the split pair (>=0.36) -- the
        sequence-length rule below covers both.
FLUX.2  8 dual + 48 single blocks. Same [text, image] ordering. Single blocks
        are *parallel* blocks: attention and MLP share one input projection and
        one output projection, so there is no separate post-attention residual.
FLUX.2 text length is prompt-dependent (Mistral text encoder), not a fixed 512.
PixArt  28 blocks, each with attn1 (image self-attention -- where sinks live)
        and attn2 (cross-attention into text). The image sequence never carries
        text tokens, so image->image attention is not diluted by a text sink.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class LayerRef:
    index: int                 # global layer index, 0-based, dual blocks first
    kind: str                  # "dual" | "single" | "block"
    local_id: int              # index within its own ModuleList
    block: Any                 # nn.Module
    attns: Dict[str, Any] = field(default_factory=dict)   # role -> attention module

    @property
    def name(self) -> str:
        return f"{self.kind}{self.local_id}"


class ModelAdapter:
    family = "base"
    # Human-readable meaning of each captured residual-stream stage.
    stage_notes: Dict[str, str] = {}
    # Which attention role carries image->image attention.
    image_attn_role = "attn"

    # ---------------------------------------------------------------- loading
    def load_pipeline(self, spec, cfg):
        import diffusers

        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
        cls = getattr(diffusers, spec.pipeline_cls)
        kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if cfg.quantize_4bit:
            from diffusers.quantizers import PipelineQuantizationConfig

            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": dtype,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )
        pipe = cls.from_pretrained(spec.repo_id, **kwargs)
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.quantize_4bit:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def transformer(self, pipe):
        return pipe.transformer

    # ---------------------------------------------------------------- layers
    def layers(self, transformer) -> List[LayerRef]:
        raise NotImplementedError

    def call_kwargs(self, cfg, spec) -> Dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------- token geometry
    def num_image_tokens(self, transformer_kwargs, cfg) -> Optional[int]:
        """Image-token count, read off the tensors the transformer was called with."""
        hs = transformer_kwargs.get("hidden_states")
        if hs is None:
            return None
        if hs.ndim == 3:
            return int(hs.shape[1])
        return None

    def num_text_tokens(self, transformer_kwargs, cfg) -> int:
        ehs = transformer_kwargs.get("encoder_hidden_states")
        if ehs is not None and hasattr(ehs, "ndim") and ehs.ndim == 3:
            return int(ehs.shape[1])
        return 0

    # ------------------------------------------------------------ generation
    def generate(self, pipe, prompt, seed, cfg, spec):
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        kwargs = self.call_kwargs(cfg, spec)
        out = pipe(prompt=prompt, generator=gen, output_type="pil", **kwargs)
        return out.images[0]


# --------------------------------------------------------------------- FLUX.1
class Flux1Adapter(ModelAdapter):
    family = "flux1"
    stage_notes = {
        "pre_block": "image residual stream entering the block",
        "attn_out": "dual: projected attention update added to the image residual; "
                    "single: attention branch before proj_out mixes it with the MLP branch",
        "post_attn_residual": "dual blocks only: residual after the attention sub-layer, before the MLP",
        "post_block": "image residual stream leaving the block",
    }

    def layers(self, transformer) -> List[LayerRef]:
        refs: List[LayerRef] = []
        for i, blk in enumerate(transformer.transformer_blocks):
            refs.append(LayerRef(index=len(refs), kind="dual", local_id=i, block=blk, attns={"attn": blk.attn}))
        for i, blk in enumerate(transformer.single_transformer_blocks):
            refs.append(LayerRef(index=len(refs), kind="single", local_id=i, block=blk, attns={"attn": blk.attn}))
        return refs

    def call_kwargs(self, cfg, spec) -> Dict[str, Any]:
        kw = dict(
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
        )
        if spec.max_sequence_length:
            kw["max_sequence_length"] = spec.max_sequence_length
        kw.update(spec.extra_call_kwargs)
        return kw


# --------------------------------------------------------------------- FLUX.2
class Flux2Adapter(Flux1Adapter):
    family = "flux2"
    stage_notes = {
        "pre_block": "image residual stream entering the block",
        "attn_out": "dual: projected attention update; single: the FUSED attention+MLP update "
                    "(FLUX.2 single blocks are parallel blocks, so the two paths share one output projection)",
        "post_attn_residual": "dual blocks only",
        "post_block": "image residual stream leaving the block",
    }


# ---------------------------------------------------------------- PixArt-Sigma
class PixArtAdapter(ModelAdapter):
    family = "pixart"
    stage_notes = {
        "pre_block": "image token stream entering the block",
        "attn_out": "projected output of attn1 (image self-attention)",
        "post_attn_residual": "residual after self- AND cross-attention, before the MLP "
                              "(PixArt applies norm2 at the feed-forward stage)",
        "post_block": "image token stream leaving the block",
    }

    def layers(self, transformer) -> List[LayerRef]:
        refs: List[LayerRef] = []
        for i, blk in enumerate(transformer.transformer_blocks):
            attns = {"attn": blk.attn1}
            if getattr(blk, "attn2", None) is not None:
                attns["cross"] = blk.attn2
            refs.append(LayerRef(index=i, kind="block", local_id=i, block=blk, attns=attns))
        return refs

    def num_image_tokens(self, transformer_kwargs, cfg) -> Optional[int]:
        # PixArt's transformer takes unpatchified latents [B, C, H, W].
        hs = transformer_kwargs.get("hidden_states")
        if hs is None:
            return None
        if hs.ndim == 4:
            return None  # resolved from the first block's token count instead
        if hs.ndim == 3:
            return int(hs.shape[1])
        return None

    def num_text_tokens(self, transformer_kwargs, cfg) -> int:
        return 0  # text never enters the image stream; it arrives via cross-attention

    def call_kwargs(self, cfg, spec) -> Dict[str, Any]:
        kw = dict(
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
        )
        if spec.max_sequence_length:
            kw["max_sequence_length"] = spec.max_sequence_length
        kw.update(spec.extra_call_kwargs)
        return kw


_ADAPTERS = {"flux1": Flux1Adapter, "flux2": Flux2Adapter, "pixart": PixArtAdapter}


def get_adapter(family: str) -> ModelAdapter:
    try:
        return _ADAPTERS[family]()
    except KeyError:
        raise KeyError(f"No adapter for family {family!r}; known: {sorted(_ADAPTERS)}") from None


# --------------------------------------------------------------- shared utils
def image_slice(seq_len: int, n_img: int) -> Optional[slice]:
    """Where the image tokens sit in a sequence of length `seq_len`.

    The invariant across FLUX.1, FLUX.2 and PixArt is that image tokens are the
    *tail* of whatever sequence they share (text is prepended, never appended).
    Returns None when the sequence carries no image tokens at all (e.g. the key
    axis of PixArt cross-attention).
    """
    if n_img <= 0:
        return None
    if seq_len == n_img:
        return slice(0, seq_len)
    if seq_len > n_img:
        return slice(seq_len - n_img, seq_len)
    return None


def extract_image_tokens(obj, n_img: int) -> Optional[torch.Tensor]:
    """Pull the [B, n_img, C] image stream out of a block's input or output."""
    candidates: List[torch.Tensor] = []
    if torch.is_tensor(obj):
        candidates = [obj]
    elif isinstance(obj, (tuple, list)):
        candidates = [t for t in obj if torch.is_tensor(t) and t.ndim == 3]
    for t in candidates:
        if t.ndim == 3 and t.shape[1] == n_img:
            return t
    for t in candidates:
        sl = image_slice(int(t.shape[1]), n_img) if t.ndim == 3 else None
        if sl is not None:
            return t[:, sl, :]
    return None
