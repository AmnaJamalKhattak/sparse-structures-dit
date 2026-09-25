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
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


class InterventionPoint(str, Enum):
    """Stable, architecture-aware names for causal intervention sites."""

    BLOCK_INPUT = "block_input"
    BLOCK_OUTPUT = "block_output"
    PRE_MLP_RESIDUAL = "pre_mlp_residual"
    MLP_HIDDEN = "mlp_hidden"
    ADALN_MODULATION = "adaln_modulation"
    PRE_KEY_NORM_RESIDUAL = "pre_key_norm_residual"
    KEY_PRE_POSITION = "key_pre_position"
    FINAL_KEY = "final_key"
    WRITER_RESIDUAL = "writer_residual"
    # The tensor the image self-attention module is CALLED with -- after the adaptive
    # norm, after its scale/shift, and after any block-level positional embedding.
    # Distinct from PRE_KEY_NORM_RESIDUAL, which is the norm MODULE's output and is the
    # same tensor only where the modulation lives inside that module. It does on FLUX
    # (AdaLayerNormZero applies scale and shift itself) and it does NOT on PixArt, whose
    # BasicTransformerBlock computes `norm1(x) * (1 + scale_msa) + shift_msa` in the
    # block body. Anything asking what the computation actually received must use this.
    ATTENTION_INPUT = "attention_input"


@dataclass(frozen=True)
class HookCapability:
    point: InterventionPoint
    supported: bool
    module_path: Optional[str] = None
    representation: str = ""
    reason: str = ""

    def require(self) -> "HookCapability":
        if not self.supported:
            raise NotImplementedError(f"{self.point.value} is unsupported: {self.reason}")
        return self


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

    def intervention_capabilities(self, ref: LayerRef) -> Dict[InterventionPoint, HookCapability]:
        """Describe only sites that exist as clean module boundaries.

        Missing/fused stages are deliberately reported as unsupported: callers must
        never substitute a nearby tensor and call it the requested representation.
        """
        b = ref.block
        caps = {
            p: HookCapability(p, False, reason="not exposed as a separable module by this architecture")
            for p in InterventionPoint
        }
        caps[InterventionPoint.BLOCK_INPUT] = HookCapability(InterventionPoint.BLOCK_INPUT, True, "", "block forward input")
        caps[InterventionPoint.BLOCK_OUTPUT] = HookCapability(InterventionPoint.BLOCK_OUTPUT, True, "", "block forward output")
        caps[InterventionPoint.FINAL_KEY] = HookCapability(InterventionPoint.FINAL_KEY, True, representation="key passed unchanged to the native attention kernel")
        # Attribute names follow diffusers; checking the instance makes this
        # checkpoint/version aware rather than merely family aware.
        for point, names, rep in (
            (InterventionPoint.MLP_HIDDEN, ("ff", "mlp"), "feed-forward module output (architecture-equivalent MLP activation)"),
            (InterventionPoint.PRE_KEY_NORM_RESIDUAL, ("norm1",),
             "output of the normalisation MODULE -- equal to what the QKV projection "
             "receives only where the adaptive scale/shift is applied inside it"),
            (InterventionPoint.ATTENTION_INPUT, ("attn", "attn1"),
             "the tensor the image self-attention is called with: what its QKV "
             "projection actually receives"),
        ):
            name = next((n for n in names if getattr(b, n, None) is not None), None)
            if name:
                caps[point] = HookCapability(point, True, name, rep)
        return caps

    def intervention_capability(self, ref: LayerRef, point: InterventionPoint) -> HookCapability:
        """Return (and allow callers to ``require``) one checkpoint-aware capability."""
        return self.intervention_capabilities(ref)[InterventionPoint(point)]

    def intervention_module(self, ref: LayerRef, point: InterventionPoint):
        """Resolve the exact module boundary advertised for ``point``."""
        cap = self.intervention_capability(ref, point).require()
        if point in (InterventionPoint.BLOCK_INPUT, InterventionPoint.BLOCK_OUTPUT):
            return ref.block
        obj = ref.block
        if not cap.module_path:
            raise NotImplementedError(
                f"{point.value} is observable only inside the attention dispatcher; "
                "use AttentionTap rather than a module hook"
            )
        for component in cap.module_path.split("."):
            obj = getattr(obj, component)
        return obj

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

    @staticmethod
    def _advertise(caps, block, point: InterventionPoint, attr: str, representation: str) -> None:
        """Advertise a point at ``attr``, but only when that module is really there.

        The block kind decides which *name* to look for; the instance decides
        whether it exists.  Reporting a point unsupported because we looked under
        the wrong name would be a claim about the architecture, not about this
        checkpoint, so a genuine absence says which module was missing.
        """
        if getattr(block, attr, None) is not None:
            caps[point] = HookCapability(point, True, attr, representation)
        else:
            caps[point] = HookCapability(
                point, False,
                reason=f"this block exposes no {attr!r} module, which is where "
                       f"{point.value} lives in this architecture")

    def intervention_capabilities(self, ref: LayerRef) -> Dict[InterventionPoint, HookCapability]:
        """Name each site by block kind, because FLUX.1 has two block layouts.

        The base class finds these by attribute name, which is checkpoint-aware
        but assumes one layout.  A FLUX.1 *single* block computes the same
        quantities under different names -- its adaptive norm is ``norm`` rather
        than ``norm1``, and its feed-forward is ``proj_mlp`` into ``act_mlp``
        rather than ``ff`` -- so a name-only probe misses them and reports the
        architecture as incapable of something it does perfectly well.  Only
        ``pre_mlp_residual`` is genuinely absent there, and it says why.
        """
        caps = super().intervention_capabilities(ref)
        block = ref.block
        if ref.kind == "dual":
            self._advertise(caps, block, InterventionPoint.MLP_HIDDEN, "ff",
                            "feed-forward hidden activation, inside the image feed-forward stack")
            self._advertise(caps, block, InterventionPoint.ADALN_MODULATION, "norm1",
                            "adaptive-norm output: the normalised stream and five timestep gates")
            self._advertise(caps, block, InterventionPoint.PRE_KEY_NORM_RESIDUAL, "norm1",
                            "normalised residual supplied to the QKV projection")
            self._advertise(caps, block, InterventionPoint.PRE_MLP_RESIDUAL, "norm2",
                            "norm2 input: post-attention image residual before feed-forward")
            self._advertise(caps, block, InterventionPoint.WRITER_RESIDUAL, "attn",
                            "projected attention contribution, BEFORE the gate_msa scaling and before the "
                            "residual addition")
        else:
            self._advertise(caps, block, InterventionPoint.MLP_HIDDEN, "act_mlp",
                            "feed-forward hidden activation, after proj_mlp and its nonlinearity; "
                            "text tokens lead the sequence and the width is the expanded one")
            self._advertise(caps, block, InterventionPoint.ADALN_MODULATION, "norm",
                            "adaptive-norm output: the normalised stream and one shared gate, "
                            "a single block having no separate feed-forward gate")
            self._advertise(caps, block, InterventionPoint.PRE_KEY_NORM_RESIDUAL, "norm",
                            "normalised stream supplied to the QKV projection, shared with the "
                            "feed-forward branch; text tokens lead the sequence")
            self._advertise(caps, block, InterventionPoint.WRITER_RESIDUAL, "proj_out",
                            "fused attention and feed-forward writer contribution, BEFORE the gate scaling "
                            "and before the residual addition; text tokens lead the sequence")
            # The one that really is absent: both branches read the same normalised
            # stream and are concatenated before a single proj_out, so no tensor in
            # the block is "after attention and before the feed-forward".
            caps[InterventionPoint.PRE_MLP_RESIDUAL] = HookCapability(
                InterventionPoint.PRE_MLP_RESIDUAL, False,
                reason="FLUX.1 single blocks feed attention and the feed-forward from one "
                       "normalised stream and concatenate them before a single proj_out, so "
                       "there is no post-attention stage before the feed-forward to read")
        # RoPE is applied by a function call inside the attention processor, between
        # the key norm and the attention dispatch, so no module's input or output
        # ever holds the key before position is applied.
        caps[InterventionPoint.KEY_PRE_POSITION] = HookCapability(
            InterventionPoint.KEY_PRE_POSITION, False,
            reason="RoPE is applied inside the attention processor rather than at a module "
                   "boundary, so the key before positional encoding is never a module's "
                   "input or output")
        return caps

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

    def intervention_capabilities(self, ref: LayerRef) -> Dict[InterventionPoint, HookCapability]:
        caps = super().intervention_capabilities(ref)
        caps[InterventionPoint.ADALN_MODULATION] = HookCapability(
            InterventionPoint.ADALN_MODULATION, False,
            reason="PixArt computes scale/shift/gates inline from timestep; no separable module output")
        caps[InterventionPoint.PRE_MLP_RESIDUAL] = HookCapability(InterventionPoint.PRE_MLP_RESIDUAL, True, "norm2", "residual after self/cross attention, before feed-forward")
        caps[InterventionPoint.WRITER_RESIDUAL] = HookCapability(
            InterventionPoint.WRITER_RESIDUAL, True, "ff",
            "feed-forward contribution, BEFORE the gate_mlp scaling and before the "
            "residual addition")
        caps[InterventionPoint.KEY_PRE_POSITION] = HookCapability(InterventionPoint.KEY_PRE_POSITION, True, "attn1.to_k", "self-attention key projection; PixArt has no image-key RoPE")
        # PixArt's norm1 is a PLAIN LayerNorm and the adaptive modulation is applied in
        # the block body, not inside it:
        #     norm_hidden_states = self.norm1(hidden_states)
        #     norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        #     attn_output = self.attn1(norm_hidden_states, ...)
        # so norm1's output is NOT the tensor the QKV projection receives -- a per-channel
        # scale stands between them, and a per-channel scale is not a rotation, so it
        # changes each token's alignment with a direction differently. Anything measuring
        # what the computation received must use ATTENTION_INPUT. FLUX does not have this
        # gap: AdaLayerNormZero applies its scale and shift itself.
        caps[InterventionPoint.PRE_KEY_NORM_RESIDUAL] = HookCapability(
            InterventionPoint.PRE_KEY_NORM_RESIDUAL, True, "norm1",
            "LayerNorm output BEFORE the adaptive scale and shift, which PixArt applies "
            "in the block body; this is NOT what the QKV projection receives")
        caps[InterventionPoint.ATTENTION_INPUT] = HookCapability(
            InterventionPoint.ATTENTION_INPUT, True, "attn1",
            "the tensor attn1 is called with: modulated, and what its QKV projection "
            "actually receives")
        return caps

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
