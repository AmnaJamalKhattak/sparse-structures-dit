"""Execution engine for the causal questions.

The observational sweep (``ditsinks.capture``) answers *where* the register
phenomena live.  This module answers *what happens when we change them*, and it
is built around three rules the causal experiments require.

**Targets are frozen.**  Every intervention edits token identities chosen from a
clean run of the *same* prompt and seed.  Nothing is ever re-selected from a
treated activation, so a primary endpoint can never be inflated by re-defining
"the register" after treatment.

**Clean and treated trajectories are paired.**  One generation unit is one
``(prompt, seed)``: the clean pass runs first, its targets and endpoints are
stored, and every condition then re-runs the identical generation with one hook
installed.  Differences are therefore within-unit.

**A hook site is either advertised or refused.**  Intervention points resolve
through :class:`ditsinks.adapters.HookCapability`; an architecture that fuses a
stage away reports it unsupported rather than letting a caller silently patch a
neighbouring tensor and label it something it is not.

The engine records, per observed layer, everything the causal experiments need: the
projection of every image token on the frozen direction, token norms, the values
of the frozen channels, the per-head incoming image-to-image attention, and the
cosine between each image key and the mean image query.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .adapters import (InterventionPoint, LayerRef, ModelAdapter, extract_image_tokens,
                       get_adapter, image_slice)
from .attention_patch import AttentionTap, to_bhsd
from .capture import _to_float


# --------------------------------------------------------------- hook sites
# Whether a point is read/edited at a module's input or its output.  The choice
# is part of the point's definition: ``pre_mlp_residual`` is norm2's *input*
# (the post-attention residual), while ``pre_key_norm_residual`` is norm1's
# *output* (the normalised stream that the QKV projection actually consumes).
_POINT_SIDE: Dict[InterventionPoint, str] = {
    InterventionPoint.BLOCK_INPUT: "pre",
    InterventionPoint.BLOCK_OUTPUT: "post",
    InterventionPoint.PRE_MLP_RESIDUAL: "pre",
    InterventionPoint.PRE_KEY_NORM_RESIDUAL: "post",
    InterventionPoint.ADALN_MODULATION: "post",
    InterventionPoint.MLP_HIDDEN: "post",
    InterventionPoint.WRITER_RESIDUAL: "post",
    InterventionPoint.KEY_PRE_POSITION: "post",
    # A module's INPUT, because the point is defined as what the attention was handed.
    InterventionPoint.ATTENTION_INPUT: "pre",
}


def _resolve_module(root, path: str):
    """Follow a dotted attribute path, indexing numeric components."""
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def hook_site(adapter: ModelAdapter, ref: LayerRef, point: InterventionPoint) -> Tuple[Any, str]:
    """Return ``(module, side)`` for an advertised point, or refuse."""
    point = InterventionPoint(point)
    if point is InterventionPoint.FINAL_KEY:
        raise NotImplementedError(
            "final_key is patched through AttentionTap(mode='patch'), not a module hook")
    cap = adapter.intervention_capability(ref, point).require()
    if point in (InterventionPoint.BLOCK_INPUT, InterventionPoint.BLOCK_OUTPUT):
        return ref.block, _POINT_SIDE[point]
    module = _resolve_module(ref.block, cap.module_path)
    # The advertised feed-forward module is the whole FF stack; the hidden
    # activation lives one level in, at its first (projection+nonlinearity) stage.
    if point is InterventionPoint.MLP_HIDDEN and getattr(module, "net", None) is not None:
        try:
            module = module.net[0]
        except (IndexError, TypeError):
            pass
    return module, _POINT_SIDE[point]


def image_span(tensor: torch.Tensor, n_img: Optional[int], n_txt: int = 0) -> Optional[slice]:
    """Where the image tokens sit, or ``None`` when the answer is not certain.

    A sequence qualifies only when its length is exactly the image-token count or
    exactly image+text.  Anything else (a text-only stream, a pooled projection,
    a hidden width that merely happens to be long enough) is refused, because
    editing the tail of the wrong tensor would look like a result.
    """
    if not torch.is_tensor(tensor) or tensor.ndim != 3 or not n_img:
        return None
    length = int(tensor.shape[1])
    if length == int(n_img):
        return slice(0, length)
    if n_txt and length == int(n_img) + int(n_txt):
        return slice(length - int(n_img), length)
    return None


# ------------------------------------------------------------ observations
@dataclass
class LayerObservation:
    """Everything one observed layer contributes at one denoising step."""

    layer: int
    step: int
    projection: Optional[torch.Tensor] = None        # [N]  x . v*
    cosine: Optional[torch.Tensor] = None            # [N]  x . v* / ||x||
    norm: Optional[torch.Tensor] = None              # [N]
    channel_values: Optional[torch.Tensor] = None    # [len(channels), N]
    top_channel: Optional[torch.Tensor] = None       # [N] argmax_c |x_c|
    top_channel_value: Optional[torch.Tensor] = None # [N]
    incoming: Optional[torch.Tensor] = None          # [H, N] renormalised image->image
    qk_cosine: Optional[torch.Tensor] = None         # [H, N] key vs mean image query
    entropy: Optional[torch.Tensor] = None           # [H] nats
    keys: Optional[torch.Tensor] = None              # [H, N, D] final image keys, where asked
    queries: Optional[torch.Tensor] = None           # [H, N, D] final image queries, where asked
    attention_probs: Optional[torch.Tensor] = None   # [H, N, S] probabilities from captured Q/K
    attention_logits: Optional[torch.Tensor] = None  # [H, N, S] logits from captured Q/K
    values: Optional[torch.Tensor] = None            # [H, N, D] final image values, where asked
    # Set only when a virtual register is appended: attention that left the image for it.
    virtual_mass: Optional[float] = None
    virtual_mass_per_head: Optional[torch.Tensor] = None     # [H]
    virtual_mass_per_register: Optional[torch.Tensor] = None # [R]
    n_virtual: Optional[int] = None
    # `incoming` is renormalised over the image tokens, so it is exactly invariant to a
    # virtual register drawing mass proportionally, and the mass it divides out is the
    # very thing such a register takes. These two carry the missing scale: the share of
    # the full softmax the image tokens hold, per head, before that renormalisation, and
    # the full key length. With them a sink can be scored against uniform over the whole
    # sequence instead, which does register the loss.
    image_mass: Optional[torch.Tensor] = None        # [H] pre-renormalisation image share
    n_keys: Optional[int] = None                     # full key length, registers included
    states: Optional[torch.Tensor] = None            # [N, C] kept only where asked
    token_states: Optional[torch.Tensor] = None      # [len(keep_tokens), C]

    @property
    def median_norm(self) -> float:
        return float(self.norm.median()) if self.norm is not None else float("nan")


@dataclass
class Trace:
    """One generation's observations, indexed by ``(step, layer)``."""

    prompt_id: int
    seed: int
    condition: str = "clean"
    n_img: int = 0
    n_txt: int = 0
    n_heads: int = 0
    d_model: int = 0
    channels: Tuple[int, ...] = ()
    keep_tokens: Tuple[int, ...] = ()
    grid: Tuple[int, int] = (0, 0)
    rows: Dict[Tuple[int, int], LayerObservation] = field(default_factory=dict)
    # States read at an intervention point rather than at the block output, so a
    # frozen replacement is drawn from exactly the tensor a hook will overwrite.
    probe_states: Dict[Tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    # Lossless manipulation checks. Keys are (step, layer, point, stage, call).
    diagnostics: Dict[Tuple[int, int, str, str, int], torch.Tensor] = field(default_factory=dict)
    image: Any = None

    def layers(self, step: Optional[int] = None) -> List[int]:
        return sorted({l for (s, l) in self.rows if step is None or s == step})

    def steps(self) -> List[int]:
        return sorted({s for (s, _) in self.rows})

    def at(self, step: int, layer: int) -> Optional[LayerObservation]:
        return self.rows.get((int(step), int(layer)))

    def probe(self, step: int, layer: int,
              point: "InterventionPoint" = None) -> Optional[torch.Tensor]:
        """Image-token states read at an intervention point of one layer."""
        point = InterventionPoint.BLOCK_INPUT if point is None else InterventionPoint(point)
        return self.probe_states.get((int(step), int(layer), point.value))

    def position(self, token: Optional[int]) -> Optional[Tuple[int, int]]:
        """Grid coordinates of an image token, when the grid is known."""
        if token is None or self.grid[1] <= 0:
            return None
        return (int(token) // self.grid[1], int(token) % self.grid[1])


# --------------------------------------------------------------- the tracer
class CausalTracer:
    """Record the causal-experiment observables at a fixed set of layers and steps."""

    def __init__(self, adapter: ModelAdapter, transformer, *, direction: torch.Tensor,
                 layers: Sequence[int], steps: Sequence[int],
                 channels: Sequence[int] = (), keep_tokens: Sequence[int] = (),
                 full_state_layers: Sequence[int] = (), key_layers: Sequence[int] = (),
                 n_img_hint: Optional[int] = None,
                 grid: Tuple[int, int] = (0, 0), cfg=None, batch_row: int = -1):
        self.adapter = adapter
        self.transformer = transformer
        # -1: the last batch row (the conditional one under CFG); 0: the first.
        self.batch_row = int(batch_row)
        v = torch.as_tensor(direction).float().flatten()
        self.v = v / v.norm().clamp_min(1e-9)
        self.layers = sorted({int(l) for l in layers})
        self.steps = sorted({int(s) for s in steps})
        self.channels = tuple(int(c) for c in channels)
        self.keep_tokens = tuple(int(t) for t in keep_tokens)
        self.full_state_layers = {int(l) for l in full_state_layers}
        # Keeping the exact keys the kernel saw is what makes a final-key
        # transplant possible: everything after this point is the kernel itself.
        self.key_layers = {int(l) for l in key_layers}
        # Virtual-register columns appended to the keys and values when a run injects
        # one. The tracer has to know the count: the image tokens are the TAIL of the
        # real sequence, so with R extra columns on the end `image_slice` would either
        # refuse or silently slice the registers instead of the image. Every readout
        # that follows (sinkhood, incoming mass, the qk cosine) depends on that slice.
        self.n_virtual = 0
        # The post-append key length the injector produced for the call in flight. Two
        # attention taps nest, and whichever patches LAST is outermost, so a tap that
        # enters after this one appends rows that this one's callback never sees, while
        # `n_virtual` is already set. That combination silently slices the wrong columns.
        # Recording the length the injector actually produced turns it into a refusal.
        self.virtual_kv_len = 0
        self.cfg = cfg
        self.refs: List[LayerRef] = adapter.layers(transformer)
        self._wanted = set(self.layers)
        self._by_attn: Dict[int, LayerRef] = {}
        for ref in self.refs:
            mod = ref.attns.get(adapter.image_attn_role)
            if mod is not None:
                self._by_attn[id(mod)] = ref
        self.trace = Trace(prompt_id=0, seed=0, channels=self.channels,
                           keep_tokens=self.keep_tokens, grid=grid)
        self.n_img: Optional[int] = n_img_hint
        self.n_txt: int = 0
        self._n_img_hint = n_img_hint
        self.step = -1
        self.pass_idx = 0
        self._last_timestep = None
        self._active: Optional[LayerRef] = None
        self._heads: Optional[int] = None
        # Set by a caller that patches final keys; a single tap then both
        # replaces the key and observes the geometry the kernel really saw.
        self.key_transform: Optional[Callable] = None
        # Substituted alongside the key, in the SAME tap, so that two runs sharing a key
        # patch provably share their logits and differ only in the value delivered.
        self.value_transform: Optional[Callable] = None
        self._handles: List[Any] = []
        self._tap: Optional[AttentionTap] = None
        # The steps at which edit installers also keep their FULL-TENSOR manipulation
        # checks (x_pre_hook / x_post_hook / x_next_module_input). None keeps them at
        # every recorded step by default. A caller recording many
        # steps must narrow this: each check is a whole [N, C] slice (about 50 MB at 1024px
        # on FLUX) per hook call, and at every step of every block that is a hundred
        # gigabytes of tensors nobody reads.
        self.diagnostic_steps: Optional[set] = None
        # The steps at which the attention readout (incoming mass, sinkhood, the qk
        # cosine) is also recorded. None records it at every recorded step, the default.
        # The residual-state readout is cheap and runs at every recorded step
        # regardless; the attention readout materialises a [heads, N, N] probability
        # tensor per block, which at 1024px dominates the cost of a traced generation, so
        # a caller that needs states at many steps and attention at a few narrows this.
        self.attention_steps: Optional[set] = None

    def wants_diagnostics(self) -> bool:
        """Whether the current step keeps full-tensor manipulation checks."""
        return self.recording and (self.diagnostic_steps is None
                                   or int(self.step) in self.diagnostic_steps)

    # ------------------------------------------------------------ lifecycle
    def begin_generation(self, prompt_id: int, seed: int, condition: str = "clean") -> None:
        self.trace = Trace(prompt_id=int(prompt_id), seed=int(seed), condition=condition,
                           channels=self.channels, keep_tokens=self.keep_tokens,
                           grid=self.trace.grid)
        self.step = -1
        self.pass_idx = 0
        self._last_timestep = None
        self.n_img = self._n_img_hint
        self.n_txt = 0
        self.n_virtual = 0
        self.virtual_kv_len = 0

    def __enter__(self) -> "CausalTracer":
        tr = self.transformer
        self._handles.append(tr.register_forward_pre_hook(self._transformer_pre, with_kwargs=True))
        for ref in self.refs:
            if ref.index not in self._wanted:
                continue
            self._handles.append(ref.block.register_forward_hook(self._make_block_post(ref),
                                                                 with_kwargs=True))
            mod = ref.attns.get(self.adapter.image_attn_role)
            if mod is not None:
                self._handles.append(mod.register_forward_pre_hook(self._make_attn_pre(ref),
                                                                   with_kwargs=True))
        if self.key_transform is None and self.value_transform is None:
            self._tap = AttentionTap(self.adapter.family, self._on_attention)
        else:
            self._tap = AttentionTap(self.adapter.family, self._on_attention, mode="patch",
                                     key_transform=self.key_transform,
                                     value_transform=self.value_transform)
        self._tap.__enter__()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        if self._tap is not None:
            self._tap.__exit__(None, None, None)
            self._tap = None
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    # ------------------------------------------------------------- plumbing
    def _transformer_pre(self, module, args, kwargs):
        value = _to_float(kwargs.get("timestep", None))
        if value is None or value != self._last_timestep:
            self.step += 1
            self.pass_idx = 0
            self._last_timestep = value
        else:
            self.pass_idx += 1
        merged = dict(kwargs)
        if args and "hidden_states" not in merged:
            merged["hidden_states"] = args[0]
        n = self.adapter.num_image_tokens(merged, self.cfg)
        if n is not None:
            self.n_img = int(n)
        self.n_txt = int(self.adapter.num_text_tokens(merged, self.cfg) or 0)
        return None

    @property
    def recording(self) -> bool:
        return self.step in self.steps and self.pass_idx == 0

    def _row(self, x: torch.Tensor) -> int:
        """The batch row this tracer records. Classifier-free guidance duplicates the
        batch with the conditional row last, which is the default (``batch_row = -1``);
        ``batch_row = 0`` records the unconditional row instead."""
        wanted = int(getattr(self, "batch_row", -1))
        n = int(x.shape[0])
        return n - 1 if wanted < 0 else min(wanted, n - 1)

    def _observation(self, layer: int) -> LayerObservation:
        key = (self.step, int(layer))
        row = self.trace.rows.get(key)
        if row is None:
            row = LayerObservation(layer=int(layer), step=self.step)
            self.trace.rows[key] = row
        return row

    def _make_attn_pre(self, ref: LayerRef):
        def hook(module, args, kwargs):
            self._active = ref
            heads = getattr(module, "heads", None)
            if isinstance(heads, int) and heads > 0:
                self._heads = int(heads)
            return None
        return hook

    def _make_block_post(self, ref: LayerRef):
        @torch.no_grad()
        def hook(module, args, kwargs, out):
            if not self.recording:
                return None
            if self.n_img is None:
                first = out[0] if isinstance(out, (tuple, list)) and out else out
                if torch.is_tensor(first) and first.ndim == 3:
                    self.n_img = int(first.shape[1])
            img = extract_image_tokens(out, self.n_img or 0)
            if img is None:
                return None
            h = img[self._row(img)].float()
            self._store_state(ref.index, h)
            return None
        return hook

    def _store_state(self, layer: int, h: torch.Tensor) -> None:
        row = self._observation(layer)
        v = self.v.to(h.device)
        norm = h.norm(dim=-1)
        projection = h @ v
        row.projection = projection.cpu()
        row.norm = norm.cpu()
        row.cosine = (projection / norm.clamp_min(1e-9)).cpu()
        if self.channels:
            idx = torch.as_tensor(self.channels, device=h.device, dtype=torch.long)
            idx = idx.clamp_max(h.shape[-1] - 1)
            row.channel_values = h.index_select(-1, idx).t().contiguous().cpu()
        top_value, top_channel = h.abs().max(dim=-1)
        row.top_channel = top_channel.cpu()
        row.top_channel_value = top_value.cpu()
        if self.keep_tokens:
            ids = torch.as_tensor(self.keep_tokens, device=h.device, dtype=torch.long)
            ids = ids[ids < h.shape[0]]
            if ids.numel():
                row.token_states = h.index_select(0, ids).cpu()
        if layer in self.full_state_layers:
            row.states = h.cpu()
        self.trace.n_img = int(h.shape[0])
        self.trace.n_txt = int(self.n_txt)
        self.trace.d_model = int(h.shape[-1])

    @torch.no_grad()
    def _on_attention(self, query, key, value, kwargs=None):
        ref, self._active = self._active, None
        if ref is None or not self.recording or ref.index not in self._wanted:
            return
        wanted_steps = getattr(self, "attention_steps", None)
        if wanted_steps is not None and int(self.step) not in wanted_steps:
            return
        n_img = int(self.n_img or 0)
        if n_img <= 0:
            return
        heads = self._heads or self.trace.n_heads or _head_count(query, key)
        q = to_bhsd(query, heads)
        k = to_bhsd(key, heads)
        row_idx = self._row(q)
        q = q[row_idx].float()
        k = k[row_idx].float()
        H, S_q, D = q.shape
        S_k = int(k.shape[1])
        virtual = int(self.n_virtual or 0)
        if virtual:
            expected = int(self.virtual_kv_len or 0)
            if expected and S_k != expected:
                raise RuntimeError(
                    f"the tracer sees a key of length {S_k} but the virtual-register "
                    f"injector produced {expected} ({virtual} appended column(s)). The "
                    "injector's attention tap is nested INSIDE the tracer's, so the "
                    "tracer's callback receives the key from before the append and every "
                    "readout that follows would be shifted by the register count. Enter "
                    "the tracer FIRST and the injector second -- see "
                    "run_traced_generation(extra_contexts=...)."
                )
        # The appended columns sit after the image tokens, so the image slice is taken
        # against the REAL sequence length and the registers are addressed separately.
        real_k = S_k - virtual
        if virtual and real_k <= 0:
            return
        q_slice = image_slice(S_q, n_img)
        k_slice = image_slice(real_k, n_img)
        if q_slice is None or k_slice is None:
            return
        scale = (kwargs or {}).get("scale", None) or 1.0 / math.sqrt(D)
        qi = q[:, q_slice, :]
        scores = torch.matmul(qi, k.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs.clamp_min(1e-12).log() * probs).sum(-1).mean(-1)
        incoming = probs.mean(dim=1)[:, k_slice]
        obs = self._observation(ref.index)
        image_mass = incoming.sum(-1)
        obs.incoming = (incoming / image_mass.unsqueeze(-1).clamp_min(1e-12)).cpu()
        # Kept so a later readout can undo the renormalisation. Without it, "10x uniform"
        # means a different thing in a condition that moved mass off the image entirely.
        obs.image_mass = image_mass.cpu()
        obs.n_keys = S_k
        obs.entropy = entropy.cpu()
        if virtual:
            # Mass that left the image and text for the appended slots. Read off the SAME
            # softmax the kernel used, so it is a share of the real distribution rather
            # than of a renormalised one: the register competes with every real token.
            register_probs = probs[:, :, real_k:]
            obs.virtual_mass = float(register_probs.sum(-1).mean())
            obs.virtual_mass_per_head = register_probs.sum(-1).mean(-1).cpu()
            obs.virtual_mass_per_register = register_probs.mean(dim=(0, 1)).cpu()
            obs.n_virtual = virtual
        q_mean = torch.nn.functional.normalize(qi.mean(dim=1), dim=-1)
        keys = torch.nn.functional.normalize(k[:, k_slice, :], dim=-1)
        obs.qk_cosine = torch.einsum("hd,hnd->hn", q_mean, keys).cpu()
        if ref.index in self.key_layers:
            obs.keys = k[:, k_slice, :].detach().cpu().clone()
            obs.queries = qi.detach().cpu().clone()
            # Values, which nothing recorded before. A virtual register's V template is
            # the mean value of the clean register population, so it cannot be built
            # without them. They are never rotated, so unlike the keys they need no
            # pre-RoPE capture.
            v = to_bhsd(value, heads)[row_idx].float()
            if int(v.shape[1]) == S_k:
                obs.values = v[:, k_slice, :].detach().cpu().clone()
            obs.attention_probs = probs.detach().cpu().clone()
            obs.attention_logits = scores.detach().cpu().clone()
        self.trace.n_heads = int(H)


def _head_count(query, key) -> int:
    """Last-resort head-axis guess when the module does not advertise a count.

    Only reached for a processor that exposes no ``heads`` attribute; attention
    tensors always have far more sequence positions than heads, so the smaller
    of the two middle axes is the head axis.
    """
    return int(min(query.shape[1], query.shape[2]))


# ----------------------------------------------------------------- editing
@dataclass
class EditContext:
    """What a pure token edit is allowed to know."""

    layer: int
    step: int
    point: InterventionPoint
    targets: "FrozenTargets"
    trace: Optional[Trace] = None
    # Which batch row the edit is handed. Under classifier-free guidance the conditional
    # row is last and is the one every record describes; a plan with ``all_batch_rows``
    # also edits the others, which see ``"unconditional"`` here so that an edit keeps its
    # bookkeeping (edit records, suppression sites) to the conditional row.
    branch: str = "conditional"


TokenEdit = Callable[[torch.Tensor, EditContext], torch.Tensor]


@dataclass
class EditPlan:
    """One declarative intervention: an edit, a point, and the layers it runs at."""

    edit: TokenEdit
    point: InterventionPoint = InterventionPoint.BLOCK_INPUT
    layers: Sequence[int] = ()
    steps: Optional[Sequence[int]] = None      # None -> every captured step
    label: str = ""
    # A causal intervention edits the conditional branch, because that is the branch
    # whose trajectory the paired comparison is about. A *deployment policy* (an
    # activation precision allocation, say) is not a branch-specific edit: on a
    # CFG-batched model the unconditional branch runs through the same kernels, so a
    # policy that touched only the conditional row would be half-applied and its image
    # fidelity overstated. Such plans opt in here.
    all_batch_rows: bool = False

    def active(self, step: int) -> bool:
        return self.steps is None or int(step) in {int(s) for s in self.steps}


class EditInstaller:
    """Install a plan's hooks and count the energy it moves."""

    def __init__(self, adapter: ModelAdapter, transformer, plan: EditPlan,
                 targets: "FrozenTargets", tracer: CausalTracer):
        self.adapter = adapter
        self.transformer = transformer
        self.plan = plan
        self.targets = targets
        self.tracer = tracer
        self.refs = {r.index: r for r in adapter.layers(transformer)}
        self._handles: List[Any] = []
        self.calls = 0
        self.removed_energy = 0.0
        self.injected_energy = 0.0
        self.perturbation_energy = 0.0

    def __enter__(self) -> "EditInstaller":
        for layer in self.plan.layers:
            ref = self.refs[int(layer)]
            module, side = hook_site(self.adapter, ref, self.plan.point)
            hook = self._make_hook(int(layer))
            if side == "pre":
                self._handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
            else:
                self._handles.append(module.register_forward_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        return False

    # The hook edits the image slice of the conditional batch row in place and
    # writes the tensor back into whatever container the module passed it in.
    def _apply(self, tensor: torch.Tensor, layer: int) -> Optional[torch.Tensor]:
        sl = image_span(tensor, self.tracer.n_img or (tensor.shape[1] if tensor.ndim == 3 else 0),
                        self.tracer.n_txt)
        if sl is None:
            return None
        row = int(tensor.shape[0]) - 1
        before = tensor[row, sl, :]
        call = self.calls
        ctx = EditContext(layer=int(layer), step=self.tracer.step, point=self.plan.point,
                          targets=self.targets, trace=self.tracer.trace)
        after = self.plan.edit(before.float(), ctx)
        if after is None:
            return None
        after = after.to(device=before.device, dtype=before.dtype)
        if after.shape != before.shape:
            raise ValueError(f"edit changed the image-token shape at layer {layer}")
        # Energy bookkeeping is what makes an "equal-energy control" checkable:
        # how much squared magnitude the edit took out, put in, and moved.
        change = float(before.float().pow(2).sum() - after.float().pow(2).sum())
        self.removed_energy += max(change, 0.0)
        self.injected_energy += max(-change, 0.0)
        self.perturbation_energy += float((after.float() - before.float()).pow(2).sum())
        self.calls += 1
        if self.tracer.wants_diagnostics():
            key = (int(self.tracer.step), int(layer), self.plan.point.value)
            self.tracer.trace.diagnostics[(*key, "x_pre_hook", call)] = \
                before.detach().float().cpu().clone()
            self.tracer.trace.diagnostics[(*key, "x_post_hook", call)] = \
                after.detach().float().cpu().clone()
            # A forward pre-hook's returned tensor is precisely what the hooked
            # module receives. Keeping a separately named copy makes this identity a
            # numerical manipulation check rather than an assumption in a caption.
            self.tracer.trace.diagnostics[(*key, "x_next_module_input", call)] = \
                after.detach().float().cpu().clone()
        out = tensor.clone()
        out[row, sl, :] = after
        if self.plan.all_batch_rows:
            # Diagnostics and energy bookkeeping stay keyed to the conditional row, so
            # the recorded contract is unchanged; only the application widens.
            other_ctx = replace(ctx, branch="unconditional")
            for other in range(int(tensor.shape[0]) - 1):
                edited = self.plan.edit(tensor[other, sl, :].float(), other_ctx)
                if edited is not None:
                    out[other, sl, :] = edited.to(device=tensor.device, dtype=tensor.dtype)
        return out

    def _walk(self, value, layer: int):
        if torch.is_tensor(value):
            edited = self._apply(value, layer)
            return value if edited is None else edited
        if isinstance(value, tuple):
            return tuple(self._walk(v, layer) for v in value)
        if isinstance(value, list):
            return [self._walk(v, layer) for v in value]
        return value

    def _make_hook(self, layer: int):
        side = _POINT_SIDE[self.plan.point]

        @torch.no_grad()
        def pre_hook(module, args, kwargs):
            if not self.plan.active(self.tracer.step) or self.tracer.pass_idx != 0:
                return None
            if isinstance(kwargs, dict) and torch.is_tensor(kwargs.get("hidden_states")):
                edited = self._apply(kwargs["hidden_states"], layer)
                if edited is None:
                    return None
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = edited
                return args, kwargs
            if not args:
                return None
            return self._walk(tuple(args), layer), kwargs

        @torch.no_grad()
        def post_hook(module, args, kwargs, out):
            if not self.plan.active(self.tracer.step) or self.tracer.pass_idx != 0:
                return None
            return self._walk(out, layer)

        return pre_hook if side == "pre" else post_hook


class StateProbe:
    """Read, without changing, the image tokens at one intervention point.

    Frozen replacement states must come from the same representation the hook
    will later overwrite.  Reading a block *output* and writing it into a block
    *input* would silently transplant across a residual update, so the clean
    pass probes exactly the tensor the treated pass edits.
    """

    def __init__(self, adapter: ModelAdapter, transformer, point: InterventionPoint,
                 layers: Sequence[int], tracer: "CausalTracer",
                 steps: Optional[Sequence[int]] = None):
        self.adapter = adapter
        self.transformer = transformer
        self.point = InterventionPoint(point)
        self.layers = [int(l) for l in layers]
        self.tracer = tracer
        # A probe keeps the WHOLE image slice, so recording it at every step a tracer
        # records would multiply its memory by the number of steps. `steps` confines it.
        self.steps = None if steps is None else {int(s) for s in steps}
        self.refs = {r.index: r for r in adapter.layers(transformer)}
        self._handles: List[Any] = []

    def __enter__(self) -> "StateProbe":
        for layer in self.layers:
            module, side = hook_site(self.adapter, self.refs[layer], self.point)
            hook = self._make_hook(layer, side)
            if side == "pre":
                self._handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
            else:
                self._handles.append(module.register_forward_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        return False

    def _wanted(self) -> bool:
        return self.tracer.recording and (self.steps is None
                                          or int(self.tracer.step) in self.steps)

    def _record(self, value, layer: int) -> None:
        tensors = [value] if torch.is_tensor(value) else [
            t for t in (value if isinstance(value, (tuple, list)) else []) if torch.is_tensor(t)]
        for tensor in tensors:
            sl = image_span(tensor, self.tracer.n_img or (tensor.shape[1] if tensor.ndim == 3 else 0),
                            self.tracer.n_txt)
            if sl is None:
                continue
            row = int(tensor.shape[0]) - 1
            self.tracer.trace.probe_states[(self.tracer.step, layer, self.point.value)] = \
                tensor[row, sl, :].detach().float().cpu().clone()
            return

    def _make_hook(self, layer: int, side: str):
        @torch.no_grad()
        def pre_hook(module, args, kwargs):
            if self._wanted():
                value = kwargs.get("hidden_states") if isinstance(kwargs, dict) else None
                self._record(value if torch.is_tensor(value) else tuple(args), layer)
            return None

        @torch.no_grad()
        def post_hook(module, args, kwargs, out):
            if self._wanted():
                self._record(out, layer)
            return None

        return pre_hook if side == "pre" else post_hook


class FinalKeyPatch:
    """Replace the key of one image token with a clean key, at named layers.

    This is the last rung of the residual-to-key transplant ladder: after normalisation
    and any positional operation, at the tensor the attention kernel actually consumes.
    Only the recipient's key changes. Queries, values, and every other key stay
    exactly as the treated run produced them.
    """

    def __init__(self, tracer: "CausalTracer", *, layers: Sequence[int], source: int,
                 recipient: int, clean_keys: Mapping[int, torch.Tensor], move: bool = False):
        self.tracer = tracer
        self.layers = {int(l) for l in layers}
        self.source = int(source)
        self.recipient = int(recipient)
        self.clean_keys = {int(k): v for k, v in clean_keys.items()}
        self.move = bool(move)
        self.applied = 0

    @staticmethod
    def _head_axis(key: torch.Tensor, heads: int) -> int:
        """Which axis carries heads, resolved the same way as :func:`to_bhsd`."""
        if key.shape[1] == heads and key.shape[2] != heads:
            return 1
        if key.shape[2] == heads and key.shape[1] != heads:
            return 2
        return 2 if key.shape[2] == heads else 1

    def __call__(self, query, key, value, kwargs):
        ref = self.tracer._active
        if ref is None or int(ref.index) not in self.layers or not self.tracer.recording:
            return key
        clean = self.clean_keys.get(int(ref.index))
        if clean is None or key.ndim != 4:
            return key
        heads = self.tracer._heads or int(clean.shape[0])
        axis = self._head_axis(key, heads)
        sequence_axis = 2 if axis == 1 else 1
        n_img = int(clean.shape[1])
        offset = int(key.shape[sequence_axis]) - n_img
        if offset < 0 or self.recipient >= n_img or self.source >= n_img:
            return key
        out = key.clone()
        row = int(out.shape[0]) - 1
        replacement = clean[:, self.source, :].to(device=out.device, dtype=out.dtype)  # [H, D]
        index = [slice(None)] * 4
        index[0] = row
        index[sequence_axis] = offset + self.recipient
        out[tuple(index)] = replacement
        if self.move and self.source != self.recipient:
            ordinary = [t for t in range(n_img) if t not in (self.source, self.recipient)]
            if ordinary:
                neutral = clean[:, ordinary, :].mean(dim=1).to(out)
                source_index = list(index)
                source_index[sequence_axis] = offset + self.source
                out[tuple(source_index)] = neutral
        self.applied += 1
        return out


class SinkSuppression:
    """Present attention with an ordinary key at the register positions.

    This removes the sink without touching the residual stream.  The tokens keep
    their magnitude, their channel values and their direction; only the key the
    attention kernel sees at those positions is replaced, by the mean of the keys
    of the tokens that are *not* targets.  A head therefore has no reason to prefer
    a register over any other patch, and anything that changes downstream is
    attributable to the routing rather than to the state that produced it.

    Replacing the key with the ordinary mean rather than zeroing it matters: a zero
    key still scores zero against every query, which can be the largest score when
    the others are negative, and the sink would survive the intervention meant to
    remove it.
    """

    def __init__(self, tracer: "CausalTracer", *, layers: Sequence[int], tokens: Sequence[int]):
        self.tracer = tracer
        self.layers = {int(l) for l in layers}
        self.tokens = tuple(int(t) for t in tokens)
        self.applied = 0

    def __call__(self, query, key, value, kwargs):
        ref = self.tracer._active
        if ref is None or int(ref.index) not in self.layers or not self.tokens or key.ndim != 4:
            return key
        heads = self.tracer._heads or 0
        axis = FinalKeyPatch._head_axis(key, heads or int(min(key.shape[1], key.shape[2])))
        sequence_axis = 2 if axis == 1 else 1
        n_img = int(self.tracer.n_img or 0)
        length = int(key.shape[sequence_axis])
        span = image_span(key.select(axis, 0), n_img, self.tracer.n_txt) if n_img else None
        if span is None:
            return key
        offset = span.start or 0
        positions = [offset + t for t in self.tokens if offset + t < length]
        if not positions:
            return key
        ordinary = [p for p in range(offset, length) if p not in set(positions)]
        if not ordinary:
            return key
        out = key.clone()
        row = int(out.shape[0]) - 1
        index = [slice(None)] * 4
        index[0] = row
        index[sequence_axis] = ordinary
        # Indexing the batch with an int drops that axis, so the sequence sits one
        # dimension lower than it does in the full tensor.
        mean_key = out[tuple(index)].mean(dim=sequence_axis - 1)      # -> [heads, head_dim]
        for position in positions:
            target = [slice(None)] * 4
            target[0] = row
            target[sequence_axis] = position
            out[tuple(target)] = mean_key
        self.applied += 1
        return out


# ---------------------------------------------------------- frozen targets
@dataclass
class FrozenTargets:
    """Clean-run selections that every condition of a unit must reuse verbatim."""

    prompt_id: int
    seed: int
    step: int
    layer: int
    point: InterventionPoint = InterventionPoint.BLOCK_INPUT
    n_img: int = 0
    grid: Tuple[int, int] = (0, 0)
    register_ids: Tuple[int, ...] = ()
    topk_ids: Tuple[int, ...] = ()
    matched_ordinary_ids: Tuple[int, ...] = ()
    norm_matched_ids: Tuple[int, ...] = ()
    random_ids: Tuple[int, ...] = ()
    nonregister_ids: Tuple[int, ...] = ()
    nearby_id: Optional[int] = None
    random_recipient_id: Optional[int] = None
    ordinary_norm: float = 0.0
    median_norm: float = 0.0
    register_norm: float = 0.0
    register_energy: float = 0.0
    alignment_threshold: float = 0.0
    norm_threshold: float = 0.0
    matched_states: Optional[torch.Tensor] = None      # [k, C] typical ordinary states
    register_states: Optional[torch.Tensor] = None     # [k, C] clean register states
    clean_states: Optional[torch.Tensor] = None        # [N, C] whole clean image slice
    clean_sink_by_head: Dict[int, Dict[int, int]] = field(default_factory=dict)
    matching_distances: Dict[int, float] = field(default_factory=dict)
    selection_rule: str = ""

    @property
    def all_ids(self) -> Tuple[int, ...]:
        seen: List[int] = []
        for group in (self.register_ids, self.topk_ids, self.matched_ordinary_ids,
                      self.norm_matched_ids, self.random_ids, self.nonregister_ids):
            for t in group:
                if int(t) not in seen:
                    seen.append(int(t))
        for t in (self.nearby_id, self.random_recipient_id):
            if t is not None and int(t) not in seen:
                seen.append(int(t))
        return tuple(seen)

    def ids_for(self, group: str) -> Tuple[int, ...]:
        """Token group by name, so a condition table can stay declarative."""
        table = {
            "register": self.register_ids, "topk": self.topk_ids,
            "matched": self.matched_ordinary_ids, "norm_matched": self.norm_matched_ids,
            "random": self.random_ids, "nonregister": self.nonregister_ids,
            "none": (),
        }
        try:
            return table[group]
        except KeyError:
            raise KeyError(f"unknown token group {group!r}; known: {sorted(table)}") from None

    def states_for(self, group: str) -> Optional[torch.Tensor]:
        if self.clean_states is None:
            return None
        ids = self.ids_for(group)
        return self.clean_states[list(ids)].clone() if ids else None


def _rng(*parts: Any) -> torch.Generator:
    import zlib

    seed = zlib.crc32("|".join(str(p) for p in parts).encode())
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _grid_neighbour(token: int, grid: Tuple[int, int], n: int, taken: set) -> Optional[int]:
    """A spatially adjacent patch, which is what "nearby position" has to mean."""
    rows, cols = grid
    if rows <= 0 or cols <= 0:
        candidates = [token + 1, token - 1]
    else:
        r, c = divmod(int(token), cols)
        candidates = [r * cols + (c + 1), r * cols + (c - 1),
                      (r + 1) * cols + c, (r - 1) * cols + c]
        candidates = [t for t, ok in zip(candidates,
                                         [c + 1 < cols, c - 1 >= 0, r + 1 < rows, r - 1 >= 0]) if ok]
    for candidate in candidates:
        if 0 <= candidate < n and candidate not in taken:
            return int(candidate)
    return None


def select_frozen_targets(trace: Trace, *, layer: int, step: int,
                          point: InterventionPoint = InterventionPoint.BLOCK_INPUT,
                          direction: Optional[torch.Tensor] = None, percentile: float = 99.0,
                          topk: int = 8, alignment_quantile: float = 0.999,
                          highnorm_ratio: float = 3.0) -> FrozenTargets:
    """Choose every token group the six questions need, from clean states only.

    ``register_ids`` is the percentile rule applied first (top 1% by
    residual norm); ``topk_ids`` is the threshold-free companion applied
    alongside, so no headline claim rests on one percentile.  Four control groups
    are chosen here rather than inside any condition:

    ``matched``          typical ordinary tokens, whose clean states replace the
                         registers in the "matched ordinary state" intervention;
    ``norm_matched``     the highest-norm tokens that are *not* registers, the
                         closest an ordinary token gets in magnitude;
    ``random``           count-matched ordinary tokens;
    ``nonregister``      the ordinary tokens carrying the most direction energy,
                         which makes an equal-energy removal possible off-register.
    """
    states = trace.probe(step, layer, point)
    observation = trace.at(step, layer)
    if states is None:
        if observation is None or observation.states is None:
            raise ValueError(
                f"no clean state at step {step}, layer {layer}: run the clean pass with a "
                "StateProbe at the intervention point, or ask the tracer for full states")
        states = observation.states
    states = states.float()
    norms = states.norm(dim=-1)
    n = int(norms.numel())
    if direction is not None:
        v = torch.as_tensor(direction).float().flatten()
        v = v / v.norm().clamp_min(1e-9)
        projection = states @ v if v.numel() == states.shape[-1] else None
    else:
        projection = observation.projection.float() if observation is not None and \
            observation.projection is not None and observation.projection.numel() == n else None
    cosine = None if projection is None else projection / norms.clamp_min(1e-9)

    order = torch.argsort(norms, descending=True)
    cut = float(torch.quantile(norms, min(max(percentile / 100.0, 0.0), 1.0)))
    register = [int(t) for t in order if float(norms[t]) >= cut]
    register = register[: max(1, n // 2)] or [int(order[0])]
    topk_ids = [int(t) for t in order[: min(int(topk), n)]]
    register_set = set(register) | set(topk_ids)

    ordinary = [int(t) for t in order if int(t) not in register_set] or [int(order[-1])]
    median = float(norms.median())
    # Typical ordinary partners: closest to the median norm, one per register.
    matched, distances, pool = [], {}, list(ordinary)
    for token in register:
        best = min(pool, key=lambda c: abs(float(norms[c]) - median))
        matched.append(best)
        distances[int(token)] = abs(float(norms[best]) - median)
        if len(pool) > 1:
            pool.remove(best)
    norm_matched = ordinary[: len(register)]
    generator = _rng("random", trace.prompt_id, trace.seed, step, layer)
    picks = torch.randperm(len(ordinary), generator=generator)[: len(register)]
    random_ids = sorted(int(ordinary[i]) for i in picks.tolist())
    if projection is not None:
        nonregister = sorted(ordinary, key=lambda t: -abs(float(projection[t])))[: len(register)]
    else:
        nonregister = list(norm_matched)

    nearby = _grid_neighbour(register[0], trace.grid, n, register_set)
    if nearby is None:
        nearby = int(ordinary[0])
    recipient = int(ordinary[len(ordinary) // 2])

    threshold = 0.0
    if cosine is not None:
        register_cos = torch.as_tensor([float(cosine[t]) for t in register]).abs()
        ordinary_cos = torch.as_tensor([float(cosine[t]) for t in ordinary]).abs()
        # The bar a token must clear to count as aligned like a natural register:
        # the weakest natural register, floored by the ordinary population's
        # extreme so ordinary drift alone can never cross it.
        floor = float(torch.quantile(ordinary_cos, alignment_quantile)) if ordinary_cos.numel() else 0.0
        threshold = float(max(float(register_cos.min()), floor))

    sinks: Dict[int, Dict[int, int]] = {}
    for (s, l), row in trace.rows.items():
        if s != int(step) or row.incoming is None:
            continue
        sinks[int(l)] = {int(h): int(row.incoming[h].argmax()) for h in range(row.incoming.shape[0])}

    return FrozenTargets(
        prompt_id=trace.prompt_id, seed=trace.seed, step=int(step), layer=int(layer),
        point=InterventionPoint(point), n_img=n, grid=trace.grid,
        register_ids=tuple(register), topk_ids=tuple(topk_ids),
        matched_ordinary_ids=tuple(matched), norm_matched_ids=tuple(norm_matched),
        random_ids=tuple(random_ids), nonregister_ids=tuple(nonregister),
        nearby_id=nearby, random_recipient_id=recipient,
        ordinary_norm=float(norms[list(matched)].mean()) if matched else median,
        median_norm=median, register_norm=float(norms[list(register)].mean()),
        register_energy=float(norms[list(register)].pow(2).sum()),
        alignment_threshold=threshold, norm_threshold=float(highnorm_ratio) * median,
        matched_states=states[list(matched)].clone(),
        register_states=states[list(register)].clone(), clean_states=states.clone(),
        clean_sink_by_head=sinks, matching_distances=distances,
        selection_rule=(f"top {100 - percentile:g}% of residual norm at layer {layer}, "
                        f"step {step}, read at {InterventionPoint(point).value}; "
                        f"threshold-free companion is top-{min(int(topk), n)}"),
    )


# ------------------------------------------------------------------ driver
class GenerationDriver:
    """Run one generation, synthetic or real, with hooks already installed."""

    def __init__(self, cfg, adapter: Optional[ModelAdapter] = None, pipe=None):
        from .synthetic import build_tiny, planted_direction

        self.cfg = cfg
        self.spec = cfg.spec
        self.adapter = adapter or get_adapter(self.spec.family)
        self.synthetic = self.spec.repo_id == "synthetic"
        self.pipe = pipe
        if self.synthetic:
            self.bundle = build_tiny(self.spec.family, steps=cfg.num_inference_steps,
                                     grid=max(4, int(cfg.height) // 16))
            self.transformer = self.bundle.transformer
            self.planted = planted_direction(self.bundle.d_model)
            self.grid = self.bundle.grid
            self.n_img = self.bundle.n_img
        else:
            if self.pipe is None:
                self.pipe = self.adapter.load_pipeline(self.spec, cfg)
            self.transformer = self.adapter.transformer(self.pipe)
            self.bundle = None
            self.planted = None
            self.grid = _grid_for(cfg)
            self.n_img = self.grid[0] * self.grid[1] if self.grid[0] else None
        self.refs = self.adapter.layers(self.transformer)

    @property
    def n_layers(self) -> int:
        return len(self.refs)

    def generate(self, prompt: str, seed: int, save_image: bool = False):
        with torch.no_grad():
            if self.synthetic:
                for step in range(self.cfg.num_inference_steps):
                    self.bundle.call(self.bundle.transformer, int(seed) * 10 + 1, step, self.planted)
                return None
            image = self.adapter.generate(self.pipe, prompt, seed, self.cfg, self.spec)
            return image if save_image else None


def _grid_for(cfg) -> Tuple[int, int]:
    """Latent patch grid implied by a config, for spatial sink maps."""
    height, width = int(cfg.height or 0), int(cfg.width or 0)
    if not height or not width:
        return (0, 0)
    family = cfg.family
    factor = 16 if family in ("flux1", "flux2") else 16
    return (max(height // factor, 1), max(width // factor, 1))


# ------------------------------------------------------------- run helpers
@dataclass
class RunStats:
    """What an intervention actually did, so a control can be matched to it."""

    edit_calls: int = 0
    removed_energy: float = 0.0
    injected_energy: float = 0.0
    perturbation_energy: float = 0.0

    def merge(self, installer: "EditInstaller") -> "RunStats":
        self.edit_calls += installer.calls
        self.removed_energy += installer.removed_energy
        self.injected_energy += installer.injected_energy
        self.perturbation_energy += installer.perturbation_energy
        return self

    def as_dict(self) -> Dict[str, float]:
        return {"edit_calls": float(self.edit_calls), "removed_energy": self.removed_energy,
                "injected_energy": self.injected_energy,
                "perturbation_energy": self.perturbation_energy}


def run_traced_generation(driver: GenerationDriver, tracer: CausalTracer, *, prompt_id: int,
                          prompt: str, seed: int, condition: str = "clean",
                          plans: Sequence[EditPlan] = (),
                          targets: Optional[FrozenTargets] = None,
                          probes: Sequence[Tuple[InterventionPoint, Sequence[int]]] = (),
                          key_patch: Optional[Callable] = None,
                          value_patch: Optional[Callable] = None,
                          extra_contexts: Sequence[Any] = (),
                          save_image: bool = False) -> Tuple[Trace, RunStats]:
    """One paired generation: edits installed, observations recorded.

    ``plans`` may hold more than one edit (a suppression plus a later rescue, for
    example); they are installed together and all see the same trajectory.
    ``probes`` read the tensors at named intervention points without changing
    them, which is how the clean pass supplies frozen replacement states.

    ``extra_contexts`` are entered **after** the tracer and exited before it. That
    order is the whole point of the parameter, not a detail: an
    :class:`~ditsinks.attention_patch.AttentionTap` wraps whatever is bound at the
    attention entry point when it enters, so the tap that enters LAST is the
    outermost one, and only taps that entered *earlier* see its transform. A
    virtual-register injector that appends key/value rows must therefore enter after
    the tracer, or the tracer's callback reads the key from before the append while
    believing the registers are there. Wrapping this call in ``with injector:``
    produces exactly the wrong order.
    """
    from contextlib import ExitStack

    plans = list(plans)
    tracer.key_transform = key_patch
    tracer.value_transform = value_patch
    tracer.begin_generation(prompt_id, seed, condition)
    stats = RunStats()
    if plans and targets is None:
        raise ValueError("an edit plan needs the frozen targets it edits")

    with ExitStack() as stack:
        stack.enter_context(tracer)
        for ctx in extra_contexts:
            stack.enter_context(ctx)
        for spec in probes:
            point, layers = spec[0], spec[1]
            steps = spec[2] if len(spec) > 2 else None
            stack.enter_context(StateProbe(driver.adapter, driver.transformer, point,
                                           layers, tracer, steps=steps))
        installers = [stack.enter_context(
            EditInstaller(driver.adapter, driver.transformer, plan, targets, tracer))
            for plan in plans]
        image = driver.generate(prompt, seed, save_image=save_image)
        tracer.trace.image = image
        for installer in installers:
            stats.merge(installer)

    tracer.key_transform = None
    tracer.value_transform = None
    trace = tracer.trace
    if trace.grid == (0, 0) and driver.grid[0]:
        trace.grid = driver.grid
    return trace, stats
