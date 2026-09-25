"""Causal interventions on the register tokens.

The observational sweep shows that the high-norm tokens and the attention sinks
are the same tokens. That is a correlation. These interventions edit the residual
stream during generation and ask what the sinks do in response, which is what
separates "the sinks are the loud tokens" from "the sinks are the tokens pointing
along v*".

Each condition edits a small set of image tokens at one layer, at every denoising
step, and everything downstream sees the edited trajectory.

    Magnitude ablation      rescale the target tokens to the median norm,
                            leaving their direction untouched
    Direction ablation      replace their direction with a random one, leaving
                            their norm untouched
    Direction and magnitude replace both
    Direction transfer      copy a target's direction onto an ordinary token
    Magnitude sweep         the same transfer at a range of insertion norms

Two control families run alongside every ablation, with the same number of tokens
edited at the same steps: a **matched-norm control** (the next-highest-norm tokens,
which are not registers) and a **random-token control**. Without them a change in
the image cannot be attributed to the registers rather than to editing anything.
"""
from __future__ import annotations

import json
import hashlib
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .adapters import InterventionPoint, LayerRef, get_adapter, image_slice
from .capture import SweepCapture, _to_float
from .config import SweepConfig


# --------------------------------------------------------------- conditions

@dataclass(frozen=True, order=True)
class TensorAddress:
    """Complete coordinate of a captured activation (no cross-run guessing)."""
    prompt: int
    seed: int
    step: int
    layer: int
    point: InterventionPoint
    token: Optional[int] = None
    head: Optional[int] = None


class CleanTensorStore:
    """CPU-backed clean-run activations indexed by experimental coordinates."""
    def __init__(self):
        self._values: Dict[TensorAddress, torch.Tensor] = {}

    def capture(self, address: TensorAddress, tensor: torch.Tensor) -> None:
        if address in self._values:
            raise KeyError(f"duplicate clean tensor: {address}")
        self._values[address] = tensor.detach().to("cpu").clone()

    def get(self, address: TensorAddress, *, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            value = self._values[address]
        except KeyError:
            raise KeyError(f"no exactly matched clean tensor for {address}") from None
        return value.to(device=like.device, dtype=like.dtype) if like is not None else value.clone()


@dataclass(frozen=True)
class RunIdentity:
    """Inputs that must match between clean and intervened trajectories."""
    noise_digest: str
    scheduler_digest: str
    prompt_encoding_digest: str
    generation_settings: Tuple[Tuple[str, str], ...]

    @staticmethod
    def _digest(value: Any) -> str:
        if torch.is_tensor(value):
            value = value.detach().contiguous().cpu()
            payload = ((str(value.dtype) + repr(tuple(value.shape))).encode()
                       + value.view(torch.uint8).numpy().tobytes())
        else:
            payload = json.dumps(value, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_run(cls, noise: torch.Tensor, scheduler_state: Mapping[str, Any],
                 prompt_encoding: Any, generation_settings: Mapping[str, Any]) -> "RunIdentity":
        """Fingerprint actual run inputs before either trajectory is executed."""
        return cls(cls._digest(noise), cls._digest(scheduler_state),
                   cls._digest(prompt_encoding),
                   tuple(sorted((str(k), json.dumps(v, sort_keys=True, default=str))
                                for k, v in generation_settings.items())))

    def assert_matches(self, other: "RunIdentity") -> None:
        if self != other:
            differing = [f for f in self.__dataclass_fields__ if getattr(self, f) != getattr(other, f)]
            raise ValueError("clean/intervened run mismatch: " + ", ".join(differing))


class FinalKeyIntervention:
    """Patch selected image keys only; AttentionTap still invokes the native kernel."""
    def __init__(self, store: CleanTensorStore, address: TensorAddress, image_tokens: slice,
                 tokens: Sequence[int], heads: Optional[Sequence[int]] = None):
        if address.point != InterventionPoint.FINAL_KEY:
            raise ValueError("FinalKeyIntervention requires a FINAL_KEY address")
        self.store, self.address, self.image_tokens = store, address, image_tokens
        self.tokens, self.heads = tuple(tokens), None if heads is None else tuple(heads)

    def __call__(self, query, key, value, kwargs):
        out = key.clone()
        # Both supported layouts place head and sequence in dimensions 1/2.
        head_dim = 1 if key.shape[1] <= key.shape[2] else 2
        seq_dim = 2 if head_dim == 1 else 1
        for token in self.tokens:
            pos = (self.image_tokens.start or 0) + token
            for head in (self.heads if self.heads is not None else range(key.shape[head_dim])):
                addr = TensorAddress(self.address.prompt, self.address.seed, self.address.step,
                                     self.address.layer, self.address.point, token, head)
                clean = self.store.get(addr, like=key)
                idx = [slice(None)] * key.ndim
                idx[head_dim], idx[seq_dim] = head, pos
                out[tuple(idx)] = clean
        return out


@dataclass(frozen=True)
class Condition:
    key: str
    label: str                 # how it appears in a figure or a table
    edit: str                  # none | magnitude | direction | both | transfer
    targets: str               # register | matched | random | none
    short: str = ""            # compact axis-tick form
    alpha: Optional[float] = None
    isolate_source: bool = False
    description: str = ""

    @property
    def tick(self) -> str:
        return self.short or self.label


CONDITIONS: Dict[str, Condition] = {c.key: c for c in [
    Condition("baseline", "Baseline", "none", "none", "baseline",
              description="Unmodified generation."),
    Condition("instrumentation_control", "Instrumentation control", "none", "register",
              "instrumentation",
              description="The hook runs and selects tokens but changes nothing. Reproduces the "
                          "baseline up to numerical noise; if it does not, the instrumentation "
                          "itself is perturbing the model and no other condition can be trusted."),
    Condition("magnitude_ablation", "Magnitude ablation", "magnitude", "register", "magnitude"),
    Condition("magnitude_ablation_matched", "Magnitude ablation, matched-norm control",
              "magnitude", "matched", "magnitude (matched)"),
    Condition("magnitude_ablation_random", "Magnitude ablation, random-token control",
              "magnitude", "random", "magnitude (random)"),
    Condition("direction_ablation", "Direction ablation", "direction", "register", "direction"),
    Condition("direction_ablation_matched", "Direction ablation, matched-norm control",
              "direction", "matched", "direction (matched)"),
    Condition("direction_ablation_random", "Direction ablation, random-token control",
              "direction", "random", "direction (random)"),
    Condition("full_ablation", "Direction and magnitude ablation", "both", "register",
              "direction + magnitude"),
    Condition("direction_transfer", "Direction transfer", "transfer", "register", "transfer",
              description="A register's unit direction is copied onto an ordinary token at that "
                          "token's own norm, so no magnitude is injected."),
    Condition("direction_transfer_isolated", "Direction transfer, source removed", "transfer",
              "register", "transfer (source removed)", alpha=12.0, isolate_source=True),
]}

# The magnitude sweep: the same transfer at a range of insertion norms.
TRANSFER_ALPHAS = (1.0, 3.0, 6.0, 12.0)
for _a in TRANSFER_ALPHAS:
    _k = f"direction_transfer_x{_a:g}"
    CONDITIONS[_k] = Condition(_k, f"Direction transfer at {_a:g}x median norm", "transfer",
                               "register", f"{_a:g}x", alpha=_a)

ABLATION_SUITE = ["baseline", "instrumentation_control",
                  "magnitude_ablation", "magnitude_ablation_matched", "magnitude_ablation_random",
                  "direction_ablation", "direction_ablation_matched", "direction_ablation_random",
                  "full_ablation", "direction_transfer"]
MAGNITUDE_SWEEP_SUITE = [f"direction_transfer_x{a:g}" for a in TRANSFER_ALPHAS] + \
                        ["direction_transfer_isolated"]


def condition_label(key: str) -> str:
    c = CONDITIONS.get(key)
    return c.label if c else key


def condition_tick(key: str) -> str:
    c = CONDITIONS.get(key)
    return c.tick if c else key


# ------------------------------------------------------------------- config
@dataclass
class InterventionConfig:
    base: SweepConfig
    # Global layer index whose *input* is edited. Default resolved at run time to
    # the first single-stream layer, which is the dual->single boundary in FLUX.
    intervention_layer: Optional[int] = None
    # Layers at which attention and norms are recorded, to see the consequences.
    observe_layers: Optional[List[int]] = None
    target_ratio: float = 3.0          # a target token has norm > ratio x median
    max_targets: int = 8
    ordinary_rank_min: int = 100       # transfer recipients must be safely ordinary
    conditions: List[str] = field(default_factory=lambda: list(ABLATION_SUITE))
    output_dir: str = ""
    resume: bool = True

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = str(Path(self.base.output_dir) / self.base.spec.key / "interventions")

    @property
    def root(self) -> Path:
        return Path(self.output_dir)


def suggest_intervention_layer(result) -> int:
    """The layer just after high-norm tokens first appear -- where editing them can still matter."""
    from .metrics import highnorm_mask

    first = None
    for rec in sorted(result.records.values(), key=lambda r: r.layer):
        norms = rec.norms.get("post_block")
        if norms is not None and bool(highnorm_mask(norms, result.cfg.highnorm_ratio).any()):
            first = rec.layer
            break
    n = int(result.meta.get("n_layers", 1))
    if first is None:
        b = result.zone_boundary()
        return int(b + 0.5) if b is not None else n // 3
    return min(first + 1, n - 1)


# --------------------------------------------------------------- the edit hook
class TokenEditHook:
    """Edits selected image tokens at the input of one layer, at every step."""

    def __init__(self, icfg: InterventionConfig):
        self.icfg = icfg
        self.condition: Condition = CONDITIONS["baseline"]
        self.prompt_id = 0
        self.seed = 0
        self.step = -1
        self.n_img: Optional[int] = None
        self._last_timestep = None
        # Ablation controls edit the same number of tokens as the ablation they
        # control for, so counts are shared across conditions of one run.
        self.target_counts: Dict[Tuple[int, int, int], int] = {}
        self.log: List[dict] = []

    # ---- generation bookkeeping
    def begin_generation(self, prompt_id: int, seed: int) -> None:
        self.prompt_id, self.seed, self.step = int(prompt_id), int(seed), -1
        self._last_timestep = None
        self.n_img = None

    def transformer_pre(self, module, args, kwargs):
        tv = _to_float(kwargs.get("timestep", None))
        if tv is None or tv != self._last_timestep:
            self.step += 1
            self._last_timestep = tv
        merged = dict(kwargs)
        if args and "hidden_states" not in merged:
            merged["hidden_states"] = args[0]
        n = self._adapter.num_image_tokens(merged, self.icfg.base)
        if n is not None:
            self.n_img = n
        return None

    def bind(self, adapter) -> "TokenEditHook":
        self._adapter = adapter
        return self

    # ---- selection
    def _rng(self, tag: str) -> torch.Generator:
        h = zlib.crc32(f"{tag}|{self.prompt_id}|{self.seed}|{self.step}".encode())
        return torch.Generator(device="cpu").manual_seed(int(h))

    def _select(self, norms: torch.Tensor) -> torch.Tensor:
        med = norms.median()
        key = (self.prompt_id, self.seed, self.step)
        order = torch.argsort(norms, descending=True)

        if self.condition.targets == "none":
            return torch.empty(0, dtype=torch.long, device=norms.device)
        if self.condition.targets == "register":
            sel = torch.nonzero(norms > self.icfg.target_ratio * med).flatten()
            if sel.numel() > self.icfg.max_targets:
                sel = sel[torch.argsort(norms[sel], descending=True)[: self.icfg.max_targets]]
            self.target_counts.setdefault(key, int(sel.numel()))
            return sel

        n = self.target_counts.get(key, 0)
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=norms.device)
        if self.condition.targets == "matched":
            return order[n: 2 * n]                       # next-highest norms, not registers
        pool = order[n:].cpu()                            # random, excluding the register ranks
        pick = torch.randperm(pool.numel(), generator=self._rng("random"))[:n]
        return pool[pick].to(norms.device)

    # ---- the hook itself
    def __call__(self, module, args, kwargs):
        in_kwargs = isinstance(kwargs, dict) and "hidden_states" in kwargs
        x = kwargs["hidden_states"] if in_kwargs else (args[0] if args else None)
        if x is None or not torch.is_tensor(x) or x.ndim != 3:
            return None
        if self.n_img is None:
            self.n_img = int(x.shape[1])
        sl = image_slice(int(x.shape[1]), int(self.n_img))
        if sl is None:
            return None

        b = int(x.shape[0]) - 1
        img = x[b, sl, :]
        norms = img.float().norm(dim=-1)
        med = float(norms.median())
        sel = self._select(norms)
        row = dict(condition=self.condition.key, prompt_id=self.prompt_id, seed=self.seed,
                   step=self.step, n_edited=int(sel.numel()), median_norm=med,
                   max_norm=float(norms.max()),
                   edited_tokens=json.dumps([int(t) for t in sel.tolist()]),
                   source_token=-1, recipient_token=-1)

        if self.condition.edit == "none" or sel.numel() == 0:
            self.log.append(row)
            return None

        x = x.clone()
        off = sl.start
        D = x.shape[-1]

        if self.condition.edit == "transfer":
            source = int(sel[torch.argmax(norms[sel])])
            unit = img[source].float() / norms[source].clamp(min=1e-6)
            # Recipients must be safely ordinary, but the floor cannot exceed the
            # sequence: on a short sequence it degrades to the lower half rather
            # than silently transferring nothing.
            rank_min = min(self.icfg.ordinary_rank_min,
                           max(int(sel.numel()) + 1, int(norms.numel()) // 2))
            pool = torch.argsort(norms, descending=True)[rank_min:]
            if pool.numel() == 0:
                row["note"] = "no ordinary recipient available"
                self.log.append(row)
                return None
            pick = int(torch.randint(pool.numel(), (1,), generator=self._rng("recipient")).item())
            recipient = int(pool[pick])
            scale = (self.condition.alpha * med if self.condition.alpha is not None
                     else float(norms[recipient]))
            x[b, off + recipient, :] = (unit * scale).to(x.dtype)
            if self.condition.isolate_source:
                v = torch.randn(D, generator=self._rng("isolate"))
                x[b, off + source, :] = (v / v.norm()).to(x.device, x.dtype) * med
            row.update(source_token=source, recipient_token=recipient, n_edited=1,
                       edited_tokens=json.dumps([recipient]))
        else:
            idx = (off + sel).to(x.device)
            if self.condition.edit == "magnitude":
                scale = (med / norms[sel].clamp(min=1e-6)).to(x.dtype)
                x[b, idx, :] = x[b, idx, :] * scale.unsqueeze(-1)
            else:
                v = torch.randn(sel.numel(), D, generator=self._rng("direction"))
                v = (v / v.norm(dim=-1, keepdim=True)).to(x.device)
                target = (torch.full((sel.numel(),), med, device=x.device)
                          if self.condition.edit == "both" else norms[sel].to(x.device))
                x[b, idx, :] = (v * target.unsqueeze(-1)).to(x.dtype)

        self.log.append(row)
        if in_kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = x
            return (args, kwargs)
        return ((x,) + tuple(args[1:]), kwargs)


# ------------------------------------------------------------------- runner
def run_intervention_suite(icfg: InterventionConfig, pipe=None, progress: bool = True) -> Dict[str, Any]:
    """Run every requested condition and write per-condition metrics, images and logs."""
    from .metrics import head_table, layer_table
    from .runner import SweepResult, _infer_grid
    from .synthetic import build_tiny, planted_direction

    cfg = icfg.base
    spec = cfg.spec
    adapter = get_adapter(spec.family)
    root = icfg.root
    root.mkdir(parents=True, exist_ok=True)

    synthetic = spec.repo_id == "synthetic"
    if synthetic:
        bundle = build_tiny(spec.family, steps=cfg.num_inference_steps,
                            grid=max(4, int(cfg.height) // 16))
        transformer = bundle.transformer
        v_planted = planted_direction(bundle.d_model)
    else:
        if pipe is None:
            pipe = adapter.load_pipeline(spec, cfg)
        transformer = adapter.transformer(pipe)

    refs: List[LayerRef] = adapter.layers(transformer)
    layer = icfg.intervention_layer
    if layer is None:
        singles = [r.index for r in refs if r.kind == "single"]
        layer = singles[0] if singles else len(refs) // 3
        icfg.intervention_layer = layer
    if icfg.observe_layers is None:
        icfg.observe_layers = sorted({min(layer + d, len(refs) - 1) for d in (0, 2, 4, 8, 16)})

    # Conditions are ordered so a register condition always precedes the controls
    # that reuse its per-step token count.
    order = {"register": 0, "none": 0, "matched": 1, "random": 1}
    todo = sorted(icfg.conditions, key=lambda k: order.get(CONDITIONS[k].targets, 1))

    hook = TokenEditHook(icfg).bind(adapter)
    (root / "images").mkdir(exist_ok=True)
    written: List[str] = []

    for key in todo:
        cond = CONDITIONS[key]
        metrics_path = root / f"head_metrics_{key}.csv"
        if icfg.resume and metrics_path.exists() and _images_complete(root, cfg, key):
            if progress:
                print(f"  [{key}] already complete, skipping")
            written.append(key)
            continue
        if progress:
            print(f"\n=== {cond.label} ===", flush=True)

        hook.condition = cond
        obs = SweepConfig(**{**{k: v for k, v in asdict(cfg).items()
                               if k not in ("layers", "focus_layers")},
                             "layers": list(icfg.observe_layers), "focus_layers": []})
        cap = SweepCapture(adapter, obs, transformer)

        handles = [transformer.register_forward_pre_hook(hook.transformer_pre, with_kwargs=True),
                   refs[layer].block.register_forward_pre_hook(hook, with_kwargs=True)]
        cap.__enter__()          # installed after the edit hook, so it sees the edited state
        try:
            img_dir = root / "images" / key
            img_dir.mkdir(parents=True, exist_ok=True)
            for pid, prompt in enumerate(cfg.prompts):
                for seed in cfg.seeds:
                    cap.begin_generation(pid, seed)
                    hook.begin_generation(pid, seed)
                    if synthetic:
                        with torch.no_grad():
                            for stp in range(cfg.num_inference_steps):
                                bundle.call(bundle.transformer, seed * 10 + pid, stp, v_planted)
                    else:
                        image = adapter.generate(pipe, prompt, seed, cfg, spec)
                        if image is not None:
                            image.save(img_dir / f"prompt{pid}_seed{seed}.png")
        finally:
            cap.close()
            for h in handles:
                h.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        res = SweepResult(cfg=obs, records=dict(cap.records))
        res.meta.update(model=spec.key, family=spec.family, n_layers=len(refs),
                        grid=_infer_grid(res, cfg))
        ht = head_table(res)
        ht.insert(0, "condition", key)
        ht.to_csv(metrics_path, index=False)
        lt = layer_table(res)
        lt.insert(0, "condition", key)
        lt.to_csv(root / f"layer_metrics_{key}.csv", index=False)
        written.append(key)
        del cap, res

    pd.DataFrame(hook.log).drop_duplicates(
        subset=["condition", "prompt_id", "seed", "step"], keep="last"
    ).to_csv(root / "edit_log.csv", index=False)
    (root / "config.json").write_text(json.dumps(
        {**asdict(cfg), "intervention_layer": icfg.intervention_layer,
         "observe_layers": icfg.observe_layers, "target_ratio": icfg.target_ratio,
         "max_targets": icfg.max_targets, "conditions": list(icfg.conditions)}, indent=2))
    if progress:
        print(f"\nWrote {len(written)} conditions to {root}")
    return dict(root=root, conditions=written, intervention_layer=icfg.intervention_layer,
                observe_layers=icfg.observe_layers)


def _images_complete(root: Path, cfg: SweepConfig, key: str) -> bool:
    d = root / "images" / key
    if not d.exists():
        return False
    for pid in range(len(cfg.prompts)):
        for seed in cfg.seeds:
            if not (d / f"prompt{pid}_seed{seed}.png").exists():
                return False
    return True


# ------------------------------------------------------------------ loading
def load_intervention_results(root) -> Dict[str, pd.DataFrame]:
    """Read whatever conditions are on disk."""
    root = Path(root)
    heads, layers = [], []
    for f in sorted(root.glob("head_metrics_*.csv")):
        heads.append(pd.read_csv(f, low_memory=False))
    for f in sorted(root.glob("layer_metrics_*.csv")):
        layers.append(pd.read_csv(f, low_memory=False))
    if not heads:
        raise FileNotFoundError(f"no intervention metrics under {root}")
    log = pd.read_csv(root / "edit_log.csv") if (root / "edit_log.csv").exists() else pd.DataFrame()
    return dict(head=pd.concat(heads, ignore_index=True),
                layer=pd.concat(layers, ignore_index=True) if layers else pd.DataFrame(),
                log=log, root=root)


# ------------------------------------------------------------------ analysis
def _edited_sets(log: pd.DataFrame) -> Dict[Tuple[str, int, int, int], set]:
    out: Dict[Tuple[str, int, int, int], set] = {}
    for r in log.itertuples():
        try:
            ids = set(json.loads(r.edited_tokens))
        except Exception:
            ids = set()
        out[(r.condition, int(r.prompt_id), int(r.seed), int(r.step))] = ids
    return out


def sink_persistence(results: Dict[str, Any], reference: str = "instrumentation_control"
                     ) -> pd.DataFrame:
    """Does each head keep sinking on the same token after the edit?

    Restricted to the heads that were actually affected -- those whose reference
    sink was one of the tokens the register ablation edits. A head that never
    sank on an edited token tells us nothing about the edit.
    """
    head, log = results["head"], results["log"]
    conds = set(head["condition"])
    ref = reference if reference in conds else "baseline"
    key = ["prompt_id", "seed", "step", "layer", "block_name", "head_id"]
    base = head[head["condition"] == ref][key + ["sink_token"]].rename(
        columns={"sink_token": "reference_sink"})
    merged = head.merge(base, on=key, how="left")

    edited = _edited_sets(log)
    target_key = "magnitude_ablation" if "magnitude_ablation" in conds else ref

    def affected(r):
        ids = edited.get((target_key, int(r["prompt_id"]), int(r["seed"]), int(r["step"])), set())
        return int(r["reference_sink"]) in ids if pd.notna(r["reference_sink"]) else False

    merged["head_was_affected"] = merged.apply(affected, axis=1)
    merged["sink_unchanged"] = merged["sink_token"] == merged["reference_sink"]
    sub = merged[merged["head_was_affected"] & (merged["condition"] != ref)]
    if sub.empty:                      # no head sank on an edited token: report all heads
        sub = merged[merged["condition"] != ref]
    return sub.groupby(["condition", "layer", "block_name"], as_index=False).agg(
        sink_unchanged=("sink_unchanged", "mean"),
        sink_strength=("sink_ratio_over_uniform", "mean"),
        n_heads=("sink_token", "size"))


def head_agreement(results: Dict[str, Any]) -> pd.DataFrame:
    """Fraction of heads in a block that converge on the same key token."""
    head = results["head"]
    grp = head.groupby(["condition", "layer", "block_name", "prompt_id", "seed", "step"])
    rows = []
    for keys, g in grp:
        counts = g["sink_token"].value_counts()
        rows.append(dict(zip(["condition", "layer", "block_name", "prompt_id", "seed", "step"], keys),
                         agreement=float(counts.iloc[0]) / max(len(g), 1)))
    df = pd.DataFrame(rows)
    return df.groupby(["condition", "layer", "block_name"], as_index=False)["agreement"].mean()


def attention_reallocation(results: Dict[str, Any]) -> pd.DataFrame:
    """Share of attention mass that moves onto text keys after the edit."""
    head = results["head"]
    if "text_mass" not in head or float(head["text_mass"].max()) == 0.0:
        return pd.DataFrame()
    return head.groupby(["condition", "layer", "block_name"], as_index=False).agg(
        text_mass=("text_mass", "mean"), sink_is_text=("sink_is_text", "mean"))


def norm_recovery(results: Dict[str, Any]) -> pd.DataFrame:
    """Outlier norm ratio downstream of the edit: does the network rebuild it?"""
    layer = results.get("layer")
    if layer is None or layer.empty:
        return pd.DataFrame()
    return layer.groupby(["condition", "layer", "block_name"], as_index=False).agg(
        norm_ratio=("norm_max_over_median", "mean"),
        n_highnorm=("n_highnorm", "mean"))


def transfer_capture(results: Dict[str, Any]) -> pd.DataFrame:
    """How often a head's strongest image key becomes the token that received the direction."""
    head, log = results["head"], results["log"]
    tl = log[log["recipient_token"] >= 0]
    if tl.empty:
        return pd.DataFrame()
    rec = {(r.condition, int(r.prompt_id), int(r.seed), int(r.step)):
           (int(r.recipient_token), int(r.source_token)) for r in tl.itertuples()}
    rows = []
    for cond in sorted(set(tl["condition"])):
        hc = head[head["condition"] == cond]
        hb = head[head["condition"] == "baseline"]
        for name, frame in (("transfer", hc), ("baseline", hb)):
            if frame.empty:
                continue
            for _, r in frame.iterrows():
                pair = rec.get((cond, int(r["prompt_id"]), int(r["seed"]), int(r["step"])))
                if pair is None:
                    continue
                rows.append(dict(condition=cond, which=name, layer=int(r["layer"]),
                                 block_name=r["block_name"],
                                 alpha=CONDITIONS[cond].alpha,
                                 captured=int(r["sink_token"]) == pair[0],
                                 source_still_sink=int(r["sink_token"]) == pair[1]))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["condition", "which", "layer", "block_name", "alpha"], dropna=False,
                      as_index=False).agg(capture_rate=("captured", "mean"),
                                          source_rate=("source_still_sink", "mean"))


def image_distances(root, cfg: SweepConfig, reference: str = "baseline") -> pd.DataFrame:
    """Perceptual distance from the reference image, per condition and generation."""
    from PIL import Image

    root = Path(root)
    try:
        import lpips

        net = lpips.LPIPS(net="alex", verbose=False)
        metric = "LPIPS"

        def dist(a, b):
            ta = torch.from_numpy(np.array(a)).permute(2, 0, 1).float() / 127.5 - 1
            tb = torch.from_numpy(np.array(b)).permute(2, 0, 1).float() / 127.5 - 1
            with torch.no_grad():
                return float(net(ta.unsqueeze(0), tb.unsqueeze(0)))
    except Exception:
        metric = "RMSE"

        def dist(a, b):
            return float(np.sqrt(np.mean((np.array(a).astype(np.float32)
                                          - np.array(b).astype(np.float32)) ** 2)))

    rows = []
    ref_dir = root / "images" / reference
    for pid in range(len(cfg.prompts)):
        for seed in cfg.seeds:
            fn = f"prompt{pid}_seed{seed}.png"
            if not (ref_dir / fn).exists():
                continue
            ref_img = Image.open(ref_dir / fn).convert("RGB")
            for d in sorted((root / "images").iterdir()):
                if not d.is_dir() or d.name == reference or not (d / fn).exists():
                    continue
                rows.append(dict(condition=d.name, prompt_id=pid, seed=seed,
                                 pair=f"p{pid}_s{seed}", metric=metric,
                                 distance=dist(ref_img, Image.open(d / fn).convert("RGB"))))
    return pd.DataFrame(rows)


def paired_comparisons(dist: pd.DataFrame,
                       pairs: Sequence[Tuple[str, str]] = (
                           ("magnitude_ablation", "magnitude_ablation_matched"),
                           ("magnitude_ablation", "magnitude_ablation_random"),
                           ("direction_ablation", "direction_ablation_matched"),
                           ("direction_ablation", "direction_ablation_random"),
                           ("direction_ablation", "magnitude_ablation"))) -> pd.DataFrame:
    """Wilcoxon signed-rank tests between conditions on the same (prompt, seed) pairs."""
    if dist is None or dist.empty or "condition" not in dist:
        return pd.DataFrame()
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return pd.DataFrame()
    rows = []
    for a, b in pairs:
        ga = dist[dist["condition"] == a].set_index("pair")["distance"]
        gb = dist[dist["condition"] == b].set_index("pair")["distance"]
        common = ga.index.intersection(gb.index)
        if len(common) < 3:
            continue
        va, vb = ga.loc[common].to_numpy(), gb.loc[common].to_numpy()
        try:
            stat, p = wilcoxon(va, vb)
        except Exception:
            stat, p = np.nan, np.nan
        rows.append(dict(condition_a=condition_label(a), condition_b=condition_label(b),
                         n_pairs=len(common), median_difference=float(np.median(va - vb)),
                         p_value=float(p)))
    return pd.DataFrame(rows)
