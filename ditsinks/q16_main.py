"""Q16 main run: the depth at which the register state exists, and what it does to the image.

The experiment
--------------
One **experimental unit** is a (prompt, seed) pair. Every condition of a unit shares the
prompt, the seed and therefore the initial noise, so every effect is a paired difference
against the unit's own unmodified generation. Prompts are the cluster for inference: a
prompt's seeds share its layout and its register positions, and are resampled together.

The manipulated variable is the interval of transformer depth over which the natural
register state exists -- the sparse set of high-norm, ``v*``-aligned image tokens that act
as attention sinks. It is set in three ways, all during the same denoising steps:

- **Depth sweep, natural register retained.** The register state is written into one window
  of blocks at a time, at the natural register's own positions and with its own ``v*``
  projection and norm, both carried as multiples of each block's median token norm. Windows
  of equal length tile the blocks before the natural formation and after the natural end;
  one window inside the natural plateau rewrites the state where it already exists, which
  measures what the write itself does.
- **Removal and relocation.** The natural state is removed from every token more
  ``v*``-aligned than the unmodified run's ordinary tokens, at every block of the natural
  interval; alone, and with the state written into the window just before formation or
  just after the natural end.
- **Extension.** The natural state is maintained from before its decline through the first
  late window, so its lifetime continues rather than being re-created after a gap.

Three **specificity controls** accompany the window just before formation: the same write
along a random direction orthogonal to ``v*``; the ``v*`` write at positions that never
carry the register; and a ``v*`` write at the largest projection that block already holds.

Two stages, in this order, both before any conditioned image exists:

1. :func:`calibrate_protocol` measures the natural interval on held-out calibration prompts
   and freezes every window by rule (:func:`plan_depth_windows`) into a
   :class:`DepthProtocol`.
2. :func:`run_unit` runs every condition of one unit against that frozen protocol, writes
   compact records, and marks the unit complete, so an interrupted run resumes.

:func:`collect_units` and :func:`effect_table` pool the units for the figures in
:mod:`ditsinks.q16_figures`.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from . import lifecycle as LC
from .adapters import InterventionPoint
from .causal_engine import CausalTracer, EditPlan, run_traced_generation, select_frozen_targets

__all__ = [
    "EDIT_BRANCHES", "run_root", "guidance_override", "branch_register_overlap",
    "register_profile",
    "late_state_table", "policy_table",
    "PROTOCOL_VERSION", "MainSettings", "MainCondition", "DepthProtocol",
    "plan_depth_windows", "measure_boundaries", "protocol_from_calibration",
    "calibrate_protocol", "condition_catalog", "evaluation_prompts", "unit_directory",
    "unit_is_complete", "run_unit", "collect_units", "effect_table", "paired_contrast",
    "writer_gain_table", "FORMAL_METRIC_NAMES", "NoRegister",
]

PROTOCOL_VERSION = 1

# Formal names for every metric a figure or table prints. Nothing reaches a figure under
# its code name.
FORMAL_METRIC_NAMES = {
    "lpips": "LPIPS to the unmodified image",
    "psnr": "PSNR to the unmodified image (dB)",
    "clip_image_similarity": "CLIP image similarity to the unmodified image",
    "clip_prompt_similarity_change": "Change in CLIP image-text similarity",
    "high_frequency_energy_change": "Change in high-frequency energy",
}


# =================================================================== settings
@dataclass(frozen=True)
class MainSettings:
    """Every value the main run uses, declared before any generation.

    The basis for each value is in ``experiments/q16_lifecycle_retiming/PARAMETERS.md``.
    The shared definitions (``highnorm_ratio``, ``alignment_quantile``,
    ``sink_threshold``) are the ones every other table of the paper uses; the range check
    (:func:`ditsinks.lifecycle.threshold_sensitivity`) is run on every unit.
    """

    checkpoint: str
    size: int
    steps: int
    guidance: Optional[float] = None          # None -> the checkpoint's default
    seeds: Tuple[int, ...] = (0, 42, 1234, 777, 3407)
    # WHEN: the denoising steps at which every intervention fires. The first third is the
    # high-noise phase in which the global composition is set.
    edit_phase: Any = "early"
    # WHERE: equal-length windows per side of the natural interval.
    windows_per_side: int = 3
    first_block: int = 1                      # block 0 reads the patch embedding
    gap: int = 1                              # blocks kept free next to the natural interval
    # Shared definitions.
    highnorm_ratio: float = 3.0
    alignment_quantile: float = 0.999
    sink_threshold: float = 10.0
    # Boundary conventions.
    formation_fraction: float = 0.5
    dissolution_fraction: float = 0.9
    bridge_lead: int = 2
    # A step contributes to calibration only if its register has at least this many tokens.
    min_carriers: int = 4
    rule_max_share_of_image: float = 0.02
    # The in-distribution control: this multiple of the largest v* projection any token
    # of the recipient block holds in the unmodified run.
    in_distribution_strength: float = 1.0
    control_direction_seed: int = 0
    ordinary_positions_seed: int = 0
    # Recording. None derives them from the edited steps (see ``attention_step_list``).
    attention_steps: Optional[Tuple[int, ...]] = None
    record_steps: Optional[Tuple[int, ...]] = None
    thumbnail_size: int = 384
    run_hooks_only_check: bool = True
    # WHICH PASS: under classifier-free guidance (PixArt-Sigma) the sampler runs the model
    # twice per step, conditional and unconditional, and extrapolates their difference by
    # the guidance scale. 'conditional' edits the conditional pass only, so the edit's
    # effect on each update is multiplied by the guidance scale (4.5); 'both' edits both
    # passes (each row with its own remainder; positions and sizes from the conditional
    # pass), which is the like-for-like comparison with FLUX.1-dev, whose guidance is an
    # input to a single pass. No effect on a model sampled without CFG.
    edit_branches: str = "conditional"

    def edit_step_list(self) -> List[int]:
        return LC.denoising_phase_steps(self.edit_phase, self.steps)

    def readout_step(self) -> int:
        """The single step a one-step figure shows: the middle of the edited phase."""
        edited = self.edit_step_list()
        return int(edited[len(edited) // 2])

    def attention_step_list(self) -> List[int]:
        """Attention is expensive at 1024px: the middle and last edited steps, and the
        trajectory midpoint to see what remains once the edits have stopped."""
        if self.attention_steps is not None:
            return sorted({int(s) for s in self.attention_steps})
        edited = self.edit_step_list()
        return sorted({self.readout_step(), int(edited[-1]), int(self.steps // 2)})

    def record_step_list(self) -> List[int]:
        """Residual states are cheap: every edited step, the attention steps, the last."""
        if self.record_steps is not None:
            return sorted({int(s) for s in self.record_steps}
                          | set(self.attention_step_list()))
        return sorted(set(self.edit_step_list()) | set(self.attention_step_list())
                      | {int(self.steps) - 1})

    def row(self) -> Dict[str, Any]:
        out = asdict(self)
        out["edit_steps"] = self.edit_step_list()
        out["attention_steps"] = self.attention_step_list()
        out["record_steps"] = self.record_step_list()
        out["readout_step"] = self.readout_step()
        return out


# ================================================================== protocol
@dataclass
class DepthProtocol:
    """The frozen depth design for one checkpoint, written before any conditioned image.

    ``formation`` is the earliest and ``natural_end`` the latest boundary measured on the
    calibration prompts, so no early window reaches a formed register and no late window
    starts while one is still present, in any calibration generation.
    """

    checkpoint: str
    n_layers: int
    seam: Optional[int]
    formation: int
    natural_end: int
    decline: Optional[int]
    selection_block: int
    windows: Dict[str, Tuple[int, int]]
    primary_early: Optional[str]
    primary_late: Optional[str]
    plateau: Tuple[int, int]
    window_length: int
    settings: Dict[str, Any]
    calibration: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    version: int = PROTOCOL_VERSION

    # ------------------------------------------------------------ access
    def window(self, name: str) -> Optional[LC.Window]:
        span = self.windows.get(name)
        return None if span is None else LC.Window(name, int(span[0]), int(span[1]))

    def early_windows(self) -> List[str]:
        return sorted((k for k in self.windows if k.startswith("E")), key=lambda k: int(k[1:]))

    def late_windows(self) -> List[str]:
        return sorted((k for k in self.windows if k.startswith("L")), key=lambda k: int(k[1:]))

    def removal_window(self) -> LC.Window:
        return LC.Window("removal", *self.windows["removal"])

    @property
    def edit_steps(self) -> List[int]:
        return list(self.settings["edit_steps"])

    @property
    def record_steps(self) -> List[int]:
        return list(self.settings["record_steps"])

    @property
    def attention_steps(self) -> List[int]:
        return list(self.settings["attention_steps"])

    @property
    def readout_step(self) -> int:
        return int(self.settings["readout_step"])

    def block_kind(self, layer: int) -> str:
        if self.seam is None:
            return "uniform"
        return "dual-stream" if int(layer) < int(self.seam) else "single-stream"

    def table(self) -> List[Dict[str, Any]]:
        rows = []
        for name, (first, last) in self.windows.items():
            rows.append(dict(window=name, first=int(first), last=int(last),
                             n_blocks=int(last) - int(first) + 1,
                             relative_depth_center=(0.5 * (first + last) / max(self.n_layers - 1, 1)),
                             block_kind=(self.block_kind(first) if self.block_kind(first) ==
                                         self.block_kind(last) else "crosses the seam")))
        return rows

    # ------------------------------------------------------------ files
    def to_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["windows"] = {k: [int(a), int(b)] for k, (a, b) in self.windows.items()}
        payload["plateau"] = [int(self.plateau[0]), int(self.plateau[1])]
        path.write_text(json.dumps(payload, indent=2, default=_jsonable))
        return path

    @classmethod
    def from_json(cls, path) -> "DepthProtocol":
        payload = json.loads(Path(path).read_text())
        if int(payload.get("version", 0)) != PROTOCOL_VERSION:
            raise ValueError(f"{path} was written by protocol version "
                             f"{payload.get('version')}, this code reads {PROTOCOL_VERSION}")
        payload["windows"] = {k: (int(v[0]), int(v[1])) for k, v in payload["windows"].items()}
        payload["plateau"] = tuple(int(x) for x in payload["plateau"])
        return cls(**payload)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


# ============================================================ window planning
def plan_depth_windows(*, n_layers: int, formation: int, natural_end: int,
                       decline: Optional[int] = None, per_side: int = 3,
                       first_block: int = 1, gap: int = 1,
                       bridge_lead: int = 2) -> Tuple[Dict[str, Tuple[int, int]], List[str]]:
    r"""Place every window by rule from the measured natural interval.

    - The **pre-formation region** is blocks ``first_block`` to ``formation - gap - 1``;
      the **post-natural region** is ``natural_end + gap + 1`` to the last block. The
      ``gap`` keeps the block next to the natural interval free: on the formation side it
      is where the writer begins, on the other it carries the removal's terminal hook.
    - Every window has the same length ``L``, the largest for which ``per_side`` windows
      fit in both regions, so every window holds the same number of hooked blocks and
      depth is not confounded with dose. Early windows ``E1..Ek`` stack outward from
      formation (``Ek`` is adjacent to it); late windows ``L1..Lk`` stack outward from the
      natural end (``L1`` is adjacent to it). Numbering runs from shallow to deep.
    - ``N`` rewrites the state inside the natural plateau, from the block after formation,
      and is shortened so it ends before the measured decline.
    - ``removal`` is the interval the removal conditions clear: from the block before
      formation (where the writer starts) through the natural end.
    - ``extend`` maintains the natural state from ``bridge_lead`` blocks before the decline
      through the last block of ``L1``, so the same late blocks are covered by a continued
      lifetime (``extend``) and by a re-created one (``L1``).
    """
    notes: List[str] = []
    formation, natural_end = int(formation), int(natural_end)
    pre = list(range(int(first_block), formation - int(gap)))
    post = list(range(natural_end + int(gap) + 1, int(n_layers)))
    requested, per_side, length = int(per_side), int(per_side), 0
    while per_side >= 1:
        length = min(len(pre) // per_side, len(post) // per_side)
        if length >= 1:
            break
        per_side -= 1
    if per_side < 1 or length < 1:
        raise ValueError(f"no room for a window of one block on both sides of the natural "
                         f"interval {formation}-{natural_end} in {n_layers} blocks")
    if per_side < requested:
        notes.append(f"only {per_side} window(s) per side fit, not {requested}")
    windows: Dict[str, Tuple[int, int]] = {}
    for k in range(per_side):                        # k = 0 is adjacent to formation
        last = pre[-1] - k * length
        windows[f"E{per_side - k}"] = (last - length + 1, last)
    for k in range(per_side):                        # k = 0 is adjacent to the natural end
        first = post[0] + k * length
        windows[f"L{k + 1}"] = (first, first + length - 1)
    in_place_last = formation + length
    if decline is not None and in_place_last >= int(decline):
        in_place_last = int(decline) - 1
        notes.append(f"the in-place window was shortened to end before the decline at "
                     f"block {decline}")
    if in_place_last >= formation + 1:
        windows["N"] = (formation + 1, in_place_last)
    else:
        notes.append("no plateau block after formation: no in-place window")
    windows["removal"] = (formation - 1, natural_end)
    bridge_start = (max(formation + 1, int(decline) - int(bridge_lead))
                    if decline is not None else formation + 1)
    windows["extend"] = (bridge_start, windows["L1"][1])
    if length < 2:
        notes.append(f"windows are {length} block long; the depth resolution is coarse")
    return windows, notes


# ======================================================= measured boundaries
def measure_boundaries(trace, *, steps: Sequence[int], layers: Sequence[int],
                       settings: MainSettings) -> List[Dict[str, Any]]:
    r"""The natural interval of one unmodified generation, at each step in ``steps``.

    Formation is the first block whose output holds half the peak register population,
    counted with the register test and no carrier set; the carriers are the register at
    its peak block; the natural end is the block before the first block after the peak at
    which none of them passes the register test; the decline is the first block after the
    peak at which their mean ``v*`` projection is below ``dissolution_fraction`` of its
    maximum.
    """
    layers = sorted(int(l) for l in layers)
    base = LC.CleanStateReference(highnorm_ratio=settings.highnorm_ratio,
                                  alignment_quantile=settings.alignment_quantile)
    base.add(trace, steps=list(steps), layers=layers)
    rows: List[Dict[str, Any]] = []
    for step in steps:
        population = LC.register_test_counts(base, step=step, layers=layers)
        formation = LC.measured_formation_by_count(population,
                                                   fraction=settings.formation_fraction)
        row = dict(step=int(step), formation=formation["onset"],
                   peak_layer=formation["peak_layer"], peak_count=formation["peak_count"])
        if formation["onset"] is None:
            rows.append(dict(row, natural_end=None, decline=None, n_carriers=0,
                             valid=False, note=formation["note"]))
            continue
        carriers = sorted(base.at(step, formation["peak_layer"]).carriers)
        excluded = LC.CleanStateReference(highnorm_ratio=settings.highnorm_ratio,
                                          alignment_quantile=settings.alignment_quantile,
                                          frozen=carriers)
        excluded.add(trace, steps=[int(step)], layers=layers)
        end = LC.measured_natural_end(
            LC.register_test_counts(excluded, step=step, layers=layers, carriers=carriers),
            after=formation["peak_layer"])
        lifecycle = LC.achieved_lifecycle(trace, step=int(step), layers=layers,
                                          carriers=carriers,
                                          sink_threshold=settings.sink_threshold,
                                          highnorm_ratio=settings.highnorm_ratio)
        decline = LC.measured_dissolution_onset(
            lifecycle, natural=LC.Window("natural", int(formation["onset"]), layers[-1]),
            fraction=settings.dissolution_fraction)
        peak_row = next((r for r in lifecycle if r["layer"] == formation["peak_layer"]), {})
        rows.append(dict(
            row, natural_end=end["natural_end"], decline=decline.get("onset"),
            n_carriers=len(carriers),
            carrier_cosine_at_peak=peak_row.get("carrier_cosine", float("nan")),
            carrier_norm_ratio_at_peak=peak_row.get("carrier_norm_ratio", float("nan")),
            valid=bool(len(carriers) >= settings.min_carriers
                       and end["natural_end"] is not None),
            note=end["note"]))
    return rows


def protocol_from_calibration(rows: Sequence[Dict[str, Any]], *, n_layers: int,
                              seam: Optional[int], checkpoint: str,
                              settings: MainSettings) -> DepthProtocol:
    """Freeze the depth design from calibration boundaries (one row per generation step)."""
    valid = [r for r in rows if r.get("valid")]
    if not valid:
        raise RuntimeError(
            "no calibration generation formed a register with at least "
            f"{settings.min_carriers} carriers and a measurable end at the edited steps. "
            "The design has nothing to retime on this checkpoint at these steps.")
    formation = int(min(r["formation"] for r in valid))
    natural_end = int(max(r["natural_end"] for r in valid))
    declines = [int(r["decline"]) for r in valid if r.get("decline") is not None]
    decline = int(np.median(declines)) if declines else None
    peaks = [int(r["peak_layer"]) for r in valid]
    selection = int(max(set(peaks), key=peaks.count))
    windows, notes = plan_depth_windows(
        n_layers=n_layers, formation=formation, natural_end=natural_end, decline=decline,
        per_side=settings.windows_per_side, first_block=settings.first_block,
        gap=settings.gap, bridge_lead=settings.bridge_lead)
    plateau_last = (decline - 1) if decline is not None else natural_end
    plateau = (formation + 1, max(formation + 1, plateau_last))
    length = windows["E1"][1] - windows["E1"][0] + 1
    early = sorted((k for k in windows if k.startswith("E")), key=lambda k: int(k[1:]))
    late = sorted((k for k in windows if k.startswith("L")), key=lambda k: int(k[1:]))
    spread = dict(formation=sorted({int(r["formation"]) for r in valid}),
                  natural_end=sorted({int(r["natural_end"]) for r in valid}),
                  selection=sorted(set(peaks)))
    notes.append(f"calibration: {len(valid)} of {len(rows)} generation steps formed a "
                 f"register; formation blocks observed {spread['formation']}, natural ends "
                 f"{spread['natural_end']}, peak blocks {spread['selection']}")
    return DepthProtocol(
        checkpoint=checkpoint, n_layers=int(n_layers), seam=seam, formation=formation,
        natural_end=natural_end, decline=decline, selection_block=selection,
        windows=windows, primary_early=early[-1] if early else None,
        primary_late=late[0] if late else None, plateau=plateau, window_length=int(length),
        settings=settings.row(), calibration=[dict(r) for r in rows], notes=notes)


def _tracer(ctx, *, layers: Sequence[int], steps: Sequence[int],
            attention_steps: Optional[Sequence[int]], batch_row: int = -1) -> CausalTracer:
    tracer = CausalTracer(ctx.driver.adapter, ctx.driver.transformer, direction=ctx.vstar,
                          layers=list(layers), steps=list(steps), channels=ctx.channels,
                          grid=ctx.driver.grid, cfg=ctx.cfg, batch_row=batch_row)
    tracer.attention_steps = None if attention_steps is None else {int(s) for s in
                                                                    attention_steps}
    # Full-tensor manipulation checks nowhere: at 1024px they are ~50 MB per hook call.
    tracer.diagnostic_steps = set()
    return tracer


def seam_of(ctx) -> Optional[int]:
    """FLUX's dual/single seam; None for a uniform stack such as PixArt's."""
    family = str(getattr(ctx.cfg.spec, "family", ""))
    if family in ("flux1", "flux2"):
        return len(getattr(ctx.driver.transformer, "transformer_blocks", []))
    return None


def calibrate_protocol(ctx, settings: MainSettings, units: Sequence[Tuple[int, str, int]],
                       *, log: Callable = print) -> DepthProtocol:
    """Stage 1: measure the natural interval on the calibration units and freeze it.

    Unmodified generations only, residual states only (no attention), at the edited steps.
    """
    layers = list(range(int(ctx.driver.n_layers)))
    rows: List[Dict[str, Any]] = []
    for prompt_id, prompt, seed in units:
        started = time.time()
        tracer = _tracer(ctx, layers=layers, steps=settings.edit_step_list(),
                         attention_steps=())
        trace, _ = run_traced_generation(ctx.driver, tracer, prompt_id=int(prompt_id),
                                         prompt=prompt, seed=int(seed),
                                         condition="calibration")
        found = measure_boundaries(trace, steps=settings.edit_step_list(), layers=layers,
                                   settings=settings)
        rows += [dict(r, prompt_id=int(prompt_id), seed=int(seed)) for r in found]
        ok = [r for r in found if r["valid"]]
        log(f"  calibration prompt {prompt_id} seed {seed}: {len(ok)}/{len(found)} steps "
            f"with a register; formation {sorted({r['formation'] for r in ok})}, end "
            f"{sorted({r['natural_end'] for r in ok})} ({time.time() - started:.0f}s)")
        del trace
    return protocol_from_calibration(rows, n_layers=len(layers), seam=seam_of(ctx),
                                     checkpoint=str(ctx.cfg.model), settings=settings)


# ================================================================ conditions
@dataclass(frozen=True)
class MainCondition:
    """One condition of the main run. ``window`` names a window of the protocol."""

    key: str
    group: str                     # reference | depth | removal | extension | control | check
    window: Optional[str] = None
    remove: bool = False
    operator: str = "matched"      # matched | random_direction | ordinary_positions |
    #                                in_distribution | none | hooks_only

    def label(self, protocol: DepthProtocol) -> str:
        """The formal name a figure prints."""
        span = protocol.windows.get(self.window) if self.window else None
        blocks = (f"blocks {span[0]}–{span[1]}" if span and span[0] != span[1]
                  else f"block {span[0]}" if span else "")
        if self.group == "reference":
            return "Unmodified"
        if self.group == "check":
            return "Hooks installed, no modification"
        if self.group == "removal" and self.window is None:
            return "Natural register removed"
        if self.group == "removal":
            return f"Removed; induced at {blocks}"
        if self.group == "extension":
            return f"Maintained through block {span[1]}"
        if self.operator == "random_direction":
            return f"Random direction, norm-matched ({blocks})"
        if self.operator == "ordinary_positions":
            return f"Non-register positions ({blocks})"
        if self.operator == "in_distribution":
            return f"v* at in-distribution strength ({blocks})"
        if self.window == "N":
            return f"Rewritten in place ({blocks})"
        return f"Induced at {blocks}"


def condition_catalog(protocol: DepthProtocol, *, include_check: bool = True
                      ) -> List[MainCondition]:
    """Every condition of the main run, in the order they are generated."""
    out = [MainCondition("reference", "reference", operator="none")]
    for name in protocol.early_windows():
        out.append(MainCondition(f"induce_{name}", "depth", name))
    if "N" in protocol.windows:
        out.append(MainCondition("induce_N", "depth", "N"))
    for name in protocol.late_windows():
        out.append(MainCondition(f"induce_{name}", "depth", name))
    out.append(MainCondition("remove", "removal", None, remove=True, operator="none"))
    if protocol.primary_early:
        out.append(MainCondition(f"remove_induce_{protocol.primary_early}", "removal",
                                 protocol.primary_early, remove=True))
    if protocol.primary_late:
        out.append(MainCondition(f"remove_induce_{protocol.primary_late}", "removal",
                                 protocol.primary_late, remove=True))
    out.append(MainCondition("extend", "extension", "extend"))
    if protocol.primary_early:
        for operator in ("random_direction", "ordinary_positions", "in_distribution"):
            out.append(MainCondition(f"control_{operator}", "control", protocol.primary_early,
                                     operator=operator))
    if include_check:
        out.append(MainCondition("hooks_only", "check", "N" if "N" in protocol.windows
                                 else protocol.primary_early, operator="hooks_only"))
    return out


# ================================================================== prompts
def evaluation_prompts(cache_dir, *, n_prompts: int, n_calibration: int,
                       exclude: Sequence[str] = (), seed: int = 1,
                       similarity_limit: float = 0.5) -> Tuple[List[str], List[str]]:
    """Held-out DiffusionDB prompts: (calibration, evaluation), disjoint from ``exclude``.

    A seeded, diversity-filtered draw (a different seed from the discovery draw), with every
    prompt that equals or nearly duplicates an excluded one removed, cached so a resumed run
    reads the same prompts.
    """
    from .prompt_sets import diffusiondb_prompts, similarity

    cache_dir = Path(cache_dir)
    needed = int(n_prompts) + int(n_calibration)
    draw = diffusiondb_prompts(cache_dir / f"q16_heldout_seed{seed}_{needed + 8}.json",
                               count=needed + 8, seed=seed)
    kept = [p for p in draw
            if all(p != q and similarity(p, q) < similarity_limit for q in exclude)]
    if len(kept) < needed:
        raise RuntimeError(f"only {len(kept)} held-out prompts survive the exclusion; "
                           f"{needed} are needed. Draw a larger pool or lower n_prompts.")
    return kept[:int(n_calibration)], kept[int(n_calibration):needed]


# ================================================================ unit runner
def unit_directory(root, prompt_id: int, seed: int) -> Path:
    return Path(root) / "units" / f"p{int(prompt_id):03d}_s{int(seed)}"


def run_root(model_root, edit_branches: str = "conditional") -> Path:
    """Where a model's main-run units live for a pass policy: the model's folder for
    ``'conditional'`` (the policy of a one-pass model such as FLUX.1-dev), and
    ``both_passes/`` under it for ``'both'``. The two runs share the model's frozen
    protocol and never overwrite each other, so a conditional-pass-only run stays on Drive
    as the ablation of a both-passes run."""
    if str(edit_branches) not in EDIT_BRANCHES:
        raise ValueError(f"edit_branches is one of {EDIT_BRANCHES}, not {edit_branches!r}")
    return Path(model_root) if str(edit_branches) == "conditional" \
        else Path(model_root) / "both_passes"


def unit_is_complete(root, prompt_id: int, seed: int, protocol: DepthProtocol,
                     edit_branches: Optional[str] = None) -> bool:
    """A unit is complete for this protocol's windows (and, when given, this pass policy;
    a unit written before the policy was recorded ran the conditional pass only)."""
    marker = unit_directory(root, prompt_id, seed) / "unit.json"
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text())
    except Exception:
        return False
    if edit_branches is not None and \
            payload.get("edit_branches", "conditional") != str(edit_branches):
        return False
    return bool(payload.get("complete")) and payload.get("windows") == {
        k: [int(a), int(b)] for k, (a, b) in protocol.windows.items()}


class NoRegister(RuntimeError):
    """The unit's unmodified run has no natural register to retime: a declared exclusion."""


@dataclass
class _UnitState:
    """What every condition of one unit is built from, read off its unmodified run."""

    carriers_at: Dict[int, List[int]]
    ordinary_at: Dict[int, List[int]]
    frozen: List[int]
    reference: LC.CleanStateReference
    targets: LC.RegisterTargets
    in_distribution_alpha: Dict[int, float]
    random_direction: torch.Tensor
    frozen_targets: Any


_LIFECYCLE_COLUMNS = (
    "layer", "carrier_projection", "carrier_cosine", "carrier_norm", "median_norm",
    "carrier_norm_ratio", "carrier_norm_vs_ordinary", "n_highnorm", "n_highnorm_frozen",
    "n_highnorm_new", "carrier_incoming_mass", "carrier_sink_strength",
    "carrier_sink_fraction", "carrier_is_sink", "n_sinks", "n_sinks_new",
    "carrier_qk_cosine")


def _prepare_unit(ctx, protocol: DepthProtocol, clean, settings: MainSettings,
                  prompt_id: int, seed: int) -> Tuple[_UnitState, List[Dict[str, Any]]]:
    layers = list(range(protocol.n_layers))
    base = LC.CleanStateReference(highnorm_ratio=settings.highnorm_ratio,
                                  alignment_quantile=settings.alignment_quantile)
    base.add(clean, steps=protocol.record_steps, layers=layers)
    carriers_at: Dict[int, List[int]] = {}
    for step in protocol.record_steps:
        bars = base.at(step, protocol.selection_block)
        if bars is None or int(bars.step) != int(step):
            continue
        ids = sorted(int(t) for t in bars.carriers)
        if len(ids) >= settings.min_carriers:
            carriers_at[int(step)] = ids
    if not carriers_at:
        raise NoRegister(f"the unmodified run has no token passing the register test at "
                         f"block {protocol.selection_block} at any recorded step "
                         f"(at least {settings.min_carriers} needed)")
    if not set(carriers_at) & {int(s) for s in protocol.edit_steps}:
        # Without a register at an EDITED step every induction is a no-op: the hooks fire
        # and write nothing, which would read as "no effect" rather than "nothing written".
        raise NoRegister(f"no register at block {protocol.selection_block} at any edited "
                         f"step {protocol.edit_steps} (registers only at steps "
                         f"{sorted(carriers_at)})")
    frozen = sorted(set().union(*carriers_at.values()))
    reference = LC.CleanStateReference(highnorm_ratio=settings.highnorm_ratio,
                                       alignment_quantile=settings.alignment_quantile,
                                       frozen=frozen)
    reference.add(clean, steps=protocol.record_steps, layers=layers)

    # Count-matched positions that never carry the register, ordinary-sized, not aligned.
    ordinary_at: Dict[int, List[int]] = {}
    for step, ids in carriers_at.items():
        obs = clean.at(step, protocol.selection_block)
        bars = reference.at(step, protocol.selection_block)
        norm, cosine = obs.norm.float(), obs.cosine.float()
        median = float(norm.median())
        pool = [t for t in range(int(norm.shape[0]))
                if t not in set(frozen) and float(norm[t]) < 2.0 * median
                and float(cosine[t]) < float(bars.ordinary_max_cosine)]
        rng = np.random.default_rng([int(settings.ordinary_positions_seed), int(prompt_id),
                                     int(seed), int(step)])
        ordinary_at[step] = sorted(int(t) for t in rng.choice(
            pool, size=min(len(ids), len(pool)), replace=False))

    # Natural-register-matched targets at every recipient block of every window.
    recipients = sorted({l for name, (a, b) in protocol.windows.items() if name != "removal"
                         for l in range(int(a), int(b) + 1)})
    targets = LC.RegisterTargets()
    match_rows: List[Dict[str, Any]] = []
    plateau = list(range(protocol.plateau[0], protocol.plateau[1] + 1))
    for step in protocol.edit_steps:
        ids = carriers_at.get(int(step))
        if not ids:
            continue
        try:
            by_block, rows = LC.calibrate_register_match(
                clean, step=int(step), natural_layers=plateau, recipient_layers=recipients,
                carriers=ids)
        except ValueError:
            continue
        for layer, target in by_block.items():
            targets.add(int(step), int(layer), target)
        match_rows += rows

    # The in-distribution control's projection: the loudest one the recipient block already
    # holds, at the input of the primary early window's first block.
    alpha: Dict[int, float] = {}
    early = protocol.window(protocol.primary_early) if protocol.primary_early else None
    if early is not None and early.first >= 1:
        for step in protocol.edit_steps:
            ids = carriers_at.get(int(step))
            natural = clean.at(int(step), protocol.selection_block)
            recipient = clean.at(int(step), early.first - 1)
            if not ids or natural is None or recipient is None:
                continue
            calibration = LC.calibrate_alpha(natural.projection, natural.norm,
                                             recipient.projection, recipient.norm, ids,
                                             mode="recipient_outlier")
            alpha[int(step)] = float(calibration.alpha_target) * float(
                settings.in_distribution_strength)

    frozen_targets = select_frozen_targets(
        clean, layer=protocol.selection_block, step=protocol.readout_step,
        direction=ctx.vstar, highnorm_ratio=settings.highnorm_ratio)
    state = _UnitState(carriers_at=carriers_at, ordinary_at=ordinary_at, frozen=frozen,
                       reference=reference, targets=targets, in_distribution_alpha=alpha,
                       random_direction=LC.unrelated_direction(
                           ctx.vstar, seed=settings.control_direction_seed),
                       frozen_targets=frozen_targets)
    return state, match_rows


def _plans(condition: MainCondition, protocol: DepthProtocol, state: _UnitState, ctx,
           edit_log: List, sites: List, *, all_rows: bool = False) -> List[EditPlan]:
    plans = _condition_plans(condition, protocol, state, ctx, edit_log, sites)
    for plan in plans:
        plan.all_batch_rows = bool(all_rows)
    return plans


def _condition_plans(condition: MainCondition, protocol: DepthProtocol, state: _UnitState,
                     ctx, edit_log: List, sites: List) -> List[EditPlan]:
    steps = protocol.edit_steps
    plans: List[EditPlan] = []
    point = InterventionPoint.BLOCK_INPUT
    if condition.operator == "hooks_only":
        window = protocol.window(condition.window)
        return [EditPlan(edit=LC.lifecycle_edit("sham", state.frozen, ctx.vstar),
                         point=point, layers=list(window.layers), steps=steps,
                         label=f"{condition.key}:hooks_only")]
    if condition.remove:
        # One suppression, shared exactly by every removal condition, built first so that a
        # block both would touch is cleaned before any write (the schedules never overlap).
        schedule = LC.suppression_schedule(protocol.removal_window(),
                                           n_layers=protocol.n_layers)
        suppressor = LC.Suppressor(ctx.vstar, mode="state",
                                   rule=LC.AlignmentRule(state.reference), sites=sites,
                                   operator="subtractive", original=state.frozen)
        plans.append(EditPlan(edit=suppressor.edit, point=point,
                              layers=schedule.hook_layers, steps=steps,
                              label=f"{condition.key}:remove"))
    if condition.window is None:
        return plans
    window = protocol.window(condition.window)
    if condition.operator == "in_distribution":
        edit = LC.lifecycle_edit("induce", state.carriers_at, ctx.vstar,
                                 alpha_target=state.in_distribution_alpha, records=edit_log)
    elif condition.operator == "random_direction":
        edit = LC.lifecycle_edit("induce_matched", state.carriers_at,
                                 state.random_direction, register_targets=state.targets,
                                 records=edit_log)
    elif condition.operator == "ordinary_positions":
        edit = LC.lifecycle_edit("induce_matched", state.ordinary_at, ctx.vstar,
                                 register_targets=state.targets, records=edit_log)
    else:
        edit = LC.lifecycle_edit("induce_matched", state.carriers_at, ctx.vstar,
                                 register_targets=state.targets, records=edit_log)
    plans.append(EditPlan(edit=edit, point=point, layers=list(window.layers), steps=steps,
                          label=f"{condition.key}:{condition.operator}"))
    return plans


def _positions(condition: MainCondition, state: _UnitState) -> Dict[int, List[int]]:
    return state.ordinary_at if condition.operator == "ordinary_positions" else state.carriers_at


def _readouts(trace, condition: MainCondition, protocol: DepthProtocol, state: _UnitState,
              settings: MainSettings, edit_log: List, sites: List,
              keys: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    layers = list(range(protocol.n_layers))
    positions = _positions(condition, state)
    out: Dict[str, List[Dict[str, Any]]] = {"lifecycle": [], "attention": [], "removal": [],
                                            "induction": []}
    for step in protocol.record_steps:
        ids = positions.get(int(step))
        if not ids:
            continue
        for row in LC.achieved_lifecycle(trace, step=int(step), layers=layers, carriers=ids,
                                         sink_threshold=settings.sink_threshold,
                                         highnorm_ratio=settings.highnorm_ratio):
            out["lifecycle"].append(dict(keys, step=int(step),
                                         **{k: row.get(k) for k in _LIFECYCLE_COLUMNS}))
    for step in protocol.attention_steps:
        for row in LC.attention_relocation_rows(trace, state.reference, step=int(step),
                                                layers=layers, carriers=state.frozen):
            out["attention"].append(dict(keys, **row))
    if condition.remove:
        interval = LC.Window("natural", protocol.formation, protocol.natural_end)
        schedule = LC.suppression_schedule(protocol.removal_window(),
                                           n_layers=protocol.n_layers)
        span = [l for l in range(protocol.formation - 1, protocol.natural_end + 3)
                if 0 <= l < protocol.n_layers]
        for step in protocol.edit_steps:
            at_step = [s for s in sites if int(s.step) == int(step)]
            received = LC.received_state_rows(trace, at_step, state.reference,
                                              step=int(step), layers=span,
                                              carriers=state.frozen)
            summary_a = LC.received_state_summary(received, interval=interval,
                                                  terminal=schedule.terminal)
            regrowth = LC.regrowth_relocation_rows(
                trace, state.reference, step=int(step),
                layers=[l for l in span if l <= protocol.natural_end],
                carriers=state.frozen)
            summary_b = LC.regrowth_relocation_summary(regrowth, window=interval)
            out["removal"].append(dict(
                keys, step=int(step),
                received_result=summary_a.get("result"),
                blocks_received=len(summary_a.get("blocks_received") or []),
                max_received_cosine_vs_natural=summary_a.get(
                    "max_received_cosine_vs_natural"),
                boundary_received=summary_a.get("boundary_received"),
                regrowth_outcome=summary_b.get("outcome"),
                blocks_original_regained=summary_b.get("blocks_original_regained"),
                unique_new_register_positions=summary_b.get(
                    "unique_new_register_positions"),
                max_original_projection_share=summary_b.get(
                    "max_original_projection_share")))
    if edit_log and condition.window is not None:
        window = protocol.window(condition.window)
        if condition.operator == "in_distribution":
            def target(record):
                a = state.in_distribution_alpha.get(int(record.step))
                return None if a is None else (a, None)
        else:
            def target(record):
                t = state.targets.at(int(record.step), int(record.layer))
                return None if t is None else t.at(int(record.token))
        for row in LC.induction_written(edit_log, target, window=window):
            out["induction"].append(dict(keys, **row))
    return out


def _thumbnail(image, size: int):
    small = image.copy()
    small.thumbnail((int(size), int(size)))
    return small


EDIT_BRANCHES = ("conditional", "both")


@contextmanager
def guidance_override(ctx, value: Optional[float]):
    """Sample with guidance scale ``value`` inside the block, and restore it after.

    On PixArt-Sigma a scale of 1.0 turns classifier-free guidance off: the sampler then
    runs one pass per step, as FLUX.1-dev does. ``None`` changes nothing.
    """
    configs = {}
    for cfg in (getattr(ctx, "cfg", None), getattr(getattr(ctx, "driver", None), "cfg", None)):
        if cfg is not None and hasattr(cfg, "guidance_scale"):
            configs[id(cfg)] = cfg
    before = {key: cfg.guidance_scale for key, cfg in configs.items()}
    try:
        if value is not None:
            for cfg in configs.values():
                cfg.guidance_scale = float(value)
        yield
    finally:
        for key, cfg in configs.items():
            cfg.guidance_scale = before[key]


def _settings(protocol: DepthProtocol) -> MainSettings:
    return MainSettings(**{k: v for k, v in protocol.settings.items()
                           if k in MainSettings.__dataclass_fields__})


def run_unit(ctx, protocol: DepthProtocol, *, prompt_id: int, prompt: str, seed: int,
             root, conditions: Optional[Sequence[MainCondition]] = None,
             lpips_net=None, clip=None, log: Callable = print,
             edit_branches: Optional[str] = None,
             guidance: Optional[float] = None) -> Dict[str, Any]:
    """Stage 2 for one (prompt, seed): every condition, its records, and a completion mark.

    Idempotent: a unit whose ``unit.json`` says complete for this protocol's windows is
    skipped; anything else is rerun from the start and overwritten.

    ``edit_branches`` and ``guidance`` override the protocol's setting and the checkpoint's
    guidance scale for this unit (the unmodified run included); they exist for the
    guidance diagnostic, which writes to its own root. Both are recorded in ``unit.json``
    and ``images.csv``.
    """
    with guidance_override(ctx, guidance):
        return _run_unit(ctx, protocol, prompt_id=prompt_id, prompt=prompt, seed=seed,
                         root=root, conditions=conditions, lpips_net=lpips_net, clip=clip,
                         log=log, edit_branches=edit_branches)


def _run_unit(ctx, protocol: DepthProtocol, *, prompt_id: int, prompt: str, seed: int,
              root, conditions, lpips_net, clip, log, edit_branches) -> Dict[str, Any]:
    import pandas as pd
    from .control_surface import image_metrics

    settings = _settings(protocol)
    branches = str(edit_branches or settings.edit_branches)
    if branches not in EDIT_BRANCHES:
        raise ValueError(f"edit_branches is one of {EDIT_BRANCHES}, not {branches!r}")
    if unit_is_complete(root, prompt_id, seed, protocol, edit_branches=branches):
        return dict(status="skipped", prompt_id=int(prompt_id), seed=int(seed))
    guidance_used = getattr(getattr(ctx.driver, "cfg", None), "guidance_scale", None)
    guidance_used = float("nan") if guidance_used is None else float(guidance_used)
    directory = unit_directory(root, prompt_id, seed)
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "thumbnails").mkdir(parents=True, exist_ok=True)
    layers = list(range(protocol.n_layers))
    conditions = list(conditions if conditions is not None
                      else condition_catalog(protocol, include_check=False))
    started = time.time()
    keys = dict(checkpoint=protocol.checkpoint, prompt_id=int(prompt_id), seed=int(seed))

    # ------------------------------------------------ the unmodified generation
    tracer = _tracer(ctx, layers=layers, steps=protocol.record_steps,
                     attention_steps=protocol.attention_steps)
    # One full [N, C] slice, at the selection block and the readout step only: the frozen
    # target record the edit installer is handed is read from it. Everything else the unit
    # needs is in the per-token statistics.
    clean, _ = run_traced_generation(
        ctx.driver, tracer, prompt_id=int(prompt_id), prompt=prompt, seed=int(seed),
        condition="reference", save_image=True,
        probes=[(InterventionPoint.BLOCK_INPUT, [protocol.selection_block],
                 [protocol.readout_step])])
    try:
        state, match_rows = _prepare_unit(ctx, protocol, clean, settings, prompt_id, seed)
    except NoRegister as reason:
        # Declared before the run: a unit whose unmodified generation forms no register has
        # nothing to retime. It is recorded, counted in the appendix and never retried.
        (directory / "unit.json").write_text(json.dumps(dict(
            complete=True, status="no_register", reason=str(reason),
            checkpoint=protocol.checkpoint, prompt_id=int(prompt_id), seed=int(seed),
            prompt=prompt, windows={k: [int(a), int(b)] for k, (a, b) in
                                    protocol.windows.items()},
            total_seconds=round(time.time() - started, 1)), indent=2))
        log(f"    excluded: {reason}")
        return dict(status="no_register", prompt_id=int(prompt_id), seed=int(seed),
                    total_seconds=round(time.time() - started, 1))
    overlap: List[Dict[str, Any]] = []
    if branches == "both":
        # 'both' edits the unconditional pass at the conditional pass's register positions.
        # One more unmodified generation, recording the unconditional row, says on every
        # unit whether those are its register positions too.
        tick = time.time()
        unconditional = _branch_trace(ctx, protocol, settings, prompt_id=prompt_id,
                                      prompt=prompt, seed=seed, batch_row=0)
        overlap = [dict(keys, **r) for r in _overlap_rows(
            protocol, _branch_reference(clean, protocol, settings), unconditional)]
        del unconditional
        overlap_seconds = round(time.time() - tick, 1)
    boundaries = [dict(keys, **r) for r in measure_boundaries(
        clean, steps=protocol.edit_steps, layers=layers, settings=settings)]
    sensitivity = [dict(keys, **r) for r in LC.threshold_sensitivity(
        clean, step=protocol.readout_step, layers=layers,
        carriers=state.carriers_at.get(protocol.readout_step, state.frozen),
        highnorm_ratio=settings.highnorm_ratio,
        alignment_quantile=settings.alignment_quantile,
        sink_threshold=settings.sink_threshold,
        interval=(protocol.formation, protocol.natural_end))]
    audit = []
    for step in protocol.edit_steps:
        for row in LC.rule_selectivity(clean, LC.AlignmentRule(state.reference), step=step,
                                       layers=list(range(protocol.formation - 2,
                                                         protocol.natural_end + 1)),
                                       direction=ctx.vstar, carriers=state.frozen):
            if isinstance(row.get("selected"), int):
                audit.append(dict(keys, step=int(step), hook_at_block=int(row["layer"]) + 1,
                                  selected=int(row["selected"]),
                                  share_of_image=float(row["share_of_image"])))
    worst_share = max((r["share_of_image"] for r in audit), default=float("nan"))

    records: Dict[str, List[Dict[str, Any]]] = {"lifecycle": [], "attention": [],
                                                "removal": [], "induction": [], "images": [],
                                                "edits": []}
    reference_condition = MainCondition("reference", "reference", operator="none")
    for name, rows in _readouts(clean, reference_condition, protocol, state, settings, [],
                                [], dict(keys, condition="reference")).items():
        records[name] += rows
    clean_image = clean.image
    if clean_image is not None:
        clean_image.save(directory / "images" / "reference.png")
        _thumbnail(clean_image, settings.thumbnail_size).save(
            directory / "thumbnails" / "reference.jpg", quality=92)
    timings = {"reference": round(time.time() - started, 1)}

    # ------------------------------------------------ every condition
    for condition in conditions:
        if condition.group == "reference":
            continue
        tick = time.time()
        edit_log: List = []
        sites: List = []
        plans = _plans(condition, protocol, state, ctx, edit_log, sites,
                       all_rows=branches == "both")
        tracer = _tracer(ctx, layers=layers, steps=protocol.record_steps,
                         attention_steps=protocol.attention_steps)
        trace, stats = run_traced_generation(ctx.driver, tracer, prompt_id=int(prompt_id),
                                             prompt=prompt, seed=int(seed),
                                             condition=condition.key, plans=plans,
                                             targets=state.frozen_targets, save_image=True)
        ckeys = dict(keys, condition=condition.key)
        for name, rows in _readouts(trace, condition, protocol, state, settings, edit_log,
                                    sites, ckeys).items():
            records[name] += rows
        perturbation = [r.perturbation_l2 for r in edit_log]
        records["edits"].append(dict(ckeys, edit_calls=int(stats.edit_calls),
                                     perturbation_energy=float(stats.perturbation_energy),
                                     n_writes=len(edit_log),
                                     mean_perturbation_l2=(float(np.mean(perturbation))
                                                           if perturbation else float("nan"))))
        image = trace.image
        row = dict(ckeys, label=condition.label(protocol), group=condition.group,
                   window=condition.window, edit_branches=branches,
                   guidance_scale=guidance_used)
        if image is not None and clean_image is not None:
            metrics = image_metrics(clean_image, image, prompt=prompt, lpips_net=lpips_net,
                                    clip=clip)
            if "clip_prompt_similarity_treated" in metrics:
                metrics["clip_prompt_similarity_change"] = (
                    metrics["clip_prompt_similarity_treated"]
                    - metrics["clip_prompt_similarity_clean"])
            row.update(metrics)
            image.save(directory / "images" / f"{condition.key}.png")
            _thumbnail(image, settings.thumbnail_size).save(
                directory / "thumbnails" / f"{condition.key}.jpg", quality=92)
        records["images"].append(row)
        timings[condition.key] = round(time.time() - tick, 1)
        log(f"    {condition.key:<32} {timings[condition.key]:>6.1f}s  "
            f"{int(stats.edit_calls)} edit calls")
        del trace

    for name, rows in records.items():
        pd.DataFrame(rows).to_csv(directory / f"{name}.csv", index=False)
    pd.DataFrame(boundaries).to_csv(directory / "boundaries.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(directory / "sensitivity.csv", index=False)
    pd.DataFrame(audit).to_csv(directory / "rule_audit.csv", index=False)
    if overlap:
        pd.DataFrame(overlap).to_csv(directory / "branch_overlap.csv", index=False)
        timings["branch_overlap"] = overlap_seconds
    pd.DataFrame([dict(keys, **r) for r in match_rows]).to_csv(
        directory / "register_match.csv", index=False)
    summary = dict(
        complete=True, status="done", checkpoint=protocol.checkpoint, prompt_id=int(prompt_id),
        seed=int(seed), prompt=prompt, edit_branches=branches, guidance_scale=guidance_used,
        conditional_register_in_unconditional_pass=_nan_min(
            r["conditional_in_unconditional"] for r in overlap),
        windows={k: [int(a), int(b)] for k, (a, b) in protocol.windows.items()},
        conditions=[c.key for c in conditions],
        carriers_per_step={int(k): len(v) for k, v in state.carriers_at.items()},
        edit_steps_with_register=[s for s in protocol.edit_steps if s in state.carriers_at],
        rule_worst_share_of_image=worst_share,
        rule_within_ceiling=bool(worst_share <= settings.rule_max_share_of_image)
        if worst_share == worst_share else None,
        timings_seconds=timings, total_seconds=round(time.time() - started, 1))
    (directory / "unit.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    del clean
    return dict(status="done", **{k: summary[k] for k in ("prompt_id", "seed",
                                                          "total_seconds")})


# ================================================================ collection
def collect_units(root) -> Dict[str, "Any"]:
    """Every complete unit under ``root``, concatenated per table."""
    import pandas as pd

    tables: Dict[str, List[Any]] = {}
    units = []
    for directory in sorted((Path(root) / "units").glob("p*_s*")):
        marker = directory / "unit.json"
        if not marker.exists():
            continue
        payload = json.loads(marker.read_text())
        if not payload.get("complete"):
            continue
        units.append(payload)
        if payload.get("status", "done") != "done":
            continue                          # a declared exclusion: listed, not pooled
        for csv in directory.glob("*.csv"):
            try:
                frame = pd.read_csv(csv)
            except Exception:
                continue
            if not frame.empty:
                # Every row names its unit. Older units wrote register_match.csv without
                # the keys; they are filled in from unit.json, which always has them.
                for key in ("checkpoint", "prompt_id", "seed"):
                    if key not in frame and key in payload:
                        frame.insert(0, key, payload[key])
                tables.setdefault(csv.stem, []).append(frame)
    out = {name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()}
    out["units"] = pd.DataFrame([{k: v for k, v in u.items()
                                  if not isinstance(v, (dict, list))} for u in units])
    return out


# ================================================================ guidance diagnostic
def branch_register_overlap(ctx, protocol: DepthProtocol, *, prompt_id: int, prompt: str,
                            seed: int, guidance: Optional[float] = None
                            ) -> List[Dict[str, Any]]:
    """Where each classifier-free-guidance pass holds its register, at every edited step.

    Two unmodified generations of the same prompt and seed, one recording the conditional
    batch row and one the unconditional row, each put through the register test at the
    selection block. With ``edit_branches='both'`` the unconditional pass is edited at the
    conditional pass's register positions, so this says whether those are its register
    positions too (``conditional_in_unconditional``) and how large its tokens there are.
    A model sampled in one pass records the same row twice (``single_pass``).
    """
    settings = _settings(protocol)
    with guidance_override(ctx, guidance):
        conditional = _branch_trace(ctx, protocol, settings, prompt_id=prompt_id,
                                    prompt=prompt, seed=seed, batch_row=-1)
        unconditional = _branch_trace(ctx, protocol, settings, prompt_id=prompt_id,
                                      prompt=prompt, seed=seed, batch_row=0)
    keys = dict(checkpoint=protocol.checkpoint, prompt_id=int(prompt_id), seed=int(seed))
    return [dict(keys, **r) for r in _overlap_rows(protocol, conditional, unconditional)]


def _nan_min(values) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and float(v) == float(v)]
    return min(finite) if finite else None


def _branch_reference(trace, protocol: DepthProtocol, settings: MainSettings):
    """``(trace, register-test reference)`` at the selection block and the edited steps."""
    reference = LC.CleanStateReference(highnorm_ratio=settings.highnorm_ratio,
                                       alignment_quantile=settings.alignment_quantile)
    reference.add(trace, steps=[int(s) for s in protocol.edit_steps],
                  layers=[int(protocol.selection_block)])
    return trace, reference


def _branch_trace(ctx, protocol: DepthProtocol, settings: MainSettings, *, prompt_id: int,
                  prompt: str, seed: int, batch_row: int):
    """One unmodified generation recording batch row ``batch_row`` at the selection block."""
    tracer = _tracer(ctx, layers=[int(protocol.selection_block)],
                     steps=[int(s) for s in protocol.edit_steps], attention_steps=(),
                     batch_row=batch_row)
    trace, _ = run_traced_generation(ctx.driver, tracer, prompt_id=int(prompt_id),
                                     prompt=prompt, seed=int(seed),
                                     condition=f"reference_row{int(batch_row)}")
    return _branch_reference(trace, protocol, settings)


def _overlap_rows(protocol: DepthProtocol, conditional, unconditional) -> List[Dict[str, Any]]:
    block = int(protocol.selection_block)

    def ratio(obs, bars, ids):
        ids = sorted(int(t) for t in ids if int(t) < int(obs.norm.shape[0]))
        if not ids:
            return float("nan")
        return float(obs.norm.float()[ids].mean()) / max(float(bars.median_norm), 1e-9)

    rows = []
    for step in [int(s) for s in protocol.edit_steps]:
        (trace_c, ref_c), (trace_u, ref_u) = conditional, unconditional
        bars_c, bars_u = ref_c.at(step, block), ref_u.at(step, block)
        obs_c, obs_u = trace_c.at(step, block), trace_u.at(step, block)
        if (bars_c is None or bars_u is None or obs_c is None or obs_u is None
                or int(bars_c.step) != step or int(bars_u.step) != step):
            continue
        c, u = set(bars_c.carriers), set(bars_u.carriers)
        rows.append(dict(
            step=step, block=block, n_conditional=len(c), n_unconditional=len(u),
            n_shared=len(c & u), jaccard=len(c & u) / max(len(c | u), 1),
            conditional_in_unconditional=(len(c & u) / len(c)) if c else float("nan"),
            conditional_norm_ratio=ratio(obs_c, bars_c, c),
            unconditional_norm_ratio=ratio(obs_u, bars_u, u),
            unconditional_norm_ratio_at_conditional_positions=ratio(obs_u, bars_u, c),
            median_norm_ratio=float(bars_u.median_norm) / max(float(bars_c.median_norm), 1e-9),
            single_pass=bool(torch.equal(obs_c.norm, obs_u.norm))))
    return rows


def register_profile(pooled: Mapping[str, Any], protocols: Mapping[str, DepthProtocol]):
    """One row per model, from the records the main run already wrote: how many tokens
    the register has, how large they are, how much of all attention they receive, how
    large the matched write is at the primary early window, and how many passes the
    sampler makes per step (classifier-free guidance runs two).

    Every value is a median over prompt-seed pairs (and edited steps, where recorded per
    step). Attention is the absolute share of each query's attention that lands on the
    register tokens, averaged over heads and queries, at the plateau blocks of the readout
    step; on FLUX.1-dev the key sequence includes the 512 text tokens, on PixArt-Sigma it
    is the image alone (text enters through a separate cross-attention).
    """
    import pandas as pd

    match = pooled.get("register_match")
    life = pooled.get("lifecycle")
    units = pooled.get("units")
    rows = []
    for checkpoint, p in protocols.items():
        settings = _settings(p)
        row: Dict[str, Any] = dict(checkpoint=checkpoint, n_layers=p.n_layers,
                                   formation=p.formation, natural_end=p.natural_end,
                                   window_length=p.window_length)
        guidance = settings.guidance
        if units is not None and not units.empty and "guidance_scale" in units:
            here = units[units["checkpoint"] == checkpoint]["guidance_scale"].dropna()
            guidance = float(here.median()) if len(here) else guidance
        row["guidance_scale"] = guidance
        row["passes_per_step"] = (2 if "pixart" in checkpoint and guidance is not None
                                  and float(guidance) > 1.0 else 1)
        if match is not None and not match.empty and "checkpoint" in match:
            m = match[match["checkpoint"] == checkpoint]
            per_step = m.drop_duplicates(["prompt_id", "seed", "step"])
            for column, name in (("n_natural_carriers", "register_tokens"),
                                 ("natural_norm_ratio", "register_norm_over_median"),
                                 ("natural_projection_ratio",
                                  "register_projection_over_median"),
                                 ("natural_cosine", "register_cosine")):
                if column in per_step:
                    row[name] = float(per_step[column].median())
            early = p.windows.get(p.primary_early) if p.primary_early else None
            if early is not None and "layer" in m:
                at = m[(m["layer"] >= early[0]) & (m["layer"] <= early[1])]
                for column, name in (("perturbation_vs_median", "early_write_over_median"),
                                     ("norm_vs_recipient_max",
                                      "early_target_over_largest_clean_token")):
                    if column in at:
                        row[name] = float(at[column].median())
        if life is not None and not life.empty and "checkpoint" in life:
            here = life[(life["checkpoint"] == checkpoint) & (life["step"] == p.readout_step)]
            plateau = here[(here["layer"] >= p.plateau[0]) & (here["layer"] <= p.plateau[1])
                           & (here["condition"] == "reference")]
            if "carrier_incoming_mass" in plateau and plateau["carrier_incoming_mass"].notna().any():
                share = float(plateau["carrier_incoming_mass"].median())
                row["attention_share_to_register"] = share
                tokens = row.get("register_tokens")
                if tokens:
                    row["attention_share_per_register_token"] = share / float(tokens)
        rows.append(row)
    return pd.DataFrame(rows)


def late_state_table(pooled: Mapping[str, Any], protocols: Mapping[str, DepthProtocol], *,
                     step: Optional[int] = None):
    """Per model and condition: is a register-sized state still at the edited positions
    after the natural end, does it draw attention there, and how much the image changed.

    Each block reads a token through a LayerNorm, so what a block adds to a token does not
    grow with the token's size. The block that ends the natural register can therefore
    remove about as much as the natural register holds. A write that adds to the natural
    register (an early write at the register positions) can leave a surplus that outlives
    the natural end, and a late write or an extension puts the state there directly, into
    blocks that never hold a register in an unmodified run. This table says, for every
    condition, whether that happened -- read from the lifecycle records, at the readout
    step (or ``step``), at the positions each condition edits (the non-register control's
    own positions for that control; the register positions otherwise).

    Columns (medians over prompt-seed pairs): ``norm_ratio_last_block`` (token norm over
    the block's median, at the last block), ``norm_ratio_after_end`` (the largest over the
    blocks after the natural end), ``attention_share_after_end`` (the share of all
    attention these positions receive, averaged over the blocks after the natural end),
    ``new_sinks_after_end`` (attention sinks at other positions, per block), and
    ``lpips``.
    """
    import pandas as pd

    life = pooled.get("lifecycle")
    images = pooled.get("images")
    if life is None or life.empty:
        return pd.DataFrame()
    rows = []
    for checkpoint, p in protocols.items():
        at_step = int(p.readout_step if step is None else step)
        here = life[(life["checkpoint"] == checkpoint) & (life["step"] == at_step)]
        if here.empty:
            continue
        after = here[here["layer"] > p.natural_end]
        last = here[here["layer"] == p.n_layers - 1]
        order = [c.key for c in condition_catalog(p, include_check=False)]
        for condition in [c for c in order if c in set(here["condition"])]:
            def per_unit(frame, column, how):
                block = frame[frame["condition"] == condition]
                if column not in block or block[column].dropna().empty:
                    return float("nan")
                grouped = block.groupby(["prompt_id", "seed"])[column]
                return float(getattr(grouped, how)().median())

            row = dict(checkpoint=checkpoint, condition=condition,
                       positions=("other" if condition == "control_ordinary_positions"
                                  else "register"),
                       norm_ratio_last_block=per_unit(last, "carrier_norm_ratio", "mean"),
                       norm_ratio_after_end=per_unit(after, "carrier_norm_ratio", "max"),
                       attention_share_after_end=per_unit(after, "carrier_incoming_mass",
                                                          "mean"),
                       new_sinks_after_end=per_unit(after, "n_sinks_new", "mean"))
            if images is not None and not images.empty and "lpips" in images:
                values = images[(images["checkpoint"] == checkpoint)
                                & (images["condition"] == condition)]["lpips"].dropna()
                row["lpips"] = float(values.median()) if len(values) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def policy_table(images_by_policy: Mapping[str, Any], *, metric: str = "lpips"):
    """``{policy: images table}`` -> one row per condition, one column per policy: the
    mean of ``metric`` over the prompt-seed pairs every policy ran (so the columns are
    paired), with the number of pairs."""
    import pandas as pd

    frames = {k: v for k, v in images_by_policy.items() if v is not None and not v.empty
              and metric in v}
    if not frames:
        return pd.DataFrame()
    keys = None
    for frame in frames.values():
        units = set(zip(frame["prompt_id"], frame["seed"]))
        keys = units if keys is None else keys & units
    out = {}
    for policy, frame in frames.items():
        paired = frame[[(int(p), int(s)) in keys for p, s in zip(frame["prompt_id"],
                                                                  frame["seed"])]]
        out[policy] = paired.groupby("condition")[metric].mean()
    table = pd.DataFrame(out)
    table.insert(0, "n_pairs", len(keys or ()))
    return table.reset_index()


def effect_table(images, *, metrics: Sequence[str] = ("lpips", "clip_image_similarity",
                                                       "clip_prompt_similarity_change"),
                 iterations: int = 2000, seed: int = 0):
    """Per condition and metric: the mean over units with a prompt-clustered 95% interval.

    Every metric is already a within-unit comparison with the unit's unmodified image, so
    this is the paired effect of each condition.
    """
    import pandas as pd
    from .causal_stats import bootstrap_mean

    rows = []
    if images is None or images.empty:
        return pd.DataFrame()
    for (checkpoint, condition), block in images.groupby(["checkpoint", "condition"],
                                                         sort=False):
        for metric in metrics:
            if metric not in block or block[metric].notna().sum() == 0:
                continue
            estimate = bootstrap_mean(block, metric, iterations=iterations, seed=seed)
            rows.append(dict(checkpoint=checkpoint, condition=condition,
                             label=block["label"].iloc[0] if "label" in block else condition,
                             group=block["group"].iloc[0] if "group" in block else "",
                             window=block["window"].iloc[0] if "window" in block else None,
                             metric=metric, mean=estimate.value, ci_low=estimate.ci_low,
                             ci_high=estimate.ci_high, n_units=estimate.n_units,
                             n_prompts=estimate.n_clusters))
    return pd.DataFrame(rows)


def paired_contrast(images, *, treatment: str, reference: str, metric: str = "lpips",
                    checkpoint: Optional[str] = None, iterations: int = 2000, seed: int = 0):
    """``treatment - reference`` on a per-unit metric, with a prompt-clustered interval."""
    from .causal_stats import paired_effect

    frame = images if checkpoint is None else images[images["checkpoint"] == checkpoint]
    return paired_effect(frame, metric, treatment=treatment, reference=reference,
                         iterations=iterations, seed=seed)


def writer_gain_table(lifecycle, protocol: DepthProtocol, *, step: Optional[int] = None):
    """What the natural writer adds along v* at the carriers, per condition and unit.

    The change in the carriers' mean v* projection across the formation block (its output
    minus its input), divided by the same change in the unit's unmodified run. 1 means the
    writer wrote as it naturally does; near 0 means it passed over those tokens.
    """
    import pandas as pd

    if lifecycle is None or lifecycle.empty:
        return pd.DataFrame()
    step = protocol.readout_step if step is None else int(step)
    frame = lifecycle[(lifecycle["checkpoint"] == protocol.checkpoint)
                      & (lifecycle["step"] == step)
                      & lifecycle["layer"].isin([protocol.formation - 1, protocol.formation])]
    pivot = frame.pivot_table(index=["prompt_id", "seed", "condition"], columns="layer",
                              values="carrier_projection", aggfunc="first").reset_index()
    if protocol.formation not in pivot or protocol.formation - 1 not in pivot:
        return pd.DataFrame()
    pivot["gain"] = pivot[protocol.formation] - pivot[protocol.formation - 1]
    base = pivot[pivot["condition"] == "reference"][["prompt_id", "seed", "gain"]].rename(
        columns={"gain": "reference_gain"})
    merged = pivot.merge(base, on=["prompt_id", "seed"], how="left")
    merged["writer_gain_vs_reference"] = merged["gain"] / merged["reference_gain"].where(
        merged["reference_gain"].abs() > 1e-9)
    merged["checkpoint"] = protocol.checkpoint
    return merged[["checkpoint", "prompt_id", "seed", "condition", "gain", "reference_gain",
                   "writer_gain_vs_reference"]]
