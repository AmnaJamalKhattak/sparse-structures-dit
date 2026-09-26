"""Executable answers to six causal questions about the register mechanism.

Each ``run_qN`` performs the whole experiment: it runs the clean pass for one
prompt-seed unit, freezes the targets from it, replays the identical generation
once per condition, extracts the predeclared endpoints, and returns a tidy table
plus a plain-language verdict.  The condition grids include their controls, and
every grid is a data structure a caller can shorten (for a debug run) or extend
(for a new control).

The six experiments, and the arrow each closes in the causal model
``writer -> c* -> v* -> key geometry -> sink routing -> generation``:

``run_q1``  natural-register removal              -> is the routing anchor causal?
``run_q2``  live dominant-channel suppression      -> c* -> v* and routing
``run_q3``  direction destruction and recovery     -> is the register maintained?
``run_q4``  reciprocal upstream-feature patching   -> what selects the positions?
``run_q5``  residual-to-key sufficiency ladder     -> what makes v* sufficient?
``run_q6``  dissolution: suppress or refresh late competition -> what ends the state?
"""
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from . import endpoints as EP
from . import diagnostics as DG
from .adapters import InterventionPoint
from .causal_engine import (CausalTracer, EditContext, EditPlan, FrozenTargets, GenerationDriver,
                            RunStats, Trace, run_traced_generation, select_frozen_targets)
from .causal_ops import (add_matched_energy, clamp_norm, refresh_direction, refresh_parallel_component,
                         remove_direction, remove_matched_energy, replace_tokens,
                         rotate_toward_channel_preserve_norm, scale_channel, scale_direction,
                         transplant)
from .causal_records import LayerHeadMeasurement


# ------------------------------------------------------------------ context
@dataclass
class QuestionContext:
    """Everything the six runners share, resolved once per checkpoint."""

    cfg: Any                                   # SweepConfig
    driver: GenerationDriver
    vstar: torch.Tensor
    writer_layer: int
    register_layers: Tuple[int, ...]
    dissolution_layers: Tuple[int, ...]
    dominant_channel: int
    competitor_channel: int
    control_channel: int
    step: int
    prompts: Sequence[str] = ()
    seeds: Sequence[int] = ()
    prompt_split: str = "confirmatory"
    percentile: float = 99.0
    topk: int = 8
    highnorm_ratio: float = 3.0
    output_dir: Optional[Path] = None
    progress: bool = True
    # The whole writer range, not only the layer taken from its end. A range can
    # straddle an architectural boundary (FLUX.1's runs between its dual and
    # single blocks), and a question that needs a particular kind of block has to
    # be able to look inside the range rather than accept whichever end it landed
    # on. Defaults to the writer layer alone.
    writer_range: Tuple[int, int] = ()
    vstar_metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_artifact(cls, cfg, artifact, vstar, *, driver: Optional[GenerationDriver] = None,
                      step: Optional[int] = None, prompts: Optional[Sequence[str]] = None,
                      seeds: Optional[Sequence[int]] = None, prompt_split: str = "confirmatory",
                      output_dir=None, progress: bool = True, **overrides) -> "QuestionContext":
        """Build a context from a frozen discovery artifact.

        Layer ranges and channels come from discovery and are never re-chosen
        here, which is what keeps the confirmatory phase confirmatory.
        """
        driver = driver or GenerationDriver(cfg)
        ranges = artifact.layer_ranges
        last = driver.n_layers - 1
        writer = min(int(ranges.writer[-1]), last)
        writer_range = (min(int(ranges.writer[0]), last), writer)
        register = tuple(range(min(int(ranges.register[0]), last), min(int(ranges.register[1]), last) + 1))
        dissolution = tuple(range(min(int(ranges.dissolution[0]), last),
                                  min(int(ranges.dissolution[1]), last) + 1))
        width = int(vstar.numel())
        return cls(
            cfg=cfg, driver=driver, vstar=torch.as_tensor(vstar).float().flatten(),
            writer_layer=writer, writer_range=writer_range,
            register_layers=register or (writer,),
            dissolution_layers=dissolution or (last,),
            dominant_channel=min(int(artifact.dominant_register_channel.selected), width - 1),
            competitor_channel=min(int(artifact.late_growing_competitor.selected), width - 1),
            control_channel=min(int(artifact.massive_unspecific_control.selected), width - 1),
            step=int(step if step is not None else (cfg.capture_steps or [0])[0]),
            prompts=list(prompts if prompts is not None else cfg.prompts),
            seeds=[int(s) for s in (seeds if seeds is not None else cfg.seeds)],
            prompt_split=prompt_split, output_dir=None if output_dir is None else Path(output_dir),
            progress=progress,
            vstar_metadata=dict(source=getattr(artifact, "vstar_file", "frozen artifact"),
                                explained_variance=getattr(artifact, "explained_variance", None),
                                fitting_population=getattr(artifact, "fitting_population", None),
                                sign_convention=getattr(artifact, "sign_convention", None)),
            **{"highnorm_ratio": float(getattr(cfg, "highnorm_ratio", 3.0)), **overrides},
        )

    @property
    def intervention_layer(self) -> int:
        """First clear register layer: where these experiments intervene."""
        return int(self.register_layers[0])

    def layer_exposing(self, points: Sequence[Any],
                       within: Optional[Tuple[int, int]] = None) -> int:
        """The latest layer in a range that exposes every one of ``points``.

        A layer range is a claim about where something happens in depth, not about
        what the blocks there are made of, and the two can disagree: FLUX.1's
        writer range ends one layer past the dual/single boundary, where a fused
        block no longer has a separate feed-forward stage to read.  Walking back
        from the end finds the closest layer that can actually answer the
        question, rather than reporting the architecture as incapable.

        Falls back to the range's end when nothing in it qualifies, so the caller
        still gets the layer it would have used and the usual unsupported
        reporting takes over.
        """
        low, high = within or self.writer_range or (self.writer_layer, self.writer_layer)  # noqa: E501
        low, high = int(min(low, high)), int(max(low, high))
        high = min(high, self.driver.n_layers - 1)
        refs = self.driver.adapter.layers(self.driver.transformer)
        wanted = [p for p in points]
        for layer in range(high, max(low, 0) - 1, -1):
            caps = self.driver.adapter.intervention_capabilities(refs[layer])
            if all(caps[p].supported for p in wanted):
                return int(layer)
        return int(high)

    @property
    def observe_layers(self) -> Tuple[int, ...]:
        """Every layer whose downstream response the questions read."""
        layers = {self.writer_layer, self.intervention_layer}
        layers.update(self.register_layers)
        layers.update(self.dissolution_layers)
        layers.update(l for l in (self.writer_layer - 1, self.writer_layer - 2) if l >= 0)
        return tuple(sorted(l for l in layers if 0 <= l < self.driver.n_layers))

    @property
    def downstream_layers(self) -> Tuple[int, ...]:
        """Layers strictly after the intervention: where a consequence can appear."""
        return tuple(l for l in self.observe_layers if l >= self.intervention_layer)

    @property
    def channels(self) -> Tuple[int, ...]:
        return (self.dominant_channel, self.competitor_channel, self.control_channel)

    def units(self) -> List[Tuple[int, str, int]]:
        return [(pid, prompt, int(seed)) for pid, prompt in enumerate(self.prompts)
                for seed in self.seeds]

    def tracer(self, **kwargs) -> CausalTracer:
        return CausalTracer(self.driver.adapter, self.driver.transformer, direction=self.vstar,
                            layers=self.observe_layers, steps=[self.step],
                            channels=self.channels, grid=self.driver.grid, cfg=self.cfg, **kwargs)

    def say(self, message: str) -> None:
        if self.progress:
            print(message, flush=True)

    def create_run_directory(self, run_id: str, root: Path | str = ".") -> Path:
        """Create the non-overwriting confirmatory result/provenance layout."""
        from .provenance import create_run_layout
        spec = self.cfg.spec
        metadata = dict(
            notebook="notebooks/causal_mechanism.ipynb",
            model=getattr(spec, "key", repr(spec)), checkpoint=getattr(spec, "repo_id", None),
            prompt_set=self.prompt_split, prompts=list(self.prompts), seeds=list(self.seeds),
            denoising_steps=[self.step], writer_layer=self.writer_layer,
            intervention_layer=self.intervention_layer,
            register_layers=list(self.register_layers), dissolution_layers=list(self.dissolution_layers),
            vstar={**dict(self.vstar_metadata), "dimensions": int(self.vstar.numel())},
            dominant_channel=self.dominant_channel, competing_channel=self.competitor_channel)
        run = create_run_layout(root, run_id, config=self.cfg, metadata=metadata)
        self.output_dir = run
        return run


@dataclass
class QuestionResult:
    """A question's tidy table, its supporting frames, and its verdict."""

    question: str
    tidy: pd.DataFrame
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    verdict: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    # Generated images, keyed by unit.  They are the one part of a run a reader
    # looks at directly, so they are written beside the tables rather than left
    # in a runtime that will end.
    images: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, question: str, directory) -> "QuestionResult":
        """Rebuild a result from the files :meth:`save` wrote.

        A Colab runtime ending is the normal case, not the exception, so a
        completed question must be reusable without re-running it.  Everything
        downstream (figures, clustered statistics, the evidence table) reads
        only the tidy frame, the companion tables and the verdict, so a reloaded
        result is interchangeable with a freshly computed one.  The activation
        traces are not saved: a *new* endpoint still needs a new run.
        """
        question = str(question).lower()
        directory = Path(directory)
        tidy_path = directory / f"{question}_tidy.csv"
        if not tidy_path.exists():
            raise FileNotFoundError(f"no saved {question.upper()} at {tidy_path}")
        tidy = pd.read_csv(tidy_path, low_memory=False)
        tables = {}
        for path in sorted(directory.glob(f"{question}_*.csv")):
            name = path.stem[len(question) + 1:]
            if name != "tidy":
                tables[name] = pd.read_csv(path, low_memory=False)
        verdict, meta = "", {}
        sidecar = directory / f"{question}_verdict.json"
        if sidecar.exists():
            payload = json.loads(sidecar.read_text())
            verdict, meta = payload.get("verdict", ""), payload.get("meta", {})
        tensors = {}
        tensor_path = directory / f"{question}_diagnostics.pt"
        if tensor_path.exists():
            tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        return cls(question, tidy, tables, verdict, meta, _read_images(directory, question), tensors)

    def save(self, directory) -> Dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = {"tidy": directory / f"{self.question}_tidy.csv"}
        self.tidy.to_csv(written["tidy"], index=False)
        for name, frame in self.tables.items():
            path = directory / f"{self.question}_{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = path
        payload = {"question": self.question, "verdict": self.verdict, "meta": _jsonable(self.meta)}
        written["meta"] = directory / f"{self.question}_verdict.json"
        written["meta"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        for key, path in _write_images(directory, self.question, self.images).items():
            written[f"image_{key}"] = path
        if self.diagnostics:
            written["diagnostics"] = directory / f"{self.question}_diagnostics.pt"
            torch.save({str(k): v.detach().cpu() for k, v in self.diagnostics.items()},
                       written["diagnostics"])
        return written


def _image_dir(directory, question: str) -> Path:
    return Path(directory) / f"{question}_images"


def _write_images(directory, question: str, images: Mapping[str, Any]) -> Dict[str, Path]:
    """Write the generated images beside the tables, so a run leaves them behind.

    Anything without a ``save`` method is skipped rather than guessed at: a
    result may carry an array, a tensor or nothing at all, and a wrong guess here
    would fail a run that had already done the expensive part.
    """
    if not images:
        return {}
    folder = _image_dir(directory, question)
    written: Dict[str, Path] = {}
    for key, image in images.items():
        save = getattr(image, "save", None)
        if save is None:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}.png"
        save(path)
        written[str(key)] = path
    return written


def _read_images(directory, question: str) -> Dict[str, Any]:
    """Read back whatever :func:`_write_images` left, when Pillow is available."""
    folder = _image_dir(directory, question)
    if not folder.is_dir():
        return {}
    try:
        from PIL import Image
    except ImportError:
        return {}
    return {path.stem: Image.open(path).copy() for path in sorted(folder.glob("*.png"))}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if torch.is_tensor(value):
        return list(value.flatten().tolist()[:16])
    return value


# ----------------------------------------------------------------- helpers
def _rows_to_frame(rows: Sequence[LayerHeadMeasurement], **columns) -> pd.DataFrame:
    """Flatten measurements, stamping the run coordinates onto every row."""
    from dataclasses import fields as dataclass_fields

    records = []
    for row in rows:
        record = {f.name: getattr(row, f.name) for f in dataclass_fields(row) if f.name != "tensors"}
        if record.get("sink_position") is not None:
            record["sink_row"], record["sink_col"] = record["sink_position"]
        record.pop("sink_position", None)
        record.update(columns)
        records.append(record)
    return pd.DataFrame(records)


def _clean_pass(ctx: QuestionContext, unit: Tuple[int, str, int], *,
                probe_points: Sequence[Tuple[InterventionPoint, Sequence[int]]] = (),
                tracer: Optional[CausalTracer] = None) -> Tuple[Trace, CausalTracer]:
    prompt_id, prompt, seed = unit
    tracer = tracer or ctx.tracer()
    trace, _ = run_traced_generation(ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt,
                                     seed=seed, condition="clean", probes=list(probe_points))
    return trace, tracer


def _mean(values) -> float:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(values)) if values else float("nan")


def layer_level(frame: pd.DataFrame) -> pd.DataFrame:
    """The one row per layer that carries the layer-level rollups.

    ``head`` is also the name of a DataFrame method, so this split is always done
    by explicit column lookup rather than attribute access.
    """
    if frame is None or frame.empty or "head" not in frame.columns:
        return pd.DataFrame(columns=list(frame.columns) if frame is not None else [])
    return frame[frame["head"] == -1]


def head_level(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-head rows only."""
    if frame is None or frame.empty or "head" not in frame.columns:
        return pd.DataFrame(columns=list(frame.columns) if frame is not None else [])
    return frame[frame["head"] >= 0]


def _fraction(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(series.mean()) if len(series) else float("nan")


def diagnostic_rows(trace: Trace, *, question: str, condition: str,
                    vstar: torch.Tensor, channels: Sequence[int],
                    token_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """Scalar, CSV-safe manipulation checks derived from lossless hook tensors."""
    rows = []
    unit = vstar.float() / vstar.float().norm().clamp_min(1e-12)
    ids = [int(token) for token in token_ids]
    for (step, layer, point, stage, call), states in trace.diagnostics.items():
        valid = [token for token in ids if token < states.shape[0]]
        for token in valid:
            state = states[token].float()
            row = dict(question=question, condition=condition, prompt_id=trace.prompt_id,
                       seed=trace.seed, step=step, layer=layer, intervention_point=point,
                       temporal_endpoint=stage, hook_call=call, token_id=token,
                       norm=float(state.norm()), projection=float(state @ unit),
                       cosine=float((state @ unit) / state.norm().clamp_min(1e-12)))
            for channel in channels:
                if int(channel) < state.numel():
                    row[f"channel_{int(channel)}"] = float(state[int(channel)])
            rows.append(row)
    return rows


def diagnostic_tensors(trace: Trace, *, question: str, condition: str) -> Dict[str, torch.Tensor]:
    return {f"{question}/{condition}/step{step}/layer{layer}/{point}/{stage}/call{call}": tensor
            for (step, layer, point, stage, call), tensor in trace.diagnostics.items()}


def temporal_endpoint(layer: int, intervention_layer: int, end_layer: int) -> str:
    offset = int(layer) - int(intervention_layer)
    if int(layer) == int(end_layer):
        return "end_of_zone"
    return "current_block_output" if offset == 0 else f"plus_{offset}_blocks"


def retention_summary(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Publish macro/all-head and pooled affected-head estimates side-by-side."""
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "affected_head_macro", "all_head_macro",
                                     "affected_head_pooled", "n_affected_heads",
                                     "affected_head_ids"])
    layers, heads = layer_level(frame), head_level(frame)
    rows = []
    for keys, group in layers.groupby(list(group_columns), dropna=False, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        selector = pd.Series(True, index=heads.index)
        for column, value in zip(group_columns, keys):
            selector &= heads[column].eq(value)
        affected = heads[selector & heads["head_was_affected"].fillna(False)]
        rows.append({**dict(zip(group_columns, keys)),
                     "affected_head_macro": float(group["head_retention"].mean()),
                     "all_head_macro": float(group["sink_retention_all"].mean()),
                     "affected_head_pooled": float(affected["original_sink_retained"].mean()),
                     "n_affected_heads": int(len(affected)),
                     "affected_head_ids": json.dumps(sorted(set(int(v) for v in affected["head"])))})
    return pd.DataFrame(rows)


def _resumable(question: str):
    """Wrap a runner so a completed question is reloaded instead of recomputed."""
    def decorate(runner):
        @functools.wraps(runner)
        def wrapped(ctx: QuestionContext, *args, resume: bool = True, **kwargs):
            directory = None if ctx.output_dir is None else Path(ctx.output_dir) / question
            if resume and directory is not None:
                try:
                    result = QuestionResult.load(question, directory)
                    ctx.say(f"  [{question.upper()}] reusing the saved run at {directory} "
                            f"({len(result.tidy):,} rows); pass resume=False to recompute")
                    return result
                except FileNotFoundError:
                    pass
            result = runner(ctx, *args, **kwargs)
            if directory is not None:
                result.save(directory)
            return result
        return wrapped
    return decorate


def _clean_reference(ctx: "QuestionContext", clean: Trace, targets: FrozenTargets,
                     layers: Sequence[int]) -> "EP.CleanReference":
    """The clean run's per-layer bars, built once and reused by every condition."""
    return EP.build_clean_reference(clean, targets, layers=layers, step=ctx.step,
                                    highnorm_ratio=ctx.highnorm_ratio)


# ======================================================  natural-register removal
@dataclass(frozen=True)
class Q1Condition:
    """One condition row: which tokens are edited and how."""

    key: str
    label: str
    edit: str          # sham | remove_vstar | matched_state | norm_clamp | zero
    group: str         # register | topk | random | norm_matched | nonregister
    role: str = "intervention"     # intervention | control


Q1_CONDITIONS: Tuple[Q1Condition, ...] = (
    Q1Condition("sham", "Sham hook", "sham", "register", "control"),
    Q1Condition("direction_removal", "Register direction removed", "remove_vstar", "register"),
    Q1Condition("matched_ordinary_state", "Replaced by matched ordinary state", "matched_state", "register"),
    Q1Condition("ordinary_norm_clamp", "Clamped to ordinary norm", "norm_clamp", "register"),
    Q1Condition("state_zeroed", "Register state zeroed", "zero", "register"),
    Q1Condition("random_tokens_zeroed", "Count-matched random tokens zeroed", "zero", "random", "control"),
    Q1Condition("norm_matched_tokens_zeroed", "Norm-matched ordinary tokens zeroed", "zero",
                "norm_matched", "control"),
    Q1Condition("offregister_direction_removal", "Matched-magnitude removal off register",
                "remove_vstar", "nonregister", "control"),
    # Zeroing a token is far more violent than removing its direction, and an
    # equal-energy nudge is far gentler; neither is matched in strength to the
    # headline intervention.  This one applies the identical operation to the
    # highest-norm tokens that are not registers, so a difference between them is
    # about which tokens were edited rather than how hard.
    Q1Condition("normmatched_direction_removal", "Direction removed from norm-matched tokens",
                "remove_vstar", "norm_matched", "control"),
)

Q1_SELECTION_RULES = ("percentile", "topk")


def run_direction_magnitude_gate(ctx: QuestionContext) -> QuestionResult:
    """Reproduce the live-selection direction-versus-magnitude semantics.

    This is an acceptance gate, not a condition of the main experiments: targets
    are selected from each live block input at every denoising step, exactly as
    ``TokenEditHook`` does, while the affected-head denominator is defined by the
    paired clean run at the fixed reporting step.
    """
    layer, point = ctx.intervention_layer, InterventionPoint.BLOCK_INPUT
    tracer = ctx.tracer()
    frames = []
    for prompt_id, prompt, seed in ctx.units():
        clean, _ = _clean_pass(ctx, (prompt_id, prompt, seed),
                               probe_points=[(point, [layer])], tracer=tracer)
        frozen = select_frozen_targets(clean, layer=layer, step=ctx.step, point=point,
                                       direction=ctx.vstar, percentile=100 * (1 - 0.01),
                                       topk=ctx.topk, highnorm_ratio=ctx.highnorm_ratio)
        reference = _clean_reference(ctx, clean, frozen, [layer])

        def live_edit(kind):
            def edit(image: torch.Tensor, edit_ctx: EditContext) -> torch.Tensor:
                norms = image.float().norm(dim=-1)
                ids = torch.nonzero(norms > ctx.highnorm_ratio * norms.median()).flatten()
                if kind == "norm":
                    return clamp_norm(image, ids.tolist(), float(norms.median()))
                generator = torch.Generator(device="cpu").manual_seed(
                    int(seed) * 100003 + int(edit_ctx.step) * 101 + int(edit_ctx.layer))
                random = torch.randn((len(ids), image.shape[-1]), generator=generator).to(image)
                random = random / random.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
                out = image.clone()
                out[ids] = random * norms[ids, None].to(random)
                return out
            return edit

        for condition in ("norm", "direction"):
            plan = EditPlan(edit=live_edit(condition), point=point, layers=[layer],
                            label=f"direction_magnitude_gate_{condition}")
            treated, _ = run_traced_generation(
                ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
                condition=f"direction_magnitude_gate_{condition}", plans=[plan], targets=frozen)
            rows = EP.measure_trace(clean, treated, frozen, layers=[layer], step=ctx.step,
                                    token_ids=frozen.register_ids, channels=ctx.channels,
                                    reference=reference)
            frames.append(_rows_to_frame(rows, question="direction_magnitude_gate",
                                         condition=condition, prompt_id=prompt_id, seed=seed,
                                         step=ctx.step))
    tidy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    heads = head_level(tidy)
    layer_rows = layer_level(tidy)
    summary = []
    for condition, group in layer_rows.groupby("condition"):
        affected = heads[(heads["condition"] == condition) & heads["head_was_affected"].fillna(False)]
        summary.append(dict(
            condition=condition,
            affected_head_macro=float(group["head_retention"].mean()),
            all_head_macro=float(group["sink_retention_all"].mean()),
            affected_head_pooled=float(affected["original_sink_retained"].mean()),
            n_affected_heads=int(len(affected)),
            affected_head_ids=json.dumps(sorted(set(int(v) for v in affected["head"])))))
    return QuestionResult("direction_magnitude_gate", tidy,
                          {"summary": pd.DataFrame(summary)},
                          "Acceptance requires direction retention < norm retention.",
                          meta=dict(layer=layer, point=point.value,
                                    target_rule=f"live norm > {ctx.highnorm_ratio} x median"))


def run_diagnostic_harness(ctx: QuestionContext) -> QuestionResult:
    """One-unit fail-loud identity/manipulation gate run before any sweep."""
    if not ctx.units():
        raise ValueError("diagnostic harness needs one prompt-seed unit")
    prompt_id, prompt, seed = ctx.units()[0]
    layer, point = ctx.intervention_layer, InterventionPoint.BLOCK_INPUT
    tracer = ctx.tracer(key_layers=[layer], full_state_layers=[layer])
    clean, _ = run_traced_generation(ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt,
                                     seed=seed, condition="diagnostic_clean",
                                     probes=[(point, [layer])], save_image=True)
    targets = select_frozen_targets(clean, layer=layer, step=ctx.step, point=point,
                                    direction=ctx.vstar, percentile=ctx.percentile,
                                    topk=ctx.topk, highnorm_ratio=ctx.highnorm_ratio)
    ids = list(targets.register_ids)
    if not ids:
        raise AssertionError("diagnostic clean run selected no register tokens")
    clean_input = clean.probe(ctx.step, layer, point)
    checks = []

    def passed(name, value=0.0):
        checks.append(dict(check=name, status="PASS", max_error=float(value),
                           prompt_id=prompt_id, seed=seed, layer=layer, step=ctx.step,
                           target_token_ids=json.dumps(ids)))

    clamped = clamp_norm(clean_input, ids, targets.ordinary_norm)
    values = DG.assert_norm_clamp(clean_input, clamped, ctx.vstar, ids, targets.ordinary_norm)
    passed("norm_clamp", max(values.values()))
    removed = remove_direction(clean_input, ctx.vstar, ids, preserve_norm=True)
    passed("direction_removal", DG.assert_direction_removed(removed, ctx.vstar, ids))
    zeroed = scale_channel(clean_input, ctx.dominant_channel, 0.0, ids)
    passed("channel_zero", DG.assert_channel_zero(zeroed, ctx.dominant_channel, ids))

    sham = EditPlan(edit=lambda image, _: image.clone(), point=point, layers=[layer], label="sham")
    treated, _ = run_traced_generation(ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt,
                                       seed=seed, condition="diagnostic_sham", plans=[sham],
                                       targets=targets, save_image=True)
    for name in ("states", "keys", "queries", "attention_logits", "attention_probs"):
        passed(f"sham_{name}", DG.assert_identity(getattr(clean.at(ctx.step, layer), name),
                                                   getattr(treated.at(ctx.step, layer), name), name=name))
    clean_sinks = clean.at(ctx.step, layer).incoming.argmax(dim=-1)
    treated_sinks = treated.at(ctx.step, layer).incoming.argmax(dim=-1)
    passed("sham_sink_ids", DG.assert_identity(clean_sinks, treated_sinks, name="sink ids", atol=0))
    if clean.image is not None and treated.image is not None:
        clean_bytes = clean.image.tobytes() if hasattr(clean.image, "tobytes") else None
        treated_bytes = treated.image.tobytes() if hasattr(treated.image, "tobytes") else None
        if clean_bytes is not None and clean_bytes != treated_bytes:
            raise AssertionError("sham final image identity failed")
        if clean_bytes is not None:
            passed("sham_final_image")

    clean_keys = {layer: clean.at(ctx.step, layer).keys}
    from .causal_engine import FinalKeyPatch
    key_patch = FinalKeyPatch(tracer, layers=[layer], source=ids[0], recipient=ids[0],
                              clean_keys=clean_keys)
    self_key, _ = run_traced_generation(ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt,
                                        seed=seed, condition="diagnostic_key_self_patch",
                                        key_patch=key_patch)
    for name in ("keys", "attention_logits", "attention_probs"):
        passed(f"final_key_self_patch_{name}", DG.assert_identity(
            getattr(clean.at(ctx.step, layer), name), getattr(self_key.at(ctx.step, layer), name),
            name=f"final key self-patch {name}"))

    residual = EditPlan(edit=lambda image, _: transplant(image, ids[0], ids[0],
                                                          stage="full_residual"),
                        point=point, layers=[layer], label="residual_self_patch")
    self_residual, _ = run_traced_generation(
        ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
        condition="diagnostic_residual_self_patch", plans=[residual], targets=targets)
    passed("full_residual_self_patch", DG.assert_identity(
        clean.at(ctx.step, layer).states, self_residual.at(ctx.step, layer).states,
        name="full residual self-patch"))

    mapping = pd.DataFrame(DG.token_index_table(
        sequence_length=clean.n_img + clean.n_txt, image_tokens=clean.n_img,
        text_tokens=clean.n_txt, grid_cols=clean.grid[1]))
    passed("token_indexing")
    checks_frame = pd.DataFrame(checks)
    raw = {}
    for name, trace in (("clean", clean), ("sham", treated), ("key_self", self_key),
                        ("residual_self", self_residual)):
        raw.update(diagnostic_tensors(trace, question="diagnostics", condition=name))
        observation = trace.at(ctx.step, layer)
        for tensor_name in ("states", "keys", "queries", "attention_logits", "attention_probs"):
            value = getattr(observation, tensor_name) if observation is not None else None
            if value is not None:
                raw[f"diagnostics/{name}/layer{layer}/{tensor_name}"] = value
    return QuestionResult("diagnostics", checks_frame,
                          {"token_index": mapping}, "All one-unit numerical gates passed.",
                          meta=dict(prompt_id=prompt_id, seed=seed, layer=layer, step=ctx.step),
                          diagnostics=raw)


def _q1_edit(condition: Q1Condition, vstar: torch.Tensor, ids: Sequence[int],
             targets: FrozenTargets) -> Callable:
    ids = [int(t) for t in ids]

    def edit(image: torch.Tensor, ctx: EditContext) -> torch.Tensor:
        if condition.edit == "sham" or not ids:
            return image.clone()
        if condition.edit == "remove_vstar":
            # Norm-preserving: x - <x,v*>v* has norm ||x||*sqrt(1-cos^2), so for a
            # register nearly collinear with v*, plain removal would leave the token
            # both directionless and tiny, and the condition would stop isolating
            # direction from magnitude.
            return remove_direction(image, vstar, ids, preserve_norm=True)
        if condition.edit == "norm_clamp":
            return clamp_norm(image, ids, targets.ordinary_norm)
        if condition.edit == "matched_state":
            states = targets.states_for("matched")
            if states is None or states.shape[0] != len(ids):
                states = torch.zeros(len(ids), image.shape[-1])
            return replace_tokens(image, ids, states.to(image))
        if condition.edit == "zero":
            return replace_tokens(image, ids, torch.zeros(len(ids), image.shape[-1]).to(image))
        raise KeyError(condition.edit)

    return edit


@_resumable("q1")
def run_q1(ctx: QuestionContext, *, conditions: Sequence[Q1Condition] = Q1_CONDITIONS,
           selection_rules: Sequence[str] = Q1_SELECTION_RULES) -> QuestionResult:
    """Remove the natural register population and follow the sinks.

    The four interventions separate *which property* of the register matters:
    direction removal keeps magnitude, the norm clamp keeps direction, the
    matched-ordinary replacement removes both while injecting a realistic state,
    and zeroing is the extreme perturbation the controls calibrate against.
    """
    layer = ctx.intervention_layer
    point = InterventionPoint.BLOCK_INPUT
    downstream = ctx.downstream_layers
    frames, fate_rows, diagnostics, raw_diagnostics = [], [], [], {}

    for prompt_id, prompt, seed in ctx.units():
        ctx.say(f"  [Q1] prompt {prompt_id} seed {seed}: clean pass")
        tracer = ctx.tracer()
        clean, _ = _clean_pass(ctx, (prompt_id, prompt, seed),
                               probe_points=[(point, [layer])], tracer=tracer)
        clean_concentration = EP.clean_head_concentration(clean, layers=downstream, step=ctx.step)
        for rule in selection_rules:
            targets = select_frozen_targets(
                clean, layer=layer, step=ctx.step, point=point, direction=ctx.vstar,
                percentile=ctx.percentile, topk=ctx.topk, highnorm_ratio=ctx.highnorm_ratio)
            ids = targets.register_ids if rule == "percentile" else targets.topk_ids
            reference = _clean_reference(ctx, clean, targets, downstream)
            for condition in conditions:
                group_ids = ids if condition.group in ("register", "topk") else targets.ids_for(condition.group)
                plan = EditPlan(edit=_q1_edit(condition, ctx.vstar, group_ids, targets),
                                point=point, layers=[layer], label=condition.key)
                treated, stats = run_traced_generation(
                    ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
                    condition=condition.key, plans=[plan], targets=targets)
                samples = {key[3]: value for key, value in treated.diagnostics.items()
                           if key[0] == ctx.step and key[1] == layer}
                if group_ids and {"x_pre_hook", "x_post_hook"} <= set(samples):
                    if condition.edit == "norm_clamp":
                        DG.assert_norm_clamp(samples["x_pre_hook"], samples["x_post_hook"],
                                             ctx.vstar, group_ids, targets.ordinary_norm)
                    elif condition.edit == "remove_vstar":
                        DG.assert_direction_removed(samples["x_post_hook"], ctx.vstar, group_ids)
                diagnostics.extend(diagnostic_rows(
                    treated, question="q1", condition=condition.key, vstar=ctx.vstar,
                    channels=ctx.channels, token_ids=group_ids))
                raw_diagnostics.update(diagnostic_tensors(
                    treated, question="q1", condition=condition.key))
                rows = EP.measure_trace(clean, treated, targets, layers=downstream,
                                        step=ctx.step, token_ids=ids, channels=ctx.channels,
                                        reference=reference)
                recovery_layer, recovery_token, kind = EP.recovery_outcome(
                    treated, targets, layers=downstream, step=ctx.step, reference=reference)
                fate = EP.sink_fate(rows, targets, clean_concentration=clean_concentration,
                                    recovery_kind=kind)
                frame = _rows_to_frame(
                    rows, question="q1", condition=condition.key,
                    condition_label=condition.label, role=condition.role,
                    edit=condition.edit, token_group=condition.group, selection_rule=rule,
                    prompt_id=prompt_id, seed=seed, step=ctx.step, fate=fate,
                    n_targets=len(group_ids), target_token_ids=json.dumps(group_ids),
                    perturbation_energy=stats.perturbation_energy,
                    removed_energy=stats.removed_energy)
                frame["temporal_endpoint"] = frame["layer"].map(
                    lambda value: temporal_endpoint(value, layer, max(downstream)))
                frames.append(frame)
                fate_rows.append(dict(
                    question="q1", condition=condition.key, condition_label=condition.label,
                    role=condition.role, selection_rule=rule, prompt_id=prompt_id, seed=seed,
                    fate=fate, recovery_kind=kind, first_recovery_layer=recovery_layer,
                    recovery_token=recovery_token,
                    head_retention=_mean([r.head_retention for r in rows if r.head == -1]),
                    perturbation_energy=stats.perturbation_energy))

    tidy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    fates = pd.DataFrame(fate_rows)
    if not tidy.empty:
        tidy = tidy.merge(fates[["condition", "selection_rule", "prompt_id", "seed",
                                 "recovery_kind", "first_recovery_layer"]],
                          on=["condition", "selection_rule", "prompt_id", "seed"], how="left",
                          suffixes=("", "_run"))
    return QuestionResult("q1", tidy, {"fates": fates,
                                        "diagnostics": pd.DataFrame(diagnostics),
                                        "retention_summary": retention_summary(
                                            tidy, ["condition", "selection_rule", "temporal_endpoint"])},
                          _q1_verdict(fates, tidy),
                          meta=dict(layer=layer, point=point.value, downstream=list(downstream),
                                    selection_rules=list(selection_rules)), diagnostics=raw_diagnostics)


def _q1_verdict(fates: pd.DataFrame, tidy: pd.DataFrame) -> str:
    if fates.empty:
        return "Q1 produced no runs."
    primary = fates[fates["role"] == "intervention"]
    if primary.empty:
        return "Q1 ran controls only; no intervention to interpret."
    share = primary["fate"].value_counts(normalize=True)
    leading = share.index[0]
    reading = {
        "same_position": "the same positions regain the treated-only register criterion; paired-clean "
                         "novelty is not tested here",
        "relocated": "a different position wins the treated scan; this does not establish that the "
                     "position was newly generated relative to clean",
        "reserve_takeover": "sinks move to non-target tokens; this endpoint does not prove a "
                            "predeclared ranked reserve",
        "diffuse": "incoming attention becomes diffuse, so natural registers are indispensable "
                   "routing anchors",
        "none": "no single downstream outcome dominates",
    }[leading]
    layer_rows = layer_level(tidy)
    retention = _fraction(layer_rows[layer_rows["role"] == "intervention"], "head_retention")
    control = _fraction(layer_rows[layer_rows["role"] == "control"], "head_retention")
    headline = (f"Q1: after removing the frozen register population, {share[leading]:.0%} of runs end in "
                f"'{leading.replace('_', ' ')}' -- {reading}.")
    if math.isnan(retention):
        affected = layer_rows["affected_heads"].fillna(0).sum() if "affected_heads" in layer_rows else 0
        return (headline + f" No head sank on a frozen target in the clean run "
                f"({int(affected)} affected head-layers), so sink retention is undefined here and "
                "the routing claim rests on the fate labels alone.")
    return (headline + f" Affected heads keep the clean sink in {retention:.0%} of cases against "
            f"{control:.0%} for matched controls.")


# ==================================================  live dominant-channel scaling
Q2_GAMMAS: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5)
Q2_SCOPES: Tuple[str, ...] = ("writer_only", "maintenance")
Q2_TARGETS: Tuple[str, ...] = ("dominant_channel", "competing_channel", "unspecific_channel",
                               "random_channel", "matched_energy_direction", "ordinary_positions")
Q2_TARGET_LABELS = {
    "dominant_channel": "Dominant register channel",
    "competing_channel": "Late-growing competing channel",
    "unspecific_channel": "Massive but register-unspecific channel",
    "random_channel": "Random channel",
    "matched_energy_direction": "Magnitude-matched direction removal",
    "ordinary_positions": "Dominant channel at ordinary positions",
}


def _random_unit(width: int, seed: int, orthogonal_to: Optional[torch.Tensor] = None) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    v = torch.randn(width, generator=generator)
    if orthogonal_to is not None:
        u = orthogonal_to.float().flatten()
        u = u / u.norm().clamp_min(1e-9)
        v = v - (v @ u) * u
    return v / v.norm().clamp_min(1e-9)


def _q2_edit(ctx: QuestionContext, target: str, gamma: float, ids: Sequence[int],
             targets: FrozenTargets, ordinary_ids: Sequence[int]) -> Callable:
    ids = [int(t) for t in ids]
    ordinary = [int(t) for t in ordinary_ids]
    random_channel = int(_rng_channel(ctx))

    def edit(image: torch.Tensor, _: EditContext) -> torch.Tensor:
        if target == "dominant_channel":
            return scale_channel(image, ctx.dominant_channel, gamma, ids)
        if target == "competing_channel":
            return scale_channel(image, ctx.competitor_channel, gamma, ids)
        if target == "unspecific_channel":
            return scale_channel(image, ctx.control_channel, gamma, ids)
        if target == "random_channel":
            return scale_channel(image, random_channel, gamma, ids)
        if target == "ordinary_positions":
            return scale_channel(image, ctx.dominant_channel, gamma, ordinary)
        if target == "matched_energy_direction":
            # Same squared magnitude the dominant-channel scaling would remove,
            # taken along a direction orthogonal to v* instead.
            removed = float((1.0 - gamma ** 2) *
                            image[ids, ctx.dominant_channel].float().pow(2).sum())
            axis = _random_unit(image.shape[-1], 11, orthogonal_to=ctx.vstar)
            return remove_matched_energy(image, axis, ids, max(removed, 0.0))
        raise KeyError(target)

    return edit


def _rng_channel(ctx: QuestionContext) -> int:
    """A reproducible random channel that is not one of the frozen three."""
    generator = torch.Generator(device="cpu").manual_seed(int(ctx.dominant_channel) + 9973)
    width = int(ctx.vstar.numel())
    for _ in range(64):
        candidate = int(torch.randint(width, (1,), generator=generator))
        if candidate not in ctx.channels:
            return candidate
    return (ctx.dominant_channel + 1) % width


@_resumable("q2")
def run_q2(ctx: QuestionContext, *, gammas: Sequence[float] = Q2_GAMMAS,
           scopes: Sequence[str] = Q2_SCOPES, targets: Sequence[str] = Q2_TARGETS,
           control_gammas: Sequence[float] = (0.0,), maintenance_span: int = 4) -> QuestionResult:
    """Scale the dominant channel live and follow v*, key geometry, sinks.

    This tests whether suppressing the channel *during* the forward pass
    propagates downstream. ``writer_only`` tests a single write, ``maintenance``
    keeps suppressing it through the register zone, which separates a one-off
    write from a state the later blocks keep rebuilding.
    """
    point = InterventionPoint.WRITER_RESIDUAL
    writer = ctx.writer_layer
    maintenance_layers = [l for l in ctx.register_layers if writer <= l < writer + maintenance_span] or [writer]
    downstream = ctx.downstream_layers
    measurement_layers = tuple(sorted(set([writer, *downstream])))
    frames, summary, diagnostics, raw_diagnostics = [], [], [], {}

    for prompt_id, prompt, seed in ctx.units():
        ctx.say(f"  [Q2] prompt {prompt_id} seed {seed}: clean pass")
        tracer = ctx.tracer()
        clean, _ = _clean_pass(ctx, (prompt_id, prompt, seed),
                               probe_points=[(InterventionPoint.BLOCK_INPUT, [ctx.intervention_layer])],
                               tracer=tracer)
        frozen = select_frozen_targets(clean, layer=ctx.intervention_layer, step=ctx.step,
                                       point=InterventionPoint.BLOCK_INPUT, direction=ctx.vstar,
                                       percentile=ctx.percentile, topk=ctx.topk,
                                       highnorm_ratio=ctx.highnorm_ratio)
        ids = frozen.register_ids
        reference = _clean_reference(ctx, clean, frozen, measurement_layers)
        for target in targets:
            sweep = list(gammas) if target == "dominant_channel" else list(control_gammas)
            for gamma in sweep:
                for scope in scopes:
                    layers = [writer] if scope == "writer_only" else maintenance_layers
                    plan = EditPlan(edit=_q2_edit(ctx, target, float(gamma), ids, frozen,
                                                  frozen.matched_ordinary_ids),
                                    point=point, layers=layers,
                                    label=f"{target}_g{gamma:g}_{scope}")
                    condition = f"{target}__gamma{gamma:g}__{scope}"
                    treated, stats = run_traced_generation(
                        ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
                        condition=condition, plans=[plan], targets=frozen)
                    samples = {key[3]: value for key, value in treated.diagnostics.items()
                               if key[0] == ctx.step and key[1] in layers}
                    if target == "dominant_channel" and float(gamma) == 0.0 and ids and \
                            "x_post_hook" in samples:
                        DG.assert_channel_zero(samples["x_post_hook"], ctx.dominant_channel, ids)
                    diagnostics.extend(diagnostic_rows(
                        treated, question="q2", condition=condition, vstar=ctx.vstar,
                        channels=ctx.channels, token_ids=ids))
                    raw_diagnostics.update(diagnostic_tensors(
                        treated, question="q2", condition=condition))
                    rows = EP.measure_trace(clean, treated, frozen, layers=measurement_layers,
                                            step=ctx.step, token_ids=ids, channels=ctx.channels,
                                            reference=reference)
                    frame = _rows_to_frame(
                        rows, question="q2", condition=condition, gamma=float(gamma),
                        scope=scope, control=target, control_label=Q2_TARGET_LABELS[target],
                        role="intervention" if target == "dominant_channel" else "control",
                        prompt_id=prompt_id, seed=seed, step=ctx.step,
                        target_token_ids=json.dumps(list(ids)),
                        perturbation_energy=stats.perturbation_energy)
                    frame["temporal_endpoint"] = frame["layer"].map(
                        lambda value: temporal_endpoint(value, writer, max(measurement_layers)))
                    frames.append(frame)
                    layer_rows = layer_level(frame)
                    immediate = layer_rows[layer_rows["layer"] == writer]
                    summary.append(dict(
                        question="q2", condition=condition, gamma=float(gamma), scope=scope,
                        control=target, control_label=Q2_TARGET_LABELS[target],
                        prompt_id=prompt_id, seed=seed,
                        immediate_vstar_change=_fraction(immediate, "vstar_projection_change"),
                        vstar_projection_change=_fraction(layer_rows, "vstar_projection_change"),
                        original_sink_retained=_fraction(layer_rows, "head_retention"),
                        key_rank=_fraction(head_level(frame), "key_rank"),
                        query_key_advantage=_fraction(head_level(frame), "query_key_advantage"),
                        takeover_channel=_mode(layer_rows, "takeover_channel"),
                        channel_takeover=_fraction(layer_rows, "channel_takeover")))

    tidy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    dose = pd.DataFrame(summary)
    return QuestionResult("q2", tidy, {"dose_response": dose,
                                        "diagnostics": pd.DataFrame(diagnostics),
                                        "retention_summary": retention_summary(
                                            tidy, ["condition", "temporal_endpoint"])}, _q2_verdict(dose),
                          meta=dict(writer_layer=writer, point=point.value,
                                    maintenance_layers=maintenance_layers,
                                    gammas=list(gammas), dominant_channel=ctx.dominant_channel),
                          diagnostics=raw_diagnostics)


def _fmt(value, spec: str = "+.2f", missing: str = "not measurable") -> str:
    """Format a number, or say plainly that it could not be measured."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    return missing if math.isnan(number) else format(number, spec)


def _mode(frame: pd.DataFrame, column: str):
    if frame.empty or column not in frame:
        return None
    values = frame[column].dropna()
    return None if values.empty else values.mode().iloc[0]


def _q2_verdict(dose: pd.DataFrame) -> str:
    if dose.empty:
        return "Q2 produced no runs."
    primary = dose[dose["control"] == "dominant_channel"]
    if primary.empty:
        return "Q2 ran controls only; no dominant-channel sweep to interpret."
    curve = primary.groupby("gamma")["vstar_projection_change"].mean().sort_index()
    at_zero = float(curve.iloc[0]) if len(curve) else float("nan")
    sham = primary[np.isclose(primary["gamma"], 1.0)]["vstar_projection_change"].mean()
    controls = dose[dose["control"] != "dominant_channel"].groupby("control")["vstar_projection_change"].mean()
    strongest_control = controls.abs().max() if len(controls) else float("nan")
    retention = primary[np.isclose(primary["gamma"], 0.0)]["original_sink_retained"].mean()
    monotone = bool(len(curve) > 2 and curve.is_monotonic_increasing or curve.is_monotonic_decreasing)
    reading = ("dose-dependently" if monotone else "non-monotonically")
    beats = ("larger than every matched control" if abs(at_zero) > abs(strongest_control)
             else "no larger than the strongest matched control, so channel identity is not "
                  "established by this endpoint")
    tail = (f"Affected heads keep the clean sink in {retention:.0%} of cases at full suppression."
            if not math.isnan(retention) else
            "No head sank on a frozen target in the clean run, so sink retention is undefined here.")
    return (f"Q2: suppressing the dominant channel moves the register projection by "
            f"{_fmt(at_zero)} ordinary-token standard deviations at gamma=0 ({reading} across the "
            f"sweep; the gamma=1 sham gives {_fmt(sham)}), which is {beats}. " + tail)


# ==============================================  direction destruction and recovery
Q3_SCOPES: Tuple[str, ...] = ("single_shot", "fixed_carrier_repeated",
                              "dynamic_register_state")


@_resumable("q3")
def run_q3(ctx: QuestionContext, *, scopes: Sequence[str] = Q3_SCOPES,
           rescue_layers: Optional[Sequence[int]] = None, max_rescues: int = 3) -> QuestionResult:
    """Destroy v* at its natural home and watch whether it comes back.

    Magnitude is preserved, so a token that recovers cannot have
    recovered merely by shrinking.  The readout is not only *whether* the sink
    returns but *where*: the same position means a writer is bound to those
    positions, a new position means the model keeps a register function without a
    fixed spatial identity.
    """
    point = InterventionPoint.BLOCK_INPUT
    layer = ctx.intervention_layer
    downstream = ctx.downstream_layers
    later = [l for l in ctx.register_layers if l > layer]
    if rescue_layers is None:
        step = max(1, len(later) // max(max_rescues, 1))
        rescue_layers = later[::step][:max_rescues]
    rescue_layers = [int(l) for l in rescue_layers]
    probe_layers = sorted({layer, *later, *rescue_layers})
    frames, outcomes, histories, raw_diagnostics = [], [], [], {}

    def destroy(ids):
        def edit(image: torch.Tensor, _: EditContext) -> torch.Tensor:
            return remove_direction(image, ctx.vstar, ids, preserve_norm=True)
        return edit

    def restore(ids, states):
        def edit(image: torch.Tensor, _: EditContext) -> torch.Tensor:
            return replace_tokens(image, ids, states.to(image))
        return edit

    for prompt_id, prompt, seed in ctx.units():
        ctx.say(f"  [Q3] prompt {prompt_id} seed {seed}: clean pass")
        tracer = ctx.tracer()
        clean, _ = _clean_pass(ctx, (prompt_id, prompt, seed),
                               probe_points=[(point, probe_layers)], tracer=tracer)
        frozen = select_frozen_targets(clean, layer=layer, step=ctx.step, point=point,
                                       direction=ctx.vstar, percentile=ctx.percentile,
                                       topk=ctx.topk, highnorm_ratio=ctx.highnorm_ratio)
        ids = list(frozen.register_ids)
        reference = _clean_reference(ctx, clean, frozen, downstream)
        input_bars = {}
        excluded = set(frozen.register_ids) | set(frozen.topk_ids)
        unit_v = ctx.vstar.float() / ctx.vstar.float().norm().clamp_min(1e-12)
        for candidate in probe_layers:
            states_at_input = clean.probe(ctx.step, candidate, point)
            if states_at_input is None:
                continue
            norms_at_input = states_at_input.float().norm(dim=-1)
            cos_at_input = (states_at_input.float() @ unit_v).abs() / norms_at_input.clamp_min(1e-12)
            ordinary = [t for t in range(len(norms_at_input)) if t not in excluded] or list(range(len(norms_at_input)))
            input_bars[candidate] = (ctx.highnorm_ratio * float(norms_at_input.median()),
                                     float(torch.quantile(cos_at_input[ordinary], 0.999)))
        conditions: List[Tuple[str, str, List[EditPlan]]] = [
            ("sham", "control", [EditPlan(edit=lambda image, _: image.clone(), point=point,
                                          layers=[layer], label="sham")]),
        ]
        for scope in scopes:
            edit_layers = [layer] if scope == "single_shot" else [layer, *later]
            if scope == "dynamic_register_state":
                def dynamic_destroy(image: torch.Tensor, edit_ctx: EditContext) -> torch.Tensor:
                    norm_bar, alignment_bar = input_bars[edit_ctx.layer]
                    norms = image.float().norm(dim=-1)
                    v = ctx.vstar.to(image)
                    v = v / v.norm().clamp_min(1e-12)
                    cosine = (image.float() @ v.float()).abs() / norms.clamp_min(1e-12)
                    current = torch.nonzero((norms >= norm_bar) &
                                            (cosine >= alignment_bar)).flatten().tolist()
                    out = remove_direction(image, ctx.vstar, current, preserve_norm=True)
                    if current:
                        DG.assert_direction_removed(out, ctx.vstar, current)
                    return out
                edit_fn = dynamic_destroy
            else:
                edit_fn = destroy(ids)
            conditions.append((f"direction_destroyed__{scope}", "intervention",
                               [EditPlan(edit=edit_fn, point=point, layers=edit_layers,
                                         label=scope)]))
        for rescue in rescue_layers:
            states = clean.probe(ctx.step, int(rescue), point)
            if states is None:
                continue
            conditions.append((
                f"clean_state_restored_at_layer_{rescue}", "rescue",
                [EditPlan(edit=destroy(ids), point=point, layers=[layer], label="destroy"),
                 EditPlan(edit=restore(ids, states[ids].clone()), point=point,
                          layers=[int(rescue)], label=f"restore@{rescue}")]))

        for condition, role, plans in conditions:
            treated, stats = run_traced_generation(
                ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
                condition=condition, plans=plans, targets=frozen)
            rows = EP.measure_trace(clean, treated, frozen, layers=downstream, step=ctx.step,
                                    token_ids=ids, channels=ctx.channels, reference=reference)
            recovery_layer, recovery_token, kind = EP.recovery_outcome(
                treated, targets=frozen, layers=downstream, step=ctx.step, reference=reference)
            scope = ("dynamic_register_state" if condition.endswith("dynamic_register_state")
                     else "fixed_carrier_repeated" if condition.endswith("fixed_carrier_repeated")
                     else "rescue" if role == "rescue" else "single_shot")
            frames.append(_rows_to_frame(
                rows, question="q3", condition=condition, role=role, scope=scope,
                prompt_id=prompt_id, seed=seed, step=ctx.step, recovery_kind=kind,
                target_token_ids=json.dumps(ids),
                first_recovery_layer=recovery_layer, recovery_token=recovery_token,
                perturbation_energy=stats.perturbation_energy))
            outcomes.append(dict(
                question="q3", condition=condition, role=role, scope=scope,
                prompt_id=prompt_id, seed=seed, recovery_kind=kind,
                first_recovery_layer=recovery_layer, recovery_token=recovery_token,
                recovered_at_original_position=bool(recovery_token in set(ids))
                if recovery_token is not None else False,
                head_retention=_mean([r.head_retention for r in rows if r.head == -1]),
                max_cosine=_mean([r.max_cosine for r in rows if r.head == -1])))
            if role == "intervention":
                raw_diagnostics.update(diagnostic_tensors(
                    treated, question="q3", condition=condition))
                histories.extend(_q3_histories(clean, treated, reference, input_bars, ctx, condition,
                                               prompt_id, seed,
                                               [layer] if scope in ("single_shot", "rescue")
                                               else [layer, *later]))

    tidy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    recovery = pd.DataFrame(outcomes)
    history = pd.DataFrame(histories)
    summary_rows = []
    for condition, group in recovery.groupby("condition", observed=True):
        token_group = history[history["condition"] == condition] if not history.empty else history
        summary_rows.append(dict(
            condition=condition, scope=str(group["scope"].iloc[0]),
            fixed_carrier_takeover_rate=(float(group["recovery_kind"].eq("relocated").mean())
                                         if str(group["scope"].iloc[0]) == "fixed_carrier_repeated"
                                         else float("nan")),
            paired_clean_regeneration_rate=(float(token_group.groupby(
                ["prompt_id", "seed", "layer"], observed=True)["newly_regenerated"].any().mean())
                                            if not token_group.empty else float("nan"))))
    return QuestionResult("q3", tidy, {"recovery": recovery,
                                        "token_history": history,
                                        "strict_summary": pd.DataFrame(summary_rows)}, _q3_verdict(recovery),
                          meta=dict(layer=layer, rescue_layers=rescue_layers,
                                    repeated_layers=later, point=point.value),
                          diagnostics=raw_diagnostics)


def _q3_histories(clean: Trace, treated: Trace, reference: "EP.CleanReference",
                  input_bars: Mapping[int, Tuple[float, float]], ctx: QuestionContext,
                  condition: str, prompt_id: int, seed: int,
                  edit_layers: Sequence[int]) -> List[Dict[str, Any]]:
    """Paired-clean, token-level evidence required before saying regenerated."""
    rows = []
    for layer in sorted(set(int(v) for v in edit_layers)):
        before = [v for key, v in treated.diagnostics.items()
                  if key[0] == ctx.step and key[1] == layer and key[3] == "x_pre_hook"]
        after = [v for key, v in treated.diagnostics.items()
                 if key[0] == ctx.step and key[1] == layer and key[3] == "x_post_hook"]
        clean_pre = clean.probe(ctx.step, layer, InterventionPoint.BLOCK_INPUT)
        clean_obs, treated_obs = clean.at(ctx.step, layer), treated.at(ctx.step, layer)
        if not before or not after or clean_pre is None or clean_obs is None or treated_obs is None:
            continue
        pre, post = before[0], after[0]
        v = ctx.vstar.float() / ctx.vstar.float().norm().clamp_min(1e-12)
        bars = reference.at(layer)
        input_norm_bar, input_alignment_bar = input_bars[layer]
        def register_mask(states):
            norms = states.float().norm(dim=-1)
            cosine = (states.float() @ v).abs() / norms.clamp_min(1e-12)
            return (norms >= input_norm_bar) & (cosine >= input_alignment_bar), cosine
        pre_mask, pre_cos = register_mask(pre)
        post_mask, post_cos = register_mask(post)
        clean_pre_mask, _ = register_mask(clean_pre)
        clean_mask = ((clean_obs.norm >= bars.norm_threshold) &
                      (clean_obs.cosine.float().abs() >= bars.alignment_threshold))
        treated_mask = ((treated_obs.norm >= bars.norm_threshold) &
                        (treated_obs.cosine.float().abs() >= bars.alignment_threshold))
        for token in range(min(len(pre_mask), len(clean_mask), len(treated_mask))):
            newly = (not bool(pre_mask[token]) and not bool(clean_pre_mask[token]) and
                     bool(treated_mask[token]) and not bool(post_mask[token]))
            rows.append(dict(question="q3", condition=condition, prompt_id=prompt_id, seed=seed,
                             layer=layer, token=token, pre_register=bool(pre_mask[token]),
                             post_register=bool(post_mask[token]),
                             clean_pre_register=bool(clean_pre_mask[token]),
                             clean_register=bool(clean_mask[token]),
                             treated_register=bool(treated_mask[token]), newly_regenerated=bool(newly),
                             pre_cosine=float(pre_cos[token]), post_cosine=float(post_cos[token])))
    return rows


def _q3_verdict(recovery: pd.DataFrame) -> str:
    if recovery.empty:
        return "Q3 produced no runs."
    primary = recovery[recovery["role"] == "intervention"]
    if primary.empty:
        return "Q3 ran controls only."
    share = primary["recovery_kind"].value_counts(normalize=True)
    leading = share.index[0]
    reading = {
        "same_position": "an original carrier wins again downstream",
        "relocated": "another carrier wins downstream; paired-clean token histories, not this label, "
                     "determine whether it is newly regenerated",
        "none": "v* does not return within the observed depth, so the register is written once "
                "rather than actively maintained",
    }[leading]
    layers = primary["first_recovery_layer"].dropna()
    when = f" First recovery occurs at layer {layers.mean():.1f} on average." if len(layers) else ""
    single = primary[primary["scope"] == "single_shot"]["recovery_kind"].eq("none").mean()
    repeated = primary[primary["scope"] == "fixed_carrier_repeated"]["recovery_kind"].eq("none").mean()
    contrast = ""
    if not math.isnan(single) and not math.isnan(repeated):
        contrast = (f" Repeated removal prevents recovery in {repeated:.0%} of runs against "
                    f"{single:.0%} for a single shot.")
    return (f"Q3: after norm-preserving direction destruction, {share[leading]:.0%} of runs show "
            f"'{leading.replace('_', ' ')}' -- {reading}.{when}{contrast}")


# ==============================================  reciprocal upstream-feature patching
@dataclass(frozen=True)
class Q4Feature:
    """One candidate upstream cause of register selection."""

    key: str
    label: str
    point: InterventionPoint
    kind: str = "full"          # full | vstar_only | topk_units


Q4_FEATURES: Tuple[Q4Feature, ...] = (
    Q4Feature("pre_mlp_residual", "Pre-feed-forward residual", InterventionPoint.PRE_MLP_RESIDUAL),
    Q4Feature("feedforward_activation", "Feed-forward hidden activation",
              InterventionPoint.MLP_HIDDEN, "topk_units"),
    Q4Feature("modulated_stream", "Timestep-modulated normalised stream",
              InterventionPoint.ADALN_MODULATION),
    Q4Feature("preexisting_direction", "Pre-existing register-direction component",
              InterventionPoint.PRE_MLP_RESIDUAL, "vstar_only"),
)
Q4_DIRECTIONS = ("register_to_ordinary", "ordinary_to_register")
Q4_DIRECTION_LABELS = {"register_to_ordinary": "Transfer into an ordinary token",
                       "ordinary_to_register": "Prevent at an eventual register"}
Q4_ENDPOINTS = ("dominant_channel_value", "target_norm_ratio", "cosine", "capture_rate")
Q4_ENDPOINT_LABELS = {
    "dominant_channel_value": "Dominant-channel magnitude",
    "target_norm_ratio": "Residual norm over layer median",
    "cosine": "Alignment with register direction",
    "capture_rate": "Becomes the attention sink",
}


def _transplant_edit(source: int, recipient: int, kind: str, vstar: torch.Tensor,
                     units: Sequence[int] = ()) -> Callable:
    """Copy one upstream feature from ``source`` onto ``recipient``."""
    units = [int(u) for u in units]

    def edit(image: torch.Tensor, _: EditContext) -> torch.Tensor:
        out = image.clone()
        if source >= out.shape[0] or recipient >= out.shape[0]:
            return out
        if kind == "full":
            out[recipient] = out[source]
        elif kind == "vstar_only":
            v = vstar.to(out).flatten()
            if v.numel() != out.shape[-1]:
                return out
            v = v / v.norm().clamp_min(1e-9)
            alpha_source = float(out[source] @ v)
            alpha_recipient = float(out[recipient] @ v)
            out[recipient] = out[recipient] + (alpha_source - alpha_recipient) * v
        elif kind == "topk_units":
            index = [u for u in units if u < out.shape[-1]]
            if index:
                out[recipient, index] = out[source, index]
        return out

    return edit


def _candidate_units(states: Optional[torch.Tensor], token: int, k: int = 32) -> Tuple[int, ...]:
    """Hidden units where the eventual register is most extreme.

    Localisation only: the causal claim comes from patching these units, not
    from the fact that they stood out.
    """
    if states is None or token >= states.shape[0]:
        return ()
    contrast = states[token].abs() - states.abs().median(dim=0).values
    return tuple(int(i) for i in torch.argsort(contrast, descending=True)[:k].tolist())


@_resumable("q4")
def run_q4(ctx: QuestionContext, *, features: Sequence[Q4Feature] = Q4_FEATURES,
           directions: Sequence[str] = Q4_DIRECTIONS, candidate_units: int = 32,
           layer: Optional[int] = None) -> QuestionResult:
    """What selects the sparse positions that receive the register write?

    Two halves.  The backward-prediction half asks whether pre-writer features
    already separate eventual registers from matched ordinary tokens, and is a
    localisation tool only.  The reciprocal-patching half is the causal claim: a
    feature counts only when copying it *creates* a register at an ordinary token
    and removing it *prevents* one at an eventual register.

    ``layer`` defaults to the latest layer of the writer range that exposes every
    feature, which is not always the range's end: a writer range is a claim about
    depth, and depth can cross an architectural boundary.  FLUX.1's range ends one
    layer into its fused single blocks, where there is no separate feed-forward
    stage to patch, while the dual block just before it has one.
    """
    writer = int(layer) if layer is not None else ctx.layer_exposing(
        [f.point for f in features], ctx.writer_range)
    downstream = [l for l in ctx.downstream_layers if l >= writer]
    supported: Dict[str, bool] = {}
    reference = ctx.driver.adapter.layers(ctx.driver.transformer)[writer]
    reference_kind = str(getattr(reference, "kind", "") or "")
    probe_points: List[Tuple[InterventionPoint, List[int]]] = [
        (InterventionPoint.BLOCK_INPUT, [ctx.intervention_layer])]
    for feature in features:
        capability = ctx.driver.adapter.intervention_capability(reference, feature.point)
        supported[feature.key] = bool(capability.supported)
        if capability.supported:
            probe_points.append((feature.point, [writer]))
    unsupported = {f.key: ctx.driver.adapter.intervention_capability(reference, f.point).reason
                   for f in features if not supported[f.key]}

    frames, patch_rows, separation_rows = [], [], []
    for prompt_id, prompt, seed in ctx.units():
        ctx.say(f"  [Q4] prompt {prompt_id} seed {seed}: clean pass")
        tracer = ctx.tracer()
        clean, _ = _clean_pass(ctx, (prompt_id, prompt, seed), probe_points=probe_points,
                               tracer=tracer)
        frozen = select_frozen_targets(clean, layer=ctx.intervention_layer, step=ctx.step,
                                       point=InterventionPoint.BLOCK_INPUT, direction=ctx.vstar,
                                       percentile=ctx.percentile, topk=ctx.topk,
                                       highnorm_ratio=ctx.highnorm_ratio)
        if not frozen.register_ids or not frozen.matched_ordinary_ids:
            continue
        register = int(frozen.register_ids[0])
        ordinary = int(frozen.matched_ordinary_ids[0])
        reference = _clean_reference(ctx, clean, frozen, downstream)

        # ---- backward prediction: do pre-writer features already separate them?
        for feature in features:
            if not supported[feature.key]:
                separation_rows.append(dict(question="q4", feature=feature.key,
                                            feature_label=feature.label, supported=False,
                                            prompt_id=prompt_id, seed=seed, separation=float("nan"),
                                            reason=unsupported.get(feature.key, "")))
                continue
            states = clean.probe(ctx.step, writer, feature.point)
            separation_rows.append(dict(
                question="q4", feature=feature.key, feature_label=feature.label, supported=True,
                prompt_id=prompt_id, seed=seed,
                separation=_separation(states, frozen.register_ids, frozen.matched_ordinary_ids),
                token_dependence=_token_dependence(states), reason=""))

        # ---- causal patching, with a sham for the paired contrast.  The sham
        # runs first and is measured at both recipients, so every patch effect
        # is a within-unit difference against the same untouched trajectory.
        sham_plan = EditPlan(edit=lambda image, _: image.clone(),
                             point=InterventionPoint.BLOCK_INPUT, layers=[writer], label="sham")
        conditions: List[Tuple[str, str, str, EditPlan]] = [
            ("sham", "none", f"{ordinary},{register}", sham_plan)]
        for feature in features:
            if not supported[feature.key]:
                # A refusal is a row, not a silence. Without it the feature simply
                # vanishes from the figure, which reads as "we tested it and found
                # nothing" rather than "this architecture could not be asked".
                for direction in directions:
                    for endpoint in Q4_ENDPOINTS:
                        patch_rows.append(dict(
                            question="q4", patch=feature.key, patch_label=feature.label,
                            direction=direction, direction_label=Q4_DIRECTION_LABELS[direction],
                            endpoint=endpoint, endpoint_label=Q4_ENDPOINT_LABELS[endpoint],
                            prompt_id=prompt_id, seed=seed, recipient=-1, value=float("nan"),
                            sham_value=float("nan"), effect=float("nan"), supported=False,
                            reason=unsupported.get(feature.key, "")))
                continue
            units = _candidate_units(clean.probe(ctx.step, writer, feature.point), register,
                                     candidate_units) if feature.kind == "topk_units" else ()
            for direction in directions:
                source, recipient = ((register, ordinary) if direction == "register_to_ordinary"
                                     else (ordinary, register))
                plan = EditPlan(edit=_transplant_edit(source, recipient, feature.kind, ctx.vstar, units),
                                point=feature.point, layers=[writer],
                                label=f"{feature.key}:{direction}")
                conditions.append((feature.key, direction, str(recipient), plan))

        sham_values: Dict[Tuple[str, int], float] = {}
        for feature_key, direction, recipient_text, plan in conditions:
            measured = [int(t) for t in recipient_text.split(",")]
            condition = f"{feature_key}__{direction}"
            treated, stats = run_traced_generation(
                ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
                condition=condition, plans=[plan], targets=frozen)
            for recipient in measured:
                rows = EP.measure_trace(clean, treated, frozen, layers=downstream, step=ctx.step,
                                        token_ids=[recipient], channels=ctx.channels,
                                        reference=reference)
                capture = EP.capture_rate(treated, recipient, layers=downstream, step=ctx.step)
                frame = _rows_to_frame(
                    rows, question="q4", condition=condition, patch=feature_key,
                    direction=direction, recipient=recipient, prompt_id=prompt_id, seed=seed,
                    step=ctx.step, capture_rate=capture,
                    perturbation_energy=stats.perturbation_energy)
                frames.append(frame)
                layer_rows = layer_level(frame)
                values = {"dominant_channel_value": _fraction(layer_rows, "dominant_channel_value"),
                          "target_norm_ratio": _fraction(layer_rows, "target_norm_ratio"),
                          "cosine": _fraction(layer_rows, "cosine"), "capture_rate": capture}
                if feature_key == "sham":
                    sham_values.update({(endpoint, recipient): value
                                        for endpoint, value in values.items()})
                    continue
                for endpoint, value in values.items():
                    baseline = sham_values.get((endpoint, recipient))
                    patch_rows.append(dict(
                        question="q4", patch=feature_key,
                        patch_label=next(f.label for f in features if f.key == feature_key),
                        direction=direction, direction_label=Q4_DIRECTION_LABELS[direction],
                        endpoint=endpoint, endpoint_label=Q4_ENDPOINT_LABELS[endpoint],
                        prompt_id=prompt_id, seed=seed, recipient=recipient, value=value,
                        sham_value=baseline,
                        effect=float("nan") if baseline is None else float(value - baseline),
                        supported=True, reason=""))

    # Empty frames are dropped rather than concatenated: pandas warns that it will
    # stop inferring dtypes across them, and a zero-row frame carries no columns to
    # infer from anyway.
    frames = [f for f in frames if not f.empty]
    tidy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    patches = pd.DataFrame(patch_rows)
    separation = pd.DataFrame(separation_rows)
    return QuestionResult("q4", tidy, {"patch_effects": patches, "separation": separation},
                          _q4_verdict(patches, separation, unsupported),
                          meta=dict(writer_layer=writer, writer_range=list(ctx.writer_range),
                                    layer_kind=reference_kind, unsupported=unsupported,
                                    checkpoint=str(getattr(ctx.cfg, "model", "")),
                                    candidate_units=candidate_units))


def _separation(states: Optional[torch.Tensor], positive: Sequence[int],
                negative: Sequence[int]) -> float:
    """Rank-based separability (AUC) of feature magnitude between two token sets.

    0.5 means the pre-writer feature carries no information about which tokens
    will become registers; 1.0 means it separates them perfectly.
    """
    if states is None or not len(positive) or not len(negative):
        return float("nan")
    limit = int(states.shape[0])
    a = [float(states[t].norm()) for t in positive if int(t) < limit]
    b = [float(states[t].norm()) for t in negative if int(t) < limit]
    if not a or not b:
        return float("nan")
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return float(wins / (len(a) * len(b)))


def _token_dependence(states: Optional[torch.Tensor]) -> float:
    """How much a feature varies across tokens, relative to its own scale.

    A modulation term that is identical at every token cannot select which token
    becomes a register; this number makes that visible rather than assumed.
    """
    if states is None or states.ndim != 2 or states.shape[0] < 2:
        return float("nan")
    spread = states.std(dim=0).mean()
    scale = states.abs().mean().clamp_min(1e-9)
    return float(spread / scale)


def _q4_verdict(patches: pd.DataFrame, separation: pd.DataFrame,
                unsupported: Mapping[str, str]) -> str:
    notes = ""
    if unsupported:
        listed = ", ".join(f"{k} ({v})" for k, v in unsupported.items())
        notes = f" Unsupported on this architecture and therefore not tested: {listed}."
    if patches.empty:
        return "Q4 produced no patching runs." + notes
    capture = patches[patches["endpoint"] == "capture_rate"]
    grid = capture.pivot_table(index="patch", columns="direction", values="effect", aggfunc="mean")
    reciprocal = {}
    for patch in grid.index:
        transfer = grid.loc[patch].get("register_to_ordinary", float("nan"))
        prevent = grid.loc[patch].get("ordinary_to_register", float("nan"))
        if not math.isnan(transfer) and not math.isnan(prevent):
            reciprocal[patch] = float(transfer) - float(prevent)
    if not reciprocal:
        return ("Q4: patching ran but no feature has both directions, so no reciprocal claim is "
                "supported yet." + notes)
    best = max(reciprocal, key=reciprocal.get)
    score = reciprocal[best]
    strong = score > 0
    best_separation = float("nan")
    if not separation.empty:
        by_feature = separation.groupby("feature")["separation"].mean()
        if best in by_feature.index:
            best_separation = float(by_feature[best])
    verdict = (f"Q4: the strongest reciprocal candidate is '{best}' (transfer minus prevention "
               f"{score:+.2f} in sink-capture rate; pre-writer separability {best_separation:.2f}). ")
    verdict += ("Copying it creates and removing it prevents register behaviour in the predicted "
                "directions, which is what a writer feature has to do."
                if strong else
                "The reciprocal pattern does not go in the predicted direction, so no upstream "
                "feature tested here transfers register formation.")
    return verdict + notes


# ======================================================  sufficiency factorization
@dataclass(frozen=True)
class Q5Stage:
    """One rung of the residual-to-key ladder."""

    key: str
    label: str
    point: InterventionPoint
    kind: str          # residual transplant stage, or "final_key"


Q5_STAGES: Tuple[Q5Stage, ...] = (
    Q5Stage("direction_at_ordinary_norm", "Direction only, at the recipient's own norm",
            InterventionPoint.BLOCK_INPUT, "vstar_ordinary_norm"),
    Q5Stage("direction_at_register_norm", "Direction only, at the register's norm",
            InterventionPoint.BLOCK_INPUT, "vstar_natural_norm"),
    Q5Stage("full_residual_state", "Whole residual state",
            InterventionPoint.BLOCK_INPUT, "full_residual"),
    Q5Stage("normalised_residual_state", "Normalised state entering key projection",
            InterventionPoint.PRE_KEY_NORM_RESIDUAL, "full_residual"),
    Q5Stage("key_before_position", "Key before positional encoding",
            InterventionPoint.KEY_PRE_POSITION, "full_residual"),
    Q5Stage("final_key", "Final key presented to attention",
            InterventionPoint.FINAL_KEY, "final_key"),
)
Q5_POSITIONS = ("natural_register", "adjacent_patch", "random_ordinary")
Q5_POSITION_LABELS = {"natural_register": "The natural register itself",
                      "adjacent_patch": "A spatially adjacent patch",
                      "random_ordinary": "A random ordinary token"}


@_resumable("q5")
def run_q5(ctx: QuestionContext, *, stages: Sequence[Q5Stage] = Q5_STAGES,
           positions: Sequence[str] = Q5_POSITIONS,
           transfer_modes: Sequence[str] = ("copy", "move")) -> QuestionResult:
    """Where on the residual-to-key path does the register become sufficient?

    Copying only the direction onto an arbitrary token fails to reproduce capture.
    The ladder climbs from that failure towards the tensor attention actually
    consumes; the first rung whose transplant reproduces natural capture names the
    missing ingredient (residual context, normalisation, positional processing,
    or head-specific key structure).
    """
    layer = ctx.intervention_layer
    downstream = [l for l in ctx.downstream_layers if l >= layer]
    reference = ctx.driver.adapter.layers(ctx.driver.transformer)[layer]
    support: Dict[str, Tuple[bool, str]] = {}
    for stage in stages:
        if stage.point is InterventionPoint.FINAL_KEY:
            support[stage.key] = (True, "")
            continue
        capability = ctx.driver.adapter.intervention_capability(reference, stage.point)
        support[stage.key] = (bool(capability.supported), capability.reason)

    rows, raw_diagnostics = [], {}
    for prompt_id, prompt, seed in ctx.units():
        ctx.say(f"  [Q5] prompt {prompt_id} seed {seed}: clean pass")
        tracer = ctx.tracer(key_layers=[layer])
        probe_points = [(InterventionPoint.BLOCK_INPUT, [layer])]
        probe_points += [(stage.point, [layer]) for stage in stages
                         if support[stage.key][0] and stage.point not in
                         (InterventionPoint.BLOCK_INPUT, InterventionPoint.FINAL_KEY)]
        clean, _ = _clean_pass(ctx, (prompt_id, prompt, seed), probe_points=probe_points,
                               tracer=tracer)
        frozen = select_frozen_targets(clean, layer=layer, step=ctx.step,
                                       point=InterventionPoint.BLOCK_INPUT, direction=ctx.vstar,
                                       percentile=ctx.percentile, topk=ctx.topk,
                                       highnorm_ratio=ctx.highnorm_ratio)
        if not frozen.register_ids:
            continue
        source = int(frozen.register_ids[0])
        natural = EP.clean_capture_rate(clean, frozen, layers=downstream, step=ctx.step)
        clean_keys = {int(l): clean.at(ctx.step, l).keys for l in [layer]
                      if clean.at(ctx.step, l) is not None and clean.at(ctx.step, l).keys is not None}
        recipients = {"natural_register": source, "adjacent_patch": frozen.nearby_id,
                      "random_ordinary": frozen.random_recipient_id}

        for stage in stages:
            ok, reason = support[stage.key]
            for position in positions:
                recipient = recipients.get(position)
                modes = ("self_patch",) if position == "natural_register" else tuple(transfer_modes)
                for transfer_mode in modes:
                    base = dict(question="q5", stage=stage.key, stage_label=stage.label,
                            position=position, position_label=Q5_POSITION_LABELS[position],
                            transfer_mode=transfer_mode,
                            prompt_id=prompt_id, seed=seed, recipient=recipient,
                            target_token_ids=json.dumps(list(frozen.register_ids)), source=source,
                            supported=ok, reason=reason, any_register_clean_rate=natural,
                            matched_clean_rate=(float("nan") if recipient is None else
                                                EP.token_capture_rate(clean, int(recipient),
                                                                      layer=layer, step=ctx.step)))
                    if not ok or recipient is None:
                        rows.append({**base, "temporal_endpoint": "same_operation", "layer": layer,
                                     "capture_rate": float("nan")})
                        continue
                    if stage.kind == "final_key":
                        if not clean_keys:
                            rows.append({**base, "temporal_endpoint": "same_operation", "layer": layer,
                                         "capture_rate": float("nan"),
                                         "reason": "no clean keys captured at this layer"})
                            continue
                        from .causal_engine import FinalKeyPatch
                        patch = FinalKeyPatch(tracer, layers=[layer], source=source,
                                              recipient=int(recipient), clean_keys=clean_keys,
                                              move=transfer_mode == "move")
                        treated, _ = run_traced_generation(
                            ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
                            condition=f"{stage.key}__{position}__{transfer_mode}", key_patch=patch)
                    else:
                        plan = EditPlan(edit=_q5_edit(stage.kind, source, int(recipient), ctx.vstar,
                                                      frozen, move=transfer_mode == "move"),
                                        point=stage.point, layers=[layer],
                                        label=f"{stage.key}:{position}:{transfer_mode}")
                        treated, _ = run_traced_generation(
                            ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
                            condition=f"{stage.key}__{position}__{transfer_mode}", plans=[plan],
                            targets=frozen)
                    if position == "natural_register" and stage.key in (
                            "final_key", "full_residual_state"):
                        clean_layer, treated_layer = clean.at(ctx.step, layer), treated.at(ctx.step, layer)
                        if clean_layer is None or treated_layer is None:
                            raise AssertionError("self-patch layer was not captured")
                        if stage.key == "final_key":
                            DG.assert_identity(clean_layer.keys, treated_layer.keys,
                                               name="final key self-patch")
                            DG.assert_identity(clean_layer.attention_probs,
                                               treated_layer.attention_probs,
                                               name="final key attention probabilities")
                            DG.assert_identity(clean_layer.attention_logits,
                                               treated_layer.attention_logits,
                                               name="final key attention logits")
                        elif clean_layer.keys is not None and treated_layer.keys is not None:
                            DG.assert_identity(clean_layer.keys, treated_layer.keys,
                                               name="full residual self-patch downstream keys")
                    raw_diagnostics.update(diagnostic_tensors(
                        treated, question="q5",
                        condition=f"{stage.key}__{position}__{transfer_mode}"))
                    observed = treated.at(ctx.step, layer)
                    if observed is not None:
                        for name in ("keys", "queries", "attention_logits", "attention_probs"):
                            value = getattr(observed, name)
                            if value is not None:
                                raw_diagnostics[
                                    f"q5/{stage.key}/{position}/{transfer_mode}/layer{layer}/{name}"] = value
                    profile = EP.capture_profile(treated, int(recipient), intervention_layer=layer,
                                                 layers=downstream, step=ctx.step)
                    inserted_cosine = float("nan")
                    post_states = [value for key, value in treated.diagnostics.items()
                                   if key[0] == ctx.step and key[1] == layer and
                                   key[3] == "x_post_hook"]
                    if post_states and int(recipient) < post_states[0].shape[0] and \
                            post_states[0].shape[-1] == ctx.vstar.numel():
                        state = post_states[0][int(recipient)].float()
                        unit = ctx.vstar.float() / ctx.vstar.float().norm().clamp_min(1e-12)
                        inserted_cosine = float((state @ unit) / state.norm().clamp_min(1e-12))
                    for endpoint in profile:
                        observation = treated.at(ctx.step, int(endpoint["layer"]))
                        key_similarity = float("nan")
                        recipient_rank = float("nan")
                        if observation is not None and observation.keys is not None:
                            source_keys = observation.keys[:, source, :].float()
                            recipient_keys = observation.keys[:, int(recipient), :].float()
                            key_similarity = float(torch.nn.functional.cosine_similarity(
                                source_keys, recipient_keys, dim=-1).mean())
                        if observation is not None and observation.incoming is not None:
                            order = torch.argsort(observation.incoming, dim=-1, descending=True)
                            recipient_rank = float((order == int(recipient)).nonzero()[:, 1].float().mean() + 1)
                        rows.append({**base, **endpoint,
                                     "matched_clean_rate": EP.token_capture_rate(
                                         clean, int(recipient), layer=int(endpoint["layer"]), step=ctx.step),
                                     "any_register_clean_rate": EP.clean_capture_rate(
                                         clean, frozen, layers=[int(endpoint["layer"])], step=ctx.step),
                                     "recipient_cosine_immediately_after_patch": inserted_cosine,
                                     "recipient_source_key_similarity": key_similarity,
                                     "recipient_attention_rank": recipient_rank,
                                     "source_capture_rate": EP.token_capture_rate(
                                         treated, source, layer=int(endpoint["layer"]), step=ctx.step)})

    ladder = pd.DataFrame(rows)
    return QuestionResult("q5", ladder, {"support": _support_frame(support)},
                          _q5_verdict(ladder, support), meta=dict(layer=layer, support={
                              k: {"supported": v[0], "reason": v[1]} for k, v in support.items()}),
                          diagnostics=raw_diagnostics)


def _q5_edit(kind: str, source: int, recipient: int, vstar: torch.Tensor,
             frozen: FrozenTargets, *, move: bool = False) -> Callable:
    def edit(image: torch.Tensor, _: EditContext) -> torch.Tensor:
        if source >= image.shape[0] or recipient >= image.shape[0]:
            return image.clone()
        if kind in ("vstar_ordinary_norm", "vstar_natural_norm") and \
                int(vstar.numel()) != int(image.shape[-1]):
            # A direction defined in residual space cannot be written into a key
            # space of different width; refusing beats transplanting nonsense.
            return image.clone()
        out = transplant(image, source, recipient, stage=kind, vstar=vstar,
                         natural_norm=frozen.register_norm if kind == "vstar_natural_norm" else None)
        if move and source != recipient:
            replacement = frozen.states_for("matched")
            if replacement is not None and len(replacement):
                out[source] = replacement[0].to(out)
        return out
    return edit


def _support_frame(support: Mapping[str, Tuple[bool, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"stage": key, "supported": value[0], "reason": value[1]}
                         for key, value in support.items()])


def _q5_verdict(ladder: pd.DataFrame, support: Mapping[str, Tuple[bool, str]]) -> str:
    if ladder.empty:
        return "Q5 produced no runs."
    arbitrary = ladder[ladder["position"] != "natural_register"]
    natural = float(ladder["any_register_clean_rate"].mean())
    immediate = (arbitrary[arbitrary["temporal_endpoint"] == "same_operation"]
                 if "temporal_endpoint" in arbitrary else arbitrary)
    means = immediate.groupby("stage")["capture_rate"].mean().dropna()
    skipped = [k for k, v in support.items() if not v[0]]
    note = (f" Not exposed by this architecture, so untested: {', '.join(skipped)}."
            if skipped else "")
    if means.empty:
        return "Q5: no stage produced a measurable capture rate." + note
    if not natural > 0:
        return ("Q5: no head sinks on a natural register in the clean run, so there is no capture "
                f"reference to compare against; the best transplant reaches {means.max():.0%} at "
                f"'{means.idxmax()}'. The ladder is uninterpretable until the clean run shows "
                "natural capture." + note)
    reaching = [stage for stage in [s.key for s in Q5_STAGES] if stage in means.index
                and means[stage] >= 0.8 * natural]
    if not reaching:
        return (f"Q5: no transplanted stage reaches the any-register clean rate ({natural:.0%}); "
                f"the best same-operation recipient capture is '{means.idxmax()}' at "
                f"{means.max():.0%}. This does not locate an ingredient beyond the final key; "
                "query competition and relational attention context remain possible." + note)
    first = reaching[0]
    reading = {
        "direction_at_ordinary_norm": "direction alone is sufficient once it is present at all",
        "direction_at_register_norm": "direction plus register magnitude is sufficient, so magnitude "
                                      "is the missing factor",
        "full_residual_state": "the whole residual context is needed, so surrounding channels matter "
                               "beyond the register direction",
        "normalised_residual_state": "normalisation is the missing step: the state only becomes "
                                     "sufficient after the layer norm the key projection consumes",
        "final_key": "only the final key suffices, so position-dependent and head-specific key "
                     "processing is the missing ingredient",
        "key_before_position": "the key before positional encoding suffices, so key projection rather "
                               "than position carries the advantage",
    }.get(first, "this stage is the earliest sufficient representation")
    return (f"Q5: capture becomes reliable at '{first}' ({means[first]:.0%} against a natural "
            f"{natural:.0%}), so {reading}." + note)


# ==========================================================  dissolution
@dataclass(frozen=True)
class Q6Condition:
    key: str
    label: str
    kind: str          # sham | suppress | amplify | refresh | suppress_random | suppress_energy
    zone: str          # dissolution | register
    gamma: float = 0.0
    role: str = "intervention"


Q6_CONDITIONS: Tuple[Q6Condition, ...] = (
    Q6Condition("sham", "Sham hook", "sham", "dissolution", 1.0, "control"),
    Q6Condition("competing_channel_suppressed", "Competing channel suppressed late",
                "suppress", "dissolution", 0.0),
    Q6Condition("competing_channel_amplified_early", "Competing channel amplified early",
                "amplify", "register", 2.0),
    Q6Condition("pure_vstar_replacement", "Pure-v* replacement at unchanged norm",
                "pure_vstar", "dissolution", 1.0),
    Q6Condition("direction_refreshed", "Parallel component refreshed; context preserved",
                "refresh_component", "dissolution", 1.0),
    Q6Condition("competing_channel_rotated_early", "Norm-preserving rotation toward competitor",
                "rotate_competitor", "register", 2.0, "control"),
    Q6Condition("unspecific_channel_amplified_early", "Unspecific channel amplified early",
                "amplify_unspecific", "register", 2.0, "control"),
    Q6Condition("orthogonal_energy_amplified_early", "Orthogonal energy amplified early",
                "amplify_orthogonal", "register", 2.0, "control"),
    Q6Condition("random_channel_suppressed", "Random channel suppressed late",
                "suppress_random", "dissolution", 0.0, "control"),
    Q6Condition("matched_energy_removed", "Magnitude-matched removal late",
                "suppress_energy", "dissolution", 0.0, "control"),
)


def _q6_edit(ctx: QuestionContext, condition: Q6Condition, ids: Sequence[int],
             targets: Optional[FrozenTargets] = None) -> Callable:
    ids = [int(t) for t in ids]
    random_channel = _rng_channel(ctx)

    def edit(image: torch.Tensor, _: EditContext) -> torch.Tensor:
        if condition.kind == "sham" or not ids:
            return image.clone()
        if condition.kind in ("suppress", "amplify"):
            return scale_channel(image, ctx.competitor_channel, condition.gamma, ids)
        if condition.kind == "suppress_random":
            return scale_channel(image, random_channel, condition.gamma, ids)
        if condition.kind == "pure_vstar":
            return refresh_direction(image, ctx.vstar, ids)
        if condition.kind == "refresh_component":
            states = targets.states_for("register") if targets is not None else None
            v = ctx.vstar.to(states) if states is not None else None
            v = v / v.norm().clamp_min(1e-12) if v is not None else None
            alpha_target = float((states @ v).mean()) if states is not None else 0.0
            return refresh_parallel_component(image, ctx.vstar, ids, alpha_target)
        if condition.kind == "rotate_competitor":
            return rotate_toward_channel_preserve_norm(
                image, ctx.competitor_channel, ids, condition.gamma)
        if condition.kind == "amplify_unspecific":
            return scale_channel(image, ctx.control_channel, condition.gamma, ids)
        if condition.kind == "amplify_orthogonal":
            source = image[ids, ctx.competitor_channel].float()
            energy = float((condition.gamma ** 2 - 1.0) * source.pow(2).sum())
            axis = _random_unit(image.shape[-1], 47, orthogonal_to=ctx.vstar)
            return add_matched_energy(image, axis, ids, max(energy, 0.0))
        if condition.kind == "suppress_energy":
            energy = float((1.0 - condition.gamma ** 2) *
                           image[ids, ctx.competitor_channel].float().pow(2).sum())
            axis = _random_unit(image.shape[-1], 23, orthogonal_to=ctx.vstar)
            return remove_matched_energy(image, axis, ids, max(energy, 0.0))
        raise KeyError(condition.kind)

    return edit


@_resumable("q6")
def run_q6(ctx: QuestionContext, *, conditions: Sequence[Q6Condition] = Q6_CONDITIONS) -> QuestionResult:
    """What ends the register state?

    This asks whether the late-growing competitor *causes* dissolution.
    Suppressing it late should extend the
    register's life, amplifying it early should shorten it, and a direction
    refresh at unchanged magnitude separates "the state rotates away" from "one
    channel outgrows it".
    """
    point = InterventionPoint.BLOCK_INPUT
    late = list(ctx.dissolution_layers)
    early = [l for l in ctx.register_layers if l < late[0]] or list(ctx.register_layers)
    lifetime_layers = sorted(set(ctx.register_layers) | set(late))
    frames, lifetimes, trajectories, diagnostics, raw_diagnostics = [], [], [], [], {}

    for prompt_id, prompt, seed in ctx.units():
        ctx.say(f"  [Q6] prompt {prompt_id} seed {seed}: clean pass")
        tracer = ctx.tracer()
        clean, _ = _clean_pass(ctx, (prompt_id, prompt, seed),
                               probe_points=[(point, [ctx.intervention_layer])], tracer=tracer)
        frozen = select_frozen_targets(clean, layer=ctx.intervention_layer, step=ctx.step,
                                       point=point, direction=ctx.vstar, percentile=ctx.percentile,
                                       topk=ctx.topk, highnorm_ratio=ctx.highnorm_ratio)
        ids = list(frozen.register_ids)
        reference = _clean_reference(ctx, clean, frozen, lifetime_layers)
        clean_lifetime = EP.register_lifetime(clean, frozen, layers=lifetime_layers, step=ctx.step,
                                              reference=reference)
        clean_at_start = clean.at(ctx.step, frozen.layer)
        projection_threshold = (float(clean_at_start.projection[list(frozen.register_ids)].abs().mean()) * 0.5
                                if clean_at_start is not None and clean_at_start.projection is not None
                                and frozen.register_ids else None)
        clean_lifetimes = EP.register_lifetimes(
            clean, frozen, layers=lifetime_layers, step=ctx.step, reference=reference,
            projection_threshold=projection_threshold)
        clean_half_life = EP.alignment_half_life(clean, frozen, layers=lifetime_layers, step=ctx.step)
        trajectories.extend(_decomposition_rows(clean, frozen, ids, ctx, condition="clean",
                                                prompt_id=prompt_id, seed=seed,
                                                layers=lifetime_layers))

        for condition in conditions:
            layers = late if condition.zone == "dissolution" else early
            plan = EditPlan(edit=_q6_edit(ctx, condition, ids, frozen), point=point, layers=layers,
                            label=condition.key)
            treated, stats = run_traced_generation(
                ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
                condition=condition.key, plans=[plan], targets=frozen)
            diagnostics.extend(diagnostic_rows(
                treated, question="q6", condition=condition.key, vstar=ctx.vstar,
                channels=ctx.channels, token_ids=ids))
            raw_diagnostics.update(diagnostic_tensors(
                treated, question="q6", condition=condition.key))
            samples = {key[3]: value for key, value in treated.diagnostics.items()
                       if key[0] == ctx.step and key[1] in layers}
            if condition.key == "competing_channel_suppressed" and "x_post_hook" in samples:
                DG.assert_channel_zero(samples["x_post_hook"], ctx.competitor_channel, ids)
            if condition.kind == "refresh_component" and {"x_pre_hook", "x_post_hook"} <= set(samples):
                pre, post = samples["x_pre_hook"], samples["x_post_hook"]
                unit = ctx.vstar.float() / ctx.vstar.float().norm().clamp_min(1e-12)
                pre_perp = pre[ids] - (pre[ids].float() @ unit)[:, None].to(pre) * unit.to(pre)
                post_perp = post[ids] - (post[ids].float() @ unit)[:, None].to(post) * unit.to(post)
                DG.assert_identity(pre_perp, post_perp, name="component-preserving refresh")
            rows = EP.measure_trace(clean, treated, frozen, layers=lifetime_layers, step=ctx.step,
                                    token_ids=ids, channels=ctx.channels, reference=reference)
            treated_lifetime = EP.register_lifetime(treated, frozen, layers=lifetime_layers,
                                                    step=ctx.step, reference=reference)
            treated_half_life = EP.alignment_half_life(treated, frozen, layers=lifetime_layers,
                                                       step=ctx.step)
            separate_lifetimes = EP.register_lifetimes(
                treated, frozen, layers=lifetime_layers, step=ctx.step, reference=reference,
                projection_threshold=projection_threshold)
            lifetime_shifts = {
                f"{name}_shift": (None if value is None or clean_lifetimes[name] is None
                                   else int(value - clean_lifetimes[name]))
                for name, value in separate_lifetimes.items()}
            shift = (None if treated_lifetime is None or clean_lifetime is None
                     else int(treated_lifetime - clean_lifetime))
            half_life_shift = (None if treated_half_life is None or clean_half_life is None
                               else float(treated_half_life - clean_half_life))
            frame = _rows_to_frame(
                rows, question="q6", condition=condition.key, condition_label=condition.label,
                role=condition.role, zone=condition.zone, gamma=condition.gamma,
                prompt_id=prompt_id, seed=seed, step=ctx.step,
                target_token_ids=json.dumps(ids),
                register_lifetime=treated_lifetime, clean_lifetime=clean_lifetime,
                lifetime_shift=shift, alignment_half_life=treated_half_life,
                half_life_shift=half_life_shift, perturbation_energy=stats.perturbation_energy)
            frame["sink_retention"] = frame["head_retention"].fillna(frame["sink_retention_all"])
            frames.append(frame)
            trajectories.extend(_decomposition_rows(treated, frozen, ids, ctx,
                                                    condition=condition.key, prompt_id=prompt_id,
                                                    seed=seed, layers=lifetime_layers))
            lifetimes.append(dict(question="q6", condition=condition.key,
                                  condition_label=condition.label, role=condition.role,
                                  prompt_id=prompt_id, seed=seed, clean_lifetime=clean_lifetime,
                                  register_lifetime=treated_lifetime, lifetime_shift=shift,
                                  clean_half_life=clean_half_life,
                                  alignment_half_life=treated_half_life,
                                  half_life_shift=half_life_shift,
                                  **{f"clean_{k}": v for k, v in clean_lifetimes.items()},
                                  **separate_lifetimes, **lifetime_shifts))

    tidy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    lifetime = pd.DataFrame(lifetimes)
    trajectory = pd.DataFrame(trajectories)
    return QuestionResult("q6", tidy, {"lifetime": lifetime, "trajectory": trajectory,
                                        "diagnostics": pd.DataFrame(diagnostics)},
                          _q6_verdict(lifetime), meta=dict(dissolution_layers=late,
                                                           early_layers=early, point=point.value),
                          diagnostics=raw_diagnostics)


def _decomposition_rows(trace: Trace, frozen: FrozenTargets, ids: Sequence[int],
                        ctx: QuestionContext, *, condition: str, prompt_id: int, seed: int,
                        layers: Sequence[int]) -> List[Dict[str, Any]]:
    """x = alpha v* + x_perp through the late register zone, per layer."""
    out = []
    previous_cosine = None
    for layer in sorted(int(l) for l in layers):
        observation = trace.at(ctx.step, layer)
        if observation is None or observation.projection is None:
            continue
        valid = [t for t in ids if int(t) < observation.projection.numel()]
        if not valid:
            continue
        alphas = observation.projection[valid].float()
        alpha = float(alphas.mean())
        mean_abs_alpha = float(alphas.abs().mean())
        rms_alpha = float(alphas.pow(2).mean().sqrt())
        norm = float(observation.norm[valid].pow(2).mean().sqrt())
        # Exact per-token orthogonal norms. Do not fold between-token variance in
        # alpha into a quantity labelled perpendicular norm.
        perp = (observation.norm[valid].float().pow(2) - alphas.pow(2)).clamp_min(0).sqrt()
        cosine = float(observation.cosine[valid].mean())
        angular = (float("nan") if previous_cosine is None
                   else float(math.acos(max(-1.0, min(1.0, cosine))) -
                              math.acos(max(-1.0, min(1.0, previous_cosine)))))
        previous_cosine = cosine
        row = dict(question="q6", condition=condition, prompt_id=prompt_id, seed=seed,
                   layer=layer, alpha=alpha, cosine=cosine,
                   mean_abs_alpha=mean_abs_alpha, rms_alpha=rms_alpha,
                   mean_perp_norm=float(perp.mean()), rms_perp_norm=float(perp.pow(2).mean().sqrt()),
                   # Backward-compatible alias now has the literal per-token meaning.
                   perpendicular_norm=float(perp.mean()),
                   angular_velocity=angular, norm=norm)
        if observation.channel_values is not None and observation.channel_values.shape[0] >= 2:
            row["dominant_channel"] = float(observation.channel_values[0][valid].abs().mean())
            row["competing_channel"] = float(observation.channel_values[1][valid].abs().mean())
        if observation.incoming is not None:
            row["sink_strength"] = float(observation.incoming[:, valid].sum(-1).mean())
        if observation.qk_cosine is not None:
            row["key_rank"] = _mean([EP.key_rank(observation.qk_cosine, h, valid)
                                     for h in range(observation.qk_cosine.shape[0])])
        out.append(row)
    return out


def _q6_verdict(lifetime: pd.DataFrame) -> str:
    if lifetime.empty:
        return "Q6 produced no runs."
    # The threshold-based lifetime is the predeclared endpoint; the alignment
    # half-life stands in only when no layer crossed that threshold at all.
    column, basis = "lifetime_shift", "high-norm register lifetime"
    if lifetime["lifetime_shift"].dropna().empty:
        column, basis = "half_life_shift", "alignment half-life (the threshold endpoint was never crossed)"
    shift = lifetime.groupby("condition")[column].mean()
    if shift.dropna().empty:
        return ("Q6: neither the register lifetime nor its alignment half-life could be measured on "
                "this run, so the dissolution question is untested here rather than answered.")
    suppression = shift.get("competing_channel_suppressed", float("nan"))
    refresh = shift.get("direction_refreshed", float("nan"))
    amplify = shift.get("competing_channel_amplified_early", float("nan"))
    control = _mean([shift.get("random_channel_suppressed", float("nan")),
                     shift.get("matched_energy_removed", float("nan"))])
    prolongs = (lambda x: not math.isnan(x) and x > max(0.0, control if not math.isnan(control) else 0.0))
    if prolongs(suppression):
        reading = ("suppressing the late-growing competitor extends the register's life beyond every "
                   "matched control, which supports the channel-competition hypothesis")
    elif prolongs(refresh):
        reading = ("re-projecting the state toward the register direction at unchanged magnitude "
                   "extends its life while channel suppression does not, so dissolution is a "
                   "geometric rotation rather than the growth of one channel")
    else:
        reading = ("neither competing-channel suppression nor direction refresh extends the state, "
                   "so the model appears to erase or re-encode the register through a broader "
                   "subspace than the one channel tested")
    return (f"Q6: mean shift in {basis} is {_fmt(suppression)} layers for late competitor "
            f"suppression, {_fmt(refresh)} for direction refresh and {_fmt(amplify)} for early "
            f"amplification, against {_fmt(control)} for matched controls -- {reading}.")


QUESTION_RUNNERS: Dict[str, Callable[..., QuestionResult]] = {
    "q1": run_q1, "q2": run_q2, "q3": run_q3, "q4": run_q4, "q5": run_q5, "q6": run_q6,
}

QUESTION_TITLES = {
    "q1": "Where do the sinks go when the natural registers are removed?",
    "q2": "What happens downstream after live dominant-channel suppression?",
    "q3": "Does the network regenerate or relocate the register direction?",
    "q4": "What selects the sparse token positions that receive the register write?",
    "q5": "What makes the register direction sufficient at a natural register only?",
    "q6": "What causes the register state to dissolve?",
}


def run_questions(ctx: QuestionContext, questions: Sequence[str] = tuple(QUESTION_RUNNERS),
                  **kwargs) -> Dict[str, QuestionResult]:
    """Run several questions against one context, saving each as it finishes."""
    results: Dict[str, QuestionResult] = {}
    for question in questions:
        key = question.lower()
        ctx.say(f"\n=== {key.upper()}: {QUESTION_TITLES[key]} ===")
        result = QUESTION_RUNNERS[key](ctx, **kwargs.get(key, {}))
        results[key] = result
        ctx.say(result.verdict)
    return results


# ======================================================= structure interaction
# The three sparse structures under study, as things you can see on the image
# grid, and the three ways of taking one away.
STRUCTURES: Tuple[Tuple[str, str], ...] = (
    ("high_norm_tokens", "High-norm tokens"),
    ("dominant_channel", "Dominant channel"),
    ("attention_sinks", "Attention sinks"),
)
REMOVALS: Tuple[Tuple[str, str], ...] = (
    ("clean", "Nothing removed"),
    ("high_norm_tokens", "High-norm tokens removed"),
    ("dominant_channel", "Dominant channel suppressed"),
    ("attention_sinks", "Attention sinks suppressed"),
)
STRUCTURE_NOTES = {
    "high_norm_tokens": "residual-stream norm of each patch, over the median patch norm of the "
                        "untouched run at this layer",
    "dominant_channel": "absolute activation of the dominant register channel at each patch",
    "attention_sinks": "share of incoming image-to-image attention the strongest head sends "
                       "to each patch",
}

# Two ways of asking "what would this look like with that structure gone", and
# they answer different questions.
#
# SUBTRACT is bookkeeping.  The model is never re-run: the quantity is recomputed
# from the states the untouched pass already produced, with the structure masked
# out of the arithmetic.  It answers "how much of what we are looking at is only
# this structure being counted twice", and it is exact: nothing downstream had
# a chance to react, so there is nothing to confound it.
#
# CAUSAL runs the generation again with the structure taken away, so every block
# after the intervention is free to respond.  It answers "what does the network
# do about it", at the price of every downstream reaction being part of the answer.
#
# Subtraction is defined only where the measured quantity is a function of the
# captured residual-stream states *and* the removal is a mask on those same
# states.  Attention is neither: it is formed inside the block from queries and
# keys, so masking a channel out of the residual stream afterwards says nothing
# about where a head would have looked, and key-space sink suppression is not a
# mask on the residual stream at all.  Those panels can only be filled by running
# the model, and the figure labels the row rather than quietly mixing the two.
SUBTRACTABLE: Tuple[Tuple[str, str], ...] = (
    ("high_norm_tokens", "high_norm_tokens"),
    ("high_norm_tokens", "dominant_channel"),
    ("high_norm_tokens", "attention_sinks"),
    ("dominant_channel", "high_norm_tokens"),
    ("dominant_channel", "dominant_channel"),
)
METHOD_NOTES = {
    "clean": "the untouched run",
    "subtract": "recomputed from the untouched states with the structure masked out of the "
                "arithmetic; the model is not re-run, so nothing downstream can react",
    "causal": "the generation re-run with the structure taken away, so every later block is "
              "free to respond",
}


def ablation_method(removal: str, structure: str) -> str:
    """Which of the two methods can fill the panel for this pair."""
    if removal == "clean":
        return "clean"
    return "subtract" if (removal, structure) in SUBTRACTABLE else "causal"


def structure_map(observation, structure: str, *, scale: Optional[float] = None
                  ) -> Optional[torch.Tensor]:
    """One structure as a value per image patch, ready to lay on the grid.

    ``scale`` is the median patch norm to divide by.  Pass the untouched run's
    value when mapping a treated run: a map that renormalises itself by its own
    median cannot show a structure shrinking, because the yardstick shrinks with
    it.
    """
    if observation is None:
        return None
    if structure == "high_norm_tokens":
        if observation.norm is None:
            return None
        divisor = float(observation.norm.median()) if scale is None else float(scale)
        return observation.norm.float() / max(divisor, 1e-9)
    if structure == "dominant_channel":
        if observation.channel_values is None or not observation.channel_values.numel():
            return None
        return observation.channel_values[0].float().abs()
    if structure == "attention_sinks":
        if observation.incoming is None:
            return None
        return observation.incoming.float().max(dim=0).values
    raise KeyError(f"unknown structure {structure!r}; known: {[k for k, _ in STRUCTURES]}")


def _norm_without(states: torch.Tensor, channels: Sequence[int]) -> torch.Tensor:
    """Norm with the listed channels removed, by masking rather than subtracting.

    ``sum(x^2) - x_c^2`` is the same number in exact arithmetic and a far worse
    one in floating point.  Exactly where the register lives, one channel holds
    almost all of a patch's squared activation, so that form is the difference of
    two nearly equal large numbers and keeps hardly any significant digits.
    Masking is exact.
    """
    keep = torch.ones(states.shape[-1], dtype=torch.bool, device=states.device)
    index = torch.as_tensor([int(c) for c in channels], device=states.device, dtype=torch.long)
    keep[index.clamp_max(states.shape[-1] - 1)] = False
    return states[:, keep].norm(dim=-1)


def _attention_without(incoming: torch.Tensor, tokens: Sequence[int]) -> torch.Tensor:
    """Incoming attention with those patches taken out of the competition.

    Dropping a column and renormalising is what "this patch was not available to
    attend to" means for a distribution, and it is the whole of the bookkeeping:
    the shares that were going to the removed patches are redistributed over the
    rest in proportion to what they already had.
    """
    kept = incoming.clone()
    ids = torch.as_tensor([int(t) for t in tokens], dtype=torch.long)
    ids = ids[ids < kept.shape[-1]]
    if ids.numel():
        kept[:, ids] = 0.0
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def subtract_map(removal: str, structure: str, *, states: Optional[torch.Tensor],
                 incoming: Optional[torch.Tensor], channel: int, token_ids: Sequence[int],
                 replacement: Optional[torch.Tensor], scale: float) -> Optional[torch.Tensor]:
    """The bookkeeping counterfactual, or ``None`` where bookkeeping cannot answer.

    ``replacement`` is what the register patches become in the books: the states
    of the matched ordinary patches, the same ones the causal edit writes there,
    so the two methods are asking the same question of the same tokens.
    """
    if (removal, structure) not in SUBTRACTABLE:
        return None
    if removal == "dominant_channel":
        if states is None:
            return None
        if structure == "high_norm_tokens":
            return _norm_without(states.float(), [channel]) / max(float(scale), 1e-9)
        # Masking the channel out of the arithmetic leaves nothing of the channel
        # to measure.  The panel is flat by construction, and is shown for that
        # reason: it is the row's manipulation check, not a result.
        return torch.zeros(states.shape[0], dtype=torch.float32)
    if structure == "attention_sinks":
        if incoming is None:
            return None
        return _attention_without(incoming.float(), token_ids).max(dim=0).values
    if states is None or replacement is None:
        return None
    edited = states.float().clone()
    ids = torch.as_tensor([int(t) for t in token_ids], dtype=torch.long)
    ids = ids[ids < edited.shape[0]]
    if ids.numel():
        rows = replacement.float().to(edited.device)
        edited[ids] = rows[: ids.numel()] if rows.shape[0] >= ids.numel() else \
            rows[torch.arange(ids.numel()) % rows.shape[0]]
    if structure == "high_norm_tokens":
        return edited.norm(dim=-1) / max(float(scale), 1e-9)
    return edited[:, min(int(channel), edited.shape[-1] - 1)].abs()


def run_structure_maps(ctx: QuestionContext, *, layer: Optional[int] = None,
                       units: Optional[Sequence[Tuple[int, str, int]]] = None,
                       save_images: bool = True) -> QuestionResult:
    """Remove each sparse structure in turn and map what happens to all three.

    This tests whether high-norm tokens, the dominant channel and the attention
    sinks are one phenomenon rather than three.  The sharpest test of
    that is not another correlation: it is to take each away and look at the other
    two on the same patches.

    Each removal acts on a different part of the model, chosen so that it removes
    one structure without trivially removing the others by construction:

    ``high_norm_tokens``   replaces the frozen register states with matched
                           ordinary states at the residual stream;
    ``dominant_channel``   scales that channel to zero in the writer's contribution
                           and keeps it suppressed through the register zone;
    ``attention_sinks``    replaces only the keys those positions present to
                           attention, leaving their residual state untouched, so a
                           head has no reason to prefer them.

    The last is the one that separates routing from representation: if the tokens
    keep their norm and their channel while their sink is gone, then whatever
    changes downstream is caused by the routing.

    Every panel is recorded twice over where that is possible: once by *causal*
    re-running, and once by *subtraction*, which recomputes the same quantity from
    the untouched states with the structure masked out of the arithmetic and never
    touches the model.  The pair is worth having because they disagree in an
    informative way: subtraction shows what the structure was contributing to
    the measurement, the causal run shows what the network then did about losing
    it, and subtraction is free since it needs no extra generation.
    """
    layer = int(layer if layer is not None else ctx.intervention_layer)
    point = InterventionPoint.BLOCK_INPUT
    maintenance = [l for l in ctx.register_layers if l >= ctx.writer_layer] or [ctx.writer_layer]
    rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    images: Dict[str, Any] = {}

    for prompt_id, prompt, seed in (units if units is not None else ctx.units()):
        ctx.say(f"  [maps] prompt {prompt_id} seed {seed}: clean pass")
        tracer = ctx.tracer(full_state_layers=[layer])
        clean, _ = run_traced_generation(
            ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
            probes=[(point, [layer])], save_image=save_images)
        frozen = select_frozen_targets(clean, layer=layer, step=ctx.step, point=point,
                                       direction=ctx.vstar, percentile=ctx.percentile,
                                       topk=ctx.topk, highnorm_ratio=ctx.highnorm_ratio)
        ids = list(frozen.register_ids)
        if clean.image is not None:
            images[f"prompt{prompt_id}_seed{seed}"] = clean.image

        # The yardstick for "high norm" is frozen here, from the untouched run, so
        # every treated map is divided by the same number rather than by its own
        # median.  Otherwise a run whose patch norms all collapsed together would
        # look untouched, because the scale would have collapsed with them.
        observation = clean.at(ctx.step, layer)
        scale = float(observation.norm.median()) if observation is not None \
            and observation.norm is not None else 1.0
        grid = clean.grid or ctx.driver.grid

        frozen_ids = set(ids)

        def emit(values, removal: str, structure: str, method: str) -> None:
            for token, value in enumerate(values.tolist()):
                rows.append(dict(prompt_id=prompt_id, seed=seed, layer=layer,
                                 removal=removal, structure=structure, method=method,
                                 token=token,
                                 row=token // grid[1] if grid[1] else 0,
                                 col=token % grid[1] if grid[1] else token,
                                 value=float(value),
                                 is_frozen_target=bool(token in frozen_ids)))

        def record(trace, removal: str, method: str = "causal") -> None:
            row = trace.at(ctx.step, layer)
            for structure, _ in STRUCTURES:
                values = structure_map(row, structure, scale=scale)
                if values is not None:
                    emit(values, removal, structure, method)

        def follow_channel(trace, removal: str) -> None:
            """The dominant channel at the frozen patches, layer by layer.

            One map at one layer cannot tell "the edit never landed" from "the
            edit landed and the model put the channel back", and those are
            opposite conclusions: the first is a broken intervention, the second
            is evidence of active maintenance.  Following the same patches across
            depth separates them, and costs nothing since every observed layer is
            already in the trace.
            """
            for observed in ctx.observe_layers:
                row = trace.at(ctx.step, observed)
                if row is None or row.channel_values is None or not row.channel_values.numel():
                    continue
                values = row.channel_values[0].float().abs()
                if not ids or max(ids) >= values.numel():
                    continue
                elsewhere = [t for t in range(values.numel()) if t not in frozen_ids]
                channel_rows.append(dict(
                    prompt_id=prompt_id, seed=seed, removal=removal, layer=int(observed),
                    at_registers=float(values[list(ids)].mean()),
                    elsewhere=float(values[elsewhere].mean()) if elsewhere else float("nan"),
                    norm_at_registers=(float(row.norm.float()[list(ids)].mean())
                                       if row.norm is not None else float("nan"))))

        record(clean, "clean", method="clean")
        follow_channel(clean, "clean")

        # --- the bookkeeping counterfactuals, from the untouched states alone
        replacement = (observation.states[list(frozen.matched_ordinary_ids)]
                       if observation is not None and observation.states is not None
                       and frozen.matched_ordinary_ids else None)
        for removal, _ in REMOVALS:
            for structure, _ in STRUCTURES:
                if ablation_method(removal, structure) != "subtract":
                    continue
                values = subtract_map(
                    removal, structure,
                    states=None if observation is None else observation.states,
                    incoming=None if observation is None else observation.incoming,
                    channel=ctx.dominant_channel, token_ids=ids,
                    replacement=replacement, scale=scale)
                if values is not None:
                    emit(values, removal, structure, "subtract")

        # --- take the high-norm tokens away at the residual stream
        states = frozen.states_for("matched")
        plan = EditPlan(edit=lambda image, _: replace_tokens(image, ids, states.to(image)),
                        point=point, layers=[layer], label="high_norm_tokens")
        treated, _ = run_traced_generation(ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt,
                                           seed=seed, condition="high_norm_tokens",
                                           plans=[plan], targets=frozen)
        record(treated, "high_norm_tokens")
        follow_channel(treated, "high_norm_tokens")

        # --- take the dominant channel away where it is written, and keep it away
        channel_plan = EditPlan(
            edit=lambda image, _: scale_channel(image, ctx.dominant_channel, 0.0, ids),
            point=InterventionPoint.WRITER_RESIDUAL, layers=maintenance,
            label="dominant_channel")
        treated, _ = run_traced_generation(ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt,
                                           seed=seed, condition="dominant_channel",
                                           plans=[channel_plan], targets=frozen)
        record(treated, "dominant_channel")
        follow_channel(treated, "dominant_channel")

        # --- take the sink away in key space only, leaving the state alone
        from .causal_engine import SinkSuppression

        suppression = SinkSuppression(tracer, layers=ctx.observe_layers, tokens=ids)
        treated, _ = run_traced_generation(ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt,
                                           seed=seed, condition="attention_sinks",
                                           key_patch=suppression)
        record(treated, "attention_sinks")
        follow_channel(treated, "attention_sinks")

    tidy = pd.DataFrame(rows)
    channel_trace = pd.DataFrame(channel_rows)
    return QuestionResult("maps", tidy, {"survival": _structure_survival(tidy),
                                         "survival_immediate": _structure_survival(tidy, method="subtract"),
                                         "survival_propagated": _structure_survival(tidy, method="causal"),
                                         "channel_trace": channel_trace},
                          _maps_verdict(tidy),
                          meta=dict(layer=layer, structures=dict(STRUCTURES),
                                    removals=dict(REMOVALS), notes=STRUCTURE_NOTES,
                                    methods=METHOD_NOTES, subtractable=list(SUBTRACTABLE),
                                    maintenance_layers=maintenance,
                                    writer_layer=ctx.writer_layer,
                                    survival_formula={
                                        "numerator": "mean value at frozen clean target token IDs",
                                        "denominator": "mean clean value at the same token IDs",
                                        "aggregation": "one ratio per prompt-seed/removal/structure",
                                        "immediate": "clean tensor with coordinate/state algebraically masked",
                                        "propagated": "treated forward-pass observation"},
                                    observe_layers=list(ctx.observe_layers)),
                          images=images)


def _structure_survival(tidy: pd.DataFrame, method: str = "causal") -> pd.DataFrame:
    """How much of each structure is left at the frozen positions after each removal.

    Reported as a share of the clean run on the same patches, so 1.0 means the
    structure is untouched and 0.0 that it is gone.

    Only one method at a time: a table that averaged a bookkeeping panel together
    with a re-run one would be reporting two different quantities under a single
    heading.  The default is the causal run, because "what does the network do
    about it" is the question being asked.
    """
    if tidy.empty:
        return pd.DataFrame(columns=["removal", "structure", "survival"])
    targets = tidy[tidy["is_frozen_target"]]
    if "method" in targets.columns:
        targets = targets[targets["method"].isin(["clean", method])]
        if targets.empty:
            return pd.DataFrame(columns=["removal", "structure", "survival"])
    clean = (targets[targets["removal"] == "clean"]
             .groupby(["prompt_id", "seed", "structure"], observed=True)["value"].mean())
    rows = []
    for (prompt_id, seed, removal, structure), group in targets.groupby(
            ["prompt_id", "seed", "removal", "structure"], observed=True):
        reference = clean.get((prompt_id, seed, structure))
        if reference is None or abs(reference) < 1e-12:
            continue
        rows.append(dict(prompt_id=prompt_id, seed=seed, removal=removal, structure=structure,
                         survival=float(group["value"].mean() / reference)))
    return pd.DataFrame(rows)


def _maps_verdict(tidy: pd.DataFrame) -> str:
    survival = _structure_survival(tidy)
    if survival.empty:
        return "Structure maps: nothing measurable was recorded."
    table = survival.groupby(["removal", "structure"], observed=True)["survival"].mean().unstack()
    lines = []
    for removal, label in REMOVALS:
        if removal == "clean" or removal not in table.index:
            continue
        others = [f"{dict(STRUCTURES)[s].lower()} {table.loc[removal, s]:.0%}"
                  for s, _ in STRUCTURES if s in table.columns
                  and not math.isnan(table.loc[removal, s])]
        lines.append(f"{label}: " + ", ".join(others) + " of the clean value survives")
    return ("Structure maps -- taking one sparse structure away and measuring the other two at "
            "the same patches. " + "; ".join(lines) + ".")
