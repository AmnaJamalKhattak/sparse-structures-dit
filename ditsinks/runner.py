"""Run a sweep: real checkpoints, or the synthetic stand-ins."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .adapters import get_adapter
from .capture import LayerRecord, SweepCapture
from .config import SweepConfig
from .synthetic import build_tiny, planted_direction


@dataclass
class SweepResult:
    cfg: SweepConfig
    records: Dict[Tuple[int, int, int, int], LayerRecord] = field(default_factory=dict)
    images: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- access
    @property
    def layer_indices(self) -> List[int]:
        return sorted({r.layer for r in self.records.values()})

    @property
    def n_layers(self) -> int:
        return int(self.meta.get("n_layers", len(self.layer_indices)))

    def layer_name(self, layer: int) -> str:
        for r in self.records.values():
            if r.layer == layer:
                return r.block_name
        return f"layer{layer}"

    def zone_boundary(self) -> Optional[float]:
        """x position of the dual->single boundary, or None for single-stream models."""
        duals = [r.layer for r in self.records.values() if r.kind == "dual"]
        singles = [r.layer for r in self.records.values() if r.kind == "single"]
        if duals and singles:
            return max(duals) + 0.5
        return None

    def select(self, **kw) -> List[LayerRecord]:
        out = []
        for r in self.records.values():
            if all(getattr(r, k) == v for k, v in kw.items()):
                out.append(r)
        return sorted(out, key=lambda r: (r.prompt_id, r.seed, r.step, r.layer))

    # ------------------------------------------------------------------- I/O
    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else self.cfg.run_dir / "sweep_records.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"records": self.records, "meta": self.meta, "config": self.cfg.to_json()}, path)
        return path

    @classmethod
    def load(cls, path) -> "SweepResult":
        blob = torch.load(path, weights_only=False, map_location="cpu")
        import json

        d = json.loads(blob["config"])
        d.pop("resolved_repo_id", None)
        d.pop("resolved_family", None)
        return cls(cfg=SweepConfig(**d), records=blob["records"], meta=blob.get("meta", {}))


def run_sweep(cfg: SweepConfig, pipe=None, progress: bool = True) -> SweepResult:
    """Capture a full layer sweep for every (prompt, seed) in `cfg`."""
    spec = cfg.spec
    if spec.repo_id == "synthetic":
        return _run_synthetic(cfg, progress=progress)

    adapter = get_adapter(spec.family)
    if pipe is None:
        pipe = adapter.load_pipeline(spec, cfg)
    transformer = adapter.transformer(pipe)

    result = SweepResult(cfg=cfg)
    out_dir = cfg.run_dir
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    with SweepCapture(adapter, cfg, transformer) as cap:
        result.meta["n_layers"] = len(cap.refs)
        result.meta["layer_names"] = [r.name for r in cap.refs]
        result.meta["layer_kinds"] = [r.kind for r in cap.refs]
        result.meta["attention_tap"] = list(cap._tap.patched) if cap._tap else []
        for pid, prompt in enumerate(cfg.prompts):
            for seed in cfg.seeds:
                if progress:
                    print(f"  [{spec.key}] prompt {pid} seed {seed} ...", flush=True)
                cap.begin_generation(pid, seed)
                image = adapter.generate(pipe, prompt, seed, cfg, spec)
                result.images[(pid, seed)] = image
                if cfg.save_images and image is not None:
                    image.save(out_dir / "images" / f"prompt{pid}_seed{seed}.png")
        result.records = dict(cap.records)

    result.meta.update(
        model=spec.key, repo_id=spec.repo_id, family=spec.family,
        prompts=list(cfg.prompts), seeds=list(cfg.seeds),
        grid=_infer_grid(result, cfg),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _run_synthetic(cfg: SweepConfig, progress: bool = True) -> SweepResult:
    spec = cfg.spec
    adapter = get_adapter(spec.family)
    grid = max(4, int(cfg.height) // 16)
    bundle = build_tiny(spec.family, steps=cfg.num_inference_steps, grid=grid)
    v_planted = planted_direction(bundle.d_model)

    result = SweepResult(cfg=cfg)
    with SweepCapture(adapter, cfg, bundle.transformer) as cap:
        result.meta["n_layers"] = len(cap.refs)
        result.meta["layer_names"] = [r.name for r in cap.refs]
        result.meta["layer_kinds"] = [r.kind for r in cap.refs]
        result.meta["attention_tap"] = list(cap._tap.patched) if cap._tap else []
        for pid, _prompt in enumerate(cfg.prompts):
            for seed in cfg.seeds:
                if progress:
                    print(f"  [synthetic {spec.family}] prompt {pid} seed {seed} ...", flush=True)
                cap.begin_generation(pid, seed)
                with torch.no_grad():
                    for step in range(cfg.num_inference_steps):
                        bundle.call(bundle.transformer, seed * 10 + pid, step, v_planted)
        result.records = dict(cap.records)

    result.meta.update(
        model=spec.key, repo_id=spec.repo_id, family=spec.family, synthetic=True,
        prompts=list(cfg.prompts), seeds=list(cfg.seeds), grid=bundle.grid,
        planted_direction=v_planted,
    )
    return result


def _infer_grid(result: SweepResult, cfg: SweepConfig) -> Tuple[int, int]:
    """Recover the patch grid (rows, cols) from the token count and the image size."""
    n_img = 0
    for r in result.records.values():
        n_img = max(n_img, r.n_img)
    if n_img <= 0:
        return (0, 0)
    for d in (8, 16, 32, 64):
        if cfg.height % d == 0 and cfg.width % d == 0 and (cfg.height // d) * (cfg.width // d) == n_img:
            return (cfg.height // d, cfg.width // d)
    side = int(round(math.sqrt(n_img)))
    if side * side == n_img:
        return (side, side)
    return (1, n_img)


def run_projection_sweep(cfg: SweepConfig, direction, pipe=None, progress: bool = True) -> Dict[str, Any]:
    """Re-run the generations recording every token's exact projection on `direction`."""
    from .capture import ProjectionCapture

    spec = cfg.spec
    adapter = get_adapter(spec.family)
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

    out: Dict[str, Any] = {}
    with ProjectionCapture(adapter, cfg, transformer, torch.as_tensor(direction)) as cap:
        for pid, prompt in enumerate(cfg.prompts):
            for seed in cfg.seeds:
                if progress:
                    print(f"  [projection] prompt {pid} seed {seed} ...", flush=True)
                cap.begin_generation(pid, seed)
                if synthetic:
                    with torch.no_grad():
                        for step in range(cfg.num_inference_steps):
                            bundle.call(bundle.transformer, seed * 10 + pid, step, v_planted)
                else:
                    adapter.generate(pipe, prompt, seed, cfg, spec)
        out["rows"] = dict(cap.rows)
    return out
