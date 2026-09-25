"""Checkpoint-scoped discovery records for confirmatory mechanism experiments.

Discovery is deliberately separated from intervention code.  Channel and layer
choices are made from an observational discovery sweep, written to an immutable
manifest, and subsequently *loaded* by confirmatory runs.  This prevents a
failed expected reproduction from silently becoming a post-outcome selection.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LayerRanges:
    """Inclusive global-layer ranges, fixed using discovery data only."""

    writer: Tuple[int, int]
    register: Tuple[int, int]
    dissolution: Tuple[int, int]


@dataclass(frozen=True)
class ChannelChoice:
    """A selected channel and its relationship to an expected target."""

    selected: int
    expected: Optional[int]
    matches_expected: Optional[bool]
    discovery_score: float


@dataclass(frozen=True)
class DiscoveryArtifact:
    schema_version: int
    checkpoint: str
    repo_id: str
    vstar_file: str
    fitting_population: List[Dict[str, object]]
    explained_variance: float
    axis_convention: str
    sign_convention: str
    dominant_register_channel: ChannelChoice
    late_growing_competitor: ChannelChoice
    massive_unspecific_control: ChannelChoice
    layer_ranges: LayerRanges
    denoising_steps: List[int]
    discovery_prompts: List[str]
    discovery_seeds: List[int]
    confirmation_prompts: List[str]
    confirmation_seeds: List[int]
    code_config_fingerprint: str
    config_sha256: str
    selection_frozen: bool = True

    def mismatch_messages(self) -> List[str]:
        """Report failed hypotheses without changing the discovered selection."""
        messages = []
        for label, choice in (
            ("dominant register channel", self.dominant_register_channel),
            ("late-growing competitor", self.late_growing_competitor),
            ("massive unspecific control", self.massive_unspecific_control),
        ):
            if choice.matches_expected is False:
                messages.append(
                    f"{self.checkpoint}: expected {label} {choice.expected}, "
                    f"but discovery selected {choice.selected}"
                )
        return messages

    def assert_for_confirmation(self, checkpoint: str) -> None:
        """Refuse cross-checkpoint v* reuse and non-frozen selections."""
        if checkpoint != self.checkpoint:
            raise ValueError(
                f"Discovery artifact is for {self.checkpoint!r}, not {checkpoint!r}; "
                "fit a checkpoint-specific v*."
            )
        if not self.selection_frozen:
            raise ValueError("Channel selections must be frozen before confirmation")

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "discovery.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen discovery artifact: {path}")
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path) -> "DiscoveryArtifact":
        data = json.loads(Path(path).read_text())
        ranges = data["layer_ranges"]
        data["layer_ranges"] = LayerRanges(**{key: tuple(value) for key, value in ranges.items()})
        for key in ("dominant_register_channel", "late_growing_competitor", "massive_unspecific_control"):
            data[key] = ChannelChoice(**data[key])
        return cls(**data)


def code_config_fingerprint(cfg, repo_root: Optional[Path] = None) -> Tuple[str, str]:
    """Return a combined code/config fingerprint and the config digest.

    Dirty state is included because a commit SHA alone does not identify code
    being executed from an edited notebook checkout.
    """
    config_json = cfg.to_json()
    config_digest = hashlib.sha256(config_json.encode()).hexdigest()
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--exit-code"], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0
        code = f"git:{commit}{'+dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        code = "git:unavailable"
    return f"{code};config-sha256:{config_digest}", config_digest


def discover_channels(result, report) -> Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]]:
    """Select register, late-growing, and massive-unspecific channels.

    The late score is the change in mean absolute activation from the first to
    final third of layers.  The control maximises global peak activation after
    excluding the 16 coordinates with the largest v* energy, making it strong
    globally but explicitly unspecific to the register axis.
    """
    dominant = int(report.top_channel)
    dom_score = float(report.top_channel_energy[0])
    rows: Dict[int, List[Tuple[int, float, float]]] = {}
    for rec in result.records.values():
        if rec.channel_mean_abs is None or rec.channel_absmax is None:
            continue
        for channel, (mean, peak) in enumerate(zip(rec.channel_mean_abs, rec.channel_absmax)):
            rows.setdefault(channel, []).append((rec.layer, float(mean), float(peak)))
    if not rows:
        raise ValueError("Discovery requires per-channel activation summaries")
    layers = sorted({layer for values in rows.values() for layer, _, _ in values})
    third = max(1, len(layers) // 3)
    early, late = set(layers[:third]), set(layers[-third:])
    growth = {}
    massive = {}
    for channel, values in rows.items():
        e = [mean for layer, mean, _ in values if layer in early]
        l = [mean for layer, mean, _ in values if layer in late]
        growth[channel] = float(np.mean(l) - np.mean(e)) if e and l else -np.inf
        massive[channel] = float(np.mean([peak for _, _, peak in values]))
    competitor = max((c for c in rows if c != dominant), key=lambda c: growth[c])
    # Exclude the coordinates carrying most of v*, but never more than half the
    # width: on a narrow model a fixed cut of 16 can swallow every channel and
    # leave no control at all.
    excluded = min(16, max(1, int(report.v.numel()) // 2))
    register_specific = set(int(c) for c in torch.argsort(report.v.square(), descending=True)[:excluded])
    controls = [c for c in rows if c not in register_specific]
    if not controls:
        raise ValueError("No register-unspecific channel is available for a control")
    control = max(controls, key=lambda c: massive[c])
    return (dominant, dom_score), (competitor, growth[competitor]), (control, massive[control])


def create_discovery_artifact(
    result, report, directory, *, layer_ranges: LayerRanges,
    confirmation_prompts: Sequence[str], confirmation_seeds: Sequence[int],
    expected_channels: Optional[Mapping[str, int]] = None,
) -> DiscoveryArtifact:
    """Freeze checkpoint-specific discovery choices and save its fitted v*."""
    checkpoint = str(result.meta.get("model") or result.cfg.spec.key)
    if report.meta.get("model") not in (None, checkpoint):
        raise ValueError("The v* report and sweep belong to different checkpoints")
    expected = dict(expected_channels or {})
    # Prompts are the held-out axis: reusing one would let a layer, channel or
    # threshold chosen on it be "confirmed" on the same generation.  Seeds are
    # not.  A seed only fixes the initial noise, so the same seed under a
    # different prompt is an entirely different generation and leaks nothing --
    # and forcing seeds apart would halve the seeds available to each phase for
    # no statistical gain.  Overlapping seeds are therefore allowed, but only
    # because the prompts above are already guaranteed disjoint.
    shared_prompts = set(result.cfg.prompts) & set(confirmation_prompts)
    if shared_prompts:
        raise ValueError(
            f"Discovery and confirmation prompts must be disjoint; {len(shared_prompts)} shared")
    dominant, competitor, control = discover_channels(result, report)

    def choice(role: str, value: Tuple[int, float]) -> ChannelChoice:
        target = expected.get(role)
        return ChannelChoice(value[0], target, None if target is None else value[0] == target, value[1])

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vstar_file = "vstar.pt"
    vstar_path = directory / vstar_file
    manifest_path = directory / "discovery.json"
    if vstar_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen discovery in {directory}")
    torch.save({"checkpoint": checkpoint, "v": report.v.cpu()}, vstar_path)
    fingerprint, config_digest = code_config_fingerprint(result.cfg)
    population = report.meta.get("registers")
    if hasattr(population, "to_dict"):
        population = population.to_dict(orient="records")
    population = list(population or [])
    artifact = DiscoveryArtifact(
        schema_version=SCHEMA_VERSION, checkpoint=checkpoint, repo_id=result.cfg.spec.repo_id,
        vstar_file=vstar_file, fitting_population=population,
        explained_variance=float(report.explained_variance),
        axis_convention="unit top right-singular vector of uncentered, unit-normalized discovery registers",
        sign_convention="sign chosen so the summed discovery-register projection is positive",
        dominant_register_channel=choice("dominant", dominant),
        late_growing_competitor=choice("competitor", competitor),
        massive_unspecific_control=choice("control", control), layer_ranges=layer_ranges,
        denoising_steps=list(result.cfg.capture_steps), discovery_prompts=list(result.cfg.prompts),
        discovery_seeds=list(result.cfg.seeds), confirmation_prompts=list(confirmation_prompts),
        confirmation_seeds=[int(s) for s in confirmation_seeds],
        code_config_fingerprint=fingerprint, config_sha256=config_digest,
    )
    artifact.save(directory)
    return artifact


def load_vstar_for_confirmation(directory, checkpoint: str) -> Tuple[DiscoveryArtifact, torch.Tensor]:
    """Load only after validating that artifact and run checkpoint agree."""
    directory = Path(directory)
    artifact = DiscoveryArtifact.load(directory / "discovery.json")
    artifact.assert_for_confirmation(checkpoint)
    blob = torch.load(directory / artifact.vstar_file, weights_only=False, map_location="cpu")
    if blob.get("checkpoint") != checkpoint:
        raise ValueError("vstar.pt checkpoint tag does not match its discovery manifest")
    return artifact, blob["v"]
