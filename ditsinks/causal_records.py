"""Versioned, lossless records for the Q1--Q6 causal experiments.

JSON is used for human-readable provenance, while the companion ``.pt`` file is
the source of truth for tensors.  Flat CSV/Parquet files are deliberately
derived products: they must never be used to resume an experiment.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import torch


SCHEMA_VERSION = 1


class IncompatibleCausalRun(ValueError):
    """Raised before reuse of results produced under different provenance."""


def _jsonable(value: Any) -> Any:
    """Produce a stable JSON value, representing tensors by content hashes."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(value: Any) -> str:
    """SHA-256 of a canonical representation (including tensor contents)."""
    data = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class TokenGeometry:
    """Mapping between sequence token IDs and the image patch grid."""

    image_tokens: int
    text_tokens: int = 0
    grid_rows: int = 0
    grid_cols: int = 0
    image_token_offset: int = 0
    patch_height: Optional[int] = None
    patch_width: Optional[int] = None


@dataclass
class CleanTargetManifest:
    """Frozen target selection from the clean run; never infer it on resume."""

    checkpoint: str
    revision: str
    prompt: str
    prompt_split: str
    seed: int
    denoising_step: int
    layer: int
    token_geometry: TokenGeometry
    target_token_ids: List[int]
    control_token_ids: List[int]
    clean_sink_ids_by_head: Dict[int, int]
    clean_vstar_threshold: float
    vstar_hash: str
    selected_channels: List[int]
    matching_distances: Dict[int, float]
    configuration_fingerprint: str
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.clean_sink_ids_by_head = {int(k): int(v) for k, v in self.clean_sink_ids_by_head.items()}
        self.matching_distances = {int(k): float(v) for k, v in self.matching_distances.items()}

    @property
    def manifest_hash(self) -> str:
        return fingerprint(asdict(self))


@dataclass
class LayerHeadMeasurement:
    """Downstream observables for one layer/head (Q1--Q6 sufficient)."""

    layer: int
    head: int
    sink_token_id: Optional[int] = None
    new_sink_token_id: Optional[int] = None
    sink_position: Optional[Tuple[int, int]] = None
    original_sink_retained: Optional[bool] = None
    sink_displacement: Optional[float] = None
    new_sink_displacement: Optional[float] = None
    same_position_regenerated: Optional[bool] = None
    relocated: Optional[bool] = None
    recovered: Optional[bool] = None
    first_recovery_layer: Optional[int] = None
    attention_concentration: Optional[float] = None
    attention_entropy: Optional[float] = None
    alpha: Optional[float] = None
    perpendicular_norm: Optional[float] = None
    key_rank: Optional[int] = None
    query_key_advantage: Optional[float] = None
    takeover_channel: Optional[int] = None
    channel_takeover: Optional[float] = None
    register_present: Optional[bool] = None
    register_lifetime: Optional[int] = None
    dissolution_shift: Optional[int] = None
    # Direction geometry.  ``alpha`` is the signed projection on the frozen v*;
    # ``cosine`` is the same quantity after dividing out the token's magnitude,
    # which is what separates "points along v*" from "is simply large".
    cosine: Optional[float] = None
    vstar_projection_change: Optional[float] = None
    max_cosine: Optional[float] = None
    max_cosine_token: Optional[int] = None
    # Layer-level rollups, carried on the head == -1 row of each layer.
    head_retention: Optional[float] = None
    # How many heads the intervention could possibly have affected: heads whose
    # clean sink was one of the frozen targets.  Retention over zero affected
    # heads is undefined, not zero, and the count says which case a row is.
    affected_heads: Optional[int] = None
    sink_retention_all: Optional[float] = None
    sink_on_target_fraction: Optional[float] = None
    attention_to_targets: Optional[float] = None
    target_norm_ratio: Optional[float] = None
    dominant_channel_value: Optional[float] = None
    competitor_channel_value: Optional[float] = None
    sink_is_target: Optional[bool] = None
    clean_sink_token_id: Optional[int] = None
    head_was_affected: Optional[bool] = None
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)


@dataclass
class InterventionRecord:
    """One intervention and its complete downstream trajectory."""

    intervention_family: str
    condition: str
    strength: float
    hook_location: str
    affected_token_ids: List[int]
    affected_channels: List[int]
    removed_energy: float = 0.0
    injected_energy: float = 0.0
    repeated: bool = False
    measurements: Dict[Tuple[int, int], LayerHeadMeasurement] = field(default_factory=dict)
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def application_mode(self) -> str:
        return "repeated" if self.repeated else "single-shot"

    def __post_init__(self) -> None:
        normalized = {}
        for key, measurement in self.measurements.items():
            if not isinstance(measurement, LayerHeadMeasurement):
                measurement = LayerHeadMeasurement(**measurement)
            normalized[(int(key[0]), int(key[1]))] = measurement
        self.measurements = normalized


@dataclass(frozen=True)
class ResumeIdentity:
    """Every input whose change invalidates a causal run."""

    model_revision: str
    prompt_split: str
    target_manifest_hash: str
    vstar_hash: str
    layer_map: Mapping[str, int]
    intervention_configuration: Mapping[str, Any]

    @property
    def intervention_fingerprint(self) -> str:
        return fingerprint(self.intervention_configuration)

    def normalized(self) -> Dict[str, Any]:
        return {
            "model_revision": self.model_revision,
            "prompt_split": self.prompt_split,
            "target_manifest_hash": self.target_manifest_hash,
            "vstar_hash": self.vstar_hash,
            "layer_map": dict(self.layer_map),
            "intervention_fingerprint": self.intervention_fingerprint,
        }


@dataclass
class CausalResults:
    """Lossless result bundle plus strict save/load and tidy export methods."""

    manifest: CleanTargetManifest
    identity: ResumeIdentity
    interventions: List[InterventionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.identity.target_manifest_hash != self.manifest.manifest_hash:
            raise IncompatibleCausalRun("resume identity does not match target manifest")
        if self.identity.vstar_hash != self.manifest.vstar_hash:
            raise IncompatibleCausalRun("resume identity and manifest have different v* hashes")
        if self.identity.model_revision != self.manifest.revision:
            raise IncompatibleCausalRun("model revision differs from clean target manifest")
        if self.identity.prompt_split != self.manifest.prompt_split:
            raise IncompatibleCausalRun("prompt split differs from clean target manifest")

    def save(self, path: Path | str) -> Path:
        """Atomically save the tensor-bearing canonical bundle and JSON sidecar."""
        path = Path(path)
        if path.suffix != ".pt":
            path = path / "causal_records.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": SCHEMA_VERSION, "manifest": self.manifest,
                   "identity": self.identity, "interventions": self.interventions}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        sidecar = {"schema_version": SCHEMA_VERSION, "identity": self.identity.normalized(),
                   "manifest": _jsonable(asdict(self.manifest)),
                   "n_interventions": len(self.interventions)}
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str, expected: ResumeIdentity) -> "CausalResults":
        """Load only when all six resume-critical identity fields are identical."""
        path = Path(path)
        if path.is_dir():
            path = path / "causal_records.pt"
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("schema_version") != SCHEMA_VERSION:
            raise IncompatibleCausalRun("causal record schema version differs")
        actual = blob["identity"]
        mismatches = [k for k, value in expected.normalized().items()
                      if actual.normalized().get(k) != value]
        if mismatches:
            raise IncompatibleCausalRun("cannot load/resume: differing " + ", ".join(mismatches))
        return cls(manifest=blob["manifest"], identity=actual,
                   interventions=blob.get("interventions", []))

    def tidy_rows(self) -> List[Dict[str, Any]]:
        """One scalar-only row per intervention/layer/head measurement."""
        rows: List[Dict[str, Any]] = []
        excluded = {"measurements", "tensors", "metadata"}
        measurement_excluded = {"tensors"}
        for index, record in enumerate(self.interventions):
            base = {f.name: getattr(record, f.name) for f in fields(record) if f.name not in excluded}
            base.update(intervention_index=index, application_mode=record.application_mode,
                        manifest_hash=self.manifest.manifest_hash)
            base["affected_token_ids"] = json.dumps(base["affected_token_ids"])
            base["affected_channels"] = json.dumps(base["affected_channels"])
            for measurement in record.measurements.values():
                row = dict(base)
                row.update({f.name: getattr(measurement, f.name) for f in fields(measurement)
                            if f.name not in measurement_excluded})
                if row.get("sink_position") is not None:
                    row["sink_position"] = json.dumps(row["sink_position"])
                rows.append(row)
        return rows

    def export_tidy(self, directory: Path | str, parquet: bool = True) -> Dict[str, Path]:
        """Write derived CSV and, when requested, Parquet summaries."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.tidy_rows())
        csv_path = directory / "causal_summary.csv"
        frame.to_csv(csv_path, index=False)
        paths = {"csv": csv_path}
        if parquet:
            parquet_path = directory / "causal_summary.parquet"
            frame.to_parquet(parquet_path, index=False)
            paths["parquet"] = parquet_path
        return paths
