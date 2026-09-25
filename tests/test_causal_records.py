"""Lossless causal-record persistence and strict provenance checks."""
from dataclasses import replace

import pandas as pd
import pytest
import torch

from ditsinks.causal_records import (CausalResults, CleanTargetManifest,
                                     IncompatibleCausalRun, InterventionRecord,
                                     LayerHeadMeasurement, ResumeIdentity,
                                     TokenGeometry)


def _bundle():
    manifest = CleanTargetManifest(
        checkpoint="example/model", revision="abc123", prompt="a fox",
        prompt_split="evaluation", seed=7, denoising_step=3, layer=11,
        token_geometry=TokenGeometry(16, 4, 4, 4), target_token_ids=[2],
        control_token_ids=[9], clean_sink_ids_by_head={0: 2},
        clean_vstar_threshold=0.8, vstar_hash="vstar-sha256",
        selected_channels=[5], matching_distances={9: 0.125},
        configuration_fingerprint="config-sha256",
        tensors={"target_vectors": torch.arange(4)},
    )
    identity = ResumeIdentity(
        model_revision=manifest.revision, prompt_split=manifest.prompt_split,
        target_manifest_hash=manifest.manifest_hash, vstar_hash=manifest.vstar_hash,
        layer_map={"block.11": 11}, intervention_configuration={"strengths": [0.0, 1.0]},
    )
    measurement = LayerHeadMeasurement(
        layer=12, head=0, new_sink_token_id=8, new_sink_displacement=2.0,
        original_sink_retained=False, relocated=True, attention_entropy=1.2,
        alpha=3.0, perpendicular_norm=0.2, key_rank=1,
        query_key_advantage=4.5, takeover_channel=7, register_lifetime=9,
        dissolution_shift=-2, tensors={"attention": torch.eye(2)},
    )
    record = InterventionRecord(
        intervention_family="direction", condition="ablate", strength=1.0,
        hook_location="block.11.input", affected_token_ids=[2], affected_channels=[5],
        removed_energy=4.0, repeated=False, measurements={(12, 0): measurement},
    )
    return CausalResults(manifest, identity, [record])


def test_round_trip_retains_tensors_and_exports_tidy_csv(tmp_path):
    bundle = _bundle()
    path = bundle.save(tmp_path)
    loaded = CausalResults.load(path, bundle.identity)
    assert torch.equal(loaded.manifest.tensors["target_vectors"], torch.arange(4))
    assert torch.equal(loaded.interventions[0].measurements[(12, 0)].tensors["attention"],
                       torch.eye(2))
    outputs = loaded.export_tidy(tmp_path, parquet=False)
    row = pd.read_csv(outputs["csv"]).iloc[0]
    assert row["application_mode"] == "single-shot"
    assert row["new_sink_token_id"] == 8


@pytest.mark.parametrize("field,value", [
    ("model_revision", "different"), ("prompt_split", "train"),
    ("target_manifest_hash", "different"), ("vstar_hash", "different"),
    ("layer_map", {"block.12": 12}),
    ("intervention_configuration", {"strengths": [2.0]}),
])
def test_load_rejects_every_resume_identity_mismatch(tmp_path, field, value):
    bundle = _bundle()
    path = bundle.save(tmp_path)
    with pytest.raises(IncompatibleCausalRun, match="cannot load/resume"):
        CausalResults.load(path, replace(bundle.identity, **{field: value}))
