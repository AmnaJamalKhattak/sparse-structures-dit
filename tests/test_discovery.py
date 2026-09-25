import json

import pytest
import torch

from ditsinks.config import SweepConfig
from ditsinks.discovery import (
    DiscoveryArtifact, LayerRanges, create_discovery_artifact,
    load_vstar_for_confirmation,
)
from ditsinks.mock import make_mock_result
from ditsinks.vstar import fit_vstar


def test_discovery_is_checkpoint_scoped_and_frozen(tmp_path):
    result = make_mock_result()
    report = fit_vstar(result)
    artifact = create_discovery_artifact(
        result, report, tmp_path / "tiny-flux1",
        layer_ranges=LayerRanges((1, 2), (2, 3), (3, 4)),
        confirmation_prompts=["held out"], confirmation_seeds=[101],
        expected_channels={"dominant": 999, "competitor": 998},
    )
    assert artifact.selection_frozen
    assert artifact.fitting_population
    assert artifact.explained_variance == report.explained_variance
    assert artifact.dominant_register_channel.selected == report.top_channel
    assert artifact.dominant_register_channel.matches_expected is False
    assert "expected dominant register channel 999" in artifact.mismatch_messages()[0]

    loaded, direction = load_vstar_for_confirmation(tmp_path / "tiny-flux1", artifact.checkpoint)
    assert loaded == artifact
    assert torch.equal(direction, report.v)
    with pytest.raises(ValueError, match="checkpoint-specific"):
        load_vstar_for_confirmation(tmp_path / "tiny-flux1", "flux1-dev")
    with pytest.raises(FileExistsError, match="overwrite"):
        artifact.save(tmp_path / "tiny-flux1")


def test_discovery_manifest_contains_required_provenance(tmp_path):
    result = make_mock_result()
    report = fit_vstar(result)
    path = tmp_path / result.meta["model"]
    artifact = create_discovery_artifact(
        result, report, path,
        layer_ranges=LayerRanges((0, 1), (2, 20), (21, 40)),
        confirmation_prompts=["confirm"], confirmation_seeds=[100, 101],
    )
    data = json.loads((path / "discovery.json").read_text())
    assert data["checkpoint"] == artifact.checkpoint
    assert data["denoising_steps"] == result.cfg.capture_steps
    assert data["discovery_prompts"] == result.cfg.prompts
    assert data["discovery_seeds"] == result.cfg.seeds
    assert data["confirmation_prompts"] == ["confirm"]
    assert data["confirmation_seeds"] == [100, 101]
    assert data["code_config_fingerprint"].startswith("git:")
    assert len(data["config_sha256"]) == 64
