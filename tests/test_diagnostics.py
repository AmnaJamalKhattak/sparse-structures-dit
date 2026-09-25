import json

import pytest
import torch

from ditsinks import diagnostics as D
from ditsinks.causal_ops import clamp_norm, remove_direction, scale_channel


def test_mandatory_tensor_invariants_pass_on_exact_ops():
    x = torch.randn(9, 16)
    v = torch.randn(16)
    ids = [1, 5]
    D.assert_norm_clamp(x, clamp_norm(x, ids, 2.0), v, ids, 2.0)
    D.assert_direction_removed(remove_direction(x, v, ids), v, ids)
    D.assert_channel_zero(scale_channel(x, 3, 0.0, ids), 3, ids)
    D.assert_identity(x, x.clone(), name="sham")


def test_identity_gate_fails_loudly():
    with pytest.raises(AssertionError, match="final key identity failed"):
        D.assert_identity(torch.zeros(2, 3), torch.ones(2, 3), name="final key")


def test_token_index_table_distinguishes_text_and_image():
    rows = D.token_index_table(sequence_length=10, image_tokens=6, text_tokens=4, grid_cols=3)
    assert rows[3]["token_type"] == "text"
    assert rows[4] == {"global_sequence_index": 4, "token_type": "image",
                       "image_local_index": 0, "spatial_row": 0, "spatial_col": 0}
    assert rows[9]["image_local_index"] == 5 and rows[9]["spatial_row"] == 1


def test_run_layout_is_non_overwriting_and_records_provenance(tmp_path):
    from ditsinks.provenance import QUESTIONS, create_run_layout

    run = create_run_layout(tmp_path, "diagnostic", config={"step": 3},
                            metadata={"checkpoint": "tiny", "seeds": [0]})
    assert all((run / question).is_dir() for question in QUESTIONS)
    assert (run / "diagnostics").is_dir() and (run / "figures").is_dir()
    provenance = json.loads((run / "provenance.json").read_text())
    assert provenance["checkpoint"] == "tiny" and "packages" in provenance
    with pytest.raises(FileExistsError):
        create_run_layout(tmp_path, "diagnostic", config={}, metadata={})

