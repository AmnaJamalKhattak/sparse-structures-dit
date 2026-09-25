"""The main run's rules: windows placed by rule, a frozen protocol, formal names."""
import json

import pytest

from ditsinks import q16_main as Q


def test_windows_on_the_measured_flux_dev_interval():
    windows, notes = Q.plan_depth_windows(n_layers=57, formation=18, natural_end=39,
                                          decline=31, per_side=3)
    assert windows["E1"] == (2, 6) and windows["E2"] == (7, 11) and windows["E3"] == (12, 16)
    assert windows["L1"] == (41, 45) and windows["L2"] == (46, 50) and windows["L3"] == (51, 55)
    assert windows["N"] == (19, 23), "in place: inside the plateau, same length"
    assert windows["removal"] == (17, 39), "from where the writer starts to the natural end"
    assert windows["extend"] == (29, 45), "from before the decline through L1"
    lengths = {k: b - a + 1 for k, (a, b) in windows.items() if k[0] in "ELN"}
    assert set(lengths.values()) == {5}, "equal length: depth is not confounded with dose"
    # Nothing an induction window holds is inside the natural interval or its writer.
    natural = set(range(17, 40))
    for name in ("E1", "E2", "E3", "L1", "L2", "L3"):
        assert not set(range(windows[name][0], windows[name][1] + 1)) & natural
    # The block beside the interval is free on both sides (writer start, terminal hook).
    assert windows["E3"][1] == 16 and windows["L1"][0] == 41


def test_windows_on_a_shorter_uniform_stack_keep_equal_lengths():
    windows, notes = Q.plan_depth_windows(n_layers=28, formation=10, natural_end=20,
                                          decline=15, per_side=3)
    assert windows["E3"] == (7, 8) and windows["E1"] == (3, 4)
    assert windows["L1"] == (22, 23) and windows["L3"] == (26, 27)
    assert windows["N"] == (11, 12)
    assert any("coarse" in n for n in notes) is False


def test_the_in_place_window_stops_before_the_decline_and_windows_shrink_to_fit():
    windows, notes = Q.plan_depth_windows(n_layers=30, formation=8, natural_end=14,
                                          decline=10, per_side=3)
    assert windows["E1"] == (1, 2) and windows["L3"] == (20, 21), "two-block windows"
    assert windows["N"] == (9, 9) and any("decline" in n for n in notes)
    # Too little room on one side: fewer windows per side, still all the same length.
    windows, notes = Q.plan_depth_windows(n_layers=20, formation=6, natural_end=12,
                                          decline=None, per_side=3)
    assert windows["E1"][1] - windows["E1"][0] == windows["L1"][1] - windows["L1"][0]
    with pytest.raises(ValueError):
        Q.plan_depth_windows(n_layers=10, formation=2, natural_end=8, per_side=3)


def _rows():
    rows = []
    for prompt in range(3):
        for step in range(4):
            rows.append(dict(step=step, prompt_id=prompt, seed=0, formation=18 - (prompt == 2),
                             peak_layer=20, peak_count=30, natural_end=37 + prompt,
                             decline=31, n_carriers=30, valid=True, note=""))
    rows.append(dict(step=0, prompt_id=9, seed=0, formation=None, peak_layer=None,
                     peak_count=0, natural_end=None, decline=None, n_carriers=0,
                     valid=False, note="no register"))
    return rows


def test_the_protocol_takes_the_earliest_formation_and_latest_end_and_round_trips(tmp_path):
    settings = Q.MainSettings(checkpoint="flux1-dev", size=1024, steps=28)
    protocol = Q.protocol_from_calibration(_rows(), n_layers=57, seam=19,
                                           checkpoint="flux1-dev", settings=settings)
    assert protocol.formation == 17 and protocol.natural_end == 39
    assert protocol.selection_block == 20 and protocol.decline == 31
    assert protocol.primary_early == "E3" and protocol.primary_late == "L1"
    assert protocol.plateau == (18, 30)
    assert protocol.edit_steps == list(range(9)), "the first third of 28 steps"
    assert protocol.readout_step == 4 and 14 in protocol.attention_steps
    assert protocol.block_kind(5) == "dual-stream" and protocol.block_kind(45) == "single-stream"
    path = protocol.to_json(tmp_path / "protocol.json")
    again = Q.DepthProtocol.from_json(path)
    assert again.windows == protocol.windows and again.plateau == protocol.plateau
    payload = json.loads(path.read_text())
    payload["version"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        Q.DepthProtocol.from_json(path)
    with pytest.raises(RuntimeError):
        Q.protocol_from_calibration([r for r in _rows() if not r["valid"]], n_layers=57,
                                    seam=19, checkpoint="x", settings=settings)


def test_every_condition_has_a_formal_name_and_none_is_informal():
    settings = Q.MainSettings(checkpoint="flux1-dev", size=1024, steps=28)
    protocol = Q.protocol_from_calibration(_rows(), n_layers=57, seam=19,
                                           checkpoint="flux1-dev", settings=settings)
    catalog = Q.condition_catalog(protocol)
    keys = [c.key for c in catalog]
    assert keys[0] == "reference" and "remove" in keys and "extend" in keys
    assert {"induce_E1", "induce_E3", "induce_N", "induce_L1", "induce_L3",
            "remove_induce_E3", "remove_induce_L1", "control_random_direction",
            "control_ordinary_positions", "control_in_distribution", "hooks_only"} <= set(keys)
    labels = {c.key: c.label(protocol) for c in catalog}
    assert labels["reference"] == "Unmodified"
    assert labels["induce_E1"] == "Induced at blocks 1–5"
    assert labels["remove"] == "Natural register removed"
    assert labels["hooks_only"] == "Hooks installed, no modification"
    for label in labels.values():
        assert "sham" not in label.lower() and "_" not in label
