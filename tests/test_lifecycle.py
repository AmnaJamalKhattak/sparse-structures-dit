"""Q16 -- retiming the register lifecycle.

The claims these tests exist to protect: that induction and suppression do what their
names say, that the suppression used in D, E and F is literally the repository's
validated operator rather than a lookalike, and that "the state persisted" is a
measurement rather than a restatement of the schedule.
"""
import pytest
import math

import torch

from ditsinks import lifecycle as LC
from ditsinks.discovery import LayerRanges


def _states(n=16, c=32, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, c)
    v = LC._unit(torch.randn(c))
    x[3] = x[3] * 0.3 + 9.0 * v          # a register-like token
    return x, v


class _Ctx:
    layer, step = 7, 2


# ================================================================ windows
def test_windows_are_equal_length_and_do_not_overlap_the_natural_one():
    ranges = LayerRanges((17, 19), (20, 39), (40, 56))
    w = LC.windows_from_ranges(ranges, length=3, n_layers=57)
    assert w["early"].layers == (14, 15, 16)
    assert w["natural"].layers[0] == 20 and w["natural"].layers[-1] == 39
    assert w["late"].layers == (40, 41, 42)
    assert w["early"].length == w["late"].length == 3, (
        "early and late must have the same number of sites, or depth is confounded "
        "with dose")
    assert not set(w["early"].layers) & set(w["natural"].layers)
    assert not set(w["late"].layers) & set(w["natural"].layers)


def test_an_impossible_window_is_omitted_rather_than_shifted():
    """A writer range starting at block 0 leaves no room before it."""
    w = LC.windows_from_ranges(LayerRanges((0, 1), (1, 3), (3, 4)), length=3, n_layers=5)
    assert "early" not in w, "an early window was invented where none can exist"
    assert "natural" in w


def test_a_late_window_is_clipped_to_the_stack_and_dropped_if_it_does_not_fit():
    w = LC.windows_from_ranges(LayerRanges((17, 19), (20, 39), (40, 56)), length=3,
                               n_layers=42)
    assert w["late"].layers == (40, 41)
    assert "late" not in LC.windows_from_ranges(
        LayerRanges((17, 19), (20, 39), (40, 56)), length=3, n_layers=40)


def test_a_window_reports_crossing_an_architectural_boundary():
    """FLUX dual blocks end at 19; a window straddling that seam is not comparable."""
    assert LC.Window("early", 17, 21).crosses(19)
    assert not LC.Window("early", 14, 16).crosses(19)
    assert not LC.Window("late", 20, 22).crosses(19)
    assert not LC.Window("x", 1, 2).crosses(None)
    assert LC.Window("early", 14, 16).row(19)["crosses_architecture_boundary"] is False


# ================================================================ the induction operator
def test_induction_sets_the_coefficient_and_leaves_the_orthogonal_part_alone():
    x, v = _states()
    out = LC.induce_alpha(x, [5, 9], alpha_target=7.5, direction=v)
    for token in (5, 9):
        assert float(out[token] @ v) == pytest.approx(7.5, abs=1e-4)
        r_before = x[token] - (x[token] @ v) * v
        r_after = out[token] - (out[token] @ v) * v
        assert torch.allclose(r_before, r_after, atol=1e-4)
    others = [i for i in range(x.shape[0]) if i not in (5, 9)]
    assert torch.equal(x[others], out[others])


def test_induction_deliberately_does_not_preserve_norm():
    """The natural register IS a high-norm state; holding the norm would test something
    else. The resulting norm is exactly sqrt(||r||^2 + alpha_target^2)."""
    x, v = _states()
    out = LC.induce_alpha(x, [5], alpha_target=9.0, direction=v)
    r = x[5] - (x[5] @ v) * v
    assert float(out[5].norm()) == pytest.approx(float((r.norm() ** 2 + 81.0).sqrt()), rel=1e-4)
    assert float(out[5].norm()) > 2.0 * float(x[5].norm())


def test_an_unrelated_direction_is_exactly_orthogonal_and_reproducible():
    _, v = _states()
    u = LC.unrelated_direction(v, seed=3)
    assert float(u.norm()) == pytest.approx(1.0, abs=1e-6)
    assert abs(float(u @ v)) < 1e-6, "the control direction must be orthogonal, not nearly"
    assert torch.equal(u, LC.unrelated_direction(v, seed=3))
    assert not torch.equal(u, LC.unrelated_direction(v, seed=4))


# ================================================================ calibration
def test_ratio_calibration_rescales_across_depth_and_absolute_does_not():
    natural_projection = torch.zeros(16); natural_projection[3] = 60.0
    natural_norm = torch.full((16,), 10.0)
    recipient_projection = torch.full((16,), 1.0)
    recipient_norm = torch.full((16,), 2.0)           # a quieter layer

    ratio = LC.calibrate_alpha(natural_projection, natural_norm, recipient_projection,
                               recipient_norm, [3], mode="ratio")
    absolute = LC.calibrate_alpha(natural_projection, natural_norm, recipient_projection,
                                  recipient_norm, [3], mode="absolute")
    assert ratio.alpha_target == pytest.approx(60.0 / 10.0 * 2.0)      # 12.0
    assert absolute.alpha_target == pytest.approx(60.0)
    with pytest.raises(ValueError, match="mode must be"):
        LC.calibrate_alpha(natural_projection, natural_norm, recipient_projection,
                           recipient_norm, [3], mode="whatever")


def test_an_out_of_distribution_target_is_flagged_rather_than_clipped():
    """A layer that never contains such a state must say so; an image change from an
    enormous perturbation is not evidence the register acquired a function."""
    natural_projection = torch.zeros(16); natural_projection[3] = 60.0
    calibration = LC.calibrate_alpha(
        natural_projection, torch.full((16,), 10.0),
        torch.full((16,), 0.5), torch.full((16,), 10.0), [3], mode="absolute")
    assert calibration.out_of_distribution
    assert calibration.out_of_distribution_ratio == pytest.approx(60.0 / 0.5)
    assert calibration.alpha_target == 60.0, "the target must NOT be clipped"
    assert "raw alpha" in calibration.note


# ============================================ suppression IS the repository's operator
def test_the_suppression_is_bit_identical_to_the_validated_beta_gamma_operator():
    """D, E and F only compare if all three suppress identically. This asserts that the
    lifecycle suppression is the repository's beta=0, gamma=1 operator, not a lookalike.
    """
    from ditsinks.control_surface import control_surface_states, identity_error

    x, v = _states()
    tokens = [3, 7]
    mine = LC.lifecycle_edit("suppress", tokens, v)(x, _Ctx())
    theirs, _ = control_surface_states(x, v, beta=0.0, gamma=1.0)
    assert torch.allclose(mine[tokens], theirs[tokens], atol=1e-5), (
        "the lifecycle suppression has drifted from the validated operator")
    # ...and the operator it matches is the one whose centre cell reconstructs x, which
    # is what makes every other cell's effect attributable to the manipulation.
    assert identity_error(x, v) < 1e-5


def test_suppression_removes_the_alignment_and_keeps_the_norm():
    x, v = _states()
    out = LC.lifecycle_edit("suppress", [3], v)(x, _Ctx())
    assert abs(float(out[3] @ v)) < 1e-3
    assert float(out[3].norm()) == pytest.approx(float(x[3].norm()), rel=1e-4)


def test_the_sham_edit_changes_nothing():
    x, v = _states()
    assert torch.equal(LC.lifecycle_edit("sham", [3, 7], v)(x, _Ctx()), x)


def test_an_unknown_edit_kind_is_refused():
    _, v = _states()
    with pytest.raises(ValueError, match="unknown lifecycle edit kind"):
        LC.lifecycle_edit("zero_the_token", [3], v)


def test_a_direction_of_the_wrong_width_is_refused_rather_than_broadcast():
    x, _ = _states(c=32)
    edit = LC.lifecycle_edit("induce", [3], LC._unit(torch.randn(8)), alpha_target=5.0)
    assert torch.equal(edit(x, _Ctx()), x)


def test_every_site_records_what_it_actually_did():
    x, v = _states()
    log = []
    LC.lifecycle_edit("induce", [5, 9], v, alpha_target=8.0, highnorm_threshold=6.0,
                      records=log)(x, _Ctx())
    assert len(log) == 2
    for record in log:
        assert record.alpha_after == pytest.approx(8.0, abs=1e-3)
        assert record.perturbation_l2 > 0
        assert record.crossed_highnorm_threshold is True
        assert record.layer == 7 and record.step == 2
    assert set(LC.EditRecord.row(log[0])) >= {"alpha_before", "alpha_after", "norm_after",
                                              "perturbation_l2",
                                              "crossed_highnorm_threshold"}


# ================================================================ conditions
def test_the_seven_conditions_cover_the_required_schedules():
    by_key = {c.key: c for c in LC.CONDITIONS}
    assert by_key["A_clean"].schedule() == {"early": "none", "natural": "none", "late": "none"}
    assert by_key["B_early_natural_intact"].schedule()["early"] == "induce"
    assert by_key["B_early_natural_intact"].schedule()["natural"] == "none"
    assert by_key["C_late_natural_intact"].schedule() == {
        "early": "none", "natural": "none", "late": "induce"}
    assert by_key["D_late_natural_suppressed"].schedule() == {
        "early": "none", "natural": "suppress", "late": "induce"}
    assert by_key["E_early_natural_suppressed"].schedule() == {
        "early": "induce", "natural": "suppress", "late": "none"}
    assert by_key["F_suppression_only"].schedule() == {
        "early": "none", "natural": "suppress", "late": "none"}
    assert by_key["G_sham"].role == "control"


def test_d_e_and_f_declare_the_same_natural_window_action():
    """If these ever diverge, the comparisons that interpret D and E are meaningless."""
    schedule = {c.key: c.natural for c in LC.CONDITIONS}
    assert (schedule["D_late_natural_suppressed"] == schedule["E_early_natural_suppressed"]
            == schedule["F_suppression_only"] == "suppress")


# ================================================================ achieved lifecycle
class _Obs:
    def __init__(self, layer, projection, norm, incoming, qk=None):
        self.layer, self.step = layer, 2
        self.projection = projection
        self.norm = norm
        self.incoming = incoming
        self.qk_cosine = qk
        self.cosine = self.top_channel = self.top_channel_value = None
        self.image_mass = self.n_keys = None


class _Trace:
    def __init__(self, rows):
        self.rows = rows

    def at(self, step, layer):
        return self.rows.get(int(layer))


def _trace(levels, n=16, heads=2, carrier=3):
    rows = {}
    for layer, level in levels.items():
        projection = torch.full((n,), 0.1)
        projection[carrier] = float(level)
        norm = torch.full((n,), 1.0)
        norm[carrier] = 1.0 + float(level)
        incoming = torch.full((heads, n), 1.0 / n)
        incoming[:, carrier] = 0.5
        incoming = incoming / incoming.sum(-1, keepdim=True)
        rows[layer] = _Obs(layer, projection, norm, incoming, torch.zeros(heads, n))
    return _Trace(rows)


def test_the_achieved_lifecycle_counts_new_carriers_separately_from_frozen_ones():
    trace = _trace({0: 5.0, 1: 5.0, 2: 0.1})
    trace.rows[2].norm[11] = 40.0                       # a NEW high-norm token appears
    rows = LC.achieved_lifecycle(trace, step=2, layers=[0, 1, 2], carriers=[3],
                                 sink_threshold=3.0, condition="B")
    late = next(r for r in rows if r["layer"] == 2)
    assert late["n_highnorm_new"] == 1 and 11 in late["highnorm_new_ids"]
    assert late["n_highnorm_frozen"] == 0
    assert all(r["condition"] == "B" for r in rows)


def test_the_achieved_lifecycle_reports_which_attention_scale_it_used():
    rows = LC.achieved_lifecycle(_trace({0: 5.0}), step=2, layers=[0], carriers=[3],
                                 sink_threshold=3.0)
    assert rows[0]["attention_scale"] == "image_renormalised"


# ================================================================ persistence
def test_persistence_separates_maintained_carried_and_reconstructed():
    window = LC.Window("early", 0, 1)
    maintained = [dict(layer=0, carrier_projection=5.0), dict(layer=1, carrier_projection=5.0),
                  dict(layer=2, carrier_projection=0.1), dict(layer=3, carrier_projection=0.1)]
    carried = [dict(layer=0, carrier_projection=5.0), dict(layer=1, carrier_projection=5.0),
               dict(layer=2, carrier_projection=4.0)]
    reconstructed = [dict(layer=0, carrier_projection=5.0), dict(layer=1, carrier_projection=5.0),
                     dict(layer=2, carrier_projection=9.0)]
    assert LC.persistence_class(maintained, window=window)["persistence"] == "maintained"
    assert LC.persistence_class(carried, window=window)["persistence"] == "carried"
    assert LC.persistence_class(reconstructed, window=window)["persistence"] == "reconstructed"


def test_persistence_says_not_measured_when_no_block_follows_the_window():
    """A late window at the end of the stack cannot answer this, and must not pretend to."""
    window = LC.Window("late", 3, 4)
    rows = [dict(layer=3, carrier_projection=5.0), dict(layer=4, carrier_projection=5.0)]
    out = LC.persistence_class(rows, window=window)
    assert out["persistence"] == "not measured"
    assert "no unpatched block" in out["reason"]


# ========== the anchor and the baseline, both added after a saturated pilot sweep
def test_recipient_outlier_anchors_on_what_the_layer_already_holds():
    """Why this mode exists.

    At a depth with no registers, 'ratio' and 'absolute' both reproduce the natural
    population's outlier by construction -- on real FLUX the natural register sits at
    roughly 100x the median token norm, so every strength multiple came back
    astronomically out of distribution and every criterion passed for arithmetic reasons.
    Anchoring on the recipient layer's own maximum makes a multiple mean something:
    1.0 is "as loud as this layer's loudest token".
    """
    natural_projection = torch.zeros(16); natural_projection[3] = 30000.0
    natural_norm = torch.full((16,), 300.0)          # a 100x outlier, as measured
    recipient_projection = torch.full((16,), 2.0); recipient_projection[9] = 12.0
    recipient_norm = torch.full((16,), 40.0)

    outlier = LC.calibrate_alpha(natural_projection, natural_norm, recipient_projection,
                                 recipient_norm, [3], mode="recipient_outlier")
    assert outlier.alpha_target == pytest.approx(12.0)
    assert outlier.out_of_distribution_ratio == pytest.approx(1.0)
    assert not outlier.out_of_distribution

    # ...whereas the ratio mode, on the same numbers, asks for a state 333x the loudest
    # thing the layer holds. That is the behaviour that saturated the sweep.
    ratio = LC.calibrate_alpha(natural_projection, natural_norm, recipient_projection,
                               recipient_norm, [3], mode="ratio")
    assert ratio.alpha_target == pytest.approx(30000.0 / 300.0 * 40.0)
    assert ratio.out_of_distribution_ratio > 300
    assert ratio.out_of_distribution


def test_persistence_needs_the_clean_baseline_for_an_early_window():
    """The blocks after an early window are where the NATURAL register forms.

    Read absolutely, an early induction that vanished the moment the hooks stopped still
    reports "carried", because the natural register is sitting in the same measurement.
    Against the clean baseline the same rows correctly report "maintained".
    """
    window = LC.Window("early", 14, 16)
    # Induced to 5.0 in the window; afterwards the treated run matches clean exactly,
    # and clean happens to be large there because the natural register has formed.
    clean = ([dict(layer=l, carrier_projection=0.0) for l in (14, 15, 16)] +
             [dict(layer=l, carrier_projection=30.0) for l in (18, 20, 22)])
    treated = ([dict(layer=l, carrier_projection=5.0) for l in (14, 15, 16)] +
               [dict(layer=l, carrier_projection=30.0) for l in (18, 20, 22)])

    absolute = LC.persistence_class(treated, window=window)
    assert absolute["persistence"] == "reconstructed", (
        "the absolute reading is fooled by the natural register, which is the bug this "
        "baseline exists to fix")
    assert absolute["scale"] == "absolute"

    excess = LC.persistence_class(treated, window=window, baseline=clean)
    assert excess["persistence"] == "maintained"
    assert excess["after_window_peak"] == pytest.approx(0.0)
    assert excess["scale"] == "excess over clean"


def test_the_baseline_still_reports_a_genuine_carry():
    window = LC.Window("late", 40, 42)
    clean = [dict(layer=l, carrier_projection=0.0) for l in (40, 41, 42, 44, 46)]
    treated = ([dict(layer=l, carrier_projection=10.0) for l in (40, 41, 42)] +
               [dict(layer=l, carrier_projection=8.0) for l in (44, 46)])
    out = LC.persistence_class(treated, window=window, baseline=clean)
    assert out["persistence"] == "carried"
    assert out["retention_ratio"] == pytest.approx(0.8)


def test_sinkhood_is_a_population_statistic_not_an_any():
    """Why carrier_is_sink changed meaning.

    With `any`, one carrier out of sixteen clearing the bar satisfies the level -- and at
    a depth just after dissolution some carriers are ALREADY sinks in the clean run, so
    an any-based L3 fired at a near-no-op perturbation and reported baseline sinkhood as
    successful induction. On real schnell it read L3 = x0.06 for the late carriers while
    the population mean sat at the clean value of 6.97 against a bar of 10.
    """
    n, heads, carriers = 32, 4, [0, 1, 2, 3]
    absolute = torch.full((heads, n), 1.0 / n)
    absolute[:, 0] = 0.6                       # ONE carrier far above the bar
    obs = _Obs(20, torch.zeros(n), torch.ones(n),
               absolute / absolute.sum(-1, keepdim=True), torch.zeros(heads, n))
    row = LC.achieved_lifecycle(_Trace({20: obs}), step=2, layers=[20],
                                carriers=carriers, sink_threshold=10.0)[0]
    assert row["carrier_any_sink"] is True, "one carrier does clear the bar"
    assert row["carrier_sink_fraction"] == pytest.approx(0.25)
    assert row["carrier_is_sink"] is False, (
        "the level must follow the population mean, not a single token")
    assert row["carrier_sink_strength"] < 10.0


# ============================================================ suppression: two modes
def _register_scene(n=200, c=64, n_registers=4, seed=0):
    """Ordinary tokens with weak v* components, plus a sparse high-norm register set."""
    torch.manual_seed(seed)
    v = LC._unit(torch.randn(c))
    x = torch.randn(n, c)
    x = x + 0.3 * torch.randn(n, 1) * v          # ordinary tokens DO carry weaker v*
    for t in range(n_registers):
        x[t] = x[t] * 0.2 + 40.0 * v             # sparse, high-norm, strongly aligned
    return x, v, list(range(n_registers))


def test_the_conjunction_is_sparse_where_a_percentile_alone_is_not():
    """The property that makes a state-based rule usable instead of a blunt eraser.

    A projection percentile ALWAYS selects its fraction -- at p99 on 200 tokens that is
    two tokens at every block whether or not a register exists. Ordinary tokens carry
    weaker v* components, so a percentile-only rule erases the direction from the image
    stream rather than preventing the sparse state. Requiring high norm as well makes the
    rule select nothing where nothing register-like is present.
    """
    x, v, registers = _register_scene()
    rule = LC.RegisterStateRule(highnorm_ratio=3.0, projection_percentile=99.0)
    assert set(rule.select(x, v)) <= set(registers)
    assert len(rule.select(x, v)) >= 1

    # The same tokens without their register state: nothing should be selected.
    ordinary_only = x.clone()
    for t in registers:
        alpha = ordinary_only[t] @ v
        residual = ordinary_only[t] - alpha * v
        ordinary_only[t] = residual * (ordinary_only[t].norm() / residual.norm())
    assert rule.select(ordinary_only, v) == [], (
        "the rule fires on a stream with no register state, so it is not detecting one")

    # ...whereas a percentile with no norm requirement selects regardless.
    blunt = LC.RegisterStateRule(highnorm_ratio=0.0, projection_percentile=99.0)
    assert len(blunt.select(ordinary_only, v)) >= 1, (
        "a percentile always selects its fraction; that is why it needs auditing")


def test_rule_selectivity_reports_what_a_rule_would_touch_on_the_clean_run():
    x, v, registers = _register_scene()
    obs = _Obs(20, x @ v, x.norm(dim=-1), torch.full((2, x.shape[0]), 1.0 / x.shape[0]))
    obs.states = x
    rows = LC.rule_selectivity(_Trace({20: obs}), LC.RegisterStateRule(), step=2,
                               layers=[20], direction=v, carriers=registers)
    assert rows[0]["selected"] >= 1
    assert rows[0]["share_of_image"] < 0.05, "a usable rule must be sparse"
    assert rows[0]["outside_frozen_carriers"] == 0


def test_both_modes_use_the_same_norm_preserving_operator():
    """EARLY ONLY, LATE ONLY and SUPPRESSION ONLY only compare if the operator is one.

    The norm-preserving operator is now the SINK-IDENTITY control rather than the
    primary, so it is named explicitly; it must still be the validated beta/gamma one.
    """
    from ditsinks.control_surface import control_surface_states

    x, v, registers = _register_scene()
    fixed = LC.Suppressor(v, mode="fixed", tokens=registers,
                          operator="norm_preserving").edit(x, _Ctx())
    state = LC.Suppressor(v, mode="state", rule=LC.RegisterStateRule(),
                          operator="norm_preserving").edit(x, _Ctx())
    reference, _ = control_surface_states(x, v, beta=0.0, gamma=1.0)
    for out in (fixed, state):
        touched = [t for t in registers if not torch.equal(out[t], x[t])]
        assert touched, "nothing was suppressed"
        assert torch.allclose(out[touched], reference[touched], atol=1e-4)
        for t in touched:
            assert float(out[t].norm()) == pytest.approx(float(x[t].norm()), rel=1e-4)


def test_the_primary_suppression_operator_is_subtractive():
    """Q16 now removes v* without rescaling the token back to register size."""
    x, v, registers = _register_scene()
    out = LC.Suppressor(v, mode="fixed", tokens=registers).edit(x, _Ctx())
    for t in registers:
        expected = x[t] - (x[t] @ v) * v
        assert torch.allclose(out[t], expected, atol=1e-4)
        assert float(out[t].norm()) < float(x[t].norm())


def test_fixed_mode_never_touches_anything_but_the_frozen_positions():
    """That is what keeps relocation visible in the diagnostic arm."""
    x, v, registers = _register_scene()
    out = LC.Suppressor(v, mode="fixed", tokens=registers).edit(x, _Ctx())
    others = [t for t in range(x.shape[0]) if t not in registers]
    assert torch.equal(out[others], x[others])


def test_the_suppressor_logs_every_site_and_names_new_targets():
    x, v, registers = _register_scene()
    sites = []
    suppressor = LC.Suppressor(v, mode="state", rule=LC.RegisterStateRule(), sites=sites)

    class _At:
        def __init__(self, layer):
            self.layer, self.step = layer, 2

    suppressor.edit(x, _At(17))
    # A DIFFERENT token acquires the state at the next block: it must be logged as new.
    moved = x.clone()
    for t in registers:
        alpha = moved[t] @ v
        residual = moved[t] - alpha * v
        moved[t] = residual * (moved[t].norm() / residual.norm())
    moved[101] = moved[101] * 0.2 + 40.0 * v
    suppressor.edit(moved, _At(18))

    assert len(sites) == 2
    assert sites[0].layer == 17 and sites[1].layer == 18
    assert 101 in sites[1].newly_targeted
    assert sites[1].n_selected >= 1
    assert sites[1].intervention_l2 > 0
    assert sites[1].max_projection_after < sites[1].max_projection_before
    assert set(LC.SuppressionSite.row(sites[1])) >= {
        "n_selected", "n_newly_targeted", "max_projection_before",
        "max_projection_after", "intervention_l2"}


def test_the_verdict_separates_regrowth_from_what_was_consumed():
    """Regrowth is a measurement of the model; delivery is what judges the run."""
    window = LC.Window("natural", 17, 20)

    def site(layer, selected, before):
        return LC.SuppressionSite(layer=layer, step=2, mode="state", n_selected=selected,
                                  newly_targeted=[], max_projection_before=before,
                                  n_highnorm_aligned_before=float(selected),
                                  intervention_l2=1.0, max_projection_after=0.0)

    clean_alpha = 100.0
    clean = dict(delivery="prevented", peak_delivered_register_share=0.03,
                 terminal_received=False)
    quiet = [site(17, 4, 100.0), site(18, 0, 2.0), site(19, 0, 1.5), site(20, 0, 1.0)]
    assert LC.suppression_verdict(
        quiet, window=window, delivery=clean,
        clean_carrier_projection=clean_alpha)["suppression"] == "complete"

    # A moderate projection left behind must NOT count as failure: ordinary tokens carry
    # weaker v* components, and gating on it would push the intervention towards the
    # indiscriminate erasure the design forbids.
    residue = [site(17, 4, 100.0), site(18, 0, 60.0), site(19, 0, 55.0), site(20, 0, 50.0)]
    out = LC.suppression_verdict(residue, window=window, delivery=clean,
                                 clean_carrier_projection=clean_alpha)
    assert out["suppression"] == "complete"
    assert out["peak_as_share_of_clean_carrier"] == pytest.approx(0.60)

    # The SUPPRESSED CARRIERS regaining sink behaviour is a real failure...
    with_sink = LC.suppression_verdict(
        quiet, window=window, clean_carrier_projection=clean_alpha, delivery=clean,
        lifecycle_rows=[dict(layer=19, carrier_is_sink=True, n_sinks_new=0)])
    assert with_sink["suppression"] == "incomplete"
    assert with_sink["first_sink_layer"] == 19
    assert "not carried by v* alone" in with_sink["note"]

    # ...but the image merely CONTAINING sinks is not. A clean FLUX block carries 130-200
    # of them, so gating on their presence marks every condition failed, clean included.
    ordinary_sinks = [dict(layer=layer, carrier_is_sink=False, n_sinks_new=150)
                      for layer in (17, 18, 19, 20)]
    baseline = [dict(layer=layer, n_sinks_new=150) for layer in (17, 18, 19, 20)]
    out = LC.suppression_verdict(quiet, window=window, delivery=clean,
                                 clean_carrier_projection=clean_alpha,
                                 lifecycle_rows=ordinary_sinks, baseline_rows=baseline)
    assert out["suppression"] == "complete"
    assert out["blocks_where_carriers_were_sinks"] == 0
    assert out["blocks_with_sink_relocation"] == 0

    # Relocation well beyond the clean level is REPORTED, still not gated.
    busy = [dict(layer=layer, carrier_is_sink=False, n_sinks_new=900)
            for layer in (17, 18, 19, 20)]
    out = LC.suppression_verdict(quiet, window=window, delivery=clean,
                                 clean_carrier_projection=clean_alpha,
                                 lifecycle_rows=busy, baseline_rows=baseline)
    assert out["suppression"] == "complete"
    assert out["blocks_with_sink_relocation"] == 4

    # Regrowth is COUNTED at every hook and gates nothing. The model rebuilding the
    # state between blocks is the measurement this arm exists to make.
    regrowing = [site(17, 4, 100.0), site(18, 3, 80.0), site(19, 3, 75.0), site(20, 2, 60.0)]
    verdict = LC.suppression_verdict(regrowing, window=window, delivery=clean,
                                     clean_carrier_projection=clean_alpha)
    assert verdict["suppression"] == "complete"
    assert verdict["hooks_where_state_regrew"] == 3
    assert verdict["first_regrowth_layer"] == 18
    assert verdict["mean_tokens_rebuilt"] == pytest.approx(8 / 3)


def test_a_mode_needs_what_that_mode_requires():
    _, v, registers = _register_scene()
    with pytest.raises(ValueError, match="frozen clean positions"):
        LC.Suppressor(v, mode="fixed")
    with pytest.raises(ValueError, match="frozen RegisterStateRule"):
        LC.Suppressor(v, mode="state")
    with pytest.raises(ValueError, match="mode must be"):
        LC.Suppressor(v, mode="dynamic", tokens=registers)


# ================== the achieved lifetime, which the image comparisons are grouped by
def _traj(values, high=None, new=None, first=10):
    out = []
    for i, v in enumerate(values):
        row = dict(layer=first + i, carrier_projection=v)
        if high is not None:
            row["n_highnorm_frozen"] = high[i]
        if new is not None:
            row["n_highnorm_new"] = new[i]
        out.append(row)
    return out


def test_an_early_pulse_is_not_called_a_continuous_extension():
    """The distinction the primary comparisons are grouped by.

    Induced at blocks 10-12, gone by 13, and natural formation at 15-18 runs exactly as
    in the clean run. Read absolutely the treated trajectory looks identical to clean in
    the natural window and large in the induction window -- which is what an early PULSE
    looks like, and calling it a retimed lifecycle would be the central overclaim.
    """
    induction, natural = LC.Window("early", 10, 12), LC.Window("natural", 15, 18)
    clean = _traj([0, 0, 0, 0, 0, 30, 30, 30, 30], high=[0]*5 + [8]*4)
    treated = _traj([20, 20, 20, 0.5, 0, 30, 30, 30, 30], high=[0]*5 + [8]*4)
    out = LC.classify_achieved_lifetime(treated, clean, induction=induction, natural=natural)
    assert out["achieved"] == "pulse_then_natural"
    assert "EARLY PULSE" in out["note"]
    assert out["survived_fraction"] < 0.25


def test_a_state_that_survives_to_the_natural_window_is_a_continuous_extension():
    induction, natural = LC.Window("early", 10, 12), LC.Window("natural", 15, 18)
    clean = _traj([0, 0, 0, 0, 0, 30, 30, 30, 30], high=[0]*5 + [8]*4)
    treated = _traj([20, 20, 20, 18, 16, 45, 44, 43, 42], high=[0]*5 + [8]*4)
    out = LC.classify_achieved_lifetime(treated, clean, induction=induction, natural=natural)
    assert out["achieved"] == "continuous_extension"
    assert out["survived_fraction"] >= 0.25


def test_losing_the_natural_carriers_is_displacement_whatever_the_image_shows():
    induction, natural = LC.Window("early", 10, 12), LC.Window("natural", 15, 18)
    clean = _traj([0, 0, 0, 0, 0, 30, 30, 30, 30], high=[0]*5 + [8]*4)
    treated = _traj([20, 20, 20, 18, 16, 2, 2, 2, 2],
                    high=[0]*5 + [0]*4, new=[0]*5 + [300]*4)
    out = LC.classify_achieved_lifetime(treated, clean, induction=induction, natural=natural)
    assert out["achieved"] == "displacement"
    assert out["natural_carrier_presence_kept"] == pytest.approx(0.0)
    assert out["new_carriers_in_natural_window"] == 1200
    assert "did not run as it does in the clean run" in out["note"]


def test_no_excess_in_the_window_is_reported_as_no_induction():
    induction, natural = LC.Window("early", 10, 12), LC.Window("natural", 15, 18)
    clean = _traj([0, 0, 0, 0, 0, 30, 30, 30, 30])
    out = LC.classify_achieved_lifetime(clean, clean, induction=induction, natural=natural)
    assert out["achieved"] == "no_induction"


def test_cosine_matching_is_available_but_marked_secondary():
    """It must never be the primary anchor: matching the query-key preference would
    calibrate away the depth-dependence the experiment is measuring."""
    natural_projection = torch.zeros(16); natural_projection[3] = 26187.0
    calibration = LC.calibrate_alpha(
        natural_projection, torch.full((16,), 1968.0), torch.full((16,), 1163.0),
        torch.full((16,), 1334.0), [3], mode="cosine_matched")
    assert "SECONDARY CONTROL" in calibration.note
    assert "normalises away" in calibration.note
    primary = LC.calibrate_alpha(
        natural_projection, torch.full((16,), 1968.0), torch.full((16,), 1163.0),
        torch.full((16,), 1334.0), [3], mode="recipient_outlier")
    assert primary.note.startswith("PRIMARY")
    assert primary.alpha_target == pytest.approx(1163.0)


def test_the_census_tells_relocation_from_threshold_flicker():
    """Summed per-block counts cannot answer the question; unique positions can.

    Both cases below give the SAME cumulative total of 40 token-layer observations. One
    is ten positions that are high-norm throughout the window -- a genuinely relocated
    population. The other is forty positions each crossing the bar once, which is the
    threshold flickering and says nothing about relocation.
    """
    window = LC.Window("natural", 20, 23)

    stable = [dict(layer=l, highnorm_new_id_set=tuple(range(10)),
                   highnorm_frozen_id_set=()) for l in (20, 21, 22, 23)]
    flicker = [dict(layer=l, highnorm_new_id_set=tuple(range(100 + 10 * i, 110 + 10 * i)),
                    highnorm_frozen_id_set=())
               for i, l in enumerate((20, 21, 22, 23))]

    a, b = LC.carrier_census(stable, window=window), LC.carrier_census(flicker, window=window)
    assert a["observations_new"] == b["observations_new"] == 40, "same cumulative count"
    assert a["unique_new"] == 10 and b["unique_new"] == 40
    assert a["median_blocks_per_new"] == 4.0 and b["median_blocks_per_new"] == 1.0
    assert a["new_present_throughout"] == 10 and b["new_present_throughout"] == 0
    assert "stable relocated population" in a["interpretation"]
    assert "threshold flicker" in b["interpretation"]


def test_the_census_reports_nothing_when_nothing_relocated():
    window = LC.Window("natural", 20, 21)
    rows = [dict(layer=l, highnorm_new_id_set=(), highnorm_frozen_id_set=(1, 2, 3))
            for l in (20, 21)]
    out = LC.carrier_census(rows, window=window)
    assert out["unique_new"] == 0
    assert out["unique_frozen"] == 3 and out["frozen_present_throughout"] == 3
    assert out["interpretation"] == "no position outside the frozen set became high-norm"


# ============================== delivery: what the next computation actually receives
def _delivery_rows(layers, *, n=64, carriers=(3, 5), seed=0, carrier_cos=0.9,
                   ordinary_cos=0.2, carrier_norm=40.0):
    """Synthetic probe statistics: a few aligned carriers among ordinary tokens."""
    torch.manual_seed(seed)
    rows = {}
    for layer in layers:
        cos = torch.rand(n) * float(ordinary_cos)
        norm = torch.ones(n) + torch.rand(n) * 0.1
        for t in carriers:
            cos[t] = float(carrier_cos)
            norm[t] = float(carrier_norm)
        rows[(2, int(layer))] = dict(
            supplied_norm=norm, supplied_projection=cos * norm,
            supplied_cosine=cos,
            # A LayerNorm downstream: every token arrives at the same magnitude.
            consumed_norm=torch.full((n,), 5.0),
            consumed_projection=cos * 5.0, consumed_cosine=cos.clone())
    return LC.DeliveryProbe.from_rows(rows)


def test_delivered_rows_expose_that_the_consumer_cannot_see_token_magnitude():
    """The LayerNorm fact, read off the run's own numbers rather than from the source."""
    probe = _delivery_rows([20, 21])
    rows = LC.delivered_rows(probe, step=2, layers=[20, 21], carriers=[3, 5])
    for row in rows:
        assert row["supplied_norm_spread"] > 30, "the residual carries the register norm"
        assert abs(row["consumed_norm_spread"] - 1.0) < 1e-5, (
            "if the consumed spread were not ~1 the delivery gate could not be an "
            "alignment question")
        assert row["carrier_cos_max"] > row["ordinary_cos_max"]


def test_delivery_is_prevented_when_nothing_arrives_above_the_ordinary_ceiling():
    clean = LC.delivered_rows(_delivery_rows([20, 21, 22]), step=2, layers=[20, 21, 22],
                              carriers=[3, 5])
    # Suppressed: the carriers arrive no more aligned than an ordinary token.
    treated = LC.delivered_rows(
        _delivery_rows([20, 21, 22], carrier_cos=0.15, carrier_norm=1.0),
        step=2, layers=[20, 21, 22], carriers=[3, 5])
    verdict = LC.delivery_verdict(treated, clean, window=LC.Window("s", 20, 22))
    assert verdict["delivery"] == "prevented"
    assert verdict["peak_delivered_register_share"] < 0.25


def test_delivery_is_reported_when_a_consumer_receives_the_state():
    layers = [20, 21, 22]
    clean = LC.delivered_rows(_delivery_rows(layers), step=2, layers=layers,
                              carriers=[3, 5])
    leaky = _delivery_rows(layers, carrier_cos=0.15, carrier_norm=1.0)
    hot = _delivery_rows([21], carrier_cos=0.88, carrier_norm=40.0)
    leaky.rows[(2, 21)] = hot.rows[(2, 21)]
    treated = LC.delivered_rows(leaky, step=2, layers=layers, carriers=[3, 5])
    verdict = LC.delivery_verdict(treated, clean, window=LC.Window("s", 20, 22))
    assert verdict["delivery"] == "delivered"
    assert verdict["first_receiving_layer"] == 21


def test_the_terminal_boundary_is_scored_separately_from_the_interval():
    """Block 39's output reaching block 40's attention is its own finding."""
    layers = [20, 21, 22, 23]
    clean = LC.delivered_rows(_delivery_rows(layers), step=2, layers=layers,
                              carriers=[3, 5])
    treated_probe = _delivery_rows(layers, carrier_cos=0.15, carrier_norm=1.0)
    treated_probe.rows[(2, 23)] = _delivery_rows([23], carrier_cos=0.85).rows[(2, 23)]
    treated = LC.delivered_rows(treated_probe, step=2, layers=layers, carriers=[3, 5])
    verdict = LC.delivery_verdict(treated, clean, window=LC.Window("s", 20, 22),
                                  terminal=23)
    assert verdict["delivery"] == "prevented", "the interval itself was clean"
    assert verdict["terminal_received"] is True
    assert verdict["terminal_layer"] == 23
    assert "boundary" in verdict["terminal_note"] or "regrown" in verdict["terminal_note"]


def test_an_unprobed_boundary_is_reported_as_unmeasured_not_as_closed():
    layers = [20, 21]
    clean = LC.delivered_rows(_delivery_rows(layers), step=2, layers=layers,
                              carriers=[3, 5])
    treated = LC.delivered_rows(_delivery_rows(layers, carrier_cos=0.15, carrier_norm=1.0),
                                step=2, layers=layers, carriers=[3, 5])
    verdict = LC.delivery_verdict(treated, clean, window=LC.Window("s", 20, 21),
                                  terminal=22)
    assert verdict["terminal_received"] is None
    assert "UNMEASURED" in verdict["terminal_note"]


# ================================================= the precursor / coverage audit
def test_coverage_finds_a_precursor_the_conjunction_cannot_see():
    """An aligned token below the norm bar delivers what a register delivers."""
    probe = _delivery_rows([18], carriers=[3, 5])
    store = probe.rows[(2, 18)]
    # A token being written into: strongly aligned, norm still ordinary. The rule is
    # evaluated on the residual and misses it; the consumer sees only the direction.
    store["supplied_norm"][9] = 1.4
    store["supplied_cosine"][9] = 0.95
    store["supplied_projection"][9] = 0.95 * 1.4
    store["consumed_cosine"][9] = 0.95
    store["consumed_projection"][9] = 0.95 * 5.0
    rule = LC.RegisterStateRule(highnorm_ratio=3.0, projection_percentile=90.0)
    rows = LC.delivery_coverage(probe, rule, step=2, layers=[18], carriers=[3, 5])
    assert rows[0]["n_selected"] >= 2
    assert rows[0]["n_missed"] >= 1 and 9 in rows[0]["missed_ids"]
    summary = LC.coverage_recommendation(rows, rule)
    assert summary["rule_is_sufficient"] is False
    assert summary["recommended_highnorm_ratio"] < 3.0
    assert "re-running rule_selectivity" in summary["note"], (
        "a recommendation to lower the bar must carry its selectivity cost")


def test_coverage_reports_a_sufficient_rule_as_sufficient():
    probe = _delivery_rows([20, 21], carriers=[3, 5])
    rule = LC.RegisterStateRule(highnorm_ratio=3.0, projection_percentile=90.0)
    rows = LC.delivery_coverage(probe, rule, step=2, layers=[20, 21], carriers=[3, 5])
    summary = LC.coverage_recommendation(rows, rule)
    assert summary["rule_is_sufficient"] is True
    assert summary["recommended_highnorm_ratio"] is None
    assert "no case for broadening" in summary["note"]


# ======================================================= the suppression schedule
def test_the_schedule_closes_its_terminal_boundary():
    schedule = LC.suppression_schedule(LC.Window("s", 17, 39), n_layers=57)
    assert schedule.terminal == 40
    assert schedule.hook_layers[-1] == 40, (
        "without a hook at 40 the last guarded block's output reaches an unguarded "
        "attention")
    assert schedule.hook_layers[0] == 17


def test_the_schedule_stops_before_a_following_induction_and_cleans_its_first_block():
    schedule = LC.suppression_schedule(LC.Window("s", 17, 39), n_layers=57, stop_before=35)
    assert schedule.layers[-1] == 34, "suppression must not fight the induction"
    assert schedule.terminal == 35, "the induction's first block is cleaned before it"
    assert schedule.stopped_before == 35


def test_a_schedule_running_to_the_last_block_has_no_boundary_to_close():
    schedule = LC.suppression_schedule(LC.Window("s", 54, 56), n_layers=57)
    assert schedule.terminal is None
    assert schedule.hook_layers == [54, 55, 56]


def test_a_fully_overlapped_interval_is_refused_rather_than_silently_emptied():
    with pytest.raises(ValueError, match="overlap"):
        LC.suppression_schedule(LC.Window("s", 30, 39), n_layers=57, stop_before=30)


# ================================== regrowth is a measurement, delivery is the gate
def _sites(layers, *, selected):
    return [LC.SuppressionSite(layer=int(l), step=2, mode="state", n_selected=int(selected),
                               newly_targeted=[], max_projection_before=5.0,
                               n_highnorm_aligned_before=float(selected),
                               intervention_l2=1.0, max_projection_after=0.1)
            for l in layers]


def test_regrowth_alone_no_longer_fails_a_suppression_that_delivered_nothing():
    """The model rebuilding the state is a finding about the model, not a failed run."""
    sites = _sites(range(20, 30), selected=7)
    verdict = LC.suppression_verdict(
        sites, window=LC.Window("s", 20, 29), clean_carrier_projection=100.0,
        delivery=dict(delivery="prevented", peak_delivered_register_share=0.05,
                      terminal_received=False))
    assert verdict["suppression"] == "complete"
    assert verdict["hooks_where_state_regrew"] == 9
    assert verdict["regrowth_rate"] == 1.0
    assert "not a failure" in verdict["note"]


def test_delivery_is_what_fails_a_suppression():
    verdict = LC.suppression_verdict(
        _sites(range(20, 30), selected=0), window=LC.Window("s", 20, 29),
        clean_carrier_projection=100.0,
        delivery=dict(delivery="delivered", note="3 of 10 blocks",
                      peak_delivered_register_share=0.8, terminal_received=False))
    assert verdict["suppression"] == "incomplete"
    assert "INCOMPLETE SUPPRESSION" in verdict["note"]


def test_an_open_terminal_boundary_fails_an_otherwise_clean_interval():
    verdict = LC.suppression_verdict(
        _sites(range(20, 30), selected=0), window=LC.Window("s", 20, 29),
        clean_carrier_projection=100.0,
        delivery=dict(delivery="prevented", peak_delivered_register_share=0.02,
                      terminal_received=True))
    assert verdict["suppression"] == "incomplete"
    assert "terminal cleanup" in verdict["note"]


def test_suppression_without_a_delivery_measurement_is_unknown_not_complete():
    verdict = LC.suppression_verdict(
        _sites(range(20, 30), selected=0), window=LC.Window("s", 20, 29),
        clean_carrier_projection=100.0)
    assert verdict["suppression"] == "not measured"
    assert "UNKNOWN" in verdict["note"]


# ========================== continuous extension versus a second state after a gap
def _lifecycle_series(values, *, metric="carrier_projection", first=20):
    return [{"layer": first + i, metric: float(v)} for i, v in enumerate(values)]


def test_the_dissolution_onset_is_measured_not_taken_from_the_declared_range():
    clean = _lifecycle_series([100, 100, 100, 100, 100, 40, 20, 8, 3, 0.5])
    found = LC.measured_dissolution_onset(clean, natural=LC.Window("n", 20, 29))
    assert found["onset"] == 25, "the decline begins five blocks before it is gone"
    assert found["disappears_at"] == 27
    assert found["plateau_layer"] == 20


def test_the_late_window_can_never_overlap_the_natural_one():
    """Late is where the state exists in the LATE conditions; sharing a block with the
    natural range makes a late-condition effect unattributable to lateness."""
    from ditsinks.discovery import LayerRanges
    ranges = LayerRanges((17, 19), (20, 39), (40, 56))
    windows = LC.windows_from_ranges(ranges, length=3, n_layers=57)
    assert windows["late"].layers == (40, 41, 42)
    assert not set(windows["late"].layers) & set(windows["natural"].layers)
    with pytest.raises(ValueError, match="inside the natural range"):
        LC.windows_from_ranges(ranges, length=3, n_layers=57, late_first=35)
    further = LC.windows_from_ranges(ranges, length=3, n_layers=57, late_first=44)
    assert further["late"].layers == (44, 45, 46), "moving it further OUT is allowed"
    assert further["early"].length == further["late"].length


def test_extension_bridges_the_natural_decline_without_moving_the_late_window():
    """Maintenance must start where the state is still present -- the measured
    dissolution onset -- or extension becomes re-induction after a gap. That is the
    bridge's job; the late window stays after the natural range."""
    natural, late = LC.Window("natural", 20, 39), LC.Window("late", 40, 42)
    bridge = LC.extension_bridge(natural, late, 35)
    assert bridge is not None and bridge.layers == (35, 36, 37, 38, 39)
    assert bridge.last + 1 == late.first, "the bridge must hand over with no gap"
    assert LC.extension_bridge(natural, late, None) is None, "no onset, nothing to bridge"
    assert LC.extension_bridge(natural, late, 40) is None, "no room before late"
    assert LC.extension_bridge(natural, late, 5).first == natural.first


def test_continuity_distinguishes_extension_from_re_induction():
    clean = _lifecycle_series([100] * 5 + [40, 10, 1, 0.2, 0.1])
    extended = _lifecycle_series([100] * 5 + [90, 85, 80, 78, 75])
    check = LC.continuity_check(extended, clean, first=20, last=29)
    assert check["continuity"] == "continuous"
    assert check["label"] == "continuous_extension"
    assert check["n_gap_blocks"] == 0

    lapsed = _lifecycle_series([100] * 5 + [40, 2, 1, 80, 85])
    check = LC.continuity_check(lapsed, clean, first=20, last=29)
    assert check["label"] == "late_re_induction"
    assert check["first_gap"] == 25
    assert "LATE RE-INDUCTION" in check["note"]


def test_continuity_uses_a_fixed_bar_so_the_clean_run_is_not_called_continuous():
    clean = _lifecycle_series([100] * 5 + [40, 10, 1, 0.2, 0.1])
    check = LC.continuity_check(clean, clean, first=20, last=29)
    assert check["label"] == "late_re_induction", (
        "a bar that followed the clean decline would call dissolution continuous")


def test_a_degenerate_delivery_scale_is_unmeasured_rather_than_prevented():
    """If the clean carriers deliver no more alignment than ordinary tokens there is no
    scale to score on, and reporting success from an absent measurement would be worse
    than reporting nothing."""
    layers = [20, 21]
    flat = _delivery_rows(layers, carrier_cos=0.05, ordinary_cos=0.4, carrier_norm=40.0)
    clean = LC.delivered_rows(flat, step=2, layers=layers, carriers=[3, 5])
    treated = LC.delivered_rows(
        _delivery_rows(layers, carrier_cos=0.05, ordinary_cos=0.4, carrier_norm=1.0),
        step=2, layers=layers, carriers=[3, 5])
    verdict = LC.delivery_verdict(treated, clean, window=LC.Window("s", 20, 21))
    assert verdict["delivery"] == "not measured"
    assert "no register-delivery scale" in verdict["reason"]


def test_a_register_range_containing_no_block_is_refused():
    from ditsinks.discovery import LayerRanges
    with pytest.raises(ValueError, match="contains no block"):
        LC.windows_from_ranges(LayerRanges((1, 2), (5, 3), (6, 8)), length=2, n_layers=9)


# ============================ the temporal dose: WHICH denoising steps the edit fires at
def test_the_verdict_reads_one_step_and_reports_how_many_ran():
    """A schedule firing at every denoising step interleaves N sweeps down the stack.

    Without a step filter the next step's first hook is counted as regrowth after the
    previous step's last, which is not regrowth at all -- it is the schedule starting
    over. And the temporal coverage has to be reported, because an absent image effect
    from a one-step schedule says nothing about the register.
    """
    sites = [LC.SuppressionSite(layer=l, step=st, mode="state", n_selected=(3 if l > 20 else 7),
                                newly_targeted=[], max_projection_before=5.0,
                                n_highnorm_aligned_before=3.0, intervention_l2=1.0,
                                max_projection_after=0.1)
             for st in range(4) for l in range(20, 25)]
    clean = dict(delivery="prevented", peak_delivered_register_share=0.02,
                 terminal_received=False)
    one = LC.suppression_verdict(sites, window=LC.Window("s", 20, 24), step=2,
                                 n_steps_total=20, delivery=clean,
                                 clean_carrier_projection=100.0)
    assert one["n_hooks"] == 5, "one step's worth of hooks, not all four steps'"
    assert one["hooks_all_steps"] == 20
    assert one["steps_hooked"] == 4
    assert one["temporal_coverage"] == pytest.approx(4 / 20)
    # Without the filter every step's hooks are pooled and the counts are meaningless.
    pooled = LC.suppression_verdict(sites, window=LC.Window("s", 20, 24),
                                    delivery=clean, clean_carrier_projection=100.0)
    assert pooled["n_hooks"] == 20


def test_newly_targeted_is_counted_per_denoising_step():
    """Across steps a single `seen` set reports zero relocation from step 1 onward."""
    torch.manual_seed(0)
    v = LC._unit(torch.randn(32))
    rule = LC.RegisterStateRule(highnorm_ratio=2.0, projection_percentile=80.0)
    suppressor = LC.Suppressor(v, mode="fixed", tokens=[3, 7], rule=rule)

    class Ctx:
        layer, step = 20, 0

    x = torch.randn(16, 32)
    x[3] = x[3] * 0.3 + 9.0 * v
    x[7] = x[7] * 0.3 + 8.0 * v
    for step in range(3):
        Ctx.step = step
        suppressor.edit(x, Ctx)
    first = [s for s in suppressor.sites if s.step == 0]
    later = [s for s in suppressor.sites if s.step == 2]
    assert first and later
    assert len(later[0].newly_targeted) == len(first[0].newly_targeted) == 2, (
        "each denoising step starts its own relocation census; pooling them across "
        "steps reports no relocation after the first step")


# ====== what is frozen at one denoising step, and whether it still holds at the others
class _DriftTrace:
    """A clean trace whose carriers move and whose scale grows across denoising steps."""

    def __init__(self, moves=True, scale=1.0, n=64, layer=20, steps=(0, 1, 2, 3)):
        torch.manual_seed(0)
        self.v = LC._unit(torch.randn(16))
        self.rows = {}
        for i, step in enumerate(steps):
            x = torch.randn(n, 16) * 0.3
            carriers = [3, 5] if (not moves or i == 0) else [40 + i, 41 + i]
            for t in carriers:
                x[t] = 9.0 * self.v * (scale ** i)
            self.rows[(int(step), int(layer))] = x

    def at(self, step, layer):
        x = self.rows.get((int(step), int(layer)))
        if x is None:
            return None
        return type("Obs", (), dict(norm=x.norm(dim=-1), projection=x @ self.v))()


def test_drift_reports_stable_positions_and_scale_as_stable():
    trace = _DriftTrace(moves=False, scale=1.0)
    rows = LC.carrier_drift(trace, layer=20, steps=(0, 1, 2, 3), direction=trace.v,
                            reference_step=0,
                            rule=LC.RegisterStateRule(highnorm_ratio=2.0,
                                                      projection_percentile=90.0))
    verdict = LC.drift_verdict(rows)
    assert verdict["drift"] == "stable"
    assert verdict["worst_position_overlap"] == pytest.approx(1.0)
    assert "measured, not assumed" in verdict["note"]


def test_drift_catches_carriers_that_move_across_denoising_steps():
    """The induction arms write at frozen ids; if the register leaves them they induce
    into tokens that carry nothing for most of the trajectory."""
    trace = _DriftTrace(moves=True, scale=1.0)
    rows = LC.carrier_drift(trace, layer=20, steps=(0, 1, 2, 3), direction=trace.v,
                            reference_step=0,
                            rule=LC.RegisterStateRule(highnorm_ratio=2.0,
                                                      projection_percentile=90.0))
    verdict = LC.drift_verdict(rows)
    assert verdict["drift"] == "drifts"
    assert verdict["positions_stable"] is False
    assert verdict["steps_below_overlap"] == [1, 2, 3]
    assert "writing into tokens that carry nothing" in verdict["note"]


def test_drift_catches_a_scale_that_moves_even_when_positions_hold():
    """The easier failure to miss: nothing looks wrong, the coefficient is just wrong."""
    trace = _DriftTrace(moves=False, scale=2.0)
    rows = LC.carrier_drift(trace, layer=20, steps=(0, 1, 2, 3), direction=trace.v,
                            reference_step=0,
                            rule=LC.RegisterStateRule(highnorm_ratio=2.0,
                                                      projection_percentile=90.0))
    verdict = LC.drift_verdict(rows)
    assert verdict["positions_stable"] is True
    assert verdict["scale_stable"] is False
    assert verdict["worst_scale_ratio"] > 4
    assert "Calibrate per step" in verdict["note"]


def test_induction_can_take_positions_and_a_coefficient_per_step():
    x, v = _states()

    class Ctx:
        layer, step = 7, 0

    edit = LC.lifecycle_edit("induce", {0: [5], 1: [9]}, v,
                             alpha_target={0: 7.5, 1: 3.0})
    Ctx.step = 0
    out = edit(x, Ctx)
    assert float(out[5] @ v) == pytest.approx(7.5, abs=1e-4)
    assert torch.equal(out[9], x[9]), "step 0 must not touch step 1's position"
    Ctx.step = 1
    out = edit(x, Ctx)
    assert float(out[9] @ v) == pytest.approx(3.0, abs=1e-4)
    assert torch.equal(out[5], x[5])


def test_an_uncalibrated_step_is_left_alone_rather_than_given_another_steps_number():
    x, v = _states()

    class Ctx:
        layer, step = 7, 4

    edit = LC.lifecycle_edit("induce", {4: [5]}, v, alpha_target={0: 7.5})
    assert torch.equal(edit(x, Ctx), x), (
        "a step with no calibration must be skipped, not given a coefficient measured "
        "somewhere else")


def test_no_carrier_anywhere_is_unmeasured_rather_than_drifting():
    """Calling an absent population 'drift' would send a run to per-step re-selection of
    nothing, and would read as a finding about the register."""
    class _Empty:
        def __init__(self):
            torch.manual_seed(1)
            self.v = LC._unit(torch.randn(16))
            self.x = torch.randn(48, 16) * 0.3       # nothing high-norm, nothing aligned

        def at(self, step, layer):
            return type("Obs", (), dict(norm=self.x.norm(dim=-1),
                                        projection=self.x @ self.v))()

    trace = _Empty()
    rows = LC.carrier_drift(trace, layer=20, steps=(0, 1, 2), direction=trace.v,
                            reference_step=0,
                            rule=LC.RegisterStateRule(highnorm_ratio=3.0,
                                                      projection_percentile=99.0))
    verdict = LC.drift_verdict(rows)
    assert verdict["drift"] == "not measured"
    assert verdict["positions_stable"] is None
    assert "no population whose stability could be measured" in verdict["reason"]


# ========== the delivery gate must not divide by a register that is not there yet
def test_a_block_where_the_clean_register_does_not_stand_out_is_not_scored():
    """The writer blocks: the clean model has not yet put the register into the block's
    input, so its carriers deliver barely more than an ordinary token. Scoring there
    divides by a near-zero gap, and a few hundredths of cosine of trajectory-to-
    trajectory jitter reads as 'twice the natural register'. That is what made every
    suppression condition in the first dev pilot read ~1.97 'delivered'."""
    layers = [17, 20]
    clean_probe = _delivery_rows(layers, carrier_cos=0.9)
    # Block 17: the register is not written yet -- carriers barely above ordinary.
    clean_probe.rows[(2, 17)] = _delivery_rows([17], carrier_cos=0.21,
                                               ordinary_cos=0.2).rows[(2, 17)]
    clean = LC.delivered_rows(clean_probe, step=2, layers=layers, carriers=[3, 5])
    treated_probe = _delivery_rows(layers, carrier_cos=0.15, carrier_norm=1.0)
    # Ordinary jitter at 17: a max a few hundredths above the clean ceiling.
    treated_probe.rows[(2, 17)]["consumed_cosine"][11] = 0.24
    treated = LC.delivered_rows(treated_probe, step=2, layers=layers, carriers=[3, 5])

    unconditioned = LC.delivery_verdict(treated, clean, window=LC.Window("s", 17, 20),
                                        min_headroom=0.0)
    assert unconditioned["delivery"] == "delivered", (
        "without the floor, jitter over a tiny gap reads as a delivered register")
    verdict = LC.delivery_verdict(treated, clean, window=LC.Window("s", 17, 20),
                                  min_headroom=0.10)
    assert verdict["delivery"] == "prevented"
    assert verdict["blocks_without_register"] == [17]
    assert verdict["n_blocks_scored"] == 1


def test_register_tokens_outside_the_frozen_set_do_not_set_the_ordinary_ceiling():
    """A frozen set smaller than the real population leaves genuine register tokens in
    the 'ordinary' pool, and the ceiling is then a register."""
    probe = _delivery_rows([20], carriers=[3, 5, 9, 12], carrier_cos=0.9)
    frozen_only = LC.delivered_rows(probe, step=2, layers=[20], carriers=[3, 5])[0]
    assert frozen_only["ordinary_cos_max"] == pytest.approx(0.9), (
        "tokens 9 and 12 are registers the frozen set missed, and they set the ceiling")
    rule = LC.RegisterStateRule(highnorm_ratio=3.0, projection_percentile=90.0)
    widened = LC.delivered_rows(probe, step=2, layers=[20], carriers=[3, 5],
                                rule=rule)[0]
    assert widened["ordinary_cos_max"] < 0.5
    assert widened["n_carriers"] == 4
    assert widened["n_rule_selected"] == 2


# ======================================= the two counterfactuals for "no register"
def _register_token(n=32, c=16, seed=0):
    torch.manual_seed(seed)
    v = LC._unit(torch.randn(c))
    x = torch.randn(n, c)
    x[4] = x[4] + 40.0 * v                 # a register: ~10x the median, mostly along v*
    return x, v


def test_norm_preserving_removal_keeps_a_register_sized_token_subtractive_does_not():
    """The operator that produced the first dev pilot's stars.

    Norm-preserving removal scales the token's remainder up to register magnitude, so it
    stays a huge-norm token -- now pointing along ordinary content, where later blocks'
    ordinary-sized writes can barely move it. Subtractive removal leaves the token as it
    would be without its v* component: ordinary content at ordinary size.
    """
    x, v = _register_token()
    rule = LC.RegisterStateRule(highnorm_ratio=3.0, projection_percentile=90.0)
    median = float(x.norm(dim=-1).median())
    kept = LC.Suppressor(v, mode="state", rule=rule, operator="norm_preserving").edit(x, _Ctx())
    gone = LC.Suppressor(v, mode="state", rule=rule, operator="subtractive").edit(x, _Ctx())
    for out in (kept, gone):
        assert abs(float(out[4] @ v)) < 1e-3, "both remove the v* component"
    assert float(kept[4].norm()) / median > 5, "norm-preserving keeps a register-sized norm"
    assert float(gone[4].norm()) / median < 2, "subtractive leaves an ordinary-sized token"
    others = [i for i in range(x.shape[0]) if i != 4]
    assert torch.equal(kept[others], x[others]) and torch.equal(gone[others], x[others])


def test_the_site_records_how_far_the_remainder_was_scaled():
    x, v = _register_token()
    rule = LC.RegisterStateRule(highnorm_ratio=3.0, projection_percentile=90.0)
    suppressor = LC.Suppressor(v, mode="state", rule=rule, operator="subtractive")
    suppressor.edit(x, _Ctx())
    site = suppressor.sites[-1]
    assert site.operator == "subtractive"
    assert 0 < site.residual_to_median < 2
    assert site.row()["residual_to_median"] == site.residual_to_median


def test_an_unknown_operator_is_refused():
    _, v = _register_token()
    with pytest.raises(ValueError, match="operator must be"):
        LC.Suppressor(v, mode="state", rule=LC.RegisterStateRule(), operator="zero")


def test_cosine_matched_uses_the_carriers_actual_alignment():
    """It divided alpha by the MEDIAN norm -- a ratio, 13x on FLUX -- and clamped it to
    0.999, so it always resolved to the same target whatever the register looked like."""
    natural_norm = torch.full((16,), 10.0)
    natural_projection = torch.zeros(16)
    natural_norm[3], natural_projection[3] = 50.0, 40.0       # cos 0.8, alpha/median 4
    calibration = LC.calibrate_alpha(natural_projection, natural_norm,
                                     torch.zeros(16), torch.full((16,), 10.0), [3],
                                     mode="cosine_matched")
    expected = 0.8 * 10.0 / (1 - 0.8 ** 2) ** 0.5            # alpha giving cos 0.8 on r=10
    assert calibration.alpha_target == pytest.approx(expected, rel=1e-4)


# ================================================== across denoising time
def test_denoising_phases_split_the_trajectory_into_thirds():
    assert LC.denoising_phase_steps("all", 28) == list(range(28))
    early, mid, late = (LC.denoising_phase_steps(p, 28) for p in ("early", "mid", "late"))
    assert early[0] == 0 and late[-1] == 27
    assert sorted(early + mid + late) == list(range(28)), "the thirds must tile the run"
    assert LC.denoising_phase_steps([9, 3], 28) == [3, 9]
    with pytest.raises(ValueError, match="phase must be"):
        LC.denoising_phase_steps("middle", 28)


# ============================ are the image changes on the edited tokens or elsewhere?
def test_colocation_finds_changes_sitting_on_the_suppressed_tokens():
    import numpy as np
    reference = np.zeros((64, 64, 3))
    image = reference.copy()
    suppressed = {5: 3, 27: 1, 40: 2}
    for t in suppressed:                          # a bright blob on each suppressed token
        r, c = divmod(t, 8)
        image[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = 200
    out = LC.artifact_colocation(image, reference, grid=(8, 8), suppressed=suppressed,
                                 top_fraction=0.05)
    assert out["colocation"] == "on the suppressed tokens"
    assert out["top_share_suppressed"] == 1.0


def test_colocation_reports_changes_carried_to_untouched_tokens():
    import numpy as np
    reference = np.zeros((64, 64, 3))
    image = reference.copy()
    image[0:8, 56:64] = 200                       # a change at token 7, never suppressed
    out = LC.artifact_colocation(image, reference, grid=(8, 8),
                                 suppressed={40: 1, 41: 1}, top_fraction=0.02)
    assert out["colocation"] == "not concentrated on them"
    assert "through the network" in out["note"]


def test_suppressed_positions_counts_steps_not_hooks():
    sites = [LC.SuppressionSite(layer=l, step=s, mode="state", n_selected=2,
                                newly_targeted=([3, 9] if l == 20 else [11]),
                                max_projection_before=1.0, n_highnorm_aligned_before=1.0,
                                intervention_l2=1.0, max_projection_after=0.0)
             for s in (0, 1) for l in (20, 21)]
    assert LC.suppressed_positions(sites) == {3: 2, 9: 2, 11: 2}


def test_structure_extent_reads_onset_and_offset_shifts_against_clean():
    clean = [dict(step=0, layer=l, carrier_projection=(100.0 if 20 <= l <= 30 else 1.0))
             for l in range(40)]
    early = [dict(step=0, layer=l, carrier_projection=(100.0 if 16 <= l <= 30 else 1.0))
             for l in range(40)]
    extent = LC.structure_extent(early, clean, metric="carrier_projection")[0]
    assert (extent["onset"], extent["offset"]) == (16, 30)
    assert (extent["clean_onset"], extent["clean_offset"]) == (20, 30)
    assert extent["onset_shift"] == -4 and extent["offset_shift"] == 0
    gone = [dict(step=0, layer=l, carrier_projection=1.0) for l in range(40)]
    extent = LC.structure_extent(gone, clean, metric="carrier_projection")[0]
    assert extent["onset"] is None and extent["n_blocks"] == 0, (
        "a structure that is removed everywhere has no extent, not a shifted one")


def test_structure_extent_skips_a_step_with_no_positive_clean_peak():
    clean = [dict(step=0, layer=l, carrier_projection=-1.0) for l in range(10)]
    treated = [dict(step=0, layer=l, carrier_projection=5.0) for l in range(10)]
    assert LC.structure_extent(treated, clean, metric="carrier_projection") == []


# ============================== reporting that the first all-step dev run exposed
def test_a_delivery_verdict_from_a_handful_of_blocks_is_refused():
    """'Prevented, 0 of 1 scored blocks (of 23)' describes one block, not the interval."""
    layers = list(range(20, 30))
    clean_probe = _delivery_rows(layers, carrier_cos=0.9)
    for layer in layers[1:]:                  # register invisible at the consumer here
        clean_probe.rows[(2, layer)] = _delivery_rows([layer], carrier_cos=0.21,
                                                      ordinary_cos=0.2).rows[(2, layer)]
    clean = LC.delivered_rows(clean_probe, step=2, layers=layers, carriers=[3, 5])
    treated = LC.delivered_rows(_delivery_rows(layers, carrier_cos=0.15, carrier_norm=1.0),
                                step=2, layers=layers, carriers=[3, 5])
    verdict = LC.delivery_verdict(treated, clean, window=LC.Window("s", 20, 29))
    assert verdict["delivery"] == "not measured"
    assert verdict["n_blocks_scored"] == 1 and verdict["n_blocks"] == 10
    assert "would describe them, not the interval" in verdict["reason"]


def test_fixed_mask_regrowth_is_measured_by_the_rule_not_the_mask_size():
    """X reported 'rebuilding 24 tokens per block' -- which was just its mask."""
    x, v = _register_token()
    rule = LC.RegisterStateRule(highnorm_ratio=3.0, projection_percentile=90.0)
    fixed = LC.Suppressor(v, mode="fixed", tokens=[4, 7, 9], rule=rule)
    fixed.edit(x, _Ctx())
    site = fixed.sites[-1]
    assert site.n_selected == 3, "the mask size, which is what the operator edited"
    assert site.n_register_like == 1, "only token 4 actually carries the state"


def test_carriers_that_are_still_sinks_make_a_run_incomplete_even_if_delivery_is_unmeasured():
    sites = [LC.SuppressionSite(layer=l, step=2, mode="state", n_selected=0,
                                newly_targeted=[], max_projection_before=1.0,
                                n_highnorm_aligned_before=0.0, intervention_l2=0.0,
                                max_projection_after=0.0, n_register_like=0)
             for l in range(20, 24)]
    verdict = LC.suppression_verdict(
        sites, window=LC.Window("s", 20, 23), clean_carrier_projection=100.0,
        delivery=dict(delivery="not measured", reason="only 1 of 4 blocks"),
        lifecycle_rows=[dict(layer=21, carrier_is_sink=True, n_sinks_new=0)])
    assert verdict["suppression"] == "incomplete"
    assert "not carried by v* alone" in verdict["note"]
    unmeasured = LC.suppression_verdict(
        sites, window=LC.Window("s", 20, 23), clean_carrier_projection=100.0,
        delivery=dict(delivery="not measured", reason="only 1 of 4 blocks"))
    assert unmeasured["suppression"] == "not measured"
    assert "only 1 of 4 blocks" in unmeasured["note"]


def test_delivered_rows_report_shift_invariant_projection_statistics():
    probe = _delivery_rows([20], carriers=[3, 5], carrier_cos=0.9)
    row = LC.delivered_rows(probe, step=2, layers=[20], carriers=[3, 5])[0]
    assert row["register_vs_ordinary_mads"] > 5
    # Add one common vector to every token's projection: the cosine-free statistics move
    # by the same offset on both sides and the separation in MADs does not change.
    store = probe.rows[(2, 20)]
    store["consumed_projection"] = store["consumed_projection"] + 100.0
    shifted = LC.delivered_rows(probe, step=2, layers=[20], carriers=[3, 5])[0]
    assert shifted["register_vs_ordinary_mads"] == pytest.approx(
        row["register_vs_ordinary_mads"], rel=1e-5)


# ============================== natural-register-matched induction (the primary operator)
def _remainder(x, v):
    return x - (x @ v) * v


def test_matched_induction_sets_projection_and_norm_and_keeps_the_remainder_direction():
    x, v = _states()
    out = LC.induce_matched(x, [5, 9], alpha_target=7.5, norm_target=20.0, direction=v)
    for token in (5, 9):
        assert float(out[token] @ v) == pytest.approx(7.5, abs=1e-4)
        assert float(out[token].norm()) == pytest.approx(20.0, rel=1e-5)
        before, after = _remainder(x[token], v), _remainder(out[token], v)
        # The remainder is RESCALED, never rotated: the token keeps its own content.
        assert float(torch.nn.functional.cosine_similarity(before, after, dim=0)) \
            == pytest.approx(1.0, abs=1e-5)
        assert float(after.norm()) == pytest.approx((20.0 ** 2 - 7.5 ** 2) ** 0.5, rel=1e-5)
    others = [i for i in range(x.shape[0]) if i not in (5, 9)]
    assert torch.equal(x[others], out[others])


def test_matched_induction_takes_each_carriers_own_target():
    x, v = _states()
    out = LC.induce_matched(x, [5, 9], alpha_target={5: 3.0, 9: 6.0},
                            norm_target={5: 10.0, 9: 12.0}, direction=v)
    assert float(out[5] @ v) == pytest.approx(3.0, abs=1e-4)
    assert float(out[9] @ v) == pytest.approx(6.0, abs=1e-4)
    assert float(out[5].norm()) == pytest.approx(10.0, rel=1e-5)
    assert float(out[9].norm()) == pytest.approx(12.0, rel=1e-5)


def test_matched_induction_edits_each_batch_row_with_its_own_remainder():
    x, v = _states()
    y, _ = _states(seed=1)
    batch = torch.stack([y, x])               # [unconditional, conditional]
    out = LC.induce_matched(batch, [5], alpha_target=4.0, norm_target=9.0, direction=v)
    for row, source in ((0, y), (1, x)):
        assert float(out[row, 5] @ v) == pytest.approx(4.0, abs=1e-4)
        assert float(out[row, 5].norm()) == pytest.approx(9.0, rel=1e-5)
        assert float(torch.nn.functional.cosine_similarity(
            _remainder(out[row, 5], v), _remainder(source[5], v), dim=0)) \
            == pytest.approx(1.0, abs=1e-5), "one branch's content was copied into the other"


def test_a_token_with_no_remainder_is_left_alone_and_a_small_norm_target_is_clamped():
    x, v = _states()
    x[5] = 4.0 * v                             # nothing orthogonal to scale
    out = LC.induce_matched(x, [5, 9], alpha_target=6.0, norm_target=2.0, direction=v)
    assert torch.equal(out[5], x[5])
    # A norm target below the projection cannot be met with a real remainder: the token
    # becomes pure v* at the target projection rather than an imaginary length.
    assert float(out[9] @ v) == pytest.approx(6.0, abs=1e-4)
    assert float(out[9].norm()) == pytest.approx(6.0, rel=1e-4)


def test_the_matched_edit_size_is_computable_from_norm_and_projection_alone():
    x, v = _states()
    out = LC.induce_matched(x, [9], alpha_target=7.5, norm_target=20.0, direction=v)
    predicted = LC._matched_perturbation(float(x[9].norm()), float(x[9] @ v), 7.5, 20.0)
    assert predicted == pytest.approx(float((out[9] - x[9]).norm()), rel=1e-5)


def test_register_targets_prefer_the_exact_step_and_fall_back_only_when_declared():
    early = LC.RegisterTarget(alpha=1.0, norm=2.0, per_token={4: (3.0, 5.0)})
    reference = LC.RegisterTarget(alpha=10.0, norm=20.0)
    targets = LC.RegisterTargets()
    targets.add(2, 14, early)
    assert targets.at(2, 14) is early
    assert targets.at(3, 14) is None, "an uncalibrated step borrowed another step's size"
    targets.add(None, 14, reference)
    assert targets.at(3, 14) is reference and targets.at(2, 14) is early
    assert targets.at(2, 15) is None
    assert early.at(4) == (3.0, 5.0) and early.at(7) == (1.0, 2.0)
    assert targets.steps() == [2] and targets.layers() == [14] and len(targets) == 2


def test_the_matched_edit_writes_the_target_and_records_it():
    x, v = _states()
    targets = LC.RegisterTargets()
    targets.add(_Ctx.step, _Ctx.layer,
                LC.RegisterTarget(alpha=5.0, norm=12.0, per_token={9: (6.0, 15.0)}))
    log = []
    edit = LC.lifecycle_edit("induce_matched", [5, 9], v, register_targets=targets,
                             highnorm_threshold=13.0, records=log)
    out = edit(x, _Ctx())
    assert float(out[5] @ v) == pytest.approx(5.0, abs=1e-4)
    assert float(out[9].norm()) == pytest.approx(15.0, rel=1e-5)
    by_token = {r.token: r for r in log}
    assert set(by_token) == {5, 9} and all(r.kind == "induce_matched" for r in log)
    assert by_token[9].alpha_after == pytest.approx(6.0, abs=1e-4)
    assert by_token[9].norm_after == pytest.approx(15.0, rel=1e-5)
    # The high-norm flag is MEASURED against the bar, not asserted by the operator.
    assert by_token[9].crossed_highnorm_threshold is True
    assert by_token[5].crossed_highnorm_threshold is False
    assert by_token[5].perturbation_l2 == pytest.approx(float((out[5] - x[5]).norm()), rel=1e-5)


def test_the_matched_edit_leaves_an_uncalibrated_block_or_step_alone():
    x, v = _states()
    targets = LC.RegisterTargets()
    targets.add(_Ctx.step + 1, _Ctx.layer, LC.RegisterTarget(alpha=5.0, norm=12.0))
    log = []
    out = LC.lifecycle_edit("induce_matched", [5], v, register_targets=targets,
                            records=log)(x, _Ctx())
    assert torch.equal(out, x) and not log
    per_step = LC.lifecycle_edit("induce_matched", {_Ctx.step + 1: [5]}, v,
                                 register_targets=targets)
    assert torch.equal(per_step(x, _Ctx()), x), "a step with no positions was edited"
    with pytest.raises(ValueError, match="RegisterTargets"):
        LC.lifecycle_edit("induce_matched", [5], v)


class _CalibObs:
    def __init__(self, norm, projection):
        self.norm, self.projection = torch.as_tensor(norm), torch.as_tensor(projection)


def _calibration_trace():
    """Natural register at blocks 19-20's INPUTS (outputs 18, 19); recipients 14 and 41."""
    n = 32
    rows = {}
    for layer, (median, carrier_norms, carrier_projs) in {
            18: (2.0, (28.0, 24.0), (26.0, 22.0)),      # rho 14, 12; pi 13, 11
            19: (4.0, (56.0, 48.0), (52.0, 44.0))}.items():   # the same ratios, larger scale
        norm = torch.full((n,), median)
        projection = torch.full((n,), 0.1)
        norm[[3, 7]] = torch.tensor(carrier_norms)
        projection[[3, 7]] = torch.tensor(carrier_projs)
        rows[layer] = _CalibObs(norm, projection)
    quiet = torch.full((n,), 1.0)
    quiet[0] = 4.0                                   # the loudest token at block 14's input
    rows[13] = _CalibObs(quiet, torch.full((n,), 0.2))
    loud = torch.full((n,), 1.0)
    loud[0] = 40.0                                   # block 41 already holds a big token,
    loud_projection = torch.full((n,), 0.2)
    loud_projection[0] = 30.0                        # ...and it is along v*
    rows[40] = _CalibObs(loud, loud_projection)
    return _Trace(rows)


def test_register_match_carries_the_natural_ratios_to_each_recipient_scale():
    targets, rows = LC.calibrate_register_match(
        _calibration_trace(), step=2, natural_layers=[19, 20], recipient_layers=[14, 41],
        carriers=[3, 7])
    assert set(targets) == {14, 41}
    # Recipient 14's input median is 1.0, so the per-token targets ARE the natural ratios.
    assert targets[14].at(3) == pytest.approx((13.0, 14.0))
    assert targets[14].at(7) == pytest.approx((11.0, 12.0))
    assert (targets[14].alpha, targets[14].norm) == pytest.approx((12.0, 13.0))
    # A non-carrier position gets the population mean, not zero.
    assert targets[14].at(11) == pytest.approx((12.0, 13.0))
    row = {r["layer"]: r for r in rows}
    assert row[14]["natural_norm_ratio"] == pytest.approx(13.0)
    assert row[14]["natural_projection_ratio"] == pytest.approx(12.0)
    assert row[14]["natural_blocks"] == "19-20"


def test_register_match_says_when_the_target_is_unusually_large_for_the_recipient():
    _, rows = LC.calibrate_register_match(
        _calibration_trace(), step=2, natural_layers=[19, 20], recipient_layers=[14, 41],
        carriers=[3, 7])
    row = {r["layer"]: r for r in rows}
    # Block 14's loudest clean token is 4x its median; a 13x target is beyond it.
    assert row[14]["unusually_large"] is True
    assert row[14]["norm_vs_recipient_max"] == pytest.approx(13.0 / 4.0)
    assert row[14]["target_norm_percentile"] == pytest.approx(100.0)
    assert row[14]["norm_unusually_large"] and row[14]["projection_unusually_large"]
    # Block 41 already holds a 40x token: the same relative target is inside its range.
    assert row[41]["unusually_large"] is False
    # The size of the edit on the clean tensor, in the recipient's own units.
    assert row[14]["perturbation_vs_median"] > 10.0
    # Setting the projection alone already brings most of the norm when the natural
    # register is this aligned -- which is what the twins cannot separate.
    assert row[14]["projection_only_norm_ratio"] == pytest.approx(
        ((13.0 ** 2 + (1.0 - 0.04)) ** 0.5 + (11.0 ** 2 + (1.0 - 0.04)) ** 0.5) / 2, rel=1e-4)


def test_register_match_measures_what_it_does_not_match_the_remainder_direction():
    """Against an ordinary-token baseline: every token shares some common direction, and a
    raw cosine would credit that to the register."""
    torch.manual_seed(0)
    n, c = 32, 16
    v = LC._unit(torch.randn(c))
    shared = LC._unit(_remainder(torch.randn(c), v))
    content = torch.randn(n, c)               # each position's own content, both depths
    natural = content.clone()
    for t in (3, 7):                          # natural registers share one extra remainder
        natural[t] = 13.0 * v + 6.0 * shared + 0.3 * content[t]
    recipient = content.clone()               # the recipient still holds only the content
    kwargs = dict(step=2, natural_layers=[19, 20], recipient_layers=[14], carriers=[3, 7],
                  natural_states=natural, direction=v)
    _, rows = LC.calibrate_register_match(_calibration_trace(),
                                          recipient_states={14: recipient}, **kwargs)
    row = rows[0]
    # The carriers' remainders point one way far more than ordinary tokens' do...
    assert row["natural_remainder_coherence"] > 0.9
    assert row["ordinary_remainder_coherence"] < 0.5
    # ...and the matched carriers' own remainders are much further from it than an
    # ordinary token's remainder is from itself at the other depth.
    assert row["ordinary_remainder_baseline"] > 0.99
    assert row["remainder_vs_natural_cosine"] < row["ordinary_remainder_baseline"] - 0.3

    # With no register-specific remainder the two readings agree.
    plain = content.clone()
    for t in (3, 7):
        plain[t] = 13.0 * v + content[t]
    _, rows = LC.calibrate_register_match(
        _calibration_trace(), recipient_states={14: recipient},
        **{**kwargs, "natural_states": plain})
    assert rows[0]["remainder_vs_natural_cosine"] == pytest.approx(
        rows[0]["ordinary_remainder_baseline"], abs=0.05)


def test_register_match_reads_a_probed_input_where_the_trace_records_none():
    torch.manual_seed(0)
    v = LC._unit(torch.randn(16))
    states = torch.randn(32, 16) * 0.5        # block 0's input: the trace never has it
    targets, _ = LC.calibrate_register_match(
        _calibration_trace(), step=2, natural_layers=[19, 20], recipient_layers=[0],
        carriers=[3, 7], recipient_states={0: states}, direction=v)
    median = float(states.norm(dim=-1).median())
    assert targets[0].norm == pytest.approx(13.0 * median, rel=1e-5)
    assert targets[0].at(3) == pytest.approx((13.0 * median, 14.0 * median), rel=1e-5)


def test_register_match_refuses_without_a_natural_register_to_copy():
    with pytest.raises(ValueError, match="no clean statistics"):
        LC.calibrate_register_match(_calibration_trace(), step=2, natural_layers=[30],
                                    recipient_layers=[14], carriers=[3])
    with pytest.raises(ValueError, match="at least one natural carrier"):
        LC.calibrate_register_match(_calibration_trace(), step=2, natural_layers=[19],
                                    recipient_layers=[14], carriers=[999])


def _maintenance_rows(values, *, cosine=0.9, ratio=14.0, sink=12.0, first=14):
    return [dict(layer=first + i, carrier_projection=p,
                 carrier_cosine=cosine if p > 1 else 0.1,
                 carrier_norm_ratio=ratio if p > 1 else 1.0,
                 carrier_sink_strength=sink if p > 1 else 1.0,
                 carrier_is_sink=bool(p > 1 and sink >= 10.0))
            for i, p in enumerate(values)]


def test_maintenance_reads_alignment_norm_sinks_and_persistence_against_nature():
    # Window 14-16, then two blocks where it holds and one where it is gone.
    rows = _maintenance_rows([10.0, 10.0, 10.0, 8.0, 6.0, 1.0, 0.5])
    clean = _maintenance_rows([0.5] * 7)
    natural = _maintenance_rows([10.0] * 4, cosine=0.95, ratio=15.0, sink=20.0, first=20)
    summary = LC.maintenance_summary(rows, clean + natural, window=LC.Window("early", 14, 16),
                                     reference_layers=[20, 21, 22, 23])
    assert summary["cosine"] == pytest.approx(0.9)
    assert summary["norm_ratio"] == pytest.approx(14.0)
    assert summary["sink_block_share"] == pytest.approx(1.0)
    assert summary["natural_norm_ratio"] == pytest.approx(15.0)
    assert summary["norm_ratio_vs_natural"] == pytest.approx(14.0 / 15.0)
    assert summary["sink_strength_vs_natural"] == pytest.approx(12.0 / 20.0)
    # Excess 9.5 at the window end; 7.5 and 5.5 keep >= half of it, 0.5 does not.
    assert summary["excess_projection_at_window_end"] == pytest.approx(9.5)
    assert summary["projection_persists_blocks"] == 2
    assert summary["blocks_after_window_measured"] == 4


def test_maintenance_reports_zero_persistence_when_nothing_follows_the_window():
    rows = _maintenance_rows([10.0, 10.0, 10.0], first=40)
    clean = _maintenance_rows([0.5] * 3, first=40)
    summary = LC.maintenance_summary(rows, clean, window=LC.Window("late", 40, 42),
                                     reference_layers=[40])
    assert summary["projection_persists_blocks"] == 0
    assert summary["blocks_after_window_measured"] == 0


def test_the_maintenance_profile_separates_what_was_written_from_what_was_handed_on():
    rows = _maintenance_rows([4.0, 3.0, 2.0, 0.5])
    clean = _maintenance_rows([0.5] * 4)
    records = [LC.EditRecord(layer=l, step=2, kind="induce_matched", token=3,
                             alpha_before=0.5, alpha_after=8.0, norm_before=1.0,
                             norm_after=10.0, cosine_after=0.8, perturbation_l2=9.0,
                             crossed_highnorm_threshold=True) for l in (14, 15, 16)]
    records.append(LC.EditRecord(layer=14, step=5, kind="induce_matched", token=3,
                                 alpha_before=0.5, alpha_after=99.0, norm_before=1.0,
                                 norm_after=99.0, cosine_after=1.0, perturbation_l2=99.0,
                                 crossed_highnorm_threshold=True))
    profile = LC.maintenance_profile(rows, clean, window=LC.Window("early", 14, 16),
                                     records=records, step=2, after=1)
    by_layer = {r["layer"]: r for r in profile}
    assert set(by_layer) == {14, 15, 16, 17}
    assert by_layer[14]["written_alpha"] == pytest.approx(8.0), "another step leaked in"
    assert by_layer[14]["handed_on_fraction"] == pytest.approx(0.5)
    assert by_layer[17]["in_window"] is False
    assert by_layer[17]["written_alpha"] != by_layer[17]["written_alpha"]    # NaN
    assert by_layer[16]["excess_projection"] == pytest.approx(1.5)
    assert by_layer[16]["clean_carrier_projection"] == pytest.approx(0.5)


def test_the_primary_conditions_are_matched_and_each_has_a_direction_only_twin():
    for condition in LC.CONDITIONS:
        if condition.induces():
            assert condition.induction == "matched", condition.key
    twin = LC.direction_only_twin(LC.CONDITIONS[1])
    assert twin.key == "B_early_natural_intact__dironly"
    assert twin.induction == "direction_only" and twin.role == "control"
    assert twin.schedule() == LC.CONDITIONS[1].schedule()
    # The weak arm is a multiple from the direction-only sweep, so it is direction-only.
    assert LC.WEAK_EARLY.induction == "direction_only"


def test_the_achieved_lifecycle_reports_relative_norm_and_alignment():
    trace = _trace({0: 5.0})
    trace.rows[0].cosine = torch.full((16,), 0.2)
    trace.rows[0].cosine[3] = 0.95
    row = LC.achieved_lifecycle(trace, step=2, layers=[0], carriers=[3],
                                sink_threshold=3.0)[0]
    assert row["carrier_norm_ratio"] == pytest.approx(6.0)     # 6.0 against a median of 1
    assert row["carrier_cosine"] == pytest.approx(0.95)


# ================= Q1/Q4-informed revision: suppression judged by what blocks RECEIVE
class _StepObs:
    def __init__(self, norm, projection, incoming=None):
        self.norm, self.projection = norm.float(), projection.float()
        self.cosine = self.projection / self.norm.clamp_min(1e-9)
        self.incoming = incoming
        self.qk_cosine = self.top_channel = self.top_channel_value = None
        self.image_mass = self.n_keys = None


class _StepTrace:
    def __init__(self):
        self.rows = {}

    def at(self, step, layer):
        return self.rows.get((int(step), int(layer)))


def _scene(n=400, carriers=(3, 5), carrier_cos=0.95, carrier_norm=20.0, seed=0,
           extra=None, heads=4, sinks=None):
    """Ordinary tokens: norm ~1, cos ~N(0, 0.05). Carriers: large and aligned."""
    torch.manual_seed(seed)
    norm = 1.0 + 0.05 * torch.rand(n)
    cos = 0.05 * torch.randn(n)
    for t in carriers:
        norm[t], cos[t] = carrier_norm, carrier_cos
    for t, (c, nn_) in (extra or {}).items():
        cos[t], norm[t] = c, nn_
    incoming = torch.full((heads, n), 0.5 / n)
    default = [carriers[0] if carriers else 0] * heads
    for h, t in enumerate(sinks if sinks is not None else default):
        incoming[h, t] = 0.5
    incoming = incoming / incoming.sum(-1, keepdim=True)
    return _StepObs(norm, cos * norm, incoming)


def _clean_reference(layers=(19, 20, 21), steps=(0, 1), carriers=(3, 5)):
    trace = _StepTrace()
    for s in steps:
        for l in layers:
            trace.rows[(s, l)] = _scene(carriers=carriers, seed=l + 10 * s)
    ref = LC.CleanStateReference(highnorm_ratio=3.0, alignment_quantile=0.999,
                                 frozen=carriers)
    ref.add(trace, steps=steps, layers=layers)
    return trace, ref


def test_the_clean_reference_reads_q1_bars_per_step_and_block():
    trace, ref = _clean_reference()
    bars = ref.at(1, 20)
    assert 0.05 < bars.alignment_bar < 0.95, "the bar sits above ordinary tokens, below carriers"
    assert set(bars.carriers) == {3, 5}, "the clean carriers meet the Q1 register criterion"
    assert {3, 5} <= set(bars.above_bar)
    assert bars.head_sinks == (3, 3, 3, 3) and bars.concentration > 0.4
    # A block's INPUT is the previous block's output.
    assert ref.for_input_of(1, 21) is ref.at(1, 20)
    assert ref.for_input_of(1, 0) is None
    # An unrecorded step falls back to the nearest one, and says so.
    assert ref.at(7, 20).step == 1


def test_the_alignment_rule_checks_every_position_without_a_norm_bar():
    trace, ref = _clean_reference()
    rule = LC.AlignmentRule(ref)
    treated = _scene(carriers=(3, 5), extra={42: (0.9, 1.0)}, seed=99)   # small, aligned
    chosen = rule.select_from_stats(treated.norm, treated.projection, step=1, layer=21)
    assert {3, 5, 42} <= set(chosen), "a new, still-small aligned token must be caught"
    assert len(chosen) <= 3 + int(0.01 * 400), "ordinary tokens are not erased wholesale"
    # The old conjunction misses the small one -- exactly the escape route Q1 warns about.
    old = LC.RegisterStateRule(highnorm_ratio=3.0, projection_percentile=99.0)
    assert 42 not in old.select_from_stats(treated.norm, treated.projection)


def _states_from(obs, v, seed=0):
    """Full [N, C] states with the given norms and v* projections."""
    torch.manual_seed(seed)
    n, c = int(obs.norm.shape[0]), v.numel()
    r = torch.randn(n, c)
    r = r - (r @ v)[:, None] * v[None, :]
    r = r / r.norm(dim=-1, keepdim=True)
    perp = (obs.norm ** 2 - obs.projection ** 2).clamp_min(0).sqrt()
    return obs.projection[:, None] * v[None, :] + perp[:, None] * r


def test_the_suppressor_removes_every_aligned_token_and_records_what_the_block_received():
    trace, ref = _clean_reference()
    torch.manual_seed(1)
    v = LC._unit(torch.randn(24))
    obs = _scene(carriers=(3, 5), extra={42: (0.9, 1.0)}, seed=99)
    x = _states_from(obs, v)
    sites = []
    ctx = type("Ctx", (), dict(step=1, layer=21))()
    out = LC.Suppressor(v, mode="state", rule=LC.AlignmentRule(ref), sites=sites,
                        original=(3, 5)).edit(x, ctx)
    site = sites[0]
    assert site.operator == "subtractive"
    assert site.n_original == 2 and site.n_new >= 1 and 42 in site.new_ids
    assert site.received_n_above_bar == 0, "nothing over the bar reaches the block"
    assert site.received_n_register_like == 0
    assert site.received_bar == pytest.approx(ref.at(1, 20).alignment_bar)
    assert abs(site.received_carrier_cosine) < 1e-4
    # Subtractive: the edited carriers are NOT rescaled back to register size.
    assert float(out[3].norm()) < 0.5 * float(x[3].norm())


def test_the_received_state_uses_the_hook_where_there_is_one_and_the_stream_where_not():
    trace, ref = _clean_reference(layers=(18, 19, 20, 21))
    treated = _StepTrace()
    for l in (18, 19, 20, 21):
        treated.rows[(1, l)] = _scene(carriers=(3, 5), seed=200 + l)
    sites = [LC.SuppressionSite(layer=l, step=1, mode="state", n_selected=2,
                                newly_targeted=[], max_projection_before=0.0,
                                n_highnorm_aligned_before=0.0, intervention_l2=0.0,
                                max_projection_after=0.0, chosen_ids=(3, 5),
                                received_bar=ref.at(1, l - 1).alignment_bar,
                                received_max_cosine=0.01, received_n_above_bar=0,
                                received_n_register_like=0, received_carrier_cosine=0.0)
             for l in (19, 20)]
    rows = LC.received_state_rows(treated, sites, ref, step=1, layers=[19, 20, 21],
                                  carriers=(3, 5))
    by = {r["layer"]: r for r in rows}
    assert by[19]["hooked"] and by[19]["received_n_register_like"] == 0
    assert by[19]["register_like_before_hook"] >= 2, "the previous block handed it on"
    # Block 21 is the unhooked block after a terminal cleanup at 20: it receives block
    # 20's output as it is, and here that output still carries the state.
    assert not by[21]["hooked"] and by[21]["received_n_register_like"] >= 2
    assert by[21]["received_max_cosine_vs_natural"] == pytest.approx(1.0, abs=0.05)
    summary = LC.received_state_summary(rows, interval=LC.Window("s", 19, 19), terminal=20)
    assert summary["result"] == "v* state received after the interval"
    assert summary["blocks_received"] == [] and summary["boundary_received"] >= 2
    clean_only = LC.received_state_summary(rows[:2], interval=LC.Window("s", 19, 19),
                                           terminal=20)
    assert clean_only["result"] == "no v*-aligned state received"


def test_regrowth_and_relocation_are_counted_separately_and_against_paired_clean():
    trace, ref = _clean_reference(layers=(20, 21))
    treated = _StepTrace()
    # Paired with clean: the same ordinary tokens (same seed as the clean scene at that
    # step and block). Block 20: an original carrier regrows, the other stays removed.
    # Block 21: the originals are gone and a NEW token has become a carrier.
    treated.rows[(1, 20)] = _scene(carriers=(3,), seed=20 + 10)
    treated.rows[(1, 21)] = _scene(carriers=(), extra={77: (0.95, 20.0)}, seed=21 + 10)
    rows = LC.regrowth_relocation_rows(treated, ref, step=1, layers=[20, 21],
                                       carriers=(3, 5))
    by = {r["layer"]: r for r in rows}
    assert by[20]["original_register_criterion"] == 1 and by[20]["new_register_criterion"] == 0
    assert by[21]["original_register_criterion"] == 0 and 77 in by[21]["new_criterion_ids"]
    summary = LC.regrowth_relocation_summary(rows, window=LC.Window("s", 20, 21))
    assert summary["outcome"] == "original carriers regain the register criterion"
    assert summary["unique_new_register_positions"] == 1
    # The clean run's own ordinary tail -- over the removal bar by construction -- is not
    # counted as relocation.
    assert set(by[21]["new_aligned_ids"]) == {77}


def test_attention_relocation_separates_persistence_from_where_attention_went():
    trace, ref = _clean_reference(layers=(20, 21))
    treated = _StepTrace()
    treated.rows[(1, 20)] = _scene(carriers=(3, 5), extra={77: (0.95, 1.0)}, seed=20 + 10)
    # Head 0 keeps the clean sink, head 1 moves to another original carrier, head 2 to the
    # token that carried NEW v* into block 21, head 3 to an ordinary token.
    treated.rows[(1, 21)] = _scene(carriers=(3, 5), seed=21 + 10, sinks=[3, 5, 77, 200])
    rows = LC.attention_relocation_rows(treated, ref, step=1, layers=[21], carriers=(3, 5))
    row = rows[0]
    assert row["n_affected_heads"] == 4
    assert row["affected_kept"] == pytest.approx(0.25)
    assert row["affected_other_original_carrier"] == pytest.approx(0.25)
    assert row["affected_relocated_vstar_carrier"] == pytest.approx(0.25)
    assert row["affected_non_register_token"] == pytest.approx(0.25)
    summary = LC.attention_relocation_summary(rows, window=LC.Window("s", 21, 21))
    assert summary["affected_heads_kept"] == pytest.approx(0.25)


def test_a_head_on_the_clean_register_outside_the_carrier_set_is_not_called_non_register():
    # The clean run's register at this input includes token 9, which the carrier set does
    # not name -- as happens at a step other than the one the set was read at.
    trace = _StepTrace()
    for layer in (20, 21):
        trace.rows[(1, layer)] = _scene(carriers=(3, 5, 9), seed=layer + 10)
    ref = LC.CleanStateReference(highnorm_ratio=3.0, alignment_quantile=0.999,
                                 frozen=(3, 5))
    ref.add(trace, steps=[1], layers=(20, 21))
    assert 9 in ref.at(1, 20).register_like
    treated = _StepTrace()
    treated.rows[(1, 20)] = _scene(carriers=(3, 5, 9), seed=30)
    treated.rows[(1, 21)] = _scene(carriers=(3, 5, 9), seed=31, sinks=[3, 9, 9, 200])
    rows = LC.attention_relocation_rows(treated, ref, step=1, layers=[21], carriers=(3, 5))
    row = rows[0]
    assert row["affected_kept"] == pytest.approx(0.25)
    assert row["affected_clean_register_elsewhere"] == pytest.approx(0.5)
    assert row["affected_non_register_token"] == pytest.approx(0.25)


def test_attention_that_spreads_out_is_not_called_a_new_sink():
    trace, ref = _clean_reference(layers=(20, 21))
    treated = _StepTrace()
    flat = _scene(carriers=(3, 5), seed=21 + 10)
    flat.incoming = torch.full_like(flat.incoming, 1.0 / flat.incoming.shape[-1])
    flat.incoming[:, 9] += 1e-4        # a nominal argmax, but nothing concentrated
    treated.rows[(1, 21)] = flat
    treated.rows[(1, 20)] = _scene(carriers=(3, 5), seed=20 + 10)
    row = LC.attention_relocation_rows(treated, ref, step=1, layers=[21], carriers=(3, 5))[0]
    assert row["affected_spread_out"] == pytest.approx(1.0)
    assert row["concentration_vs_clean"] < 0.1


def test_the_clean_runs_own_ordinary_tail_is_never_reported_as_a_v_star_state():
    """The removal bar is a quantile, so ~0.1% of ordinary tokens sit above it in clean by
    construction. Results are judged against every ordinary token instead, so the clean
    trajectory itself shows only its registers and no relocation."""
    trace, ref = _clean_reference(layers=(19, 20, 21))
    assert len(ref.at(1, 20).above_bar) > len(ref.at(1, 20).register_like) == 2
    rows = LC.received_state_rows(trace, [], ref, step=1, layers=[20, 21], carriers=(3, 5))
    assert all(r["received_n_register_like"] == 2 for r in rows), "only the carriers"
    moved = LC.regrowth_relocation_rows(trace, ref, step=1, layers=[20, 21], carriers=(3, 5))
    assert all(r["new_aligned"] == 0 and r["original_register_criterion"] == 2 for r in moved)


def test_induction_is_established_only_if_every_hook_wrote_the_target():
    good = [LC.EditRecord(layer=14, step=s, kind="induce_matched", token=3, alpha_before=0.0,
                          alpha_after=10.0, norm_before=1.0, norm_after=11.0,
                          cosine_after=0.9, perturbation_l2=10.0,
                          crossed_highnorm_threshold=True) for s in (0, 1)]
    bad = LC.EditRecord(layer=15, step=1, kind="induce_matched", token=3, alpha_before=0.0,
                        alpha_after=5.0, norm_before=1.0, norm_after=11.0, cosine_after=0.5,
                        perturbation_l2=5.0, crossed_highnorm_threshold=True)
    rows = LC.induction_written(good + [bad], lambda r: (10.0, 11.0),
                                window=LC.Window("early", 14, 16))
    by = {r["step"]: r for r in rows}
    assert by[0]["established"] and not by[1]["established"]
    assert by[1]["worst_error"] == pytest.approx(0.5)
    # The direction-only arm controls the projection only.
    rows = LC.induction_written([bad], lambda r: (5.0, None), window=LC.Window("e", 14, 16))
    assert rows[0]["established"]


def test_formation_onset_is_measured_from_the_clean_output():
    rows = [dict(layer=l, carrier_projection=v) for l, v in
            ((15, 0.1), (16, 0.2), (17, 0.4), (18, 9.0), (19, 10.0), (20, 10.0))]
    out = LC.measured_formation_onset(rows, natural=LC.Window("natural", 20, 39))
    assert out["onset"] == 18


def test_the_schedule_follows_the_measured_flux_dev_boundaries():
    s = LC.lifecycle_schedule(n_layers=57, formation_onset=18, natural_end=39,
                              dissolution_onset=30, early=(14, 16), window_length=3,
                              bridge_lead=2, extension_blocks=3)
    assert (s.early.first, s.early.last) == (14, 16)
    assert (s.suppression.first, s.suppression.last, s.terminal) == (17, 39, 40)
    assert (s.bridge.first, s.bridge.last) == (28, 39), "maintenance starts BEFORE the decline"
    assert (s.extension.first, s.extension.last) == (40, 42), "into genuinely later blocks"
    assert (s.late_only.first, s.late_only.last) == (41, 43), "after the terminal cleanup"
    hooks = list(s.suppression.layers) + [s.terminal]
    assert LC.schedule_conflicts(s.late_only.layers, hooks) == []
    assert LC.schedule_conflicts(s.early.layers, hooks) == []
    # C's extension and D's replacement are different windows, not one forced slot.
    assert s.extension.layers != s.late_only.layers
    with pytest.raises(ValueError, match="must end before"):
        LC.lifecycle_schedule(n_layers=57, formation_onset=15, natural_end=39,
                              early=(14, 16))


def test_a_transplant_moves_the_direction_and_keeps_the_recipients_norm():
    torch.manual_seed(0)
    x = torch.randn(10, 16)
    source = torch.randn(16) * 50.0
    log = []
    out = LC.transplant_edit([4], {4: source}, torch.randn(16), records=log)(x, _Ctx())
    assert float(torch.nn.functional.cosine_similarity(out[4], source, dim=0)) \
        == pytest.approx(1.0, abs=1e-5)
    assert float(out[4].norm()) == pytest.approx(float(x[4].norm()), rel=1e-5)
    assert torch.equal(out[[0, 1, 2, 3, 5]], x[[0, 1, 2, 3, 5]])
    assert log and log[0].kind == "transplant"
    per_step = LC.transplant_edit([4], {_Ctx.step + 1: {4: source}}, torch.randn(16))
    assert torch.equal(per_step(x, _Ctx()), x), "a step without a source is left alone"


def test_an_align_edit_sets_the_cosine_and_keeps_norm_and_remainder():
    x, v = _states()
    out = LC.align_edit([5], 0.8, v)(x, _Ctx())
    assert float(out[5] @ v) / float(out[5].norm()) == pytest.approx(0.8, abs=1e-5)
    assert float(out[5].norm()) == pytest.approx(float(x[5].norm()), rel=1e-5)
    assert float(torch.nn.functional.cosine_similarity(
        _remainder(out[5], v), _remainder(x[5], v), dim=0)) == pytest.approx(1.0, abs=1e-5)


def test_norm_is_also_reported_against_the_ordinary_tokens_of_the_block():
    trace = _trace({0: 5.0})
    trace.rows[0].norm[7] = 50.0                      # another large, non-carrier token
    row = LC.achieved_lifecycle(trace, step=2, layers=[0], carriers=[3],
                                sink_threshold=3.0)[0]
    assert row["ordinary_median_norm"] == pytest.approx(1.0)
    assert row["carrier_norm_vs_ordinary"] == pytest.approx(6.0)


def test_register_match_reports_how_far_outside_the_recipient_distribution():
    _, rows = LC.calibrate_register_match(
        _calibration_trace(), step=2, natural_layers=[19, 20], recipient_layers=[14],
        carriers=[3, 7])
    assert rows[0]["norm_robust_z"] > 10.0


# ------------------------- the natural interval by the register test, and the range check
def _lifecycle_trace(n_layers=10, formed=(3, 6), carriers=(3, 5, 7), step=0, grow=None,
                     dissolved=(2.5, 0.8)):
    """A stack whose register exists at blocks ``formed`` (inclusive) and nowhere else.

    After the register has gone, its former carriers are below the high-norm bar (2.5x
    the median against a 3x bar) but still partly aligned. With ``grow`` scaling every
    token's norm with depth, as the residual stream does, their projection then never
    falls below 10% of its peak -- the FLUX.1-dev situation in which the old rule never
    fires.
    """
    trace = _StepTrace()
    for layer in range(n_layers):
        scale = 1.0 if grow is None else float(grow(layer))
        if formed[0] <= layer <= formed[1]:
            obs = _scene(carriers=carriers, seed=layer, sinks=[carriers[0]] * 4)
        else:
            obs = _scene(carriers=(), seed=layer)
            if layer > formed[1]:
                for t in carriers:                 # dissolved: below the bar, still aligned
                    obs.norm[t] = dissolved[0]
                    obs.projection[t] = dissolved[0] * dissolved[1]
                obs.cosine = obs.projection / obs.norm.clamp_min(1e-9)
        obs.norm = obs.norm * scale
        obs.projection = obs.projection * scale
        obs.cosine = obs.projection / obs.norm.clamp_min(1e-9)
        trace.rows[(step, layer)] = obs
    return trace


def test_formation_and_natural_end_are_read_off_the_register_test():
    trace = _lifecycle_trace()
    ref = LC.CleanStateReference(highnorm_ratio=3.0, alignment_quantile=0.999,
                                 frozen=(3, 5, 7))
    ref.add(trace, steps=[0], layers=range(10))
    population = LC.register_test_counts(ref, step=0, layers=range(10))
    assert population[2] == 0 and population[4] == 3 and population[8] == 0
    formation = LC.measured_formation_by_count(population)
    assert formation["onset"] == 3 and formation["peak_count"] == 3
    end = LC.measured_natural_end(
        LC.register_test_counts(ref, step=0, layers=range(10), carriers=(3, 5, 7)),
        after=formation["peak_layer"])
    assert end["natural_end"] == 6 and end["first_empty_block"] == 7
    # A step that was not recorded is not filled in from another step.
    assert LC.register_test_counts(ref, step=5, layers=range(10)) == {}


def test_the_register_test_ends_the_interval_where_the_old_ten_percent_rule_never_fires():
    # Norms grow with depth; the former carriers keep cos 0.5, so their projection stays
    # far above 10% of the peak long after they stopped being registers.
    trace = _lifecycle_trace(n_layers=12, formed=(3, 5), grow=lambda l: 1.0 + 0.5 * l)
    ref = LC.CleanStateReference(highnorm_ratio=3.0, alignment_quantile=0.999,
                                 frozen=(3, 5, 7))
    ref.add(trace, steps=[0], layers=range(12))
    rows = LC.achieved_lifecycle(trace, step=0, layers=list(range(12)), carriers=[3, 5, 7],
                                 sink_threshold=10.0, highnorm_ratio=3.0)
    old = LC.measured_dissolution_onset(rows, natural=LC.Window("natural", 3, 11))
    assert old["disappears_at"] is None, "the projection rule never fires here"
    end = LC.measured_natural_end(
        LC.register_test_counts(ref, step=0, layers=range(12), carriers=(3, 5, 7)), after=4)
    assert end["natural_end"] == 5


def test_a_register_that_never_ends_is_reported_not_guessed():
    trace = _lifecycle_trace(n_layers=6, formed=(2, 5))
    ref = LC.CleanStateReference(frozen=(3, 5, 7))
    ref.add(trace, steps=[0], layers=range(6))
    end = LC.measured_natural_end(
        LC.register_test_counts(ref, step=0, layers=range(6), carriers=(3, 5, 7)), after=3)
    assert end["natural_end"] is None and "never ends" in end["note"]
    empty = LC.measured_formation_by_count({0: 0, 1: 0})
    assert empty["onset"] is None


def test_the_range_check_moves_one_threshold_at_a_time_on_the_clean_run():
    trace = _lifecycle_trace()
    rows = LC.threshold_sensitivity(trace, step=0, layers=range(10), carriers=(3, 5, 7))
    table = {(r["parameter"], r["value"], r["statistic"]): r for r in rows}
    assert {r["parameter"] for r in rows} == {"highnorm_ratio", "alignment_quantile",
                                              "sink_threshold"}
    chosen = [r for r in rows if r["chosen"]]
    assert {(r["parameter"], r["value"]) for r in chosen} == {
        ("highnorm_ratio", 3.0), ("alignment_quantile", 0.999), ("sink_threshold", 10.0)}
    # At the chosen ratio the planted carriers are exactly the carrier set, and the
    # interval is the planted one; ratios on either side give the same answer (a plateau).
    for ratio in (2.5, 3.0, 5.0, 8.0):
        assert table[("highnorm_ratio", ratio, "carriers_at_peak")]["result"] == 3
        assert table[("highnorm_ratio", ratio, "overlap_with_chosen_carriers")]["result"] == 1
        assert table[("highnorm_ratio", ratio, "formation_block")]["result"] == 3
        assert table[("highnorm_ratio", ratio, "natural_end_block")]["result"] == 6
    # Below the plateau the ratio starts counting the dissolved carriers (2.5x the median)
    # as registers, and the natural interval no longer ends -- the edge of the plateau.
    assert math.isnan(table[("highnorm_ratio", 2.0, "natural_end_block")]["result"])
    # The removal rule still catches every carrier at the chosen quantile, and touches
    # only a sliver of the image.
    assert table[("alignment_quantile", 0.999, "min_share_of_carriers_caught")]["result"] == 1
    assert table[("alignment_quantile", 0.999, "max_share_of_image_stripped")]["result"] < 0.02
    # Every sink in the natural interval is a carrier at the chosen sink threshold.
    assert table[("sink_threshold", 10.0, "share_of_sinks_that_are_carriers")]["result"] == 1
    assert all(r["interval_first"] == 3 and r["interval_last"] == 6 for r in rows)
