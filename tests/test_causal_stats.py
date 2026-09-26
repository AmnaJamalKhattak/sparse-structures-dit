"""Uncertainty behaves the way the design promises: paired, clustered, and reported without inflation."""
import numpy as np
import pandas as pd
import pytest

from ditsinks import causal_stats as CS


def _clustered_frame(prompt_effect=0.4, treatment_effect=-0.5, prompts=10, seeds=3, noise=0.02,
                     seed=0):
    """Prompts differ from each other; the treatment shifts every unit the same way."""
    rng = np.random.default_rng(seed)
    rows = []
    for prompt in range(prompts):
        offset = rng.normal(0, prompt_effect)
        for s in range(seeds):
            base = 0.9 + offset + rng.normal(0, noise)
            rows.append(dict(prompt_id=prompt, seed=s, condition="sham", value=base))
            rows.append(dict(prompt_id=prompt, seed=s, condition="treated",
                             value=base + treatment_effect + rng.normal(0, noise)))
    return pd.DataFrame(rows)


def test_pairing_removes_prompt_to_prompt_variance():
    """The whole point of pairing: the effect must not inherit prompt spread."""
    frame = _clustered_frame(prompt_effect=0.5)
    paired = CS.paired_effect(frame, "value", treatment="treated", reference="sham")
    unpaired = CS.bootstrap_mean(frame[frame.condition == "treated"], "value")
    assert paired.value == pytest.approx(-0.5, abs=0.05)
    paired_width = paired.ci_high - paired.ci_low
    unpaired_width = unpaired.ci_high - unpaired.ci_low
    assert paired_width < unpaired_width / 5


def test_clustering_widens_the_interval_when_seeds_are_correlated():
    """Treating three seeds of one prompt as independent understates the interval."""
    frame = _clustered_frame(prompt_effect=0.5)
    sham = frame[frame.condition == "sham"]
    clustered = CS.bootstrap_mean(sham, "value", cluster="prompt_id")
    # Pretending every row is its own cluster is the naive analysis.
    naive = sham.copy()
    naive["fake_cluster"] = np.arange(len(naive))
    independent = CS.bootstrap_mean(naive, "value", unit=("prompt_id", "seed"),
                                    cluster="fake_cluster")
    assert (clustered.ci_high - clustered.ci_low) > (independent.ci_high - independent.ci_low)


def test_units_are_collapsed_before_resampling():
    """Rows arrive per layer and per head; one prompt-seed pair must count once."""
    frame = _clustered_frame(prompts=4, seeds=2)
    expanded = pd.concat([frame.assign(layer=layer) for layer in range(5)], ignore_index=True)
    assert CS.bootstrap_mean(frame, "value").n_units == 8
    assert CS.bootstrap_mean(expanded, "value").n_units == 8, \
        "repeating a unit across layers must not multiply its weight"
    assert CS.collapse_units(frame[frame.condition == "sham"], "value").shape[0] == 8


def test_too_few_clusters_yields_no_interval():
    """Two prompts cannot support a percentile bootstrap, so none is reported."""
    frame = _clustered_frame(prompts=2)
    estimate = CS.paired_effect(frame, "value", treatment="treated", reference="sham")
    assert np.isfinite(estimate.value)
    assert not estimate.has_interval
    assert not estimate.excludes_zero, "a missing interval must never certify an effect"


def test_equivalence_needs_a_tight_interval_not_merely_a_small_effect():
    frame = _clustered_frame(treatment_effect=0.0, noise=0.005)
    differences = CS.paired_differences(frame, "value", treatment="treated", reference="sham")
    tight = CS.equivalence(differences, "value", margin=0.05, label="no effect")
    assert tight.equivalent
    noisy = _clustered_frame(treatment_effect=0.0, noise=0.4, seed=3)
    wide = CS.equivalence(
        CS.paired_differences(noisy, "value", treatment="treated", reference="sham"),
        "value", margin=0.01, label="noisy")
    assert not wide.equivalent, "a wide interval must fail the equivalence test"


def test_dose_response_reports_a_curve_and_its_monotonicity():
    rng = np.random.default_rng(1)
    rows = [dict(prompt_id=p, seed=s, gamma=g, value=-2.0 * (1 - g) + rng.normal(0, 0.05))
            for p in range(6) for s in range(2) for g in (0.0, 0.25, 0.5, 0.75, 1.0)]
    curve = CS.dose_response(pd.DataFrame(rows), "value")
    assert list(curve["gamma"]) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert bool(curve["monotone"].iloc[0])
    assert curve["value"].iloc[0] < curve["value"].iloc[-1]
    assert (curve["ci_low"] <= curve["value"]).all() and (curve["value"] <= curve["ci_high"]).all()


def test_missing_condition_is_reported_rather_than_silently_dropped():
    frame = _clustered_frame()
    empty = CS.paired_differences(frame, "value", treatment="absent", reference="sham")
    assert empty.empty


def test_negligible_effects_are_not_called_significant():
    """Floating-point residue with a zero-width interval is not a finding."""
    claim = CS.Claim("q3", "tiny", "value", "a", "b", "increase")
    assert "no effect" in CS._claim_verdict(CS.Estimate(1e-9, 1e-9, 1e-9, 30, 10), claim)
    # Two prompts is too few to form an interval at all, whatever the estimate.
    assert "too few prompts" in CS._claim_verdict(CS.Estimate(-0.5, -0.5, -0.5, 6, 2), claim)
    # A real, perfectly consistent effect is reported as such, with the caveat.
    verdict = CS._claim_verdict(CS.Estimate(0.5, 0.5, 0.5, 30, 10), claim)
    assert "consistent with the prediction" in verdict and "no width" in verdict


def test_evidence_table_names_the_claims_it_could_not_evaluate():
    table = CS.evidence_table({}, checkpoint="tiny", iterations=50)
    assert set(table["verdict"]) == {"question not run"}
    for column in ("question", "checkpoint", "claim", "primary_endpoint", "effect", "verdict"):
        assert column in table.columns


def test_direction_agreement_follows_what_the_claim_predicts():
    """A claim predicting a decrease should not be scored by how often it rose."""
    from ditsinks.questions import QuestionResult

    frame = _clustered_frame(treatment_effect=-0.5)
    frame["head"] = -1
    result = QuestionResult("q1", frame)
    decrease = CS.Claim("q1", "goes down", "value", "treated", "sham", "decrease")
    increase = CS.Claim("q1", "goes up", "value", "treated", "sham", "increase")
    down = CS.evaluate_claim(result, decrease, iterations=200)
    up = CS.evaluate_claim(result, increase, iterations=200)
    assert down["predicted_direction_share"] == pytest.approx(1.0)
    assert up["predicted_direction_share"] == pytest.approx(0.0)
    assert down["verdict"].startswith("supported")
    assert "opposite" in up["verdict"]
