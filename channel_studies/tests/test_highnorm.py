"""CPU tests for the high-norm token statistics in src.common.highnorm.

Pure numpy, with no torch, diffusers or matplotlib. The planted-data tests check that
the variance-explained curve separates H1 from H2 data, and the hypergeometric test is
checked against brute-force enumeration. The method is described in docs/highnorm_tokens.md.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src.common import highnorm
from src.stage2_channel_ranking import channel_scores


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- channel selection --------------------------------------------------------


def test_top_channels_uses_abs_then_mean():
    # Channel 0 has a large mean(abs) but zero abs(mean). Ranking by abs(mean) would
    # wrongly put channel 1 first.
    x = np.zeros((4, 3), dtype=np.float64)
    x[:, 0] = [10.0, -10.0, 10.0, -10.0]  # mean(abs)=10, abs(mean)=0
    x[:, 1] = [1.0, 1.0, 1.0, 1.0]  # mean(abs)=1,  abs(mean)=1
    x[:, 2] = [0.0, 0.0, 0.0, 0.0]
    assert highnorm.top_channels(x, 1).tolist() == [0]
    assert highnorm.top_channels(x, 2).tolist() == [0, 1]


def test_top_channels_stable_tiebreak_and_bounds():
    x = np.ones((5, 4), dtype=np.float64)  # all channels tie
    assert highnorm.top_channels(x, 3).tolist() == [0, 1, 2]
    assert highnorm.top_channels(x, 0).size == 0
    with pytest.raises(ValueError):
        highnorm.top_channels(x, 5)


def test_top_channels_matches_stage2_scores():
    x = _rng(1).normal(size=(64, 32))
    expected = np.argsort(-channel_scores(x), kind="stable")[:5]
    assert highnorm.top_channels(x, 5).tolist() == expected.tolist()


# --- norm decomposition -------------------------------------------------------


def test_norm_decomposition_is_exact():
    x = _rng(2).normal(size=(40, 16))
    c = np.array([1, 5, 9])
    part = highnorm.token_norms(x[:, c])
    rest = highnorm.token_norms(x, exclude=c)
    full = highnorm.token_norms(x)
    np.testing.assert_allclose(part**2 + rest**2, full**2, rtol=1e-10, atol=1e-10)


def test_norm_fraction_bounded_and_zero_safe():
    x = _rng(3).normal(size=(20, 8))
    x[0, :] = 0.0  # zero-norm token must not produce NaN
    rho = highnorm.norm_fraction(x, np.array([0, 1]))
    assert rho.shape == (20,)
    assert np.all(np.isfinite(rho)) and np.all((rho >= 0) & (rho <= 1))
    assert rho[0] == 0.0
    np.testing.assert_allclose(highnorm.norm_fraction(x, np.arange(8))[1:], 1.0, rtol=1e-12)


def test_excluding_all_channels_gives_zero_norm():
    x = _rng(4).normal(size=(6, 3))
    np.testing.assert_allclose(highnorm.token_norms(x, exclude=np.arange(3)), 0.0)


# --- variance-explained curve (E1) --------------------------------------------


def _planted_h1(n=200, d=64, n_out=10, seed=5):
    """Planted H1 data. Outlier tokens are high-norm only because of one massive channel."""
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=1.0, size=(n, d))
    out = np.arange(n_out)
    x[out, 0] += 500.0  # one channel carries essentially all the outlier energy
    return x, out


def _planted_h2(n=200, d=64, n_out=10, seed=6):
    """Planted H2 data. Outlier tokens are elevated across all channels."""
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=1.0, size=(n, d))
    out = np.arange(n_out)
    x[out, :] *= 40.0  # elevated everywhere
    x[np.ix_(out, [0])] += 200.0  # plus a massive channel on top
    return x, out


def test_e1_separates_h1_from_h2():
    x_h1, _ = _planted_h1()
    curve = highnorm.variance_explained_curve(x_h1, ks=[1, 2, 5, 10], base_k=1)
    assert curve["rho_outlier"][0] > 0.95, "H1: one channel should own the outlier norm"
    assert curve["rho_typical"][0] < 0.1, "typical tokens must not be channel-dominated"

    x_h2, _ = _planted_h2()
    curve2 = highnorm.variance_explained_curve(x_h2, ks=[1, 2, 5, 10], base_k=1)
    assert curve2["rho_outlier"][0] < 0.9, "H2: energy is spread beyond the massive channel"


def test_e1_curve_is_monotone_in_k():
    x, _ = _planted_h2()
    curve = highnorm.variance_explained_curve(x, ks=[1, 2, 4, 8, 16], base_k=2)
    for name in ("rho_outlier", "rho_typical"):
        vals = curve[name]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), f"{name} not monotone"


def test_e1_recovers_planted_outlier_tokens():
    x, out = _planted_h1()
    found = highnorm.top_fraction_indices(
        highnorm.massive_score(x, highnorm.top_channels(x, 1)), 0.05
    )
    assert set(found.tolist()) == set(out.tolist())


# --- overlap statistics -------------------------------------------------------


def _brute_force_hypergeom_sf(obs, n_pop, n_success, n_draw):
    """P(X >= obs) by enumerating every draw of size n_draw from n_pop."""
    population = range(n_pop)
    successes = set(range(n_success))
    total = hits = 0
    for draw in itertools.combinations(population, n_draw):
        total += 1
        if len(successes & set(draw)) >= obs:
            hits += 1
    return hits / total


def test_hypergeom_matches_brute_force():
    for obs in range(0, 4):
        expected = _brute_force_hypergeom_sf(obs, n_pop=12, n_success=4, n_draw=5)
        got = highnorm.hypergeom_sf(obs, 12, 4, 5)
        assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12)


def test_overlap_stats_edge_cases():
    empty = highnorm.overlap_stats(np.array([]), np.array([]), n_tokens=100)
    assert empty["iou"] == 1.0 and empty["p_value"] == 1.0

    disjoint = highnorm.overlap_stats(np.array([0, 1, 2]), np.array([50, 51, 52]), n_tokens=100)
    assert disjoint["iou"] == 0.0
    assert disjoint["n_intersection"] == 0.0
    assert disjoint["p_value"] == pytest.approx(1.0)

    identical = highnorm.overlap_stats(np.arange(5), np.arange(5), n_tokens=1000)
    assert identical["iou"] == 1.0
    assert identical["p_value"] < 1e-9


def test_overlap_expected_intersection_is_chance_rate():
    stats = highnorm.overlap_stats(np.arange(10), np.arange(5, 15), n_tokens=1000)
    assert stats["expected_intersection"] == pytest.approx(10 * 10 / 1000)
    assert stats["n_intersection"] == 5.0


# --- rank statistics ----------------------------------------------------------


def test_auroc_matches_pair_counting():
    rng = _rng(7)
    scores = rng.normal(size=60)
    labels = rng.random(60) < 0.3
    pos, neg = scores[labels], scores[~labels]
    brute = np.mean([(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg])
    assert highnorm.auroc(scores, labels) == pytest.approx(brute, rel=1e-12)


def test_auroc_perfect_and_degenerate():
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    assert highnorm.auroc(scores, np.array([False, False, True, True])) == pytest.approx(1.0)
    assert highnorm.auroc(scores, np.array([True, True, False, False])) == pytest.approx(0.0)
    assert math.isnan(highnorm.auroc(scores, np.array([True] * 4)))


def test_auroc_ties_count_half():
    assert highnorm.auroc(np.array([1.0, 1.0]), np.array([True, False])) == pytest.approx(0.5)


def test_spearman_monotone_and_ties():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert highnorm.spearman(x, x**3) == pytest.approx(1.0)
    assert highnorm.spearman(x, -(x**3)) == pytest.approx(-1.0)
    assert math.isnan(highnorm.spearman(x, np.ones(5)))


def test_spearman_matches_pearson_on_ranks():
    rng = _rng(8)
    a, b = rng.normal(size=50), rng.normal(size=50)
    ra = highnorm._average_ranks(a)
    rb = highnorm._average_ranks(b)
    expected = np.corrcoef(ra, rb)[0, 1]
    assert highnorm.spearman(a, b) == pytest.approx(expected, rel=1e-10)


# --- scale-matched null -------------------------------------------------------


def test_scale_matched_channels_respects_window_and_exclusions():
    x = _rng(9).normal(size=(100, 600)) * np.linspace(5.0, 0.1, 600)
    excluded = highnorm.top_channels(x, 2)
    order = np.argsort(-channel_scores(x), kind="stable")
    window = set(order[50:200].tolist())

    picked = highnorm.scale_matched_channels(
        x, k=2, exclude=excluded, rng=_rng(0), rank_lo=50, rank_hi=200
    )
    assert picked.size == 2 and len(set(picked.tolist())) == 2
    assert set(picked.tolist()).isdisjoint(set(excluded.tolist()))
    assert set(picked.tolist()) <= window


def test_scale_matched_channels_is_reproducible():
    x = _rng(10).normal(size=(50, 600))
    excl = highnorm.top_channels(x, 2)
    a = highnorm.scale_matched_channels(x, 3, excl, _rng(42))
    b = highnorm.scale_matched_channels(x, 3, excl, _rng(42))
    assert a.tolist() == b.tolist()


def test_scale_matched_channels_rejects_impossible_request():
    x = _rng(11).normal(size=(20, 600))
    with pytest.raises(ValueError):
        highnorm.scale_matched_channels(
            x, k=10, exclude=np.array([]), rng=_rng(0), rank_lo=50, rank_hi=55
        )


# --- bimodality ---------------------------------------------------------------


def test_bimodality_recovers_planted_threshold():
    rng = _rng(12)
    low = rng.normal(loc=20.0, scale=2.0, size=3880)
    high = rng.normal(loc=400.0, scale=20.0, size=120)
    res = highnorm.bimodality_split(np.concatenate([low, high]))
    assert 20.0 < res["threshold"] < 400.0
    assert res["frac_above"] == pytest.approx(0.03, abs=0.02)
    assert res["bc"] > highnorm.BC_BIMODAL_CUTOFF


def test_bimodality_coefficient_is_not_fooled_by_unimodal_or_tails():
    rng = _rng(13)
    # A single Gaussian lands near the theoretical BC of 1/3.
    gauss = highnorm.bimodality_split(np.abs(rng.normal(loc=50.0, scale=5.0, size=4000)))
    assert gauss["bc"] == pytest.approx(highnorm.BC_GAUSSIAN, abs=0.08)
    assert gauss["bc"] < highnorm.BC_BIMODAL_CUTOFF

    # A heavy right tail has no separate mode and must also stay below the cutoff.
    lognormal = highnorm.bimodality_split(np.exp(rng.normal(loc=3.0, scale=0.8, size=4000)))
    assert lognormal["bc"] < highnorm.BC_BIMODAL_CUTOFF


def test_bimodality_degenerate_input():
    res = highnorm.bimodality_split(np.array([1.0]))
    assert math.isnan(res["threshold"])
    constant = highnorm.bimodality_split(np.full(10, 7.0))
    assert constant["threshold"] == pytest.approx(7.0)
    assert math.isnan(highnorm.bimodality_coefficient(np.full(10, 7.0)))
    assert math.isnan(highnorm.bimodality_coefficient(np.array([1.0, 2.0])))


# --- neighbour redundancy -----------------------------------------------------


def test_neighbor_cosine_on_identical_grid():
    x = np.tile(np.array([[1.0, 2.0, 3.0]]), (9, 1))
    sim = highnorm.neighbor_cosine_similarity(x, 3, 3)
    np.testing.assert_allclose(sim, 1.0, rtol=1e-12)


def test_neighbor_cosine_handles_edges_by_hand():
    # 1x3 grid [a, b, a] with a ⟂ b. Each end has 1 neighbour (b) -> 0, the centre has 2 -> 0.
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    sim = highnorm.neighbor_cosine_similarity(x, 1, 3)
    np.testing.assert_allclose(sim, [0.0, 0.0, 0.0], atol=1e-12)

    # 1x3 grid [a, a, b] with a ⟂ b -> [1, 0.5, 0]
    y = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(
        highnorm.neighbor_cosine_similarity(y, 1, 3), [1.0, 0.5, 0.0], atol=1e-12
    )


def test_neighbor_cosine_rejects_bad_grid():
    with pytest.raises(ValueError):
        highnorm.neighbor_cosine_similarity(np.zeros((7, 4)), 3, 3)


def test_neighbor_cosine_zero_token_is_zero_not_nan():
    x = np.ones((4, 3))
    x[0] = 0.0
    sim = highnorm.neighbor_cosine_similarity(x, 2, 2)
    assert np.all(np.isfinite(sim))
    assert sim[0] == pytest.approx(0.0)


# --- lazy imports -------------------------------------------------------------


def test_modules_import_without_heavy_deps(monkeypatch):
    import builtins
    import importlib

    blocked = {"torch", "diffusers", "transformers", "matplotlib", "sklearn", "scipy"}
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"{name} must not be imported at module scope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    for mod in ("src.common.highnorm", "src.experiments.highnorm_tokens"):
        importlib.reload(importlib.import_module(mod))
