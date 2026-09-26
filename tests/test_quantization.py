"""The quantization comparison is only worth running if it is fair.

These tests are mostly about fairness rather than about plumbing.  The headline
question, does preserving ``v*`` beat preserving large activations at the same
budget, is easy to answer accidentally in the affirmative by handicapping the
control, so the properties that keep the controls strong are asserted here.
"""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from ditsinks import quantization as QZ


@pytest.fixture
def outlier_states():
    """Register-shaped activations: ordinary noise plus one massive coordinate."""
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 2.0
    x[:, 7] += 300.0
    return x


def test_quantizing_is_lossy_but_bounded(outlier_states):
    quantized = QZ.quantize_affine(outlier_states, 4)
    assert not torch.allclose(quantized, outlier_states)
    scale = outlier_states.abs().amax(dim=-1, keepdim=True)
    assert float((quantized - outlier_states).abs().max()) <= float(scale.max())


def test_sixteen_bits_is_a_passthrough(outlier_states):
    assert torch.allclose(QZ.quantize_affine(outlier_states, 16), outlier_states)


def test_protection_helps_the_unprotected_coordinates(outlier_states):
    """Every protection scheme must buy resolution for the coordinates it spares.

    This is the regression guard for a real bug: protecting the largest coordinate
    *without* first removing it from the range computation leaves it setting the
    absmax scale, so the other coordinates gain nothing and the magnitude-aware
    control silently becomes a straw man.  A comparison against a handicapped
    control cannot answer the question the experiment is asking.
    """
    rest = [c for c in range(outlier_states.shape[-1]) if c != 7]
    direction = torch.zeros(outlier_states.shape[-1])
    direction[7] = 1.0

    def error(q):
        return float((q[:, rest] - outlier_states[:, rest]).abs().mean())

    plain = error(QZ.quantize_affine(outlier_states, 4))
    for name, protected in (
            ("largest", QZ.quantize_protecting_largest(outlier_states, 1, 4)),
            ("coordinates", QZ.quantize_protecting_coordinates(outlier_states, [7], 4)),
            ("direction", QZ.quantize_protecting_direction(outlier_states, direction, 4))):
        assert error(protected) < plain / 2, (
            f"{name} protection did not improve the unprotected coordinates: "
            f"{error(protected):.4f} against {plain:.4f} for plain quantization")


def test_protection_is_exact_on_what_it_protects(outlier_states):
    direction = torch.zeros(outlier_states.shape[-1])
    direction[7] = 1.0
    unit = direction / direction.norm()
    protected = QZ.quantize_protecting_direction(outlier_states, direction, 4)
    assert torch.allclose(protected @ unit, outlier_states.float() @ unit, atol=1e-3)

    kept = QZ.quantize_protecting_coordinates(outlier_states, [7], 4)
    assert torch.allclose(kept[:, 7], outlier_states.float()[:, 7])


def test_an_axis_aligned_vstar_collapses_onto_channel_protection(outlier_states):
    """The FLUX caveat, asserted rather than left in prose.

    When ``v*`` is a single coordinate, protecting the direction and protecting that
    channel are the same operation.  The experiment must be able to show that, because
    a reader will otherwise read the v* result as ordinary outlier protection.
    """
    direction = torch.zeros(outlier_states.shape[-1])
    direction[7] = 1.0
    by_direction = QZ.quantize_protecting_direction(outlier_states, direction, 4)
    by_channel = QZ.quantize_protecting_coordinates(outlier_states, [7], 4)
    assert torch.allclose(by_direction, by_channel, atol=1e-4)


def test_subspace_protection_preserves_its_projection(outlier_states):
    basis = torch.zeros(2, outlier_states.shape[-1])
    basis[0, 7] = 1.0
    basis[1, 3] = 1.0
    protected = QZ.quantize_protecting_subspace(outlier_states, basis, 4)
    for channel in (7, 3):
        assert torch.allclose(protected[:, channel], outlier_states.float()[:, channel],
                              atol=1e-3)


# ------------------------------------------------------------------ bit budget
def test_a_per_token_choice_pays_for_its_indices():
    """Choosing coordinates per token is strictly more information than a fixed rule.

    A magnitude-aware scheme must transmit *which* coordinates it kept; a fixed
    direction is a model constant.  If that cost were not charged the two schemes
    would appear equally priced while one of them carried extra information.
    """
    fixed = QZ.QuantScheme("v", "v", bits=4, protect="direction",
                           direction=torch.ones(1024))
    chosen = QZ.QuantScheme("m", "m", bits=4, protect="largest", count=1)
    assert chosen.budget(1024, 10).bits_per_token > fixed.budget(1024, 10).bits_per_token
    assert math.isclose(QZ.index_bits(1024), 10.0)


def test_a_lifecycle_window_costs_less_than_protecting_everywhere():
    """The lifecycle claim is partly an economic one, so the budget must reflect it."""
    everywhere = QZ.QuantScheme("all", "all", bits=4, protect="direction",
                                direction=torch.ones(64))
    windowed = QZ.QuantScheme("window", "window", bits=4, protect="direction",
                              direction=torch.ones(64), layers=frozenset({1, 2}))
    assert windowed.budget(64, 10).bits_per_token < everywhere.budget(64, 10).bits_per_token


def test_equal_budget_groups_are_reported():
    schemes = [
        QZ.QuantScheme("a", "a", bits=4, protect="direction", direction=torch.ones(64)),
        QZ.QuantScheme("b", "b", bits=4, protect="coordinates", channels=(3,)),
        QZ.QuantScheme("c", "c", bits=4),
    ]
    table = QZ.budget_table(schemes, width=64, n_layers=8)
    groups = QZ.equal_budget_groups(table)
    matched = [members for members in groups.values() if len(members) > 1]
    assert matched and set(matched[0]) == {"a", "b"}, groups


# ----------------------------------------------------------------- the scheme
def test_a_windowed_scheme_only_protects_inside_its_window(outlier_states):
    direction = torch.zeros(outlier_states.shape[-1])
    direction[7] = 1.0
    scheme = QZ.QuantScheme("w", "w", bits=4, protect="direction", direction=direction,
                            layers=frozenset({2, 3}))
    assert scheme.active_at(2) and not scheme.active_at(9)
    inside = scheme.apply(outlier_states, 2)
    outside = scheme.apply(outlier_states, 9)
    plain = QZ.quantize_affine(outlier_states, 4)
    assert torch.allclose(outside, plain)
    assert not torch.allclose(inside, plain)


def test_a_subspace_scheme_without_a_basis_quantizes_plainly(outlier_states):
    """A missing calibration basis must degrade visibly, never silently protect nothing."""
    scheme = QZ.QuantScheme("s", "s", bits=4, protect="subspace", basis_by_layer={}, count=1)
    assert torch.allclose(scheme.apply(outlier_states, 0),
                          QZ.quantize_affine(outlier_states, 4))
