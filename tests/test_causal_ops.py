import pytest
import torch

from ditsinks.causal_ops import (clamp_norm, decompose_lifecycle, regeneration,
                                 FrozenTokenHook, remove_direction, scale_channel, transplant)


def test_q1_edits_preserve_requested_invariants():
    x = torch.tensor([[[3., 4.], [1., 2.], [2., 0.]]])
    v = torch.tensor([1., 0.])
    removed = remove_direction(x, v, [0], preserve_norm=True)
    assert torch.allclose(removed[:, 0].norm(dim=-1), x[:, 0].norm(dim=-1))
    assert torch.allclose(removed[:, 0, 0], torch.zeros(1))
    clamped = clamp_norm(x, [0], 2.0)
    assert torch.allclose(clamped[:, 0].norm(dim=-1), torch.tensor([2.]))


def test_channel_and_transplant_ops():
    x = torch.arange(24.0).reshape(1, 3, 8)
    y = scale_channel(x, 2, 0, [1])
    assert y[0, 1, 2] == 0 and y[0, 0, 2] == x[0, 0, 2]
    z = transplant(x, 0, 2, stage="full_residual")
    assert torch.equal(z[0, 2], x[0, 0])
    with pytest.raises(ValueError, match="unsupported"):
        transplant(x, 0, 2, stage="final_key")


def test_regeneration_and_lifecycle():
    p = torch.tensor([[0.1, .2, .1], [.2, .3, .9]])
    r = regeneration(p, [10, 11], [0], .8)
    assert (r.first_layer, r.token, r.kind) == (11, 2, "relocated")
    states = torch.tensor([[[[1., 0.]], [[0., 2.]]]])
    d = decompose_lifecycle(states, torch.tensor([1., 0.]))
    assert d["alpha"].tolist() == [[[1.], [0.]]]
    assert d["perpendicular_norm"].tolist() == [[[0.], [2.]]]


def test_frozen_hook_edits_only_image_tail():
    x = torch.ones(1, 5, 2)
    hook = FrozenTokenHook(lambda image: scale_channel(image, 0, 0), image_token_count=3)
    _, kwargs = hook.as_pre_hook(None, (), {"hidden_states": x})
    assert torch.equal(kwargs["hidden_states"][:, :2], x[:, :2])
    assert torch.equal(kwargs["hidden_states"][:, 2:, 0], torch.zeros(1, 3))
    assert hook.calls == 1


def test_component_refresh_preserves_perpendicular_state():
    from ditsinks.causal_ops import refresh_parallel_component

    x = torch.randn(7, 12)
    v = torch.randn(12)
    v = v / v.norm()
    out = refresh_parallel_component(x, v, [1, 4], alpha_target=3.5)
    before_perp = x[[1, 4]] - (x[[1, 4]] @ v)[:, None] * v
    after_perp = out[[1, 4]] - (out[[1, 4]] @ v)[:, None] * v
    assert torch.allclose(before_perp, after_perp, atol=1e-5)
    assert torch.allclose(out[[1, 4]] @ v, torch.full((2,), 3.5), atol=1e-5)


def test_norm_preserving_channel_rotation_is_really_norm_preserving():
    from ditsinks.causal_ops import rotate_toward_channel_preserve_norm

    x = torch.randn(6, 10)
    out = rotate_toward_channel_preserve_norm(x, 3, [0, 2, 5], 2.0)
    assert torch.allclose(out[[0, 2, 5]].norm(dim=-1),
                          x[[0, 2, 5]].norm(dim=-1), atol=1e-5)
