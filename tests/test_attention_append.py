"""Appending non-spatial key/value slots to attention, without changing its output shape.

This is the seam the Virtual Register experiment needs: a memory slot that queries can
attend to but that corresponds to no image patch and is never decoded spatially. It sits
*after* QK-norm and RoPE, which is what makes an appended row genuinely position-free.

The transform returns only the rows to add, not a concatenated tensor, and one
function resolves the sequence axis (diffusers hands the FLUX kernel [B, S, H, D] and
the generic Attention path [B, H, S, D]), so these tests are mostly about that contract
holding.
"""
import math

import pytest
import torch

from ditsinks.attention_patch import apply_kv_transform, sequence_axis, to_bhsd


def test_the_sequence_axis_is_resolved_for_both_diffusers_layouts():
    """FLUX gives [B, S, H, D]; the generic Attention path gives [B, H, S, D]."""
    flux = torch.zeros(1, 1024, 24, 128)
    generic = torch.zeros(1, 24, 1024, 128)
    assert sequence_axis(flux, heads=24) == 1
    assert sequence_axis(generic, heads=24) == 2
    # And without a head count, from the fact that sequences are longer than head counts.
    assert sequence_axis(flux) == 1
    assert sequence_axis(generic) == 2


def test_an_ambiguous_shape_is_refused_rather_than_guessed():
    """Heads == sequence length. Appending to the wrong axis is the bug this prevents."""
    square = torch.zeros(1, 8, 8, 16)
    with pytest.raises(ValueError, match="cannot be resolved"):
        sequence_axis(square)
    # Even a head count cannot disambiguate when it equals the sequence length, and
    # refusing is right: `to_bhsd` may fall back to a guess because it only READS, but
    # appending to the wrong axis silently corrupts the attention.
    with pytest.raises(ValueError, match="cannot be resolved"):
        sequence_axis(square, heads=8)
    with pytest.raises(ValueError, match="4-D"):
        sequence_axis(torch.zeros(4, 4))


def test_appending_grows_the_keys_and_values_and_nothing_else():
    heads, seq, dim = 4, 40, 16
    key = torch.randn(1, seq, heads, dim)
    value = torch.randn(1, seq, heads, dim)
    query = torch.randn(1, seq, heads, dim)
    extra_key = torch.randn(1, 2, heads, dim)
    extra_value = torch.randn(1, 2, heads, dim)

    new_key, new_value, kwargs = apply_kv_transform(
        lambda q, k, v, kw: (extra_key, extra_value), query, key, value, {}, heads=heads)
    assert new_key.shape == (1, seq + 2, heads, dim)
    assert new_value.shape == (1, seq + 2, heads, dim)
    # The originals are carried through untouched, and the new rows are at the end.
    assert torch.allclose(new_key[:, :seq], key)
    assert torch.allclose(new_key[:, seq:], extra_key)
    assert torch.allclose(new_value[:, seq:], extra_value)
    assert kwargs == {}


def test_the_attention_output_keeps_its_sequence_length():
    """The property that makes this safe: nothing downstream sees a shape change."""
    heads, seq, dim, added = 4, 40, 16, 3
    query = torch.randn(1, heads, seq, dim)
    key = torch.randn(1, heads, seq, dim)
    value = torch.randn(1, heads, seq, dim)
    extra = torch.randn(1, heads, added, dim)
    new_key, new_value, _ = apply_kv_transform(
        lambda q, k, v, kw: (extra, extra.clone()), query, key, value, {}, heads=heads)
    before = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    after = torch.nn.functional.scaled_dot_product_attention(query, new_key, new_value)
    assert after.shape == before.shape == (1, heads, seq, dim)
    # And the output DID change, or the slot is inert and the experiment measures nothing.
    assert not torch.allclose(after, before)


def test_the_appended_column_receives_attention_mass():
    """The measurement the experiment turns on: mass on the non-spatial slot."""
    heads, seq, dim = 2, 20, 8
    torch.manual_seed(0)
    query = torch.randn(1, heads, seq, dim)
    key = torch.randn(1, heads, seq, dim)
    value = torch.randn(1, heads, seq, dim)
    # A key aligned with the mean query attracts more than a uniform share.
    attractive = torch.nn.functional.normalize(
        query.mean(dim=2, keepdim=True), dim=-1) * float(key.norm(dim=-1).mean())
    new_key, _, _ = apply_kv_transform(
        lambda q, k, v, kw: (attractive, value[:, :, :1]), query, key, value, {},
        heads=heads)
    scale = 1.0 / math.sqrt(dim)
    probs = torch.softmax(torch.matmul(query, new_key.transpose(-1, -2)) * scale, dim=-1)
    mass = float(probs[..., -1].mean())
    assert mass > 1.0 / (seq + 1), f"an aligned slot should beat the uniform share: {mass}"


def test_a_transform_returning_none_is_a_no_op():
    key, value, query = (torch.randn(1, 12, 2, 8) for _ in range(3))
    new_key, new_value, kwargs = apply_kv_transform(
        lambda q, k, v, kw: None, query, key, value, {"attn_mask": None}, heads=2)
    assert new_key is key and new_value is value


def test_extra_rows_that_disagree_with_the_keys_are_refused():
    heads, seq, dim = 4, 40, 16
    key = value = query = torch.randn(1, seq, heads, dim)
    def transform(shape_key, shape_value):
        return lambda q, k, v, kw: (torch.randn(*shape_key), torch.randn(*shape_value))

    # wrong head count
    with pytest.raises(ValueError, match="only differ .* on the sequence"):
        apply_kv_transform(transform((1, 2, heads + 1, dim), (1, 2, heads + 1, dim)),
                           query, key, value, {}, heads=heads)
    # wrong head dim
    with pytest.raises(ValueError, match="only differ .* on the sequence"):
        apply_kv_transform(transform((1, 2, heads, dim + 1), (1, 2, heads, dim + 1)),
                           query, key, value, {}, heads=heads)
    # keys and values of different lengths
    with pytest.raises(ValueError, match="same length"):
        apply_kv_transform(transform((1, 3, heads, dim), (1, 2, heads, dim)),
                           query, key, value, {}, heads=heads)
    # not a pair
    with pytest.raises(ValueError, match=r"\(extra_key, extra_value\)"):
        apply_kv_transform(lambda q, k, v, kw: torch.randn(1, 2, heads, dim),
                           query, key, value, {}, heads=heads)
    with pytest.raises(ValueError, match="must return tensors"):
        apply_kv_transform(lambda q, k, v, kw: (1.0, 2.0), query, key, value, {},
                           heads=heads)


def test_a_boolean_mask_is_widened_so_the_register_is_visible():
    """A register nothing may attend to is not a register."""
    heads, seq, dim, added = 2, 12, 8, 2
    key = value = query = torch.randn(1, seq, heads, dim)
    mask = torch.ones(1, 1, seq, seq, dtype=torch.bool)
    extra = torch.randn(1, added, heads, dim)
    _, _, kwargs = apply_kv_transform(
        lambda q, k, v, kw: (extra, extra.clone()), query, key, value,
        {"attn_mask": mask}, heads=heads)
    widened = kwargs["attn_mask"]
    assert widened.shape == (1, 1, seq, seq + added)
    assert bool(widened[..., -added:].all()), "the appended columns must be visible"


def test_an_additive_mask_is_widened_with_zeros_not_ones():
    heads, seq, dim, added = 2, 12, 8, 1
    key = value = query = torch.randn(1, seq, heads, dim)
    mask = torch.zeros(1, 1, seq, seq)
    mask[..., 0] = float("-inf")
    extra = torch.randn(1, added, heads, dim)
    _, _, kwargs = apply_kv_transform(
        lambda q, k, v, kw: (extra, extra.clone()), query, key, value,
        {"attn_mask": mask}, heads=heads)
    widened = kwargs["attn_mask"]
    assert widened.shape == (1, 1, seq, seq + added)
    assert float(widened[..., -1].max()) == 0.0, "an additive mask pads with 0, not 1"
    assert float(widened[..., 0].min()) == float("-inf"), "the original mask is kept"


def test_a_mask_that_cannot_be_widened_is_refused_rather_than_guessed_at():
    heads, seq, dim = 2, 12, 8
    key = value = query = torch.randn(1, seq, heads, dim)
    extra = torch.randn(1, 1, heads, dim)
    transform = lambda q, k, v, kw: (extra, extra.clone())      # noqa: E731
    with pytest.raises(ValueError, match="non-tensor mask"):
        apply_kv_transform(transform, query, key, value, {"attn_mask": "causal"},
                           heads=heads)
    with pytest.raises(ValueError, match="not a per-key mask"):
        apply_kv_transform(transform, query, key, value,
                           {"attn_mask": torch.zeros(1, 1, seq, seq + 5)}, heads=heads)


def test_the_tap_rejects_a_mode_and_transform_that_disagree():
    from ditsinks.attention_patch import AttentionTap

    with pytest.raises(ValueError, match="append_kv mode requires"):
        AttentionTap("flux1", lambda *a, **k: None, mode="append_kv")
    with pytest.raises(ValueError, match="only valid in append_kv"):
        AttentionTap("flux1", lambda *a, **k: None, mode="instrument",
                     kv_transform=lambda *a: None)
    with pytest.raises(ValueError, match="patch mode requires"):
        AttentionTap("flux1", lambda *a, **k: None, mode="patch")


# ==================================================== two taps on one entry point
def _flux_entry_point():
    """The module attribute both an injector's tap and the tracer's tap rebind."""
    import importlib

    from ditsinks.attention_patch import _PATCH_TARGETS

    path, attr = _PATCH_TARGETS["flux1"][0]
    return importlib.import_module(path), attr


def test_the_tap_that_enters_last_is_the_outermost_one():
    """Which tap sees an append depends on the order they were entered, and getting it
    backwards is silent: the inner tap's callback reports the key it passed on, so a
    tracer entered *after* an injector measures the sequence from before the append
    while believing the appended columns are there. Nothing raises; the numbers shift.
    """
    from ditsinks.attention_patch import AttentionTap

    module, attr = _flux_entry_point()
    if not hasattr(module, attr):
        pytest.skip(f"{attr} is not where this diffusers build keeps it")
    heads, seq, dim = 4, 40, 16
    query = torch.randn(1, seq, heads, dim)
    key = torch.randn(1, seq, heads, dim)
    value = torch.randn(1, seq, heads, dim)
    extra_key = torch.randn(1, 2, heads, dim)
    extra_value = torch.randn(1, 2, heads, dim)

    def append(q, k, v, kw):
        return extra_key, extra_value

    def run(injector_first: bool) -> int:
        seen = []
        observer = AttentionTap("flux1", lambda q, k, v, kw=None: seen.append(k.shape[1]))
        injector = AttentionTap("flux1", lambda q, k, v, kw=None: None,
                                mode="append_kv", kv_transform=append, heads=heads)
        order = (injector, observer) if injector_first else (observer, injector)
        with order[0], order[1]:
            getattr(module, attr)(query, key, value)
        assert seen, "the observer's tap never fired"
        return seen[-1]

    # Observer entered SECOND, so outermost: it never sees the injector's rows.
    assert run(injector_first=True) == seq
    # Observer entered FIRST, so innermost: the injector's transform reaches it.
    assert run(injector_first=False) == seq + 2
    # Both taps are removed again either way.
    real = getattr(module, attr)
    assert not getattr(real, "__name__", "").startswith("_wrapped")
