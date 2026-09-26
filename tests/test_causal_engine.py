"""The engine's guarantees: inert instrumentation, frozen targets, and refusals that say why."""
import math

import pytest
import torch

from ditsinks import SweepConfig
from ditsinks.adapters import InterventionPoint
from ditsinks.causal_engine import (CausalTracer, EditPlan, FinalKeyPatch, GenerationDriver,
                                    hook_site, image_span, run_traced_generation,
                                    select_frozen_targets)
from ditsinks.causal_ops import remove_direction
from ditsinks.synthetic import planted_direction


def _config(model="tiny-flux1"):
    return SweepConfig(model=model, prompts=["a", "b"], seeds=[0], height=64, width=64,
                       num_inference_steps=2, capture_steps=[1], dtype="float32",
                       output_dir="/tmp/ditsinks-test", save_images=False)


@pytest.fixture(scope="module")
def run():
    """One tiny FLUX.1 driver, tracer and clean trace, shared across tests."""
    cfg = _config()
    driver = GenerationDriver(cfg)
    direction = planted_direction(driver.bundle.d_model)
    tracer = CausalTracer(driver.adapter, driver.transformer, direction=direction,
                          layers=list(range(driver.n_layers)), steps=[1], channels=(3, 5),
                          grid=driver.bundle.grid, cfg=cfg)
    trace, _ = run_traced_generation(driver, tracer, prompt_id=0, prompt="a", seed=0,
                                     probes=[(InterventionPoint.BLOCK_INPUT, [1])])
    targets = select_frozen_targets(trace, layer=1, step=1, direction=direction, percentile=95.0,
                                    topk=3)
    return dict(cfg=cfg, driver=driver, tracer=tracer, trace=trace, targets=targets,
                direction=direction)


# -------------------------------------------------------------------- tracer
def test_tracer_records_every_observable_the_questions_need(run):
    observation = run["trace"].at(1, 1)
    n_img, heads = run["trace"].n_img, run["trace"].n_heads
    assert observation.projection.shape == (n_img,)
    assert observation.cosine.shape == (n_img,)
    assert observation.norm.shape == (n_img,)
    assert observation.channel_values.shape == (2, n_img)
    assert observation.incoming.shape == (heads, n_img)
    assert observation.qk_cosine.shape == (heads, n_img)
    assert observation.entropy.shape == (heads,)
    # Incoming image-to-image attention is renormalised, so it is a distribution.
    assert torch.allclose(observation.incoming.sum(-1), torch.ones(heads), atol=1e-5)


def test_attention_can_be_recorded_at_fewer_steps_than_the_state(run):
    """States at every recorded step, attention only where asked: the 1024px cost lever."""
    tracer = CausalTracer(run["driver"].adapter, run["driver"].transformer,
                          direction=run["direction"], layers=[0, 1], steps=[0, 1],
                          grid=run["driver"].bundle.grid, cfg=run["cfg"])
    tracer.attention_steps = {1}
    trace, _ = run_traced_generation(run["driver"], tracer, prompt_id=0, prompt="a", seed=0)
    assert trace.at(0, 1).norm is not None and trace.at(0, 1).incoming is None
    assert trace.at(1, 1).norm is not None and trace.at(1, 1).incoming is not None
    # The narrowed tracer measures exactly what the full one does at the shared step.
    assert torch.allclose(trace.at(1, 1).incoming, run["trace"].at(1, 1).incoming, atol=1e-6)
    assert torch.equal(trace.at(1, 1).projection, run["trace"].at(1, 1).projection)


def test_cosine_is_projection_divided_by_norm(run):
    observation = run["trace"].at(1, 2)
    expected = observation.projection / observation.norm.clamp_min(1e-9)
    assert torch.allclose(observation.cosine, expected, atol=1e-5)


def test_probe_reads_the_tensor_the_hook_will_edit(run):
    """A frozen replacement must come from the edit point, not a nearby stage."""
    probed = run["trace"].probe(1, 1, InterventionPoint.BLOCK_INPUT)
    previous_output = run["trace"].at(1, 0)
    assert probed is not None and probed.shape[0] == run["trace"].n_img
    # Block 1's input is block 0's output, so their norms must agree exactly.
    assert torch.allclose(probed.norm(dim=-1), previous_output.norm, atol=1e-4)


# ---------------------------------------------------------- inert hooks
def test_a_sham_hook_leaves_the_trajectory_bit_identical(run):
    plan = EditPlan(edit=lambda image, _: image.clone(), point=InterventionPoint.BLOCK_INPUT,
                    layers=[1], label="sham")
    treated, stats = run_traced_generation(run["driver"], run["tracer"], prompt_id=0, prompt="a",
                                           seed=0, condition="sham", plans=[plan],
                                           targets=run["targets"])
    assert stats.edit_calls > 0, "the sham hook never fired, so it proves nothing"
    for layer in run["trace"].layers(step=1):
        clean, sham = run["trace"].at(1, layer), treated.at(1, layer)
        assert torch.equal(clean.projection, sham.projection), f"layer {layer} drifted"
        assert torch.equal(clean.norm, sham.norm)


def test_a_real_edit_changes_the_trajectory(run):
    ids = run["targets"].register_ids

    def edit(image, ctx):
        return remove_direction(image, run["direction"], ids)

    treated, stats = run_traced_generation(
        run["driver"], run["tracer"], prompt_id=0, prompt="a", seed=0, condition="removal",
        plans=[EditPlan(edit=edit, point=InterventionPoint.BLOCK_INPUT, layers=[1])],
        targets=run["targets"])
    clean = run["trace"].at(1, 1).projection
    assert not torch.equal(clean, treated.at(1, 1).projection)
    assert stats.perturbation_energy > 0
    assert stats.removed_energy > 0, "removing a direction must reduce squared magnitude"


# ------------------------------------------------------------ frozen targets
def test_targets_cover_every_group_the_questions_control_for(run):
    targets = run["targets"]
    for group in ("register", "topk", "matched", "norm_matched", "random", "nonregister"):
        assert targets.ids_for(group), f"{group} group is empty"
    assert len(targets.matched_ordinary_ids) == len(targets.register_ids)
    assert len(targets.random_ids) == len(targets.register_ids)
    # Controls must not be registers, or they would not control for anything.
    registers = set(targets.register_ids) | set(targets.topk_ids)
    for group in ("matched", "norm_matched", "random", "nonregister"):
        assert not registers & set(targets.ids_for(group)), f"{group} overlaps the registers"


def test_registers_are_the_high_norm_tokens_and_controls_are_not(run):
    targets, states = run["targets"], run["targets"].clean_states
    norms = states.norm(dim=-1)
    assert float(norms[list(targets.register_ids)].min()) > float(norms.median())
    assert targets.register_norm > targets.ordinary_norm


def test_unknown_token_group_is_refused_by_name(run):
    with pytest.raises(KeyError, match="unknown token group"):
        run["targets"].ids_for("whatever")


def test_selection_refuses_a_layer_with_no_clean_state(run):
    with pytest.raises(ValueError, match="no clean state"):
        select_frozen_targets(run["trace"], layer=99, step=1)


def test_adjacent_recipient_is_a_grid_neighbour(run):
    targets = run["targets"]
    rows, cols = targets.grid
    first, nearby = int(targets.register_ids[0]), int(targets.nearby_id)
    distance = math.hypot(first // cols - nearby // cols, first % cols - nearby % cols)
    assert distance == pytest.approx(1.0), "an adjacent patch must be one step away on the grid"


# ------------------------------------------------------- refusing ambiguity
def test_image_span_refuses_a_sequence_it_cannot_identify():
    assert image_span(torch.zeros(1, 16, 8), 16) == slice(0, 16)
    assert image_span(torch.zeros(1, 22, 8), 16, n_txt=6) == slice(6, 22)
    # A text-only stream, a shorter stream, and an unexplained length are refused.
    assert image_span(torch.zeros(1, 6, 8), 16, n_txt=6) is None
    assert image_span(torch.zeros(1, 30, 8), 16, n_txt=6) is None
    assert image_span(torch.zeros(1, 8), 16) is None


def test_unsupported_intervention_point_is_refused_not_approximated(run):
    """FLUX applies rotary position encoding inside the processor, so there is no
    pre-position key module to hook; the engine must say so rather than pick one."""
    reference = run["driver"].adapter.layers(run["driver"].transformer)[1]
    with pytest.raises(NotImplementedError, match="key_pre_position"):
        hook_site(run["driver"].adapter, reference, InterventionPoint.KEY_PRE_POSITION)
    with pytest.raises(NotImplementedError, match="final_key"):
        hook_site(run["driver"].adapter, reference, InterventionPoint.FINAL_KEY)


# ------------------------------------------------ FLUX.1 has two block layouts
GENERIC_REFUSAL = "not exposed as a separable module by this architecture"


def _flux_blocks():
    """One of each real FLUX.1 block type, with the adapter's view of them."""
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock, FluxTransformerBlock)

    from ditsinks.adapters import Flux1Adapter, LayerRef

    adapter = Flux1Adapter()
    made = {}
    for kind, cls in (("dual", FluxTransformerBlock), ("single", FluxSingleTransformerBlock)):
        block = cls(dim=64, num_attention_heads=2, attention_head_dim=32)
        ref = LayerRef(index=0, kind=kind, local_id=0, block=block, attns={"attn": block.attn})
        made[kind] = (adapter, ref, adapter.intervention_capabilities(ref))
    return made


@pytest.mark.parametrize("point", [InterventionPoint.MLP_HIDDEN,
                                   InterventionPoint.ADALN_MODULATION,
                                   InterventionPoint.PRE_KEY_NORM_RESIDUAL])
def test_a_single_block_exposes_what_it_names_differently(point):
    """A FLUX.1 single block computes these under other names, its adaptive norm
    is `norm`, its feed-forward is `proj_mlp` into `act_mlp`, and reporting them
    unsupported would be a false claim about FLUX, not about our hooks."""
    for kind, (adapter, ref, caps) in _flux_blocks().items():
        capability = caps[point]
        assert capability.supported, f"{kind} block: {point.value} reported {capability.reason}"
        assert capability.module_path, f"{kind} block: {point.value} named no module"
        module, side = hook_site(adapter, ref, point)
        assert isinstance(module, torch.nn.Module)


def test_the_one_stage_a_single_block_really_lacks_says_why():
    """Both branches read the same normalised stream, so there is no post-attention
    stage before the feed-forward. That refusal is true and must stay specific."""
    blocks = _flux_blocks()
    assert blocks["dual"][2][InterventionPoint.PRE_MLP_RESIDUAL].supported
    refused = blocks["single"][2][InterventionPoint.PRE_MLP_RESIDUAL]
    assert not refused.supported
    assert "proj_out" in refused.reason, refused.reason
    assert refused.reason != GENERIC_REFUSAL, "a specific absence deserves a specific reason"


def test_no_flux_point_is_refused_with_the_generic_reason():
    """The generic string means "we looked and found nothing", which for FLUX.1 only
    ever meant "we looked under the other block type's name"."""
    offenders = [(kind, point.value) for kind, (_, _, caps) in _flux_blocks().items()
                 for point, capability in caps.items()
                 if not capability.supported and capability.reason == GENERIC_REFUSAL]
    assert not offenders, f"refused without saying why: {offenders}"


def test_a_single_block_hook_lands_on_the_image_tokens():
    """Advertising a point that then fails at hook time would be worse than the
    original bug, so run the block and check every span resolves."""
    from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock

    from ditsinks.adapters import Flux1Adapter, LayerRef

    dim, n_txt, n_img = 64, 7, 16
    block = FluxSingleTransformerBlock(dim=dim, num_attention_heads=2, attention_head_dim=32)
    ref = LayerRef(index=0, kind="single", local_id=0, block=block, attns={"attn": block.attn})
    adapter, seen, handles = Flux1Adapter(), {}, []
    points = (InterventionPoint.MLP_HIDDEN, InterventionPoint.ADALN_MODULATION,
              InterventionPoint.PRE_KEY_NORM_RESIDUAL, InterventionPoint.WRITER_RESIDUAL)
    for point in points:
        module, _ = hook_site(adapter, ref, point)

        def record(p):
            def hook(mod, args, kwargs, out):
                tensors = [out] if torch.is_tensor(out) else [t for t in out if torch.is_tensor(t)]
                seen[p] = [image_span(t, n_img, n_txt) for t in tensors]
            return hook

        handles.append(module.register_forward_hook(record(point), with_kwargs=True))
    with torch.no_grad():
        block(hidden_states=torch.randn(1, n_img, dim),
              encoder_hidden_states=torch.randn(1, n_txt, dim),
              temb=torch.randn(1, dim), image_rotary_emb=None)
    for handle in handles:
        handle.remove()

    for point in points:
        spans = [s for s in seen.get(point, []) if s is not None]
        assert spans, f"{point.value} fired but no tensor carried an identifiable image span"
        # Single blocks put text first, so the image tokens are the tail.
        assert spans[0] == slice(n_txt, n_txt + n_img), f"{point.value} sliced {spans[0]}"


def test_pixart_exposes_the_pre_position_key_that_flux_does_not():
    cfg = _config("tiny-pixart")
    driver = GenerationDriver(cfg)
    reference = driver.adapter.layers(driver.transformer)[0]
    module, side = hook_site(driver.adapter, reference, InterventionPoint.KEY_PRE_POSITION)
    assert side == "post" and isinstance(module, torch.nn.Module)


# -------------------------------------------------------------- key patching
def test_final_key_patch_touches_only_the_recipient():
    heads, sequence, dim, n_img = 2, 10, 4, 8

    class _Ref:
        index = 3

    class _Tracer:
        _active = _Ref()
        _heads = heads
        n_txt = sequence - n_img
        recording = True

    clean = torch.arange(heads * n_img * dim, dtype=torch.float32).reshape(heads, n_img, dim)
    patch = FinalKeyPatch(_Tracer(), layers=[3], source=1, recipient=5, clean_keys={3: clean})
    key = torch.zeros(1, heads, sequence, dim)
    out = patch(None, key, None, {})
    assert out.shape == key.shape and patch.applied == 1
    offset = sequence - n_img
    assert torch.allclose(out[0, :, offset + 5, :], clean[:, 1, :])
    untouched = [p for p in range(sequence) if p != offset + 5]
    assert torch.equal(out[0, :, untouched, :], key[0, :, untouched, :])


def test_final_key_patch_ignores_layers_it_was_not_given():
    class _Ref:
        index = 9

    class _Tracer:
        _active = _Ref()
        _heads = 2
        n_txt = 0
        recording = True

    key = torch.randn(1, 2, 8, 4)
    patch = FinalKeyPatch(_Tracer(), layers=[3], source=0, recipient=1,
                          clean_keys={3: torch.zeros(2, 8, 4)})
    assert torch.equal(patch(None, key, None, {}), key)
    assert patch.applied == 0


# ------------------------------------------------- per-layer clean references
def _trace_with_growing_norms(targets, layers=(10, 20, 30), scale=(1.0, 4.0, 16.0)):
    """A trace whose residual norms grow with depth, as a real transformer's do."""
    from ditsinks.causal_engine import LayerObservation, Trace

    trace = Trace(prompt_id=0, seed=0)
    count = targets.n_img
    for layer, factor in zip(layers, scale):
        norms = torch.full((count,), factor)
        norms[list(targets.register_ids)] = factor * 8.0
        cosine = torch.full((count,), 0.02)
        cosine[list(targets.register_ids)] = 0.9
        trace.rows[(0, layer)] = LayerObservation(
            layer=layer, step=0, norm=norms, cosine=cosine, projection=norms * cosine)
    return trace


def test_clean_reference_reads_each_layer_on_its_own_terms(run):
    """A bar frozen at one layer would compare late tokens with an early population."""
    from ditsinks import endpoints as EP

    targets = run["targets"]
    clean = _trace_with_growing_norms(targets)
    reference = EP.build_clean_reference(clean, targets, layers=[10, 20, 30], step=0)
    bars = [reference.at(l).norm_threshold for l in (10, 20, 30)]
    assert bars[0] < bars[1] < bars[2], f"the high-norm bar did not follow the norms: {bars}"
    assert bars[2] / bars[0] == pytest.approx(16.0, rel=1e-3)


def test_alignment_bar_is_a_null_from_the_ordinary_tokens(run):
    """A token counts as register-aligned by beating the clean ordinary population."""
    from ditsinks import endpoints as EP

    targets = run["targets"]
    clean = _trace_with_growing_norms(targets)
    reference = EP.build_clean_reference(clean, targets, layers=[10], step=0)
    bar = reference.at(10).alignment_threshold
    assert 0.0 < bar < 0.9, "the bar must sit above ordinary tokens and below the registers"


def test_recovery_never_counts_the_layer_that_was_edited(run):
    """The edited block cannot show maintenance the network has not performed yet."""
    from ditsinks import endpoints as EP

    targets = run["targets"]
    clean = _trace_with_growing_norms(targets, layers=(targets.layer, targets.layer + 1))
    reference = EP.build_clean_reference(clean, targets,
                                         layers=[targets.layer, targets.layer + 1], step=0)
    layer, token, kind = EP.recovery_outcome(clean, targets,
                                             layers=[targets.layer, targets.layer + 1], step=0,
                                             reference=reference)
    assert layer != targets.layer, "recovery was scored at the intervened layer"


# ------------------------------------------------------- sink suppression
def _sink_tracer(heads=2, n_img=8, n_txt=2, layer=3):
    """A stand-in tracer.  A class body would shadow the parameter names it
    assigns, so build the object rather than declaring it."""
    from types import SimpleNamespace

    return SimpleNamespace(_active=SimpleNamespace(index=layer), _heads=heads,
                           n_txt=n_txt, n_img=n_img, recording=True)


@pytest.mark.parametrize("layout", ["heads_first", "sequence_first"])
def test_sink_suppression_replaces_only_the_target_keys(layout):
    """The sink goes; every other key, and the whole residual stream, stays."""
    from ditsinks.causal_engine import SinkSuppression

    heads, n_img, n_txt, dim = 2, 8, 2, 4
    key = (torch.randn(1, heads, n_img + n_txt, dim) if layout == "heads_first"
           else torch.randn(1, n_img + n_txt, heads, dim))
    sequence_axis = 2 if layout == "heads_first" else 1
    tracer = _sink_tracer(heads, n_img, n_txt)
    patch = SinkSuppression(tracer, layers=[3], tokens=[1, 5])
    out = patch(None, key, None, {})

    assert out.shape == key.shape and patch.applied == 1
    offset = n_txt
    targets = [offset + 1, offset + 5]
    ordinary = [p for p in range(offset, offset + n_img) if p not in targets]
    expected = key[0].index_select(sequence_axis - 1, torch.tensor(ordinary)).mean(
        dim=sequence_axis - 1)
    for position in targets:
        assert torch.allclose(out[0].select(sequence_axis - 1, position), expected, atol=1e-6)
    assert torch.equal(out[0].index_select(sequence_axis - 1, torch.tensor(ordinary)),
                       key[0].index_select(sequence_axis - 1, torch.tensor(ordinary)))
    # Text keys are not image keys and must not be touched.
    assert torch.equal(out[0].index_select(sequence_axis - 1, torch.arange(n_txt)),
                       key[0].index_select(sequence_axis - 1, torch.arange(n_txt)))


def test_sink_suppression_uses_the_ordinary_mean_not_zero():
    """A zero key still scores zero, which can win when the rest score negative."""
    from ditsinks.causal_engine import SinkSuppression

    key = torch.full((1, 2, 6, 3), -2.0)
    key[0, :, 3, :] = 5.0                      # the sink's key
    patch = SinkSuppression(_sink_tracer(2, 6, 0), layers=[3], tokens=[3])
    out = patch(None, key, None, {})
    replaced = out[0, :, 3, :]
    assert torch.allclose(replaced, torch.full_like(replaced, -2.0), atol=1e-6)
    assert not torch.allclose(replaced, torch.zeros_like(replaced))


def test_sink_suppression_leaves_other_layers_alone():
    from ditsinks.causal_engine import SinkSuppression

    tracer = _sink_tracer(layer=9)
    key = torch.randn(1, 2, 10, 4)
    patch = SinkSuppression(tracer, layers=[3], tokens=[1])
    assert torch.equal(patch(None, key, None, {}), key) and patch.applied == 0


def test_final_key_move_neutralizes_source_and_copies_recipient():
    from ditsinks.causal_engine import FinalKeyPatch

    tracer = _sink_tracer(heads=2, n_img=6, n_txt=2, layer=3)
    clean = torch.randn(2, 6, 4)
    key = torch.randn(1, 2, 8, 4)
    patch = FinalKeyPatch(tracer, layers=[3], source=1, recipient=4,
                          clean_keys={3: clean}, move=True)
    out = patch(None, key, None, {})
    assert torch.equal(out[0, :, 6, :], clean[:, 1, :])
    expected_source = clean[:, [0, 2, 3, 5], :].mean(dim=1)
    assert torch.allclose(out[0, :, 3, :], expected_source)
