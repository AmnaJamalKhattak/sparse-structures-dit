"""Prompt selection must return prompts that are not variations on each other."""
import json
from pathlib import Path

import pytest

from ditsinks.prompt_sets import (PromptSelection, content_words, diffusiondb_prompts,
                                  diversity_report, select_diverse_prompts, similarity,
                                  _write_manifest)

# The first twenty rows of DiffusionDB in dataset order, as an earlier pilot draw
# returned them.  Thirteen are near-copies of another, which is the failure this
# module exists to prevent, so the real thing is the regression fixture.
FIRST_20 = json.loads((Path(__file__).parent / "data_diffusiondb_first20.json").read_text())


def _rows(prompts, users=None):
    return [{"prompt": p, "user_name": (users[i] if users else f"user{i}")}
            for i, p in enumerate(prompts)]


# ------------------------------------------------------------------ similarity
def test_similarity_sees_through_a_swapped_name():
    """The renaissance-portrait family differs by one celebrity name."""
    a = FIRST_20[0]     # ...portrait of dwayne johnson...
    b = FIRST_20[8]     # ...portrait of will ferrell...
    assert similarity(a, b) > 0.6
    assert similarity(a, a) == 1.0


def test_similarity_sees_through_an_appended_tag():
    """The eagle-woman family differs only in its trailing artist list."""
    assert similarity(FIRST_20[1], FIRST_20[5]) > 0.8


def test_unrelated_prompts_are_not_called_similar():
    assert similarity("a cigarette with purple tobacco", "concept art of a silent hill monster") < 0.2


def test_style_tags_do_not_make_everything_a_duplicate():
    """Nearly every prompt ends in the same render tags; they must not dominate."""
    left = "a heron in a salt marsh, artstation, trending, 8 k, octane render, highly detailed"
    right = "a tram in lisbon, artstation, trending, 8 k, octane render, highly detailed"
    assert similarity(left, right) < 0.35
    assert "artstation" not in content_words(left)
    assert "heron" in content_words(left)


# ------------------------------------------------------------------- selection
def test_the_observed_duplicate_set_is_rejected_at_full_size():
    with pytest.raises(RuntimeError, match="survived selection"):
        select_diverse_prompts(_rows(FIRST_20), count=20, seed=0)


def test_selection_returns_only_distinct_prompts():
    kept, policy = select_diverse_prompts(_rows(FIRST_20), count=6, seed=0)
    assert len(kept) == 6 == len(set(kept))
    worst = diversity_report(kept)["overlap"].max()
    assert worst < policy.similarity_limit, f"kept a pair overlapping {worst:.2f}"


def test_one_prompt_per_user_by_default():
    """A single user iterating is the cause of the duplication, so cap them at one."""
    rows = _rows([f"a photograph of subject {i} in a wide field" for i in range(12)],
                 users=["one"] * 12)
    with pytest.raises(RuntimeError, match="same user"):
        select_diverse_prompts(rows, count=3, seed=0)


def test_the_per_user_cap_is_adjustable():
    rows = [{"prompt": p, "user_name": "one"} for p in
            ["a heron in a salt marsh at dawn", "a tram climbing a hill in lisbon",
             "bread cooling on a stone windowsill"]]
    kept, _ = select_diverse_prompts(rows, count=3, seed=0, max_per_user=3)
    assert len(kept) == 3


def test_flagged_and_short_entries_are_dropped():
    rows = [{"prompt": "a heron standing in a salt marsh", "user_name": "a"},
            {"prompt": "short", "user_name": "b"},
            {"prompt": "a tram climbing a hill in lisbon", "user_name": "c", "prompt_nsfw": 0.9},
            {"prompt": "bread cooling on a stone windowsill", "user_name": "d", "image_nsfw": 0.01}]
    kept, policy = select_diverse_prompts(rows, count=2, seed=0)
    assert len(kept) == 2
    assert policy.rejected["too short"] == 1 and policy.rejected["flagged unsafe"] == 1


def test_selection_is_deterministic_given_a_seed():
    rows = _rows([f"a photograph of {n} in {p}" for n, p in
                  [("a heron", "a salt marsh"), ("a tram", "lisbon"), ("bread", "a stone oven"),
                   ("a kite", "a gale"), ("moss", "a north wall"), ("a violin", "an attic")]])
    first, _ = select_diverse_prompts(rows, count=3, seed=7)
    assert first == select_diverse_prompts(rows, count=3, seed=7)[0]
    assert first != select_diverse_prompts(rows, count=3, seed=8)[0]


def test_the_policy_records_why_candidates_were_dropped():
    _, policy = select_diverse_prompts(_rows(FIRST_20), count=5, seed=0)
    assert policy.inspected == len(FIRST_20)
    assert policy.rejected["near duplicate"] > 0
    assert "per user" in policy.describe() and "overlap" in policy.describe()


# -------------------------------------------------------------------- manifest
def test_a_stale_order_based_manifest_is_refused(tmp_path):
    """The manifest an earlier draw wrote must not quietly become a paper's prompt set."""
    manifest = tmp_path / "diffusiondb_20_prompts.json"
    manifest.write_text(json.dumps({"dataset": "poloclub/diffusiondb", "prompts": FIRST_20}))
    with pytest.raises(ValueError, match="near-duplicates"):
        diffusiondb_prompts(manifest, count=20)
    # ...but reproducing an earlier run stays possible when asked for explicitly.
    assert diffusiondb_prompts(manifest, count=20, strict=False) == FIRST_20


def test_a_fresh_manifest_round_trips_and_records_its_policy(tmp_path):
    rows = _rows([f"a photograph of {n} in {p}" for n, p in
                  [("a heron", "a salt marsh"), ("a tram", "lisbon"), ("bread", "a stone oven"),
                   ("a kite", "a winter gale"), ("moss", "a north facing wall")]])
    prompts, policy = select_diverse_prompts(rows, count=4, seed=1)
    manifest = tmp_path / "fresh.json"
    _write_manifest(manifest, prompts, policy)
    assert diffusiondb_prompts(manifest, count=4) == prompts
    payload = json.loads(manifest.read_text())
    assert payload["selection"]["max_per_user"] == 1
    assert "citation" in payload and "selection_summary" in payload


# A draw the stub can satisfy: ten prompts of the same shape, sharing no content
# word beyond the frame, so none is rejected as a near-copy of another.
_STUB_PAIRS = [("a heron", "a salt marsh"), ("a tram", "lisbon"), ("bread", "a stone oven"),
               ("a kite", "a winter gale"), ("moss", "a north facing wall"),
               ("a lighthouse", "dense sea fog"), ("a bicycle", "a garden path"),
               ("an origami crane", "folded newspaper"),
               ("a stone bridge", "a river at dusk"), ("a typewriter", "a dusty attic")]


def _stub_draw(monkeypatch):
    """Stand in for a machine that can reach DiffusionDB."""
    from ditsinks import prompt_sets

    rows = _rows([f"a photograph of {n} in {p}" for n, p in _STUB_PAIRS])
    monkeypatch.setattr(prompt_sets, "fetch_metadata_rows",
                        lambda pool=20000, seed=0: (rows, "stubbed draw"))
    return rows


def test_an_undersized_cache_is_redrawn_rather_than_padded_out(tmp_path, monkeypatch):
    """The cache exists for reproducibility, not to return however many it happens to hold.

    Stubbed rather than left to the network, so the redraw is exercised regardless of
    whether the dataset is reachable.
    """
    _stub_draw(monkeypatch)
    manifest = tmp_path / "short.json"
    manifest.write_text(json.dumps({"prompts": ["a heron in a salt marsh at dawn"]}))
    drawn = diffusiondb_prompts(manifest, count=5)
    assert len(drawn) == 5
    assert "a heron in a salt marsh at dawn" not in drawn, "the stale cache was padded out"
    assert json.loads(manifest.read_text())["prompts"] == drawn, "the redraw was not recorded"


def test_an_undersized_cache_that_cannot_be_redrawn_names_itself(tmp_path):
    """And when there is no way to redraw, it says which file to fix.

    The fetchers are severed for the whole suite (see conftest), so this is the
    unreachable case without depending on this machine being unreachable.
    """
    manifest = tmp_path / "short.json"
    manifest.write_text(json.dumps({"prompts": ["a heron in a salt marsh at dawn"]}))
    with pytest.raises(RuntimeError, match="short.json"):
        diffusiondb_prompts(manifest, count=5)


def test_the_suite_guard_stops_a_test_reaching_the_network():
    """The guard itself, so it cannot rot into a no-op without a test noticing.

    Called directly rather than through ``diffusiondb_prompts``, which catches a
    failing fetch by design and would report the guard working even if it were
    not the thing that stopped the call.
    """
    from ditsinks import prompt_sets

    with pytest.raises(AssertionError, match="reached the network"):
        prompt_sets.fetch_metadata_rows()
    with pytest.raises(AssertionError, match="reached the network"):
        prompt_sets.fetch_viewer_rows()


def test_count_must_be_a_positive_integer(tmp_path):
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            diffusiondb_prompts(tmp_path / "x.json", count=bad)


def test_diversity_report_ranks_the_worst_pair_first():
    report = diversity_report(FIRST_20)
    assert report["overlap"].is_monotonic_decreasing
    assert report["overlap"].iloc[0] > 0.9
    assert len(report) == len(FIRST_20) * (len(FIRST_20) - 1) // 2
