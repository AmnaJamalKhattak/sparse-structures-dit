"""Reproducible prompt sets for the causal experiments.

DiffusionDB is a log of real Stable Diffusion submissions, and its rows are in
submission order.  Taking the first N rows therefore does not sample N users:
it usually samples one person iterating on one idea.  A pilot draw of the first
twenty rows returned six variants of "a renaissance portrait of <celebrity>, art
in the style of rembrandt", three of "portrait of a dancing eagle woman", and two
each of two more templates: thirteen of twenty prompts were near-copies.

That is not a cosmetic problem.  The prompt is the bootstrap cluster for every
confidence interval in the causal analysis, so near-copies inflate the apparent
number of independent observations and understate uncertainty; and when the same
template lands in both the discovery and confirmation halves, the split stops
being a split.

This module therefore samples the way the dataset's own documentation
recommends, straight from the published ``metadata.parquet``, which carries the
prompt text together with the submitting ``user_name`` and NSFW scores, and never
touches the image archives, and then applies four filters that make the
resulting set defensible:

``one prompt per user``      removes the iteration-session effect at its cause;
``lexical near-duplicates``  removes shared templates across different users
                             ("greg rutkowski, artstation, trending" is ubiquitous);
``safety scores``            drops prompts the dataset itself flags;
``length``                   drops degenerate entries such as "this is a test".

Selection is deterministic given a seed, and the policy is written into the
manifest alongside the prompts so a reader can see exactly how the set was drawn.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# Provenance travels with the data: any notebook that uses these prompts should
# print the citation next to them rather than keep it in a separate document.
DIFFUSIONDB_CITATION = (
    "Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover and "
    "Duen Horng Chau. DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image "
    "Generative Models. ACL 2023, pages 893-911."
)

DIFFUSIONDB_DATASET = "poloclub/diffusiondb"
# The dataset's own recommended route for prompt-only work: one Parquet table of
# every prompt and its hyper-parameters, with no image archives involved.
METADATA_FILES = ("metadata.parquet", "metadata-large.parquet")
METADATA_COLUMNS = ("prompt", "user_name", "prompt_nsfw", "image_nsfw")
DATASETS_SERVER = "https://datasets-server.huggingface.co"

# Words that carry no topical content: almost every DiffusionDB prompt ends in a
# tail of render and artist tags, and comparing prompts on those alone would call
# every prompt a duplicate of every other.
STYLE_STOPWORDS = frozenset("""
a an and the of by in on at with for to is are as art artstation trending highly
detailed intricate ultra hyperdetailed hyper realistic realism photorealistic
sharp focus smooth illustration painting digital concept render rendering octane
unreal engine ue cinematic lighting lightning dramatic volumetric ray tracing 4k
8k hd quality masterpiece beautiful stunning epic elegant style k dof bokeh
""".split())

_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class PromptSelection:
    """The policy used to draw a prompt set, recorded with the prompts."""

    count: int
    seed: int = 0
    max_per_user: int = 1
    similarity_limit: float = 0.5
    min_words: int = 5
    max_nsfw: float = 0.2
    source: str = ""
    inspected: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        return (f"{self.count} prompts, seed {self.seed}: at most {self.max_per_user} per user, "
                f"content-word overlap below {self.similarity_limit:g}, at least {self.min_words} "
                f"words, safety scores below {self.max_nsfw:g}")


def content_words(prompt: str) -> frozenset:
    """The topical words of a prompt, with render and artist tags removed."""
    return frozenset(word for word in _WORD.findall(prompt.lower())
                     if word not in STYLE_STOPWORDS and len(word) > 2)


def similarity(left: str, right: str) -> float:
    """Overlap of two prompts' content words, from 0 (disjoint) to 1 (identical).

    Jaccard rather than a prefix or an edit distance: the near-duplicates in this
    dataset differ by a name in the middle or a tag at the end, and both of those
    leave the content-word set almost unchanged.
    """
    a, b = content_words(left), content_words(right)
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / len(a | b)


def select_diverse_prompts(rows: Sequence[Mapping[str, Any]], count: int, *,
                           selection: Optional[PromptSelection] = None,
                           seed: int = 0, **overrides) -> Tuple[List[str], PromptSelection]:
    """Choose ``count`` prompts that are not variations on each other.

    ``rows`` are candidate records carrying at least ``prompt`` and, where the
    source provides them, ``user_name`` and the two NSFW scores.  Rows are shuffled
    with a fixed seed before selection, so the result depends on the seed and the
    candidate pool but not on the order the pool happened to arrive in.

    Returns the prompts and the policy that produced them, including how many
    candidates each filter rejected, which tells you whether the pool was
    large enough.
    """
    import random

    policy = selection or PromptSelection(count=int(count), seed=int(seed), **overrides)
    if policy.count < 1:
        raise ValueError("count must be a positive integer")

    candidates = [dict(row) for row in rows]
    random.Random(policy.seed).shuffle(candidates)

    kept: List[str] = []
    per_user: Dict[str, int] = {}
    rejected = {"too short": 0, "flagged unsafe": 0, "same user": 0, "near duplicate": 0}

    for row in candidates:
        prompt = str(row.get("prompt") or "").strip()
        if len(prompt.split()) < policy.min_words:
            rejected["too short"] += 1
            continue
        scores = [row.get("prompt_nsfw"), row.get("image_nsfw")]
        if any(score is not None and float(score) > policy.max_nsfw
               for score in scores if _is_number(score)):
            rejected["flagged unsafe"] += 1
            continue
        user = row.get("user_name")
        if user is not None and per_user.get(str(user), 0) >= policy.max_per_user:
            rejected["same user"] += 1
            continue
        if any(similarity(prompt, chosen) >= policy.similarity_limit for chosen in kept):
            rejected["near duplicate"] += 1
            continue
        kept.append(prompt)
        if user is not None:
            per_user[str(user)] = per_user.get(str(user), 0) + 1
        if len(kept) == policy.count:
            break

    from dataclasses import replace

    policy = replace(policy, inspected=len(candidates), rejected=rejected)
    if len(kept) < policy.count:
        raise RuntimeError(
            f"Only {len(kept)} of {policy.count} prompts survived selection from "
            f"{len(candidates)} candidates ({policy.describe()}). Rejected: "
            + ", ".join(f"{n} {why}" for why, n in rejected.items())
            + ". Widen the candidate pool rather than relaxing the policy.")
    return kept, policy


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


# ------------------------------------------------------------------- sources
def _metadata_url(name: str) -> str:
    return f"https://huggingface.co/datasets/{DIFFUSIONDB_DATASET}/resolve/main/{name}"


def fetch_metadata_rows(pool: int = 20000, seed: int = 0,
                        columns: Sequence[str] = METADATA_COLUMNS) -> Tuple[List[Dict[str, Any]], str]:
    """Read candidate rows from the published metadata table.

    Row groups are drawn from across the file rather than from its start: the
    table is in submission order, so the opening row group is a handful of users
    and would reproduce exactly the correlation this module is meant to avoid.
    """
    import random

    import fsspec
    import pyarrow.parquet as pq

    failures = []
    for name in METADATA_FILES:
        url = _metadata_url(name)
        try:
            with fsspec.filesystem("http").open(url) as handle:
                parquet = pq.ParquetFile(handle)
                available = [c for c in columns if c in parquet.schema_arrow.names]
                if "prompt" not in available:
                    raise RuntimeError(f"{name} has no prompt column")
                groups = list(range(parquet.num_row_groups))
                random.Random(seed).shuffle(groups)
                rows: List[Dict[str, Any]] = []
                for index in groups:
                    table = parquet.read_row_group(index, columns=available)
                    rows.extend(table.to_pylist())
                    if len(rows) >= pool:
                        break
        except Exception as exc:                       # noqa: BLE001 - reported, then next source
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if rows:
            return rows[:pool], f"{name}, {len(rows)} rows from row groups sampled across the file"
    raise RuntimeError("; ".join(failures) or "no metadata file returned rows")


def fetch_viewer_rows(pool: int = 1000) -> Tuple[List[Dict[str, Any]], str]:
    """Fallback: read rows through the dataset viewer, discovering its configs.

    The viewer's configurations are derived from the files the Hub can see, and
    they have changed before (the loading-script names such as ``2m_first_1k``
    no longer resolve), so they are asked for rather than assumed.
    """
    splits = _get_json(f"{DATASETS_SERVER}/splits", dataset=DIFFUSIONDB_DATASET).get("splits", [])
    for entry in splits:
        config, split = entry.get("config"), entry.get("split")
        rows: List[Dict[str, Any]] = []
        try:
            while len(rows) < pool:
                payload = _get_json(f"{DATASETS_SERVER}/rows", dataset=DIFFUSIONDB_DATASET,
                                    config=config, split=split, offset=len(rows),
                                    length=min(100, pool - len(rows)))
                page = [entry.get("row", {}) for entry in payload.get("rows", [])]
                if not page:
                    break
                rows.extend(page)
        except Exception:                              # noqa: BLE001 - try the next configuration
            continue
        if any(row.get("prompt") for row in rows):
            return rows, f"dataset viewer, config {config}, split {split}, {len(rows)} rows"
    raise RuntimeError("no viewer configuration returned a prompt column")


def _get_json(url: str, **params) -> Dict[str, Any]:
    request = Request(f"{url}?{urlencode(params)}", headers={
        "User-Agent": "ditsinks/reproducibility", "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


# ------------------------------------------------------------------ manifest
def _read_cache(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    prompts = payload if isinstance(payload, list) else payload.get("prompts")
    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError(f"Invalid DiffusionDB prompt cache: {path}")
    return prompts, ({} if isinstance(payload, list) else payload)


def diffusiondb_prompts(cache_path, count: int, *, seed: int = 0, pool: int = 20000,
                        strict: bool = True, **policy) -> List[str]:
    """Return ``count`` mutually distinct DiffusionDB prompts, caching the draw.

    A complete cache is preferred and makes later runs offline; an undersized one
    is redrawn rather than silently returning fewer prompts.  ``strict`` re-checks
    a cache written by an older, order-based selection and refuses it, so a stale
    manifest of near-duplicates cannot quietly become the prompt set of an experiment.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    cache_path = Path(cache_path)

    if cache_path.exists():
        cached, payload = _read_cache(cache_path)
        if len(cached) >= count:
            prompts = cached[:count]
            limit = float(policy.get("similarity_limit",
                                     payload.get("selection", {}).get("similarity_limit", 0.5)))
            worst = _worst_pair(prompts)
            if not strict or worst is None or worst[0] < limit:
                return prompts
            score, left, right = worst
            raise ValueError(
                f"The cached prompt set in {cache_path} contains near-duplicates "
                f"(content-word overlap {score:.2f} >= {limit:g}):\n  {left[:70]}...\n  "
                f"{right[:70]}...\nIt was probably drawn in dataset order. Delete the file to "
                f"redraw it, or pass strict=False to use it anyway.")

    failures = []
    for source in (lambda: fetch_metadata_rows(pool=pool, seed=seed), lambda: fetch_viewer_rows()):
        try:
            rows, provenance = source()
        except Exception as exc:                       # noqa: BLE001 - reported, then next source
            failures.append(f"{type(exc).__name__}: {exc}")
            continue
        prompts, selection = select_diverse_prompts(rows, count, seed=seed, source=provenance,
                                                    **policy)
        _write_manifest(cache_path, prompts, selection)
        return prompts

    raise RuntimeError(
        "Could not reach DiffusionDB. Tried the published metadata table and the dataset "
        "viewer:\n  " + "\n  ".join(failures) +
        "\nCheck network access, or supply a complete prompt cache at " + str(cache_path))


def _worst_pair(prompts: Sequence[str]) -> Optional[Tuple[float, str, str]]:
    worst = None
    for i, left in enumerate(prompts):
        for right in prompts[i + 1:]:
            score = similarity(left, right)
            if worst is None or score > worst[0]:
                worst = (score, left, right)
    return worst


def _write_manifest(cache_path: Path, prompts: Sequence[str], selection: PromptSelection) -> None:
    payload = {
        "dataset": DIFFUSIONDB_DATASET,
        "citation": DIFFUSIONDB_CITATION,
        "selection": asdict(selection),
        "selection_summary": selection.describe(),
        "prompts": list(prompts),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(cache_path)                      # only a complete manifest is ever visible


def diversity_report(prompts: Sequence[str]) -> "Any":
    """Every pair's content-word overlap, worst first: an auditable diversity check."""
    import pandas as pd

    rows = [{"prompt_a": i, "prompt_b": j, "overlap": similarity(a, b),
             "a": a[:60], "b": b[:60]}
            for i, a in enumerate(prompts) for j, b in enumerate(prompts) if i < j]
    return pd.DataFrame(rows).sort_values("overlap", ascending=False).reset_index(drop=True)
