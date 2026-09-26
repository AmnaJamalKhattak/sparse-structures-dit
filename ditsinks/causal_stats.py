"""Uncertainty for the causal claims, in the predeclared form these experiments use.

Three ideas do all the work here.

**The prompt is the cluster, the prompt-seed pair is the unit.**  Five seeds of
one prompt are not five independent observations: they share the prompt's layout
and its register positions.  Every interval below resamples *prompts* with
replacement and carries all of a prompt's seeds along, which keeps the
interval valid when seeds are correlated within a prompt.

**Effects are paired.**  A condition is compared with a reference run of the
same prompt, seed, and noise, so the difference is within-unit and the
prompt-to-prompt variance never enters the effect at all.

**A near-zero claim needs an equivalence test.**  "Not significant" is not
evidence of no effect.  Where the claim is that something is practically
unchanged (immediate sink identity after a norm-only reduction, or a sham hook
leaving the model alone), the right test is two one-sided tests against a margin
declared before the run.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_UNIT = ("prompt_id", "seed")
DEFAULT_CLUSTER = "prompt_id"


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its clustered interval and supporting counts."""

    value: float
    ci_low: float
    ci_high: float
    n_units: int
    n_clusters: int
    positive_fraction: float = float("nan")
    label: str = ""

    @property
    def has_interval(self) -> bool:
        """Whether a usable interval exists at all.

        A percentile bootstrap over two clusters can only draw three distinct
        resamples, so it reports an interval that is either degenerate or
        meaningless; such a run has no interval, and saying so beats printing one.
        """
        return bool(np.isfinite(self.ci_low) and np.isfinite(self.ci_high)
                    and self.ci_high > self.ci_low)

    @property
    def excludes_zero(self) -> bool:
        return self.has_interval and not (self.ci_low <= 0.0 <= self.ci_high)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.value:+.3f} [{self.ci_low:+.3f}, {self.ci_high:+.3f}] (n={self.n_units})"


def _clean(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame is None or frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=[column])
    out = frame.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=[column])


def collapse_units(frame: pd.DataFrame, value: str, *, unit: Sequence[str] = DEFAULT_UNIT,
                   cluster: str = DEFAULT_CLUSTER, aggregate: str = "mean") -> pd.DataFrame:
    """One row per experimental unit, which is what the bootstrap resamples.

    Rows arrive per layer and per head; averaging within a unit first stops a
    layer-rich condition from silently outweighing a layer-poor one.
    """
    frame = _clean(frame, value)
    keys = [k for k in unit if k in frame.columns]
    if frame.empty or not keys:
        return pd.DataFrame(columns=list(unit) + [cluster, value])
    grouped = frame.groupby(keys, observed=True)[value].agg(aggregate).reset_index()
    # The cluster is often one of the unit keys already; re-merging it would give
    # a duplicated column and a two-dimensional grouper.
    if cluster not in grouped.columns and cluster in frame.columns:
        lookup = frame.groupby(keys, observed=True)[cluster].first().reset_index()
        grouped = grouped.merge(lookup, on=keys, how="left")
    return grouped


def bootstrap_mean(frame: pd.DataFrame, value: str, *, unit: Sequence[str] = DEFAULT_UNIT,
                   cluster: str = DEFAULT_CLUSTER, iterations: int = 2000,
                   alpha: float = 0.05, seed: int = 0, label: str = "") -> Estimate:
    """Mean with a prompt-clustered percentile bootstrap interval."""
    units = collapse_units(frame, value, unit=unit, cluster=cluster)
    if units.empty:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, 0, float("nan"), label)
    values = units[value].to_numpy(dtype=float)
    groups = (units[cluster].to_numpy() if cluster in units.columns
              else np.arange(len(units)))
    unique = np.unique(groups)
    point = float(values.mean())
    positive = float((values > 0).mean())
    # Three clusters is the floor for a percentile bootstrap: with two, the
    # resampling distribution has three points and the interval is an artefact.
    if len(unique) < 3 or iterations <= 0:
        return Estimate(point, float("nan"), float("nan"), len(values), len(unique), positive, label)
    index_by_cluster = {key: np.flatnonzero(groups == key) for key in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for i in range(iterations):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        picked = np.concatenate([index_by_cluster[key] for key in chosen])
        draws[i] = values[picked].mean()
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Estimate(point, float(low), float(high), len(values), len(unique), positive, label)


def paired_differences(frame: pd.DataFrame, value: str, *, condition: str = "condition",
                       treatment: str, reference: str, unit: Sequence[str] = DEFAULT_UNIT,
                       cluster: str = DEFAULT_CLUSTER) -> pd.DataFrame:
    """Within-unit ``treatment - reference`` differences, one row per unit."""
    frame = _clean(frame, value)
    if frame.empty or condition not in frame.columns:
        return pd.DataFrame(columns=list(unit) + [cluster, value])
    keys = [k for k in unit if k in frame.columns]
    left = collapse_units(frame[frame[condition] == treatment], value, unit=keys, cluster=cluster)
    right = collapse_units(frame[frame[condition] == reference], value, unit=keys, cluster=cluster)
    if left.empty or right.empty:
        return pd.DataFrame(columns=list(unit) + [cluster, value])
    merged = left.merge(right, on=keys, how="inner", suffixes=("_treatment", "_reference"))
    merged[value] = merged[f"{value}_treatment"] - merged[f"{value}_reference"]
    if cluster not in merged.columns:
        source = f"{cluster}_treatment"
        merged[cluster] = merged[source] if source in merged.columns else merged[keys[0]]
    columns = list(dict.fromkeys(keys + [cluster, value, f"{value}_treatment",
                                         f"{value}_reference"]))
    return merged[columns]


def paired_effect(frame: pd.DataFrame, value: str, *, treatment: str, reference: str,
                  condition: str = "condition", unit: Sequence[str] = DEFAULT_UNIT,
                  cluster: str = DEFAULT_CLUSTER, iterations: int = 2000,
                  alpha: float = 0.05, seed: int = 0) -> Estimate:
    """Paired effect of one condition against a reference, with a clustered CI."""
    differences = paired_differences(frame, value, condition=condition, treatment=treatment,
                                     reference=reference, unit=unit, cluster=cluster)
    return bootstrap_mean(differences, value, unit=unit, cluster=cluster, iterations=iterations,
                          alpha=alpha, seed=seed, label=f"{treatment} - {reference}")


@dataclass(frozen=True)
class EquivalenceResult:
    """Two one-sided tests against a margin declared before the run."""

    effect: float
    ci_low: float
    ci_high: float
    margin: float
    equivalent: bool
    n_units: int
    label: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        verdict = "practically equivalent" if self.equivalent else "not shown to be equivalent"
        return (f"{self.label}: {self.effect:+.3f} [{self.ci_low:+.3f}, {self.ci_high:+.3f}] "
                f"against a margin of +/-{self.margin:.3f} -- {verdict}")


def equivalence(frame: pd.DataFrame, value: str, *, margin: float,
                unit: Sequence[str] = DEFAULT_UNIT, cluster: str = DEFAULT_CLUSTER,
                iterations: int = 2000, alpha: float = 0.05, seed: int = 0,
                label: str = "") -> EquivalenceResult:
    """Declare practical equivalence only when the whole interval fits the margin.

    This is the bootstrap form of two one-sided tests: the effect is equivalent
    to zero when its clustered confidence interval lies entirely inside
    ``[-margin, +margin]``.  A wide interval therefore fails, which is the point:
    an underpowered run cannot buy a null result.
    """
    estimate = bootstrap_mean(frame, value, unit=unit, cluster=cluster, iterations=iterations,
                              alpha=alpha, seed=seed, label=label)
    inside = bool(estimate.has_interval and estimate.ci_low > -abs(margin)
                  and estimate.ci_high < abs(margin))
    return EquivalenceResult(estimate.value, estimate.ci_low, estimate.ci_high, float(abs(margin)),
                             inside, estimate.n_units, label or value)


def dose_response(frame: pd.DataFrame, value: str, *, dose: str = "gamma",
                  unit: Sequence[str] = DEFAULT_UNIT, cluster: str = DEFAULT_CLUSTER,
                  iterations: int = 1000, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Clustered mean and interval at every dose, plus a monotonicity check.

    A causal scalar claim should be a curve, not a single ablation point; the
    ``monotone`` column says whether the curve actually orders with the dose.
    """
    frame = _clean(frame, value)
    if frame.empty or dose not in frame.columns:
        return pd.DataFrame(columns=[dose, "value", "ci_low", "ci_high", "n_units", "monotone"])
    rows = []
    for level, group in frame.groupby(dose, observed=True):
        estimate = bootstrap_mean(group, value, unit=unit, cluster=cluster, iterations=iterations,
                                  alpha=alpha, seed=seed, label=f"{dose}={level}")
        rows.append(dict(**{dose: level}, value=estimate.value, ci_low=estimate.ci_low,
                         ci_high=estimate.ci_high, n_units=estimate.n_units,
                         n_clusters=estimate.n_clusters))
    out = pd.DataFrame(rows).sort_values(dose).reset_index(drop=True)
    series = out["value"]
    out["monotone"] = bool(series.is_monotonic_increasing or series.is_monotonic_decreasing)
    return out


# ------------------------------------------------------------ evidence table
@dataclass(frozen=True)
class Claim:
    """One predeclared claim: which endpoint, which contrast, which direction."""

    question: str
    name: str
    endpoint: str
    treatment: str
    reference: str
    expect: str = "decrease"          # decrease | increase | equivalent
    margin: float = 0.05
    table: str = "tidy"
    filter_layer_level: bool = True
    condition_column: str = "condition"


PRIMARY_CLAIMS: Tuple[Claim, ...] = (
    Claim("q1", "Removing the register direction moves the sink",
          "head_retention", "direction_removal", "sham", "decrease"),
    Claim("q1", "A count-matched control does not",
          "head_retention", "random_tokens_zeroed", "sham", "equivalent"),
    # The gamma=1 run of the same sweep is the channel-scaling sham. The hook fires
    # and scales the channel by one, so it is the reference the suppression is paired against.
    Claim("q2", "Suppressing the dominant channel lowers the register projection",
          "vstar_projection_change", "dominant_channel__gamma0__writer_only",
          "dominant_channel__gamma1__writer_only", "decrease", table="dose_response",
          filter_layer_level=False),
    Claim("q2", "A random channel at the same scale does not",
          "vstar_projection_change", "random_channel__gamma0__writer_only",
          "dominant_channel__gamma1__writer_only", "equivalent", margin=0.5,
          table="dose_response", filter_layer_level=False),
    Claim("q5", "The register direction alone does not capture an arbitrary token",
          "capture_rate", "direction_at_ordinary_norm", "final_key", "decrease",
          table="tidy", filter_layer_level=False, condition_column="stage"),
    Claim("q3", "Repeated removal suppresses recovery more than a single shot",
          "max_cosine", "direction_destroyed__repeated", "direction_destroyed__single_shot",
          "decrease"),
    Claim("q6", "Late competitor suppression extends the register",
          "lifetime_shift", "competing_channel_suppressed", "sham", "increase"),
)


def evaluate_claim(result, claim: Claim, *, checkpoint: str = "", iterations: int = 2000,
                   seed: int = 0) -> Dict[str, Any]:
    """Turn one claim into a row of the evidence table, or say why it could not run."""
    from .questions import layer_level

    frame = result.tidy if claim.table == "tidy" else result.tables.get(claim.table)
    if frame is None or frame.empty:
        return dict(question=claim.question.upper(), checkpoint=checkpoint, claim=claim.name,
                    primary_endpoint=claim.endpoint, effect=float("nan"), ci_low=float("nan"),
                    ci_high=float("nan"), n_units=0, n_prompts=0,
                    predicted_direction_share=float("nan"), verdict="not run")
    if claim.filter_layer_level and "head" in frame.columns:
        frame = layer_level(frame)
    conditions = set(frame[claim.condition_column].astype(str)) if claim.condition_column in frame else set()
    if claim.treatment not in conditions or claim.reference not in conditions:
        return dict(question=claim.question.upper(), checkpoint=checkpoint, claim=claim.name,
                    primary_endpoint=claim.endpoint, effect=float("nan"), ci_low=float("nan"),
                    ci_high=float("nan"), n_units=0, n_prompts=0,
                    predicted_direction_share=float("nan"),
                    verdict=f"condition not present ({claim.treatment} vs {claim.reference})")
    estimate = paired_effect(frame, claim.endpoint, treatment=claim.treatment,
                             reference=claim.reference, condition=claim.condition_column,
                             iterations=iterations, seed=seed)
    # "Positive fraction" is not what is needed for a claim predicting a
    # decrease; report the share of units moving the way the claim predicts.
    agreement = (estimate.positive_fraction if claim.expect == "increase"
                 else 1.0 - estimate.positive_fraction if claim.expect == "decrease"
                 else float("nan"))
    return dict(question=claim.question.upper(), checkpoint=checkpoint, claim=claim.name,
                primary_endpoint=claim.endpoint, effect=estimate.value, ci_low=estimate.ci_low,
                ci_high=estimate.ci_high, n_units=estimate.n_units,
                n_prompts=estimate.n_clusters, predicted_direction_share=agreement,
                verdict=_claim_verdict(estimate, claim))


# Effects below this are floating-point residue, not findings.
NEGLIGIBLE = 1e-6


MINIMUM_CLUSTERS = 3


def _claim_verdict(estimate: Estimate, claim: Claim) -> str:
    """Say what the evidence supports, and say plainly when it supports nothing.

    The order of these checks is the argument: an unmeasurable endpoint, then too
    few prompts to form an interval at all, then an effect too small to be
    anything but arithmetic, then the interval itself.  A degenerate interval
    (every unit moved by exactly the same amount) is reported as such rather
    than being read either as significance or as a null.
    """
    if not np.isfinite(estimate.value):
        return "endpoint not measurable"
    if estimate.n_clusters < MINIMUM_CLUSTERS:
        return (f"too few prompts for an interval ({estimate.n_clusters}); "
                f"point estimate {estimate.value:+.3f} only")
    negligible = abs(estimate.value) < NEGLIGIBLE
    if claim.expect == "equivalent":
        if negligible:
            return "equivalent within the declared margin"
        if not estimate.has_interval:
            return "no interval available, so equivalence is not established"
        inside = (estimate.ci_low > -abs(claim.margin) and estimate.ci_high < abs(claim.margin))
        return ("equivalent within the declared margin" if inside
                else "not shown to be equivalent; interval exceeds the margin")
    if negligible:
        return "no effect distinguishable from zero"
    signed = estimate.value < 0 if claim.expect == "decrease" else estimate.value > 0
    if not estimate.has_interval:
        return (("consistent with the prediction" if signed else "opposite to the prediction")
                + "; every unit moved identically, so the interval has no width")
    if not estimate.excludes_zero:
        return "no effect distinguishable from zero"
    return ("supported, in the predicted direction" if signed
            else "significant, but opposite to the prediction")


def evidence_table(results: Mapping[str, Any], *, checkpoint: str = "",
                   claims: Sequence[Claim] = PRIMARY_CLAIMS, iterations: int = 2000,
                   seed: int = 0) -> pd.DataFrame:
    """The claim-by-claim summary table."""
    rows = []
    for claim in claims:
        result = results.get(claim.question)
        if result is None:
            rows.append(dict(question=claim.question.upper(), checkpoint=checkpoint,
                             claim=claim.name, primary_endpoint=claim.endpoint,
                             effect=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                             n_units=0, verdict="question not run"))
            continue
        rows.append(evaluate_claim(result, claim, checkpoint=checkpoint, iterations=iterations,
                                   seed=seed))
    return pd.DataFrame(rows)
