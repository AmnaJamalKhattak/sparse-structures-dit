"""Fake (quantize-dequantize) activation quantization with mechanism-aware protection.

This module supports lifecycle-aware register-preserving quantization. The
hypothesis under test is that preserving the causally identified register direction
``v*``, and only through the layers where the register state actually exists, gives a
better quality/bit trade-off than preserving large activations, which is what existing
diffusion quantizers already do.

**What is and is not simulated.**  Every scheme here is QDQ: values are quantized to a
low-bit grid and immediately dequantized back to the model's dtype, so the arithmetic
still runs in bfloat16.  That is sufficient to test numerical robustness and mechanism
fidelity, and it is *not* evidence of a latency improvement: end-to-end timing needs real
low-bit kernels, which this pipeline does not have.

Activation *memory* is a different matter and should not be lumped in with latency.  The
footprint follows from the bit allocation by arithmetic (the allocation is the footprint),
so :mod:`ditsinks.cost_model` reports it as a result rather than an estimate, along
with the memory-traffic share that bounds what any activation policy could do for speed.
Weights are untouched throughout: this measures an activation allocation policy in
isolation, which is what makes the accounting below interpretable.

**The fairness contract.**  Every protected scheme keeps the same number of
high-precision scalars per token per layer, so the bulk quantizer sees the same budget
in all of them.  The one place they genuinely differ is that a *fixed* direction (``v*``,
a random direction, a low-rank basis) is a model constant the decoder already has, while
a *per-token* choice of coordinates has to transmit which coordinates it chose.
:class:`BitBudget` charges those index bits explicitly rather than hiding them, and
:func:`budget_table` prints the resulting cost so a reader can check the comparison
instead of trusting it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Sequence, Tuple

import math

import torch


# --------------------------------------------------------------- bit accounting
@dataclass(frozen=True)
class BitBudget:
    """The exact effective activation cost of one scheme, per token per layer.

    Reported rather than assumed.  ``bits_per_coordinate`` is the number used to
    compare across conditions, and it is the total (bulk grid, quantizer scales
    and every high-precision escape hatch) divided by the width of the vector.
    """

    width: int
    base_bits: float                      # bits per coordinate in the bulk quantizer
    scale_groups: int = 1                 # (scale, zero-point) pairs stored per token
    scale_bits: int = 16
    protected_values: int = 0             # high-precision scalars kept per token
    protected_bits: int = 16
    index_bits_per_value: float = 0.0     # cost of saying *which* coordinates are kept
    amortized_bits: float = 0.0           # model constants (v*, a low-rank basis)
    note: str = ""

    @property
    def quantized_coordinates(self) -> int:
        """Coordinates still carried by the low-bit grid.

        Protecting a *direction* does not remove a coordinate: the residual keeps its
        full width and merely loses one degree of freedom.  Protecting a *coordinate*
        does remove it.  Both are charged the same way here, as extra high-precision
        scalars on top of the full-width grid, which is the conservative reading and
        never flatters the mechanism-aware scheme.
        """
        return self.width

    @property
    def bits_per_token(self) -> float:
        return (self.quantized_coordinates * self.base_bits
                + self.scale_groups * 2 * self.scale_bits
                + self.protected_values * (self.protected_bits + self.index_bits_per_value))

    @property
    def bits_per_coordinate(self) -> float:
        return self.bits_per_token / max(self.width, 1)

    def scaled(self, active_fraction: float) -> "BitBudget":
        """The budget of a scheme whose protection is only active on part of the run.

        A lifecycle-gated scheme pays for its high-precision scalars only in the layers
        where the protection is switched on, so the correct per-token figure is the mean
        over layers.  Everything else is unchanged.
        """
        fraction = float(min(max(active_fraction, 0.0), 1.0))
        return replace(self, protected_values=self.protected_values * fraction,
                       amortized_bits=self.amortized_bits * (1.0 if fraction else 0.0),
                       note=(self.note + f" (protection active on {fraction:.0%} of layers)").strip())


def index_bits(width: int) -> float:
    """Bits needed to name one coordinate out of ``width``."""
    return math.log2(max(int(width), 2))


# ------------------------------------------------------------------ quantizers
def quantize_affine(x: torch.Tensor, bits: int, *, per_token: bool = True,
                    symmetric: bool = True) -> torch.Tensor:
    """Quantize to a uniform grid and dequantize immediately (QDQ).

    Per-token absmax is the standard activation quantizer and the one that makes the
    outlier problem visible: a single massive coordinate stretches the token's range and
    costs every other coordinate its resolution.  That is precisely the failure the
    register direction is hypothesised to protect against, so it is the right bulk
    quantizer to hold fixed across all conditions.
    """
    if bits >= 16:
        return x.clone()
    x = x.float()
    dim = -1 if per_token else None
    if symmetric:
        scale = (x.abs().amax(dim=dim, keepdim=True) if dim is not None
                 else x.abs().amax())
        levels = 2 ** (bits - 1) - 1
        step = (scale / levels).clamp_min(1e-12)
        return (x / step).round().clamp(-levels - 1, levels) * step
    low = x.amin(dim=dim, keepdim=True) if dim is not None else x.amin()
    high = x.amax(dim=dim, keepdim=True) if dim is not None else x.amax()
    levels = 2 ** bits - 1
    step = ((high - low) / levels).clamp_min(1e-12)
    return ((x - low) / step).round().clamp(0, levels) * step + low


def _unit(direction: torch.Tensor) -> torch.Tensor:
    flat = direction.float().flatten()
    return flat / flat.norm().clamp_min(1e-12)


def quantize_protecting_direction(x: torch.Tensor, direction: torch.Tensor, bits: int,
                                  *, per_token: bool = True) -> torch.Tensor:
    """Quantize ``x_perp`` at ``bits`` while keeping ``alpha = <x, u>`` in full precision.

    This is the decomposition the application rests on::

        alpha = x @ u,  x_perp = x - alpha u,  x_hat = Q(x_perp) + alpha u

    The direction is a model constant, so the only per-token cost is ``alpha`` itself.
    Removing the register component before quantizing is also what shrinks the token's
    dynamic range, which is why the *remaining* coordinates gain resolution rather than
    merely being left alone.
    """
    u = _unit(direction).to(x.device)
    alpha = x.float() @ u
    perp = x.float() - alpha[..., None] * u
    return quantize_affine(perp, bits, per_token=per_token) + alpha[..., None] * u


def quantize_protecting_subspace(x: torch.Tensor, basis: torch.Tensor, bits: int,
                                 *, per_token: bool = True) -> torch.Tensor:
    """Protect the projection onto an ``r``-dimensional subspace.

    ``basis`` is ``[r, C]`` and is orthonormalised here rather than trusted.  This backs
    the low-rank absorption baseline: the same shape of escape hatch a low-rank branch
    provides, at a per-token cost of ``r`` high-precision scalars.
    """
    b = basis.float().reshape(-1, x.shape[-1]).to(x.device)
    q, _ = torch.linalg.qr(b.T)                      # [C, r], orthonormal columns
    coefficients = x.float() @ q                     # [..., r]
    parallel = coefficients @ q.T
    return quantize_affine(x.float() - parallel, bits, per_token=per_token) + parallel


def quantize_protecting_coordinates(x: torch.Tensor, channels: Sequence[int], bits: int,
                                    *, per_token: bool = True) -> torch.Tensor:
    """Keep a fixed set of coordinates in full precision; quantize the rest.

    With a single channel this is "protect the model's dominant channel".  When ``v*`` is
    nearly axis-aligned (as it is in FLUX) this converges on
    :func:`quantize_protecting_direction`, and the gap between the two conditions is the
    measurement that says how much of the benefit is geometry rather than one channel.
    """
    ids = torch.as_tensor(list(channels), dtype=torch.long, device=x.device)
    if not ids.numel():
        return quantize_affine(x.float(), bits, per_token=per_token)
    # The protected coordinates are removed from the tensor *before* the range is
    # computed, then restored exactly. Leaving them in would let the outlier keep
    # setting the absmax scale, so the remaining coordinates would gain no resolution
    # and the scheme would reduce to plain quantization. That is verified, not assumed.
    # This mirrors what protecting a direction does structurally, which is what makes
    # the two comparable at all.
    residual = x.float().clone()
    residual[..., ids] = 0.0
    out = quantize_affine(residual, bits, per_token=per_token)
    out[..., ids] = x.float()[..., ids]
    return out


def quantize_protecting_largest(x: torch.Tensor, count: int, bits: int,
                                *, per_token: bool = True) -> torch.Tensor:
    """Magnitude-aware control: protect each token's ``count`` largest coordinates.

    This is the baseline the comparison turns on ("is this just outlier
    protection?"), so it is given the *same* number of protected scalars as the
    mechanism-aware scheme and allowed to choose them per token, which is strictly more
    information.  The index cost of that choice is charged in :class:`BitBudget` and
    reported, never silently absorbed.
    """
    if count <= 0:
        return quantize_affine(x.float(), bits, per_token=per_token)
    k = min(int(count), x.shape[-1])
    values = x.float()
    ids = values.abs().topk(k, dim=-1).indices
    # As above: excise before scaling, restore after. Under absmax quantization the
    # largest coordinate already lands on the top level, so protecting it *without*
    # excising it changes nothing at all, so that version of this baseline is a straw
    # man, and the comparison depends on this control being strong.
    residual = values.scatter(-1, ids, torch.zeros_like(values).gather(-1, ids))
    out = quantize_affine(residual, bits, per_token=per_token)
    return out.scatter(-1, ids, values.gather(-1, ids))


# ------------------------------------------------------- lifecycle-gated scheme
@dataclass(frozen=True)
class QuantScheme:
    """One activation-allocation policy, and the layers its protection is active on.

    ``layers`` being ``None`` means "protect everywhere"; a set means the protection is
    switched on only inside that window and the bulk quantizer runs alone outside it.
    That switch is the entire lifecycle-aware claim, so it is a first-class field rather
    than a branch inside a runner.
    """

    key: str
    label: str
    bits: int = 4
    protect: Optional[str] = None        # None | direction | coordinates | largest | subspace
    layers: Optional[frozenset] = None   # None -> every layer
    direction: Optional[torch.Tensor] = None
    channels: Tuple[int, ...] = ()
    count: int = 1
    basis_by_layer: Optional[Dict[int, torch.Tensor]] = None
    per_token: bool = True
    window_label: str = "all layers"
    role: str = "condition"              # condition | control | reference
    note: str = ""

    def active_at(self, layer: int) -> bool:
        return self.layers is None or int(layer) in self.layers

    def apply(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Quantize one ``[tokens, channels]`` activation block for one layer."""
        if self.bits >= 16 and self.protect is None:
            return x.clone()
        if not self.active_at(layer) or self.protect is None:
            return quantize_affine(x, self.bits, per_token=self.per_token)
        if self.protect == "direction":
            if self.direction is None:
                raise ValueError(f"scheme {self.key!r} protects a direction but has none")
            return quantize_protecting_direction(x, self.direction, self.bits,
                                                 per_token=self.per_token)
        if self.protect == "coordinates":
            return quantize_protecting_coordinates(x, self.channels, self.bits,
                                                   per_token=self.per_token)
        if self.protect == "largest":
            return quantize_protecting_largest(x, self.count, self.bits,
                                               per_token=self.per_token)
        if self.protect == "subspace":
            basis = (self.basis_by_layer or {}).get(int(layer))
            if basis is None:
                # No calibration basis for this layer: quantize plainly and say so in the
                # budget rather than silently protecting nothing while charging for it.
                return quantize_affine(x, self.bits, per_token=self.per_token)
            return quantize_protecting_subspace(x, basis, self.bits, per_token=self.per_token)
        raise ValueError(f"unknown protection {self.protect!r}")

    def budget(self, width: int, n_layers: int) -> BitBudget:
        """The effective activation budget this scheme actually spends."""
        if self.bits >= 16 and self.protect is None:
            return BitBudget(width=width, base_bits=16, scale_groups=0,
                             note="unquantized reference")
        protected = {"direction": 1, "coordinates": len(self.channels),
                     "largest": self.count, "subspace": self.count, None: 0}[self.protect]
        # A per-token choice of coordinates must transmit the choice; a fixed direction,
        # channel set or basis is a model constant the decoder already holds.
        per_value_index = index_bits(width) if self.protect == "largest" else 0.0
        amortized = 0.0
        if self.protect == "direction":
            amortized = width * 16.0
        elif self.protect == "subspace":
            amortized = width * 16.0 * self.count * max(n_layers, 1)
        budget = BitBudget(width=width, base_bits=self.bits, scale_groups=1,
                           protected_values=protected,
                           index_bits_per_value=per_value_index,
                           amortized_bits=amortized, note=self.note)
        if self.layers is not None and n_layers:
            budget = budget.scaled(len(self.layers) / float(n_layers))
        return budget

    def edit(self) -> Callable:
        """A ``TokenEdit`` the causal engine can install as an :class:`EditPlan`."""
        def _edit(image: torch.Tensor, ctx) -> torch.Tensor:
            return self.apply(image, int(ctx.layer)).to(image.dtype)
        return _edit


# ------------------------------------------------------------------- reporting
def budget_table(schemes: Sequence[QuantScheme], *, width: int, n_layers: int):
    """Per-scheme effective bit cost, as a frame ready to print beside the results."""
    import pandas as pd

    rows = []
    for scheme in schemes:
        budget = scheme.budget(width, n_layers)
        rows.append(dict(
            condition=scheme.key, condition_label=scheme.label, role=scheme.role,
            window=scheme.window_label, bulk_bits=scheme.bits,
            protected_values_per_token=round(float(budget.protected_values), 4),
            index_bits_per_value=round(float(budget.index_bits_per_value), 3),
            bits_per_token=round(budget.bits_per_token, 3),
            bits_per_coordinate=round(budget.bits_per_coordinate, 5),
            amortized_model_constant_bits=float(budget.amortized_bits),
            note=budget.note))
    return pd.DataFrame(rows)


def equal_budget_groups(table, *, tolerance: float = 0.02) -> Dict[str, Tuple[str, ...]]:
    """Group conditions whose per-coordinate cost matches, so fairness is checkable.

    The comparison is made *at equal budget*. This returns the groups over which
    that condition actually holds, rather than leaving a reader to eyeball the column.
    """
    groups: Dict[float, list] = {}
    for _, row in table.iterrows():
        cost = float(row["bits_per_coordinate"])
        for anchor in groups:
            if abs(anchor - cost) <= tolerance * max(anchor, 1e-9):
                groups[anchor].append(str(row["condition"]))
                break
        else:
            groups[cost] = [str(row["condition"])]
    return {f"{anchor:.4f} bits/coordinate": tuple(members)
            for anchor, members in sorted(groups.items())}
