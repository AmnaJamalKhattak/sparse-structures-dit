"""Tensor-level building blocks and readouts for the causal experiments.

The functions in this module are intentionally independent of Diffusers hooks.  A hook
supplies an image-token tensor ``[batch, token, channel]`` and these functions make the
requested edit.  Keeping selection outside the edit is important: ``token_ids`` must come
from the clean run and is never recomputed from the treated tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence

import torch


class FrozenTokenHook:
    """Adapt a pure tensor edit to a PyTorch pre- or post-forward hook.

    The hook never performs target selection.  It edits the first rank-3 tensor
    whose sequence axis can contain the frozen image slice.  ``as_pre_hook`` and
    ``as_post_hook`` have the exact signatures expected by PyTorch.
    """

    def __init__(self, edit: Callable[[torch.Tensor], torch.Tensor], *,
                 image_token_count: int, image_at_end: bool = True):
        self.edit = edit
        self.image_token_count = int(image_token_count)
        self.image_at_end = image_at_end
        self.calls = 0

    def _tensor(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-2] < self.image_token_count:
            return value
        out = value.clone()
        sl = (slice(-self.image_token_count, None) if self.image_at_end
              else slice(0, self.image_token_count))
        out[..., sl, :] = self.edit(out[..., sl, :])
        self.calls += 1
        return out

    def _walk(self, value):
        if torch.is_tensor(value):
            return self._tensor(value)
        if isinstance(value, tuple):
            return tuple(self._walk(v) for v in value)
        if isinstance(value, list):
            return [self._walk(v) for v in value]
        return value

    def as_pre_hook(self, module, args, kwargs):
        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = self._tensor(kwargs["hidden_states"])
            return args, kwargs
        return self._walk(args), kwargs

    def as_post_hook(self, module, args, kwargs, output):
        return self._walk(output)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).to(v.dtype)


def remove_direction(x: torch.Tensor, vstar: torch.Tensor, token_ids: Sequence[int],
                     strength: float = 1.0, preserve_norm: bool = False,
                     seed: int = 0, degenerate_ratio: float = 1e-2) -> torch.Tensor:
    """Remove ``strength`` of the v* projection at frozen token IDs.

    ``preserve_norm`` rescales what is left back to the token's original length, so
    the edit changes where the token points without changing how long it is.  That
    matters whenever v* carries most of a register's energy: plain removal would
    leave the token both directionless and tiny, and a condition meant to isolate
    direction would be confounded with a large magnitude reduction.

    Rescaling has a degenerate case of its own.  For a token almost exactly
    collinear with v*, what survives the subtraction is numerical noise, and
    stretching that back to full length amplifies the noise, and can even restore
    the original direction with its sign flipped.  When the residue is smaller than
    ``degenerate_ratio`` of the original norm the direction is therefore replaced by
    an explicit, seeded random direction orthogonal to v*, which is what "keep the
    magnitude, destroy the direction" has to mean in that limit.
    """
    out = x.clone()
    ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
    if not ids.numel():
        return out
    v = _unit(vstar.to(device=x.device, dtype=x.dtype).reshape(1, -1))[0]
    before = out[..., ids, :]
    norm = before.float().norm(dim=-1, keepdim=True)
    alpha = (before.float() * v.float()).sum(-1, keepdim=True)
    edited = before - strength * alpha.to(before.dtype) * v
    if preserve_norm:
        residue = edited.float().norm(dim=-1, keepdim=True)
        degenerate = residue <= degenerate_ratio * norm
        if bool(degenerate.any()):
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            noise = torch.randn(edited.shape, generator=generator).to(edited.device)
            noise = noise - (noise.float() * v.float()).sum(-1, keepdim=True).to(noise.dtype) * v
            edited = torch.where(degenerate.to(edited.device), _unit(noise).to(edited.dtype), edited)
        edited = _unit(edited) * norm.to(edited.dtype)
    out[..., ids, :] = edited
    return out


def clamp_norm(x: torch.Tensor, token_ids: Sequence[int], target_norm: float | torch.Tensor) -> torch.Tensor:
    """Set selected norms without changing directions."""
    out = x.clone()
    ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
    if ids.numel():
        target = torch.as_tensor(target_norm, device=x.device, dtype=x.dtype)
        out[..., ids, :] = _unit(out[..., ids, :]) * target
    return out


def replace_tokens(x: torch.Tensor, token_ids: Sequence[int], replacements: torch.Tensor) -> torch.Tensor:
    """Replace frozen tokens with clean matched-ordinary states."""
    out = x.clone()
    ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
    if replacements.shape[-2:] != out[..., ids, :].shape[-2:]:
        raise ValueError("replacement token/channel shape does not match selected states")
    out[..., ids, :] = replacements.to(out)
    return out


def scale_channel(x: torch.Tensor, channel: int, gamma: float,
                  token_ids: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Scale a live dominant/competing channel globally or at frozen tokens."""
    out = x.clone()
    if token_ids is None:
        out[..., channel] *= gamma
    else:
        ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
        out[..., ids, channel] *= gamma
    return out


def scale_direction(x: torch.Tensor, direction: torch.Tensor, gamma: float,
                    token_ids: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Scale the component along a unit direction, the channel-free analogue of
    :func:`scale_channel`.  Used for the energy-matched control that scales a
    direction rather than a coordinate."""
    out = x.clone()
    v = _unit(direction.to(x).reshape(1, -1))[0]
    if token_ids is None:
        z = out
    else:
        ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
        if not ids.numel():
            return out
        z = out[..., ids, :]
    alpha = (z.float() * v.float()).sum(-1, keepdim=True).to(z.dtype)
    edited = z + (gamma - 1.0) * alpha * v
    if token_ids is None:
        return edited
    out[..., torch.as_tensor(token_ids, device=x.device, dtype=torch.long), :] = edited
    return out


def remove_matched_energy(x: torch.Tensor, direction: torch.Tensor, token_ids: Sequence[int],
                          energy: float) -> torch.Tensor:
    """Remove a fixed squared magnitude along ``direction`` from frozen tokens.

    This is the equal-energy control: it takes exactly as much energy out of the
    residual as the intervention it controls for, but along a direction the
    register circuit does not use, so a reader cannot attribute the effect to
    perturbation size.
    """
    out = x.clone()
    ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
    if not ids.numel() or energy <= 0:
        return out
    v = _unit(direction.to(x).reshape(1, -1))[0]
    z = out[..., ids, :]
    alpha = (z.float() * v.float()).sum(-1, keepdim=True)
    share = float(energy) / max(int(ids.numel()), 1)
    # Shrink each token's component along v so the squared magnitude it loses is
    # `share`; a token holding less than that is zeroed along v rather than flipped.
    keep = (alpha.pow(2) - share).clamp_min(0.0).sqrt() * torch.sign(alpha)
    out[..., ids, :] = z + (keep - alpha).to(z.dtype) * v
    return out


def add_matched_energy(x: torch.Tensor, direction: torch.Tensor, token_ids: Sequence[int],
                       energy: float) -> torch.Tensor:
    """Add a fixed squared magnitude along an explicit control direction."""
    out = x.clone()
    ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
    if not ids.numel() or energy <= 0:
        return out
    v = _unit(direction.to(x).reshape(1, -1))[0]
    z = out[..., ids, :]
    alpha = (z.float() * v.float()).sum(-1, keepdim=True)
    share = float(energy) / max(int(ids.numel()), 1)
    enlarged = (alpha.pow(2) + share).sqrt() * torch.where(alpha < 0, -1.0, 1.0)
    out[..., ids, :] = z + (enlarged - alpha).to(z.dtype) * v
    return out


def refresh_direction(x: torch.Tensor, vstar: torch.Tensor, token_ids: Sequence[int],
                      clean_cosine: float = 1.0) -> torch.Tensor:
    """Rotate selected states toward v* while preserving their exact norms."""
    if not 0 <= clean_cosine <= 1:
        raise ValueError("clean_cosine must lie in [0, 1]")
    out = x.clone()
    ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
    if not ids.numel():
        return out
    z = out[..., ids, :]
    norms = z.float().norm(dim=-1, keepdim=True)
    v = _unit(vstar.to(z).reshape(1, -1))[0]
    perp = z - (z.float() * v.float()).sum(-1, keepdim=True).to(z.dtype) * v
    perp = _unit(perp)
    s = (1.0 - clean_cosine**2) ** 0.5
    out[..., ids, :] = norms.to(z.dtype) * (clean_cosine * v + s * perp)
    return out


def refresh_parallel_component(x: torch.Tensor, vstar: torch.Tensor,
                               token_ids: Sequence[int], alpha_target: float) -> torch.Tensor:
    """Set only ``x·v*`` while preserving every token's orthogonal component."""
    out = x.clone()
    ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
    if not ids.numel():
        return out
    v = _unit(vstar.to(x).reshape(1, -1))[0]
    z = out[..., ids, :]
    alpha = (z.float() * v.float()).sum(-1, keepdim=True).to(z.dtype)
    out[..., ids, :] = z + (torch.as_tensor(alpha_target, device=x.device, dtype=x.dtype) - alpha) * v
    return out


def rotate_toward_channel_preserve_norm(x: torch.Tensor, channel: int,
                                        token_ids: Sequence[int], gamma: float) -> torch.Tensor:
    """Increase one coordinate, then rescale the full token to its exact old norm."""
    out = scale_channel(x, channel, gamma, token_ids)
    ids = torch.as_tensor(token_ids, device=x.device, dtype=torch.long)
    if ids.numel():
        old_norm = x[..., ids, :].float().norm(dim=-1, keepdim=True)
        out[..., ids, :] = _unit(out[..., ids, :]) * old_norm.to(out.dtype)
    return out


def transplant(x: torch.Tensor, source: int, recipient: int, *, stage: str,
               vstar: Optional[torch.Tensor] = None, natural_norm: Optional[float] = None) -> torch.Tensor:
    """Residual-to-key transplant ladder, for residual-like representations.

    ``pre_key`` and ``final_key`` tensors should be patched at their advertised hook
    locations with :class:`ditsinks.interventions.FinalKeyIntervention`. This function
    refuses to treat a residual tensor as a key.
    """
    out = x.clone()
    src, dst = out[..., source, :], out[..., recipient, :]
    if stage == "vstar_ordinary_norm" or stage == "vstar_natural_norm":
        if vstar is None:
            raise ValueError("vstar is required for a direction transplant")
        scale = dst.float().norm(dim=-1, keepdim=True)
        if stage == "vstar_natural_norm":
            scale = (src.float().norm(dim=-1, keepdim=True) if natural_norm is None
                     else torch.full_like(scale, natural_norm))
        value = _unit(vstar.to(out).reshape(1, -1))[0] * scale.to(out.dtype)
    elif stage == "full_residual":
        value = src
    elif stage == "normalized_residual":
        value = _unit(src) * dst.float().norm(dim=-1, keepdim=True).to(out.dtype)
    else:
        raise ValueError(f"unsupported residual transplant stage: {stage}")
    out[..., recipient, :] = value
    return out


@dataclass(frozen=True)
class Recovery:
    first_layer: Optional[int]
    token: Optional[int]
    kind: str


def regeneration(projections: torch.Tensor, layers: Sequence[int], original_tokens: Iterable[int],
                 threshold: float) -> Recovery:
    """Classify first threshold crossing in ``[layer, token]`` projections."""
    original = set(map(int, original_tokens))
    for row, layer in zip(projections, layers):
        hits = torch.nonzero(row >= threshold).flatten()
        if hits.numel():
            token = int(hits[row[hits].argmax()])
            return Recovery(int(layer), token, "same_position" if token in original else "relocated")
    return Recovery(None, None, "none")


def decompose_lifecycle(states: torch.Tensor, vstar: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Lifecycle decomposition for ``[..., layer, token, channel]`` states."""
    v = _unit(vstar.to(states).reshape(1, -1))[0]
    alpha = (states.float() * v.float()).sum(-1)
    perp = states - alpha.to(states.dtype).unsqueeze(-1) * v
    cosine = alpha / states.float().norm(dim=-1).clamp_min(1e-12)
    angular_velocity = torch.zeros_like(cosine)
    if states.shape[-3] > 1:
        angular_velocity[..., 1:, :] = torch.acos(cosine[..., 1:, :].clamp(-1, 1)) - torch.acos(cosine[..., :-1, :].clamp(-1, 1))
    return {"alpha": alpha, "perpendicular_norm": perp.float().norm(dim=-1),
            "cosine": cosine, "angular_velocity": angular_velocity}
