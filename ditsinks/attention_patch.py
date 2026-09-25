"""Observe attention without changing it.

Every diffusers model family funnels its attention through exactly one call --
`dispatch_attention_fn(q, k, v, ...)` for the FLUX families, and
`F.scaled_dot_product_attention(q, k, v, ...)` for the PixArt/`Attention` path.
By rebinding that name on the *module that calls it* we see the final queries
and keys (post QK-norm, post RoPE, exactly what produces the logits) while the
model still computes its output with its own kernel.

That matters: reimplementing a processor to expose probabilities changes the
numerics, and therefore the image. Here the forward pass is untouched -- the
statistics are computed alongside it, in fp32.
"""
from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Any, Callable, List, Literal, Optional, Tuple

import torch

# module path -> attribute holding the attention entry point, in priority order.
_PATCH_TARGETS = {
    "flux1": [("diffusers.models.transformers.transformer_flux", "dispatch_attention_fn")],
    "flux2": [("diffusers.models.transformers.transformer_flux2", "dispatch_attention_fn")],
    "pixart": [
        ("diffusers.models.attention_processor", "dispatch_attention_fn"),
        ("diffusers.models.attention", "dispatch_attention_fn"),
    ],
}

# Families whose processors may instead call torch.nn.functional directly. We
# then swap the module's `F` alias for a shim (never torch.nn.functional itself).
_F_FALLBACK_MODULES = {
    "flux1": ["diffusers.models.transformers.transformer_flux"],
    "flux2": ["diffusers.models.transformers.transformer_flux2"],
    "pixart": ["diffusers.models.attention_processor", "diffusers.models.attention"],
}


class _FunctionalShim:
    """Proxies torch.nn.functional, intercepting scaled_dot_product_attention."""

    def __init__(self, real, callback: Callable, key_transform: Optional[Callable] = None,
                 kv_transform: Optional[Callable] = None, heads: Optional[int] = None,
                 value_transform: Optional[Callable] = None):
        self._real = real
        self._callback = callback
        self._key_transform = key_transform
        self._value_transform = value_transform
        self._kv_transform = kv_transform
        self._heads = heads

    def __getattr__(self, name):
        return getattr(self._real, name)

    def scaled_dot_product_attention(self, query, key, value, *args, **kwargs):
        original_key = key
        if self._key_transform is not None:
            key = self._key_transform(query, key, value, kwargs)
        if self._value_transform is not None:
            value = self._value_transform(query, original_key, value, kwargs)
        if self._kv_transform is not None:
            key, value, kwargs = apply_kv_transform(
                self._kv_transform, query, key, value, kwargs, args, heads=self._heads)
            args = ()
        out = self._real.scaled_dot_product_attention(query, key, value, *args, **kwargs)
        if args:
            kwargs = dict(kwargs)
            kwargs.setdefault("attn_mask", args[0])
        self._callback(query, key, value, kwargs)
        return out


def sequence_axis(x: torch.Tensor, heads: Optional[int] = None) -> int:
    r"""Which axis of a 4-D attention tensor carries the sequence.

    diffusers hands the FLUX kernel :math:`[B, S, H, D]` and the generic ``Attention``
    path :math:`[B, H, S, D]`, so the sequence is axis 1 in one and axis 2 in the other.
    :func:`to_bhsd` exists to normalise that for *reading*; this resolves it for
    *writing*, where transposing is not an option -- the kernel must receive the layout
    its own processor built.

    Resolved from the head count where one is known, and otherwise from the fact that an
    attention tensor always has far more sequence positions than heads. Getting this
    wrong appends to the head axis, which is how the first version of this failed.
    """
    if x.ndim != 4:
        raise ValueError(f"expected a 4-D attention tensor, got shape {tuple(x.shape)}")
    _, d1, d2, _ = x.shape
    if heads:
        if d2 == int(heads) and d1 != int(heads):
            return 1
        if d1 == int(heads) and d2 != int(heads):
            return 2
    if d1 == d2:
        raise ValueError(
            f"axes 1 and 2 are both {d1}, so the sequence axis cannot be resolved from "
            "the shape. Pass the head count explicitly.")
    return 1 if d1 > d2 else 2


def apply_kv_transform(transform: Callable, query, key, value, kwargs, args=(), *,
                       heads: Optional[int] = None):
    r"""Append extra keys and values supplied by ``transform``, on the sequence axis.

    This is the one seam in the model where a *non-spatial* memory slot can be added.
    The queries are untouched, so the attention output keeps its sequence length exactly:
    softmax over :math:`[\ldots, S_q, S_{kv}+R]` times :math:`[\ldots, S_{kv}+R, D]` is
    still :math:`[\ldots, S_q, D]`.  Nothing downstream sees a shape change, and no image
    patch corresponds to the new rows -- they exist only as attention targets.

    It is also *after* QK-norm and RoPE, which is what makes the slots genuinely
    non-spatial: the model's positional encoding has already been applied to the real
    keys, so a row added here carries no position at all.  Appending before RoPE would
    give the slot a spatial address and defeat the purpose.

    **The transform returns only the rows to add**, not the concatenated result. The
    layout differs between model families and transposing is not an option here, so the
    axis is resolved once, in this function, and a caller cannot get it wrong. The first
    version of this took a concatenated tensor and appended to the head axis instead.
    """
    result = transform(query, key, value, kwargs)
    if result is None:
        return key, value, kwargs
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("a kv transform must return (extra_key, extra_value) or None")
    extra_key, extra_value = result
    if not torch.is_tensor(extra_key) or not torch.is_tensor(extra_value):
        raise ValueError("a kv transform must return tensors")
    axis = sequence_axis(key, heads)
    for name, extra, base in (("key", extra_key, key), ("value", extra_value, value)):
        if extra.ndim != base.ndim:
            raise ValueError(f"the extra {name} has {extra.ndim} axes, not {base.ndim}")
        for position in range(base.ndim):
            if position == axis:
                continue
            if int(extra.shape[position]) != int(base.shape[position]):
                raise ValueError(
                    f"the extra {name} may only differ from the {name}s on the sequence "
                    f"axis {axis}: got {tuple(extra.shape)} against {tuple(base.shape)}")
    added = int(extra_key.shape[axis])
    if added != int(extra_value.shape[axis]):
        raise ValueError(
            f"the appended keys and values must be the same length: {added} keys "
            f"against {int(extra_value.shape[axis])} values")
    if added <= 0:
        return key, value, kwargs
    new_key = torch.cat([key, extra_key.to(device=key.device, dtype=key.dtype)], dim=axis)
    new_value = torch.cat(
        [value, extra_value.to(device=value.device, dtype=value.dtype)], dim=axis)

    merged = dict(kwargs)
    mask = merged.get("attn_mask", args[0] if args else None)
    if mask is not None:
        # A mask has to grow with the keys or the call is silently wrong, and guessing at
        # a layout would mean masking the wrong tokens. FLUX passes None here.
        if not torch.is_tensor(mask):
            raise ValueError(
                "this attention call carries a non-tensor mask, so appended keys cannot "
                "be masked correctly. Virtual registers are refused rather than added to "
                "a call whose mask cannot be widened.")
        if int(mask.shape[-1]) != int(key.shape[axis]):
            raise ValueError(
                f"the attention mask's last axis is {int(mask.shape[-1])} but the keys "
                f"are {int(key.shape[axis])} long, so it is not a per-key mask this can "
                "widen. Virtual registers are refused rather than guessed at.")
        # Visible to every query: a register nothing may attend to is not a register.
        pad = (torch.ones((*mask.shape[:-1], added), dtype=torch.bool, device=mask.device)
               if mask.dtype == torch.bool else
               torch.zeros((*mask.shape[:-1], added), dtype=mask.dtype,
                           device=mask.device))
        merged["attn_mask"] = torch.cat([mask, pad], dim=-1)
    return new_key, new_value, merged


class AttentionTap:
    """Observe or selectively replace final keys without replacing the kernel.

    ``instrument`` and ``identity`` are deliberately inert validation modes.
    ``patch`` applies ``key_transform`` and/or ``value_transform`` immediately before
    the real model kernel; the callback then observes the exact query/key/value supplied
    to that kernel.

    **Why both transforms live in ONE tap.** The value-dependence test asks what changes
    when the attention distribution is held fixed and only the value differs. That is
    only meaningful if the logits are provably identical between the two runs, which
    requires the key patch and the value substitution to be applied at the same call, in
    a fixed order, by the same object. Splitting them across two taps would make the
    result depend on which tap entered last -- an ordering that is invisible at the call
    site and has already produced one wrong answer in this project.
    """

    def __init__(self, family: str, callback: Callable[..., None], *,
                 mode: Literal["instrument", "identity", "patch", "append_kv"] = "instrument",
                 key_transform: Optional[Callable[..., torch.Tensor]] = None,
                 value_transform: Optional[Callable[..., torch.Tensor]] = None,
                 kv_transform: Optional[Callable[..., Any]] = None,
                 heads: Optional[int] = None):
        if mode == "patch" and key_transform is None and value_transform is None:
            raise ValueError("patch mode requires key_transform, value_transform, or both")
        if mode not in ("patch",) and key_transform is not None:
            raise ValueError("key_transform is only valid in patch mode")
        if mode not in ("patch",) and value_transform is not None:
            raise ValueError("value_transform is only valid in patch mode")
        if mode == "append_kv" and kv_transform is None:
            raise ValueError("append_kv mode requires kv_transform")
        if mode != "append_kv" and kv_transform is not None:
            raise ValueError("kv_transform is only valid in append_kv mode")
        self.family = family
        self.callback = callback
        self.mode = mode
        self.key_transform = key_transform
        self.value_transform = value_transform
        self.kv_transform = kv_transform
        self.heads = heads
        self._undo: List[Tuple[object, str, object]] = []
        self.patched: List[str] = []

    def __enter__(self) -> "AttentionTap":
        for mod_path, attr in _PATCH_TARGETS.get(self.family, []):
            mod = _try_import(mod_path)
            if mod is None or not hasattr(mod, attr):
                continue
            real = getattr(mod, attr)

            def _wrapped(query, key, value, *args, _real=real, **kwargs):
                if self.mode == "patch":
                    # Key first, then value, always. The value transform is handed the
                    # ORIGINAL key so that what it substitutes cannot depend on whether a
                    # key patch ran -- which is what lets two runs share a distribution.
                    original_key = key
                    if self.key_transform is not None:
                        replacement = self.key_transform(query, key, value, kwargs)
                        if not torch.is_tensor(replacement) or replacement.shape != key.shape:
                            raise ValueError("final-key transform must preserve key tensor shape")
                        key = replacement
                    if self.value_transform is not None:
                        substitute = self.value_transform(query, original_key, value, kwargs)
                        if not torch.is_tensor(substitute) or substitute.shape != value.shape:
                            raise ValueError("value transform must preserve value tensor shape")
                        value = substitute
                elif self.mode == "append_kv":
                    key, value, kwargs = apply_kv_transform(
                        self.kv_transform, query, key, value, kwargs, args,
                        heads=self.heads)
                    args = ()
                out = _real(query, key, value, *args, **kwargs)
                self.callback(query, key, value, kwargs)
                return out

            setattr(mod, attr, _wrapped)
            self._undo.append((mod, attr, real))
            self.patched.append(f"{mod_path}.{attr}")

        if not self.patched:
            for mod_path in _F_FALLBACK_MODULES.get(self.family, []):
                mod = _try_import(mod_path)
                if mod is None or not hasattr(mod, "F"):
                    continue
                real = getattr(mod, "F")
                setattr(mod, "F", _FunctionalShim(
                    real, self.callback,
                    self.key_transform if self.mode == "patch" else None,
                    self.kv_transform if self.mode == "append_kv" else None,
                    self.heads,
                    self.value_transform if self.mode == "patch" else None))
                self._undo.append((mod, "F", real))
                self.patched.append(f"{mod_path}.F.scaled_dot_product_attention")

        if not self.patched:
            raise RuntimeError(
                f"Could not find an attention entry point to observe for family {self.family!r}. "
                "This usually means the installed diffusers version moved it; check "
                "ditsinks.attention_patch._PATCH_TARGETS against your diffusers build."
            )
        return self

    def __exit__(self, *exc):
        for mod, attr, real in reversed(self._undo):
            setattr(mod, attr, real)
        self._undo.clear()
        return False


def _try_import(path: str):
    try:
        return importlib.import_module(path)
    except Exception:
        return None


def to_bhsd(x: torch.Tensor, heads: int, seq_len: Optional[int] = None) -> torch.Tensor:
    """Normalise a 4-D attention tensor to [B, H, S, D].

    diffusers uses [B, S, H, D] in the FLUX processors and [B, H, S, D] in the
    generic `Attention` path, so the layout is resolved from the head count
    (and, when that is ambiguous, from the known sequence length).
    """
    if x.ndim != 4:
        raise ValueError(f"expected a 4-D attention tensor, got shape {tuple(x.shape)}")
    _, d1, d2, _ = x.shape
    if d1 == heads and d2 != heads:
        return x
    if d2 == heads and d1 != heads:
        return x.transpose(1, 2)
    if seq_len is not None:
        if d2 == seq_len and d1 != seq_len:
            return x
        if d1 == seq_len and d2 != seq_len:
            return x.transpose(1, 2)
    # Square case (S == H): fall back to the FLUX layout, which is the common one.
    return x.transpose(1, 2) if d2 == heads else x


@contextmanager
def null_tap():
    yield None
