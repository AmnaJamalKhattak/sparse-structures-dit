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
from typing import Callable, List, Optional, Tuple

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

    def __init__(self, real, callback: Callable):
        self._real = real
        self._callback = callback

    def __getattr__(self, name):
        return getattr(self._real, name)

    def scaled_dot_product_attention(self, query, key, value, *args, **kwargs):
        out = self._real.scaled_dot_product_attention(query, key, value, *args, **kwargs)
        if args:
            kwargs = dict(kwargs)
            kwargs.setdefault("attn_mask", args[0])
        self._callback(query, key, value, kwargs)
        return out


class AttentionTap:
    """Context manager that reports (query, key, value) for every attention call.

    The callback runs *after* the real kernel, so an exception in the callback
    cannot corrupt a generation half-way through; it is re-raised as usual.
    """

    def __init__(self, family: str, callback: Callable[..., None]):
        self.family = family
        self.callback = callback
        self._undo: List[Tuple[object, str, object]] = []
        self.patched: List[str] = []

    def __enter__(self) -> "AttentionTap":
        for mod_path, attr in _PATCH_TARGETS.get(self.family, []):
            mod = _try_import(mod_path)
            if mod is None or not hasattr(mod, attr):
                continue
            real = getattr(mod, attr)

            def _wrapped(query, key, value, *args, _real=real, **kwargs):
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
                setattr(mod, "F", _FunctionalShim(real, self.callback))
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
