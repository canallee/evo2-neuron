"""
Rotary position embeddings, ported from `vortex/model/rotary.py`.

Arc's source imports flash-attn's Triton `apply_rotary`; the pure-PyTorch
`apply_rotary_emb_torch` fallback in the same file is what runs when that is
unavailable, which is the case on this box.

**It is NOT what the reference ran.** flash-attn IS available on the H200
(measured 2026-08-03; the earlier "not installed" claim was a load-order
artifact), so the reference used the Triton kernel. This is a substitution.
Measured cost of removing rotary entirely on CUDA at 200 bp is cosine 0.978, so
the path matters; the two implementations are expected to agree to rounding, but
this port has not verified that directly.

One detail worth stating because it looks like a bug and is not: the checkpoint
stores `rotary_emb.inv_freq` in **bf16**, but vortex ignores those values --
`_update_cos_sin_cache` recomputes `inv_freq` in fp32 whenever the buffer is not
already fp32 (vortex/model/rotary.py:446-449). We do the same, so the stored
bf16 buffer is loaded for key-compatibility and never used numerically.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Non-interleaved (GPT-NeoX) halves layout; `rotary_emb_interleaved` is False."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """`x` is (B, S, H, D); `cos`/`sin` are (S, D/2)."""
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]

    # (S, D/2) -> (S, 1, D) by duplicating each frequency across the two halves.
    cos = cos.repeat(1, 2).unsqueeze(1)
    sin = sin.repeat(1, 2).unsqueeze(1)

    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin, x[..., ro_dim:]],
        dim=-1,
    )


class RotaryEmbedding(nn.Module):
    """vortex `rotary.RotaryEmbedding`, prefill-only (no `scale_base`/xPos)."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=True)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _compute_inv_freq(self, device=None) -> Tensor:
        """`1 / base^(2i/dim)`, written as `exp(-2i/dim * log base)`.

        The obvious `base ** tensor` form is `aten::pow.Scalar`, which has no
        Neuron lowering and silently runs on the host -- measured: it is the only
        fallback left in a bf16 run (in fp32 it never fires, because the loaded
        `inv_freq` buffer is already fp32 and vortex's cache short-circuits the
        recompute). `exp`/`mul` lower cleanly and are exact to fp32 here.
        """
        exponents = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim

        return torch.exp(-exponents * math.log(self.base))

    def _update_cos_sin_cache(self, seqlen: int, device, dtype) -> None:
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            # See module docstring: recomputed in fp32, matching vortex, because
            # the checkpoint's inv_freq buffer is bf16.
            inv_freq = (
                self.inv_freq
                if self.inv_freq.dtype == torch.float32
                else self._compute_inv_freq(device=device)
            )
            freqs = torch.outer(t, inv_freq.to(device))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, qkv: Tensor) -> Tensor:
        """`qkv` is (B, S, 3, H, D); rotary is applied to q and k, not v."""
        seqlen = qkv.shape[1]
        self._update_cos_sin_cache(seqlen, qkv.device, qkv.dtype)

        cos, sin = self._cos_cached[:seqlen], self._sin_cached[:seqlen]
        q = apply_rotary_emb_torch(qkv[:, :, 0], cos, sin)
        k = apply_rotary_emb_torch(qkv[:, :, 1], cos, sin)

        # Rebuilt with `stack` rather than an in-place write into `qkv`: an
        # in-place update of a tensor that autograd already saved is a known
        # Neuron failure mode ("modified by an inplace operation").
        return torch.stack([q, k, qkv[:, :, 2]], dim=2)
