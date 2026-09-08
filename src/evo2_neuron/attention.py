"""
Multi-head self-attention, ported from `vortex/model/attention.py`.

Arc's `MHA` picks `FlashSelfAttention` when `use_flash_attn` is set and
`SelfAttention` otherwise.

**The reference used FlashSelfAttention, not this path.** An earlier revision of
this docstring claimed flash-attn was absent from the H200 env and that
`SelfAttention` was therefore the reference path. That was wrong, inherited from
`2026-07-31_evo2-7b-trainium-lora-alignment.md`. Measured on the H200
(`2026-08-03_vortex-cpu-cuda-findings.md`): the `flash_attn` *package* is absent,
but vortex vendors its own extension and `import flash_attn_2_cuda` succeeds once
torch is loaded -- the earlier failure was a load-order artifact. `vortex.ops`
imports and `use_flash_attn: True`, so the reference ran `FlashSelfAttention`.

So this eager implementation IS a substitution, and a real CUDA-only path the
port replaces. Both are plain causal attention and should agree to rounding, but
that is an assumption this port has not independently verified against
`FlashSelfAttention`.

Only the QKV/out projections and the rotary buffer carry parameters, and their
names (`Wqkv.weight`, `out_proj.{weight,bias}`, `rotary_emb.inv_freq`) match
Arc's checkpoint exactly, so no key mapping is needed.

**QKV layout.** Arc packs `Wqkv`'s output three-major -- `(three, heads, dim)` --
and unpacks with `rearrange(qkv, "... (three h d) -> ... three h d", three=3)` --
i.e. **three-major**, which `MHA.forward` reproduces with
`qkv.unflatten(-1, (3, num_heads, head_dim))`.

Arc's checkpoint, however, stores `Wqkv` **head-major** `(heads, 3, head_dim)`.
vortex converts it at load time (`vortex/model/model.py:880-904`, gated on
`column_split: True`) and so does this port -- see
`loading.convert_wqkv_column_split`. Skipping the conversion scrambles q/k/v
across heads in every attention block (evo2-1b has four, evo2-7b five) and moves
`blocks.10.mlp.l3` by
rel_err ~1.37 (measured on evo2-1b). (This is also why the `Taykhoom/Evo2-1B-8K` checkpoint looks like
a permutation of Arc's: it stores the post-conversion weights.)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from evo2_neuron.rotary import RotaryEmbedding


def causal_self_attention(qkv: Tensor, use_sdpa: bool = True) -> Tensor:
    """vortex `attention.SelfAttention.forward`, prefill only.

    `qkv` is (B, S, 3, H, D); returns (B, S, H, D).

    `use_sdpa=False` selects an explicit softmax formulation. It is
    mathematically the same thing and exists because the fused SDPA *backward*
    is a known Neuron failure at longer sequence lengths
    (`knowledge_base/error_codes.md`: `NCC_INLA001` at seq 4096, even in eager).
    The two are checked against each other on CPU before either goes on device.
    """
    q, k, v = qkv.unbind(dim=2)
    q = q.permute(0, 2, 1, 3)  # (B, H, S, D)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    if use_sdpa:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        seqlen = q.shape[-2]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        # Additive -10000.0 rather than -inf: matches vortex's own masking
        # constant and avoids inf*0 -> nan in the backward on device.
        mask = torch.full((seqlen, seqlen), -10000.0, device=q.device, dtype=scores.dtype)
        mask = torch.triu(mask, diagonal=1)
        out = torch.softmax(scores + mask, dim=-1) @ v

    return out.permute(0, 2, 1, 3)


class MHA(nn.Module):
    """vortex `attention.MHA`, restricted to self-attention prefill.

    Dropped, none of which is on the Evo2 prefill path: cross-attention, MQA/GQA
    (`proj_groups` is 1), ALiBi, sliding window, dwconv, fused dense, KV cache.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        rotary_emb_dim: int,
        rotary_emb_base: float,
        qkv_proj_bias: bool,
        out_proj_bias: bool,
        layer_idx: int | None = None,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_sdpa = use_sdpa

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_proj_bias)
        self.rotary_emb = RotaryEmbedding(rotary_emb_dim, base=rotary_emb_base)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (3, self.num_heads, self.head_dim))

        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(qkv)

        context = causal_self_attention(qkv, use_sdpa=self.use_sdpa)

        return self.out_proj(context.flatten(-2, -1))
