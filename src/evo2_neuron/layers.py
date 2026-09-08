"""
Norm, MLP and embedding, ported 1:1 from `vortex/model/layers.py`.

The only divergence from Arc's source is that `TELinear` (TransformerEngine,
FP8-capable) becomes a plain `nn.Linear`. The parameter NAME and shape are kept
identical (`projections.weight`, `[3*hidden, hidden]`) so Arc's checkpoint loads
without a key map -- see `knowledge_base/op_rewrites.md`'s rule to keep the CUDA
module's parameter layout and replace only the computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """vortex `layers.RMSNorm`.

    Note the eps placement: it is added to the *norm*, outside the reciprocal,
    not inside a sqrt as in the more common formulation. Reproduced exactly --
    at Evo2's activation magnitudes the two differ measurably.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.eps = config.eps
        self.hidden_size = config.hidden_size
        self.scale = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
        return self.scale * y


class ParallelGatedMLP(nn.Module):
    """vortex `layers.ParallelGatedMLP` -- GLU with l1/l2 up, l3 down.

    `evo2_style_activations` makes every block past the first use `Identity` in
    place of the gate activation; only block 0 keeps gelu. That is not a
    simplification, it is what the checkpoint was trained with.

    `l1`/`l2`/`l3` are the modules the repo's LoRA regex targets
    (`.*blocks\\.\\d+\\.mlp\\.(l1|l2|l3)$`) and `l3`'s output is the read-out the
    CNN head consumes, so these names are load-bearing.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        act_type = config.get("mlp_activation", "gelu")
        if act_type == "gelu":
            self.act = F.gelu
        elif act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError(f"unknown mlp_activation {act_type!r}")

        if layer_idx > 0 and config.get("evo2_style_activations", False):
            self.act = nn.Identity()

        multiple_of = config.get("inner_size_multiple_of", 64) * config.model_parallel_size
        inner_size = int(2 * config.hidden_size * 4 / 3)
        inner_size = multiple_of * ((inner_size + multiple_of - 1) // multiple_of)
        inner_size = config.get("inner_mlp_size", inner_size)

        self.l1 = nn.Linear(config.hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(config.hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, config.hidden_size, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        return self.l3(self.act(self.l1(z)) * self.l2(z))


class VocabParallelEmbedding(nn.Embedding):
    """vortex `layers.VocabParallelEmbedding`, single-process branch only.

    Kept as an `nn.Embedding` subclass so the checkpoint key stays
    `embedding_layer.weight`. The distributed branch is dropped: Trainium
    sharding is not in scope and a `torch.distributed` call would not lower.
    """

    def __init__(self, config) -> None:
        super().__init__(
            config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.get("padding_idx", None),
        )

    def unembed(self, u: Tensor) -> Tensor:
        return u @ self.weight.T
