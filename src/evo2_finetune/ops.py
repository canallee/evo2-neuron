"""
Device-lowerable primitives for the ops that otherwise fall back to the host.

`torch_neuronx.get_fallback_ops()` after a full forward + backward +
`optimizer.step()` reports ops that ran on the CPU instead of the device. They
are numerically correct — they just round-trip through the host, which is
invisible in the outputs and expensive in the step time. Two of the three fixes
below are for exactly that; the third is a hard compile failure.

Every rewrite here is an algebraic identity, and every one is validated against
the naive form on CPU in `tests/test_ops.py`. The naive forms are kept in this
module (prefixed `naive_`) so that test has something to compare against.

Unlike a monkey-patching context manager, these are used *directly* by
`heads.py`, on every device. The CPU arm of a parity run therefore exercises the
identical code path as the Neuron arm — a rewrite that is only active on device
is a rewrite whose CPU validation proves nothing.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "first_set_index",
    "gather_window",
    "logaddexp",
    "windowed_conv_as_matmul",
    "naive_first_set_index",
    "naive_logaddexp",
]


# ---------------------------------------------------------------------------
# aten::argmax.out  ->  cumsum / clamp / sum
# ---------------------------------------------------------------------------

def naive_first_set_index(mask: Tensor) -> Tensor:
    """The obvious form. Falls back to the host on Neuron, silently."""
    return mask.argmax(dim=1)


def first_set_index(mask: Tensor) -> Tensor:
    """Index of the first 1 in each row of `mask`, without `argmax`.

    Requires the 1s to be **contiguous**, which is what makes the identity hold:

        cumsum(mask).clamp(max=1)  ->  0 before the first 1, 1 from it onward
        (1 - that).sum(1)          ->  how many positions precede the first 1
                                       == the index of the first 1

    cumsum / clamp / sum all lower to the device. Returns int64 to match
    `argmax`, because the result indexes a `gather`.

    An all-zero row yields `mask.shape[1]`, not 0 — `argmax` would return 0 and
    silently gather from the start of the sequence. Callers that can produce an
    empty mask should check; this at least makes the failure indexable.
    """
    seen = mask.cumsum(dim=1).clamp(max=1)

    return (1 - seen).sum(dim=1).to(torch.int64)


def gather_window(hidden: Tensor, mask: Tensor, width: int) -> Tensor:
    """Extract `(B, width, D)` starting at the first 1 of each row of `mask`."""
    starts = first_set_index(mask)
    offsets = torch.arange(width, device=hidden.device)
    positions = (starts.unsqueeze(1) + offsets.unsqueeze(0)).unsqueeze(-1)

    return hidden.gather(dim=1, index=positions.expand(-1, -1, hidden.shape[-1]))


# ---------------------------------------------------------------------------
# aten::logaddexp.out  ->  max-shift identity
# ---------------------------------------------------------------------------

def naive_logaddexp(a: Tensor, b: Tensor) -> Tensor:
    """The fused op. Falls back to the host on Neuron, silently."""
    return torch.logaddexp(a, b)


def logaddexp(a: Tensor, b: Tensor) -> Tensor:
    """`log(e^a + e^b) = max(a,b) + log(1 + e^-|a-b|)`.

    Stable for the same reason the fused op is: the exponent is always <= 0, so
    `exp` cannot overflow, and the correction lies in (0, log 2].

    `log(1 + x)` rather than `log1p(x)` because **`log1p` does not lower either**
    — rewriting in terms of it merely moves the fallback from
    `aten::logaddexp.out` to `aten::log1p.out` (measured). The accuracy `log1p`
    protects is not needed here: where `x` is small enough for `log(1+x)` to lose
    it, the term is already negligible against `max(a,b)`.

    Used by any log-sum-exp objective. Not on the default path of the objectives
    shipped here; kept because it is the second of the two silent fallbacks and
    the next objective someone writes will want it.
    """
    return torch.maximum(a, b) + torch.log(1.0 + torch.exp(-(a - b).abs()))


# ---------------------------------------------------------------------------
# aten::convolution_backward  ->  addmm     (a HARD failure, not a fallback)
# ---------------------------------------------------------------------------

def windowed_conv_as_matmul(features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """A `Conv1d` whose kernel spans the entire input, expressed as one `addmm`.

    `features` is `(B, C, W)`, `weight` is `(out, C, W)`, and the kernel covers
    all `W` positions, so the convolution produces exactly one output position:

        out[b, o] = sum_{c,k} weight[o, c, k] * features[b, c, k] + bias[o]

    Flattening both operands in the same `(channel, position)` order makes that a
    single matmul. The reduction order differs from `F.conv1d`'s; the math does
    not.

    This is not a preference. The forward compiles fine; the **backward** does
    not:

        RuntimeError: Compilation error occurred on Neuron for
          operation=aten::convolution_backward;
          [NCC_ITIN902] TensorInitialization error

    and `NEURON_CC_FLAGS=--optlevel=1`, the documented workaround for that class,
    does not clear it for these shapes. `addmm` lowers cleanly and has a
    well-supported backward.
    """
    return torch.addmm(bias, features.flatten(1), weight.flatten(1).t())
