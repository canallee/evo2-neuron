"""
Objectives: turn a head's output plus labels into a scalar loss.

The loss is a property of the *label*, not of the model, so it lives here rather
than on the head. Every objective returns the same dict:

    {"prediction": Tensor, "loss": Tensor | None}

`loss` is `None` when no labels were supplied, which is what makes the same
`forward` usable for inference and for training without a second code path.

Pairings that make sense:

| objective                   | head                            | labels                    |
|-----------------------------|---------------------------------|---------------------------|
| `next_token_ce`             | (none — uses the tied LM head)  | `input_ids` itself        |
| `sequence_regression`       | `MLPReadoutHead` / `ConvReadoutHead` | `(B, out_dim)` float  |
| `sequence_classification`   | `MLPReadoutHead` / `ConvReadoutHead` | `(B,)` int64          |
| `window_regression`         | `WindowTrackHead`               | `(B, n_win, out_dim)`     |
| `per_position_classification` | `PerPositionHead`             | `(B, L)` int64            |
| `per_position_regression`   | `PerPositionHead`               | `(B, L, out_dim)`         |

Masked variants take a `(B, ...)` weight of 1/0 and normalise by its sum, so a
batch of mostly-padding does not report an artificially small loss.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "next_token_ce",
    "sequence_regression",
    "sequence_classification",
    "window_regression",
    "per_position_classification",
    "per_position_regression",
    "OBJECTIVE_REGISTRY",
    "get_objective",
]

Result = dict[str, Any]


def _masked_mean(per_element: Tensor, weight: Tensor | None) -> Tensor:
    if weight is None:
        return per_element.mean()

    weight = weight.to(per_element.dtype)

    while weight.dim() < per_element.dim():
        weight = weight.unsqueeze(-1)

    return (per_element * weight).sum() / weight.expand_as(per_element).sum().clamp_min(1.0)


def next_token_ce(logits: Tensor, *, input_ids: Tensor,
                  attention_mask: Tensor | None = None, **_: Any) -> Result:
    """Evo 2's own autoregressive objective: predict byte `t+1` from `1..t`.

    Needs no labels beyond the sequence, which makes it the objective to reach
    for when adapting the backbone to a new genome or clade before there is any
    downstream task. `logits` is `(B, L, vocab)` from `Evo2ArcForCausalLM`.

    The shift drops the last position (nothing to predict) and the first label
    (nothing predicts it), so an `attention_mask` is sliced to match.
    """
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.shape)

    weight = None if attention_mask is None else attention_mask[:, 1:]

    return {"prediction": logits, "loss": _masked_mean(per_token, weight)}


def sequence_regression(prediction: Tensor, *, labels: Tensor | None = None,
                        weights: Tensor | None = None, **_: Any) -> Result:
    """Mean squared error against `(B, out_dim)` continuous targets."""
    if labels is None:
        return {"prediction": prediction, "loss": None}

    per_element = (prediction.float() - labels.to(prediction.device).float()) ** 2

    return {"prediction": prediction, "loss": _masked_mean(per_element, weights)}


def sequence_classification(prediction: Tensor, *, labels: Tensor | None = None,
                            weights: Tensor | None = None, **_: Any) -> Result:
    """Cross-entropy over `out_dim` classes, one label per sequence.

    Binary tasks: use `out_dim=2` rather than a single logit with BCE. It costs
    4096 extra head parameters and removes a whole class of threshold bugs.
    """
    if labels is None:
        return {"prediction": prediction, "loss": None}

    per_row = F.cross_entropy(
        prediction.float(), labels.to(prediction.device).long(), reduction="none"
    )

    return {"prediction": prediction, "loss": _masked_mean(per_row, weights)}


def window_regression(prediction: Tensor, *, labels: Tensor | None = None,
                      weights: Tensor | None = None, **_: Any) -> Result:
    """MSE against `(B, n_windows, out_dim)` per-window targets.

    `weights` is `(B, n_windows)` — 0 for windows with no measurement, so an
    unobserved bin contributes nothing rather than being trained toward zero.
    """
    if labels is None:
        return {"prediction": prediction, "loss": None}

    per_element = (prediction.float() - labels.to(prediction.device).float()) ** 2

    return {"prediction": prediction, "loss": _masked_mean(per_element, weights)}


def per_position_classification(prediction: Tensor, *, labels: Tensor | None = None,
                                weights: Tensor | None = None, **_: Any) -> Result:
    """Cross-entropy at every nucleotide. `prediction` `(B, L, C)`, `labels` `(B, L)`.

    `weights` is `(B, L)` — typically the attention mask, so padding is excluded
    from the mean rather than scored against whatever class id happens to sit in
    the padded label slots.
    """
    if labels is None:
        return {"prediction": prediction, "loss": None}

    labels = labels.to(prediction.device).long()
    per_position = F.cross_entropy(
        prediction.reshape(-1, prediction.shape[-1]).float(),
        labels.reshape(-1),
        reduction="none",
    ).view(labels.shape)

    return {"prediction": prediction, "loss": _masked_mean(per_position, weights)}


def per_position_regression(prediction: Tensor, *, labels: Tensor | None = None,
                            weights: Tensor | None = None, **_: Any) -> Result:
    """MSE at every nucleotide. `prediction` and `labels` are `(B, L, out_dim)`."""
    if labels is None:
        return {"prediction": prediction, "loss": None}

    per_element = (prediction.float() - labels.to(prediction.device).float()) ** 2

    return {"prediction": prediction, "loss": _masked_mean(per_element, weights)}


OBJECTIVE_REGISTRY: dict[str, Callable[..., Result]] = {
    "next-token-ce": next_token_ce,
    "sequence-regression": sequence_regression,
    "sequence-classification": sequence_classification,
    "window-regression": window_regression,
    "per-position-classification": per_position_classification,
    "per-position-regression": per_position_regression,
}

#: Which head each objective expects. `next-token-ce` uses the tied LM head.
OBJECTIVE_HEADS: dict[str, str | None] = {
    "next-token-ce": None,
    "sequence-regression": "mlp",
    "sequence-classification": "mlp",
    "window-regression": "window",
    "per-position-classification": "per-position",
    "per-position-regression": "per-position",
}


def get_objective(name: str) -> Callable[..., Result]:
    try:
        return OBJECTIVE_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown objective {name!r}; known: {sorted(OBJECTIVE_REGISTRY)}"
        ) from None
