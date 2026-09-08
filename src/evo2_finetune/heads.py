"""
Read-out heads: turn the backbone's per-position hidden states into predictions.

All four are task-agnostic — they emit `(B, ...)` real numbers, and
`objectives.py` decides whether those are logits or a regression target. Pick by
the granularity of the label you have:

| label granularity      | head              | output shape        |
|------------------------|-------------------|---------------------|
| one per sequence       | `MLPReadoutHead`  | `(B, out_dim)`      |
| one per sequence, but position matters | `ConvReadoutHead` | `(B, out_dim)` |
| one per fixed-size window (e.g. 256 bp) | `WindowTrackHead` | `(B, n_win, out_dim)` |
| one per nucleotide     | `PerPositionHead` | `(B, L, out_dim)`   |

Every head takes `(hidden, mask)`. `mask` is `(B, L)` with 1 at valid positions:
the attention mask for the pooled heads, a window mask for `ConvReadoutHead`.

Sizing note, because it surprises people: `ConvReadoutHead` at `width=200`,
`input_dim=4096`, `hidden_dim=64` is 52.4 M parameters — an order of magnitude
more than the LoRA adapters it sits on top of. `MLPReadoutHead` at the same
widths is 0.26 M. If the head dominates your trainable count, that is the reason.
"""

from __future__ import annotations

from torch import Tensor, nn

from evo2_finetune.ops import gather_window, windowed_conv_as_matmul

__all__ = [
    "MLPReadoutHead",
    "ConvReadoutHead",
    "WindowTrackHead",
    "PerPositionHead",
    "HEAD_REGISTRY",
    "build_head",
]


def _mlp(hidden_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = [nn.ReLU()]

    for _ in range(depth):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]

    return nn.Sequential(*layers[:-1])


class MLPReadoutHead(nn.Module):
    """Masked mean-pool -> MLP -> `out_dim`. Sequence-level regression/classification."""

    def __init__(self, input_dim: int, out_dim: int, *, hidden_dim: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.project = nn.Linear(input_dim, hidden_dim)
        self.mlp = _mlp(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = pooled.to(self.project.weight.dtype)

        return self.output_layer(self.mlp(self.project(pooled)))


class ConvReadoutHead(nn.Module):
    """Gather a `width`-position window, convolve it, MLP, project to `out_dim`.

    The kernel spans the whole window, so the convolution collapses to a single
    summary vector per row — equivalent to a position-specific
    `Linear(width * input_dim, hidden_dim)`, but with the spatial structure made
    explicit. Strictly more expressive than mean-pooling: positions are not
    averaged away.

    `mask` must carry exactly `width` **contiguous** 1s per row, marking the
    region of interest inside a possibly much longer context.

    The conv is evaluated as one `addmm` (see `ops.windowed_conv_as_matmul`) —
    `aten::convolution_backward` is a hard compile failure on Neuron. `self.conv`
    still exists and still owns the parameters, so a checkpoint is interchangeable
    with a plain `nn.Conv1d` implementation and CPU-only users can compare
    against `forward_reference`.
    """

    def __init__(
        self, input_dim: int, out_dim: int, *, width: int, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.width = width
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=width)
        self.mlp = _mlp(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def _features(self, hidden: Tensor, mask: Tensor) -> Tensor:
        window = gather_window(hidden, mask, self.width)

        return window.to(self.conv.weight.dtype).transpose(1, 2)   # (B, C, width)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        features = self._features(hidden, mask)

        if features.shape[-1] == self.width:
            summary = windowed_conv_as_matmul(features, self.conv.weight, self.conv.bias)
        else:
            # The collapse-to-one-position equivalence does not hold. Fall back
            # rather than silently computing something else.
            summary = self.conv(features).squeeze(-1)

        return self.output_layer(self.mlp(summary))

    def forward_reference(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """`forward` with the real `nn.Conv1d`. For CPU equivalence tests only."""
        summary = self.conv(self._features(hidden, mask)).squeeze(-1)

        return self.output_layer(self.mlp(summary))


class WindowTrackHead(nn.Module):
    """Bin the sequence into fixed `window` chunks and predict `out_dim` per bin.

    Mean-pools each window, then a shared MLP. `(B, L, D) -> (B, L // window,
    out_dim)`. This is the shape most genome-track objectives want: a prediction
    every 128 or 256 bp rather than one per sequence or one per base.

    `L` must be divisible by `window`. Deliberately strict — the alternative is
    padding, and a partial trailing window is a silently-downweighted label. Pad
    or crop upstream where the choice is visible.
    """

    def __init__(
        self, input_dim: int, out_dim: int, *, window: int = 256, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.window = window
        self.project = nn.Linear(input_dim, hidden_dim)
        self.mlp = _mlp(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        batch, length, dim = hidden.shape

        if length % self.window:
            raise ValueError(
                f"sequence length {length} is not divisible by window {self.window}; "
                "pad or crop upstream so the trailing bin is not silently partial"
            )

        n_windows = length // self.window
        weights = mask.unsqueeze(-1).to(hidden.dtype).view(batch, n_windows, self.window, 1)
        binned = (hidden.view(batch, n_windows, self.window, dim) * weights).sum(dim=2)
        pooled = binned / weights.sum(dim=2).clamp_min(1.0)
        pooled = pooled.to(self.project.weight.dtype)

        return self.output_layer(self.mlp(self.project(pooled)))


class PerPositionHead(nn.Module):
    """Per-nucleotide MLP. `(B, L, D) -> (B, L, out_dim)`.

    No pooling, so `mask` is not consumed here — masking a per-position
    prediction is the objective's job, since only it knows whether an ignored
    position should be dropped from the mean or scored as a padding class.
    """

    def __init__(self, input_dim: int, out_dim: int, *, hidden_dim: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.project = nn.Linear(input_dim, hidden_dim)
        self.mlp = _mlp(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, hidden: Tensor, mask: Tensor | None = None) -> Tensor:
        hidden = hidden.to(self.project.weight.dtype)

        return self.output_layer(self.mlp(self.project(hidden)))


HEAD_REGISTRY = {
    "mlp": MLPReadoutHead,
    "conv": ConvReadoutHead,
    "window": WindowTrackHead,
    "per-position": PerPositionHead,
}


def build_head(name: str, input_dim: int, out_dim: int, **kwargs) -> nn.Module:
    """`build_head("window", 4096, 1, window=256)`. Unknown names list the options."""
    try:
        cls = HEAD_REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown head {name!r}; known: {sorted(HEAD_REGISTRY)}") from None

    return cls(input_dim, out_dim, **kwargs)
