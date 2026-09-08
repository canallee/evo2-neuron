"""
LoRA for Evo 2 — standalone, no `peft`.

Two reasons this is not `peft`: peft's LoRA path expects an HF-style
`config.to_dict()` on the base module, which `StripedHyena` does not expose; and
adapter injection has to be scoped by *block index*, not by leaf name, because
adapters placed downstream of a mid-stack read-out receive no gradient at all
and would silently inflate the trainable-parameter count.

Design follows Arc/Zymrael Savanna's `savanna/lora.py`:
- Kaiming-uniform `lora_A` (a = sqrt(5)), zero `lora_B`, scaling `alpha / r`.
- An `_is_lora_param = True` sentinel on adapter tensors, so the state-dict
  filter and the freeze loop find them without walking module types.
- `get/load_lora_state_dict` for adapter-only checkpoints.
- `merge/unmerge` for inference deployment.

Reference: Hu et al., LoRA (ICLR 2022), https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import logging
import math
import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

#: Every `nn.Linear` inside a StripedHyena gated MLP, at any block depth.
#: `apply_lora_scoped` narrows the `\d+` to the blocks that actually train.
DEFAULT_TARGET_REGEX = r".*blocks\.\d+\.mlp\.(l1|l2|l3)$"


class LoRALinear(nn.Module):
    """`y = base(x) + (alpha/r) * B @ A @ dropout(x)`.

    `lora_A` is `(r, in_features)`, `lora_B` is `(out_features, r)`. `lora_B` is
    zero-initialised, so the adapter is *exactly* the identity at step 0.

    That zero-init has a consequence worth knowing before you debug a gradient:
    `lora_B.grad` is genuinely tiny, and an *absent* gradient is indistinguishable
    from a small one by magnitude alone. Check for `None` explicitly
    (`torch.autograd.grad(..., allow_unused=True)`), not for smallness.
    """

    def __init__(self, base: nn.Linear, *, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_A._is_lora_param = True
        self.lora_B._is_lora_param = True

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)

        return base_out + delta.to(base_out.dtype) * self.scaling


def _parent_and_attr(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    """Split `a.b.c` into `(root.a.b, "c")` so the child can be replaced."""
    if "." not in dotted:
        return root, dotted

    parent_path, attr = dotted.rsplit(".", 1)
    parent = root

    for part in parent_path.split("."):
        parent = getattr(parent, part)

    return parent, attr


def apply_lora(
    root: nn.Module, *, r: int, alpha: float, dropout: float, target_regex: str
) -> list[str]:
    """Wrap every `nn.Linear` whose full dotted path `re.fullmatch`es the regex.

    Returns the wrapped paths. A module that matches but is not an `nn.Linear` is
    warned and skipped rather than silently ignored — an over-broad regex is a
    much more common mistake than an under-broad one.
    """
    if not isinstance(r, int) or isinstance(r, bool) or r < 1:
        raise ValueError(f"LoRA `r` must be a positive integer, got {r!r}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0:
        raise ValueError(f"LoRA `alpha` must be a positive number, got {alpha!r}")
    if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
            or not 0.0 <= dropout < 1.0):
        raise ValueError(f"LoRA `dropout` must be a float in [0, 1), got {dropout!r}")

    pattern = re.compile(target_regex)
    wrapped: list[str] = []
    skipped: list[str] = []

    # Snapshot before mutation: `setattr` during `named_modules()` would resurface
    # the freshly-inserted wrapper as a new descendant and wrap it again.
    for name, module in list(root.named_modules()):
        if not pattern.fullmatch(name):
            continue

        if not isinstance(module, nn.Linear):
            skipped.append(f"{name}::{type(module).__name__}")
            continue

        parent, attr = _parent_and_attr(root, name)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        wrapped.append(name)

    if skipped:
        logger.warning(
            "apply_lora: %d modules matched the regex but are not nn.Linear (first few: %s)",
            len(skipped), skipped[:5],
        )

    logger.info("apply_lora: wrapped %d modules (r=%d, alpha=%s, dropout=%s)",
                len(wrapped), r, alpha, dropout)

    return wrapped


def scope_regex_to_blocks(target_regex: str, max_block: int) -> str:
    """Narrow a `blocks\\.\\d+\\.` regex to `blocks\\.(0|1|...|max_block)\\.`.

    Adapters placed after the read-out block get no gradient — the read-out is
    taken mid-stack, so nothing downstream is on the path to the loss. They would
    still be counted as trainable, still be allreduced by DDP, and still be
    written to the checkpoint. Scoping is what keeps the trainable set honest.
    """
    alternation = "|".join(str(i) for i in range(max_block + 1))

    return target_regex.replace(r"blocks\.\d+\.", rf"blocks\.({alternation})\.")


def freeze_base(root: nn.Module) -> tuple[int, int]:
    """Freeze everything without the `_is_lora_param` marker.

    Returns `(trainable_tensors, trainable_params)`. Call this *after*
    `apply_lora` and *before* attaching a head — a head must stay trainable.
    """
    tensors = params = 0

    for param in root.parameters():
        if getattr(param, "_is_lora_param", False):
            param.requires_grad_(True)
            tensors += 1
            params += param.numel()
        else:
            param.requires_grad_(False)

    return tensors, params


def get_lora_state_dict(root: nn.Module) -> dict[str, Tensor]:
    """Adapter-only state dict — megabytes, even for evo2-7b."""
    return {
        key: value.detach().clone()
        for key, value in root.state_dict().items()
        if key.endswith(".lora_A") or key.endswith(".lora_B")
    }


def load_lora_state_dict(root: nn.Module, state: dict[str, Tensor]) -> None:
    """Non-strict load. Missing base keys are expected; missing or unexpected
    *LoRA* keys are warned, because either means the checkpoint's target-module
    set disagrees with this model's, and a partially-loaded adapter otherwise
    trains on quietly.
    """
    missing, unexpected = root.load_state_dict(state, strict=False)

    if unexpected:
        logger.warning("load_lora_state_dict: %d unexpected keys (target set mismatch?): %s",
                       len(unexpected), unexpected[:5])

    lora_missing = [k for k in missing if k.endswith((".lora_A", ".lora_B"))]

    if lora_missing:
        logger.warning("load_lora_state_dict: %d LoRA keys missing (target set mismatch?): %s",
                       len(lora_missing), lora_missing[:5])


def merge_lora_weights(root: nn.Module) -> None:
    """Fold the delta into the base weight, in place, idempotently.

    The pre-merge weight is stashed as a registered buffer so `unmerge` restores
    it *exactly* rather than subtracting the delta back off — subtraction in bf16
    does not round-trip.

    **Merge is for exporting `base.weight`, not for running.** `forward` still
    adds the adapter, so calling it on a merged module double-counts the delta.
    """
    for module in root.modules():
        if not isinstance(module, LoRALinear) or getattr(module, "_lora_merged", False):
            continue

        with torch.no_grad():
            module.register_buffer("_original_weight", module.base.weight.data.clone())
            delta = (module.lora_B @ module.lora_A) * module.scaling
            module.base.weight.data += delta.to(module.base.weight.dtype)

        module._lora_merged = True


def unmerge_lora_weights(root: nn.Module) -> None:
    """Restore the saved pre-merge weight. Exact, including in bf16."""
    for module in root.modules():
        if not isinstance(module, LoRALinear) or not getattr(module, "_lora_merged", False):
            continue

        with torch.no_grad():
            module.base.weight.data.copy_(module._original_weight)
            del module._original_weight        # also drops the buffer registration

        module._lora_merged = False
