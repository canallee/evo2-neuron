"""
`Evo2FineTune` — backbone + head + objective, with the loss computed in `forward`.

Computing the loss inside `forward` is what lets the same module drop into a
plain loop, into DDP, and into a compiled step without a second code path, and it
keeps the loss on device: pulling logits to the host to compute a loss outside
the module is the single easiest way to turn a 3.8 s step into a 6 s one.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from evo2_finetune.backbone import Evo2Backbone
from evo2_finetune.heads import build_head
from evo2_finetune.objectives import OBJECTIVE_HEADS, get_objective

logger = logging.getLogger(__name__)


class Evo2FineTune(nn.Module):
    """Evo 2 + LoRA + a read-out head, trained against one objective.

    `forward(**batch)` returns `{"prediction": ..., "loss": ...}`. `loss` is
    `None` when the batch carries no labels, so evaluation and inference use the
    same call.
    """

    def __init__(self, backbone: Evo2Backbone, head: nn.Module | None, objective: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.objective_name = objective
        self.objective = get_objective(objective)
        self.causal_lm = objective == "next-token-ce"

        if self.causal_lm and head is not None:
            raise ValueError("next-token-ce uses the tied LM head; pass head=None")
        if not self.causal_lm and head is None:
            raise ValueError(f"objective {objective!r} needs a read-out head")

    def forward(self, **batch: Any) -> dict[str, Any]:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        if self.causal_lm:
            stack = self.backbone.module
            # Tied embedding, so there is no separate LM head to load: `unembed`
            # reuses the 512-row embedding table.
            return self.objective(stack.unembed(stack(input_ids)), **batch)

        hidden = self.backbone(input_ids, attention_mask)
        mask = batch.get("window_mask", attention_mask)
        prediction = self.head(hidden, mask)

        return self.objective(prediction, **batch)

    # -- passthroughs the training loop expects ----------------------------

    def gradient_checkpointing_enable(self, **_: Any) -> None:
        self.backbone.enable_activation_checkpointing()

    def trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def trainable_summary(self) -> str:
        trainable = self.trainable_parameters()
        total = sum(p.numel() for p in self.parameters())
        count = sum(p.numel() for _, p in trainable)

        return (f"{len(trainable)} tensors, {count:,} of {total:,} params "
                f"({100.0 * count / max(total, 1):.2f}%)")


def build_model(
    *,
    objective: str,
    alias: str = "evo2-7b",
    out_dim: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    lora_rank: int = 8,
    lora_alpha: float | None = None,
    lora_dropout: float = 0.0,
    head: str | None = None,
    head_kwargs: dict[str, Any] | None = None,
    random_init: bool = False,
    readout_block: int | None = None,
    max_blocks: int | None = None,
    full_finetune: bool = False,
    checkpoint: str | None = None,
    hcl_kernel: str | None = None,
    gradient_checkpointing: bool = False,
) -> Evo2FineTune:
    """One call from an objective name to a trainable model on `device`.

    The head is chosen from the objective unless `head` overrides it. A head
    always stays trainable: `apply_lora` freezes the base *before* the head is
    attached, precisely so that ordering cannot be got wrong here.
    """
    causal_lm = objective == "next-token-ce"

    # The LM objective reads logits off the top of the stack, so by default
    # nothing is pruned -- that is the whole 13 GB of evo2-7b. A read-out head
    # sees only blocks 0..readout_block. `max_blocks` overrides both, which is
    # how the CPU examples run a truncated LM in seconds; a truncated LM is a
    # different model, so use it for shape work, not for numbers.
    if max_blocks is not None:
        readout_block = min(max_blocks - 1, readout_block if readout_block is not None else 10**9)

    backbone = Evo2Backbone(
        alias,
        dtype=dtype,
        random_init=random_init,
        readout_block=readout_block,
        prune_blocks=(max_blocks is not None) or not causal_lm,
        checkpoint=checkpoint,
        hcl_kernel=hcl_kernel,
    )
    # `freeze=False` leaves the backbone trainable: full fine-tuning, with the
    # adapters still present and still trained. Understand the cost before using
    # it — see the `--full-finetune` help in scripts/train_lora_ddp.py.
    backbone.apply_lora(r=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
                        freeze=not full_finetune)

    module = None

    if not causal_lm:
        name = head or OBJECTIVE_HEADS[objective]
        module = build_head(name, backbone.hidden_size, out_dim, **(head_kwargs or {}))
        module = module.to(dtype)

    model = Evo2FineTune(backbone, module, objective).to(torch.device(device))

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info("build_model: objective=%s trainable=%s", objective, model.trainable_summary())

    return model
