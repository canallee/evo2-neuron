"""
`StripedHyena` backbone, ported from `vortex/model/model.py`.

Module names are deliberately identical to vortex's, because two things key off
them by string:

  * the LoRA target regex `.*blocks\\.\\d+\\.mlp\\.(l1|l2|l3)$`
  * the read-out path `blocks.<mid>.mlp.l3`

Keeping the names means `evo2_finetune.lora.apply_lora`, the read-out hook and
adapter save/load all work against this module unmodified -- and that an Arc
checkpoint loads with no key renaming at all.

Dropped from Arc's version, none of it on the prefill path: multi-GPU layer
placement (`block_idx_to_device`, `cross_device_transfer`), FlashFFTConv, the KV
cache / `stateful_forward`, and the activation-logging instrumentation.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from evo2_neuron.hyena import get_block
from evo2_neuron.layers import RMSNorm, VocabParallelEmbedding


class StripedHyena(nn.Module):
    """Evo 2 backbone: byte embedding -> N blocks -> final RMSNorm.

    `max_blocks` builds only `blocks[:max_blocks]`. Anything reading out at
    `blocks.<mid>.mlp.l3` never touches what is downstream, so building 11 blocks
    skips ~56% of the weights on evo2-1b (of 25) and ~66% on evo2-7b (of 32),
    plus all of their activation memory.
    This is pruning applied at construction rather than after load -- and unlike
    vortex's read-out, it needs no
    exception raised from a forward hook to short-circuit the stack, which is
    hostile to graph capture on Neuron.
    """

    def __init__(self, config, max_blocks: int | None = None) -> None:
        super().__init__()
        self.config = config

        num_layers = config.num_layers if max_blocks is None else min(max_blocks, config.num_layers)
        self.num_built_blocks = num_layers

        self.embedding_layer = VocabParallelEmbedding(config)
        self.blocks = nn.ModuleList(get_block(config, idx) for idx in range(num_layers))
        self.norm = RMSNorm(config) if config.get("final_norm", True) else None

        #: Block indices whose forward is recomputed in the backward instead of
        #: having its intermediates saved. Empty = off, which is the default.
        #: Written by `enable_gradient_checkpointing`.
        self._checkpoint_blocks: frozenset[int] = frozenset()

        #: Contiguous runs of blocks compiled as ONE graph, as
        #: `[(start, stop, callable), ...]`. Empty = off. Written by
        #: `compile_block_groups`.
        self._block_groups: list = []

    def enable_gradient_checkpointing(self, skip: Iterable[int] = ()) -> None:
        """Recompute the listed blocks' forwards during the backward pass.

        Trades block-forward FLOPs for activation memory. Sized on evo2-7b at
        seq 8192 bf16: autograd was holding ~5.9 GB of saved intermediates
        across 11 blocks, which is what put the training step over a
        NeuronCore's 24 GiB.

        `skip` excludes blocks from checkpointing. The caller is expected to
        skip the read-out block: `Evo2TrnBackbone.hidden_states` in its default
        `"hook"` mode captures `blocks[mid].mlp.l3`'s output, which is an
        intermediate *inside* the block rather than the block's return value.
        Under non-reentrant checkpointing that tensor's `grad_fn` points into a
        region whose saved tensors are placeholders, so differentiating it
        drives a recomputation from outside the checkpoint's own bookkeeping.
        Correct in principle, needlessly subtle in practice, and worth ~1/11 of
        the savings -- so the read-out block just keeps its activations.

        **`use_reentrant=False` is load-bearing, not stylistic.** The whole
        backbone is frozen under LoRA, so `embedding_layer`'s output has
        `requires_grad=False` and *no input to any block requires grad* -- only
        parameters inside them do. The reentrant implementation runs its forward
        under `no_grad` and hangs the backward off a `Function` whose inputs are
        exactly those non-requiring tensors, so its output carries no `grad_fn`
        and every LoRA gradient below it is dropped. The non-reentrant one
        records the graph during the forward and reconnects through the
        parameters, which is the only reason this works at all.

        Measured, not assumed -- with `use_reentrant=True` on a 4-block stand-in
        (`test_reentrant_checkpointing_would_drop_lora_gradients`): the loss
        computes normally, the uncheckpointed read-out block still yields a
        gradient, and blocks 0-2 come back **None**. A run in that state trains
        the top adapter and leaves the other 30 at their init. With
        `use_reentrant=False` the gradients match the uncheckpointed ones to
        exactly 0.0 and the forward is bit-identical.

        The tests ask `torch.autograd.grad(..., allow_unused=True)` directly
        rather than comparing magnitudes, because a magnitude gate cannot see
        this: `lora_B` is zero-initialised, so an absent gradient reads as
        "small". That is the same instrument failure that let a
        non-differentiable NKI kernel ship with wrong gradients.
        """
        blocks = frozenset(range(self.num_built_blocks)) - frozenset(skip)

        # `preserve_rng_state=False` below is only sound if the recomputed
        # forward is deterministic. The port's single RNG consumer on the
        # trainable path is LoRA dropout, which every gated run pins to 0. Fail
        # loudly rather than let a nonzero `p` desynchronise the recomputation
        # from the forward and produce quietly wrong gradients.
        active = [
            name
            for name, mod in self.named_modules()
            if isinstance(mod, nn.Dropout) and mod.p > 0
        ]
        if active:
            raise ValueError(
                "gradient checkpointing here assumes a deterministic recomputation "
                f"(preserve_rng_state=False), but {len(active)} dropout module(s) have "
                f"p > 0, e.g. {active[0]!r}. Re-run with --lora-dropout 0, or teach "
                "this method to preserve RNG state per device."
            )

        self._checkpoint_blocks = blocks

    def disable_gradient_checkpointing(self) -> None:
        self._checkpoint_blocks = frozenset()

    def compile_block_groups(self, size: int, backend: str = "neuron",
                             skip: Iterable[int] = ()) -> int:
        """Compile contiguous runs of `size` blocks as single graphs.

        Between two failing extremes. `torch.compile(model)` is one graph and
        exceeds `neuronx-cc`'s per-graph instruction limit above seq 512
        (`NCC_EBVF030`). Compiling every block separately fits, but costs
        **1.67x** against the whole-model graph at seq 200 because each break is
        paid for. Grouping trades one against the other: 11 blocks at size 3 is
        4 graphs instead of 11.

        Measured compilable at seq 8192: size 2 (160 s) and size 3 (439 s).
        Sizes 4 and 6 are **unmeasured** -- that probe died on
        `ConnectToService failed ... errno=2`, a compile-service failure under
        four concurrent heavy compiles, NOT the instruction limit. Compile cost
        rises steeply with group size (2.7x from size 2 to 3), so this is a
        one-off to amortise against a persisted NEFF cache.

        `skip` excludes blocks, and the read-out block must be in it: its output
        is captured by a forward hook, and once the enclosing graph is compiled
        dynamo traces through and the hook stops firing.

        Returns the number of groups created.
        """
        skip_set = frozenset(skip)
        self._block_groups = []

        run: list[int] = []

        def flush() -> None:
            if not run:
                return
            group = nn.Sequential(*[self.blocks[i] for i in run])
            group.compile(backend=backend)
            self._block_groups.append((run[0], run[-1] + 1, group))
            run.clear()

        for index in range(self.num_built_blocks):
            if index in skip_set:
                flush()
                continue

            run.append(index)
            if len(run) == size:
                flush()

        flush()
        return len(self._block_groups)

    def forward(self, x: Tensor, readout_idx: int | None = None) -> Tensor:
        """`readout_idx` returns `blocks[readout_idx].mlp.l3`'s output instead of
        the normed final hidden state -- the mid-stack read-out a downstream head
        consumes.

        Done by return value rather than a forward hook. The hook version cost
        two dynamo graph breaks (`RemovableHandle.next_id += 1` is a class
        `setattr`, which dynamo cannot trace), splitting backbone / head / loss
        into three compiled graphs.

        Every block is still executed, matching the hook version exactly: with
        `prune_dead_blocks` on, `readout_idx` IS the last built block, so nothing
        is wasted. The block's normal output is still produced and passed on, so
        a forward hook on that block still observes the post-residual tensor.
        """
        x = self.embedding_layer(x)
        readout: Tensor | None = None

        # `is_grad_enabled` rather than `self.training`: under `no_grad` there is
        # nothing to recompute for, and `torch.utils.checkpoint.checkpoint` is
        # decorated `@torch._disable_dynamo`, so leaving it in would cost the
        # inference path a graph break per block for no benefit.
        checkpointing = bool(self._checkpoint_blocks) and torch.is_grad_enabled()

        if self._block_groups:
            # Grouped path. Checkpointing applies to the GROUP rather than each
            # block, which stores fewer boundaries and so saves slightly more
            # memory, at the cost of recomputing a whole group.
            index = 0
            group_at = {start: (stop, fn) for start, stop, fn in self._block_groups}

            while index < self.num_built_blocks:
                if index in group_at:
                    stop, fn = group_at[index]
                    if checkpointing and index in self._checkpoint_blocks:
                        x = checkpoint(fn, x, use_reentrant=False, preserve_rng_state=False)
                    else:
                        x = fn(x)
                    index = stop
                    continue

                block = self.blocks[index]
                if index == readout_idx:
                    x, readout = block(x, return_readout=True)
                elif checkpointing and index in self._checkpoint_blocks:
                    x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
                else:
                    x = block(x)
                index += 1

        else:
            for index, block in enumerate(self.blocks):
                if index == readout_idx:
                    x, readout = block(x, return_readout=True)
                elif checkpointing and index in self._checkpoint_blocks:
                    x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
                else:
                    x = block(x)

        if readout_idx is not None:
            if readout is None:
                raise IndexError(
                    f"readout_idx {readout_idx} is out of range for "
                    f"{len(self.blocks)} built block(s)"
                )

            return readout

        return self.norm(x) if self.norm is not None else x

    def unembed(self, u: Tensor) -> Tensor:
        """Tied-embedding LM head (`tie_embeddings: True` in every Evo2 config)."""
        return self.embedding_layer.unembed(u)


class Evo2ArcForCausalLM(nn.Module):
    """Backbone + tied LM head.

    Needed for the next-token-CE objective. A read-out head uses `StripedHyena`
    alone and never materialises logits -- which matters, because the unembed is
    `(vocab 512 x hidden)` applied at every position.
    """

    def __init__(self, config, max_blocks: int | None = None) -> None:
        super().__init__()
        self.backbone = StripedHyena(config, max_blocks=max_blocks)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.backbone.unembed(self.backbone(input_ids))
