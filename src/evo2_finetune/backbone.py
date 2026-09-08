"""
`Evo2Backbone` — the port, wrapped for fine-tuning.

Adds four things to `evo2_neuron.StripedHyena` and nothing else:

  * block pruning to the read-out depth,
  * a mid-stack read-out with two implementations (`readout_mode`),
  * LoRA injection, scoped so no adapter lands past the read-out,
  * activation checkpointing, with the two settings that are load-bearing.

`dtype` defaults to **fp32**, not bf16. Evo 2 emits very large activations in its
deeper blocks that the final norm absorbs; in bf16 on Neuron those collapse to
exactly zero. fp32 is the source of truth and bf16 is a separate, separately
gated escalation — which is worth taking, because it is where the throughput is.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from evo2_finetune.lora import (
    DEFAULT_TARGET_REGEX,
    apply_lora,
    freeze_base,
    scope_regex_to_blocks,
)
from evo2_neuron import (
    ByteTokenizer,
    Evo2ArcConfig,
    default_checkpoint_path,
    load_into,
    stabilize_random_init,
)
from evo2_neuron.model import StripedHyena

logger = logging.getLogger(__name__)


class Evo2Backbone(nn.Module):
    """Evo 2 on CPU or Trainium, with a mid-stack read-out."""

    #: How `read_out` extracts the hidden states. Both produce **bit-identical**
    #: output (verified across 245 tensors), so this is purely a performance knob.
    #:
    #: ``"hook"``   — register a forward hook on `blocks.<mid>.mlp.l3` per call.
    #: ``"return"`` — `StripedHyena.forward(readout_idx=...)` returns it directly.
    #:
    #: `"return"` is the structurally clean one: the hook costs 2 dynamo graph
    #: breaks, because `RemovableHandle.__init__` does
    #: `RemovableHandle.next_id += 1` — a `setattr` on a *class*, which dynamo
    #: cannot trace. Removing it takes the compiled step from 3 graphs to 1.
    #:
    #: It is nevertheless not the default, because fusing made the backward
    #: SLOWER under whole-model compile on evo2-7b at seq 200 (warm, synchronised,
    #: n>=10): fp32 0.2672 -> 0.3314 s (+24%), bf16 0.2136 -> 0.3413 s (+60%).
    #: The forward was unchanged; the entire cost was the fused backward.
    #:
    #: Re-measure after any change that alters the graph. Under **per-block**
    #: compile (the production setting) there is no whole-model graph to fuse
    #: into, so the tradeoff there is open.
    readout_mode: str = "hook"

    def __init__(
        self,
        alias: str = "evo2-7b",
        *,
        dtype: torch.dtype = torch.float32,
        random_init: bool = False,
        readout_block: int | None = None,
        prune_blocks: bool = True,
        checkpoint: str | None = None,
        hcl_chunk: int | None = None,
        hcl_kernel: str | None = None,
    ) -> None:
        super().__init__()

        config = Evo2ArcConfig.for_alias(alias, hcl_chunk=hcl_chunk, hcl_kernel=hcl_kernel)
        mid = config.get("readout_block", 10) if readout_block is None else readout_block

        if mid >= config.num_layers:
            raise ValueError(
                f"{alias}: read-out block {mid} does not exist; the config has "
                f"{config.num_layers} layers"
            )

        max_blocks = mid + 1 if prune_blocks else None
        module = StripedHyena(config, max_blocks=max_blocks)

        if random_init:
            # Without this the modal filters diverge to `nan` within three
            # blocks -- see `stabilize_random_init`. Plumbing harness only.
            stabilize_random_init(module)
        else:
            path = checkpoint or default_checkpoint_path(config.get("huggingface_id"))
            load_into(
                module, path, dtype=dtype, max_blocks=max_blocks,
                column_split=config.get("column_split", True),
            )

        # Unconditional, and NOT redundant with `load_into(dtype=...)`.
        # `load_into` casts the state dict, but `load_state_dict` COPIES into
        # already-constructed fp32 parameters — so on the checkpoint path that
        # argument only rounds the VALUES while leaving the module fp32, i.e.
        # bf16 weights with fp32 compute. The random-init path did cast. Same
        # call, two different dtypes, depending on a flag that has nothing to do
        # with dtype. Cost of that confusion, once: a bf16 calibration run that
        # silently produced fp32 numbers.
        module = module.to(dtype)

        self.module = module
        self.config = config
        self.tokenizer = ByteTokenizer()
        self.readout_block = mid
        self.readout_name = f"blocks.{mid}.mlp.l3"
        self.hidden_size = config.hidden_size
        self.context_size = config.get("max_seqlen", 8192)
        self.readout_mode = type(self).readout_mode
        self._lora_targets: list[str] = []

        logger.info(
            "Evo2Backbone: built %d/%d blocks (read-out %s), dtype=%s",
            module.num_built_blocks, config.num_layers, self.readout_name,
            next(module.parameters()).dtype,
        )

    # -- read-out ----------------------------------------------------------

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Per-position read-out: the output of `blocks.<mid>.mlp.l3`.

        The MLP down-projection, taken **before** the residual add. A port that
        reads out after the add will not match the saved activations.

        `attention_mask` is accepted and ignored: Evo 2 is causal and consumes no
        mask. It is in the signature so callers can pass a batch dict straight
        through, and so heads and objectives are the only place masking lives.
        """
        del attention_mask

        if self.readout_mode == "return":
            return self.module(input_ids, readout_idx=self.readout_block)

        captured: dict[str, Tensor] = {}

        def hook(_module: nn.Module, _inputs: tuple, output: object) -> None:
            captured["h"] = output[0] if isinstance(output, tuple) else output

        handle = self.module.get_submodule(self.readout_name).register_forward_hook(hook)
        try:
            self.module(input_ids)
        finally:
            handle.remove()

        if "h" not in captured:
            raise RuntimeError(
                f"the forward hook on {self.readout_name} never fired. That is what "
                "happens once the read-out block is compiled: dynamo traces THROUGH "
                "the submodule, so the hook is never called. Set "
                "readout_mode='return' -- compile_blocks() does this for you."
            )

        return captured["h"]

    # -- LoRA --------------------------------------------------------------

    def apply_lora(
        self, *, r: int = 8, alpha: float | None = None, dropout: float = 0.0,
        target_regex: str = DEFAULT_TARGET_REGEX, freeze: bool = True,
    ) -> list[str]:
        """Inject adapters into blocks `0..readout_block`, then freeze the base.

        The regex is **scoped to the read-out depth** before matching. Adapters
        past the read-out are unreachable from the loss: they would take no
        gradient, yet still be counted as trainable, still be allreduced by DDP
        every step, and still be written to every checkpoint.

        `alpha` defaults to `r`, which makes `scaling = 1`.
        """
        scoped = scope_regex_to_blocks(target_regex, self.readout_block)
        self._lora_targets = apply_lora(
            self.module, r=r, alpha=r if alpha is None else alpha,
            dropout=dropout, target_regex=scoped,
        )

        if freeze:
            tensors, params = freeze_base(self.module)
            logger.info("Evo2Backbone: %d LoRA tensors trainable (%d params); base frozen",
                        tensors, params)

        return self._lora_targets

    # -- memory ------------------------------------------------------------

    def enable_activation_checkpointing(self) -> None:
        """Recompute every block except the read-out one during the backward.

        At seq 8192 on a 24 GiB logical NeuronCore this is not an optimisation,
        it is what makes the step fit at all. It costs ~3% on the compiled step
        and is bit-identical (verified: 245/245 tensors at exactly 0.0).

        The read-out block is excluded because its output is consumed directly by
        the head; recomputing it would recompute the tensor the graph is already
        holding.

        See `StripedHyena.enable_gradient_checkpointing` for the two settings
        that are load-bearing: `use_reentrant=False`, and the refusal to enable
        when a `nn.Dropout(p > 0)` is present.
        """
        self.module.enable_gradient_checkpointing(skip=(self.readout_block,))
        logger.info(
            "Evo2Backbone: checkpointing on for %d/%d blocks (read-out block %d excluded)",
            len(self.module._checkpoint_blocks), self.module.num_built_blocks,
            self.readout_block,
        )

    def compile_blocks(self, backend: str = "neuron") -> int:
        """Compile each block as its own graph. See `docs/OPTIMIZATIONS.md`.

        Whole-model `torch.compile` exceeds `neuronx-cc`'s instruction limit at
        seq >= 1024. The limit is **per graph**, so compiling blocks separately
        fits — and is worth 1.72x on the 8192 training step.

        Switches `readout_mode` to `"return"`. The hook path is structurally
        incompatible with compiling the read-out block: dynamo traces *through*
        `blocks.<mid>.mlp.l3`, so the forward hook never fires and the read-out
        comes back empty. `"return"` is bit-identical (verified across 245
        tensors) and it is what makes the eleventh block compilable at all
        rather than having to be left in eager.
        """
        import torch._dynamo

        if self.readout_mode != "return":
            self.readout_mode = "return"
            logger.info(
                "Evo2Backbone: readout_mode -> 'return' (a forward hook does not "
                "fire through a compiled block; the output is bit-identical)"
            )

        # Default is 8. Eleven blocks plus warmup variants exceed it, and the
        # consequence of exceeding it is a SILENT fall back to eager.
        torch._dynamo.config.recompile_limit = max(
            64, getattr(torch._dynamo.config, "recompile_limit", 8)
        )

        count = 0

        for block in self.module.blocks:
            # `Module.compile()`, not `blocks[i] = torch.compile(blocks[i])`.
            # The latter wraps the module in an `OptimizedModule`, which inserts
            # `_orig_mod.` into every parameter name — renaming most of the
            # trainable set and breaking adapter checkpoints and any comparison
            # keyed by name.
            block.compile(backend=backend)
            count += 1

        logger.info("Evo2Backbone: compiled %d blocks with backend=%r", count, backend)

        return count
