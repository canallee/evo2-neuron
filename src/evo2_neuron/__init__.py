"""
Evo 2 (StripedHyena 2) on CPU and AWS Trainium — Arc Institute's own definition.

Re-implemented against `github.com/ArcInstitute/evo2` + `github.com/Zymrael/vortex`
(Apache-2.0) in plain PyTorch: no vortex runtime, no TransformerEngine, no
flash-attn, no Triton, no CUDA. The same package runs unchanged on `cpu` and on
`torch.device("neuron")`. See `VENDORED.md` for source commits and the patch list.

CAVEAT. This is a port, not Arc's code, and it is not endorsed by Arc Institute.
Only `evo2-7b` is validated: `evo2-1b`'s config enables FP8 input projections,
which this port has no path for, so its numbers cannot be gated. The port is NOT
bit-exact against CUDA — the read-out differs from an H200 reference by 1.6e-02
(CPU) and 3.4e-02 (Trainium) in bf16. That is consistent with reduction order
rather than a defect, and it is still large enough to move a downstream score.
Validate against the CUDA path before relying on a result. See docs/PARITY.md.

    from evo2_neuron import load, ByteTokenizer

    model = load("evo2-7b", device="neuron", max_blocks=11)
    ids = ByteTokenizer().encode_batch(["ACGTACGT"])
    hidden = model(ids)                      # (B, L, config.hidden_size)
"""

from evo2_neuron.config import Evo2ArcConfig
from evo2_neuron.loading import (
    default_checkpoint_path,
    load_arc_state_dict,
    load_into,
    stabilize_random_init,
)
from evo2_neuron.model import Evo2ArcForCausalLM, StripedHyena
from evo2_neuron.tokenizer import ByteTokenizer

__all__ = [
    "ByteTokenizer",
    "Evo2ArcConfig",
    "Evo2ArcForCausalLM",
    "StripedHyena",
    "load",
    "load_arc_state_dict",
    "load_into",
    "stabilize_random_init",
    "default_checkpoint_path",
]


def load(
    alias: str = "evo2-7b",
    *,
    device: str = "cpu",
    dtype=None,
    max_blocks: int | None = None,
    checkpoint: str | None = None,
    causal_lm: bool = False,
    random_init: bool = False,
):
    """Build the ported Evo2 and load Arc's weights.

    Arguments:
        alias: config key (`evo2-1b`, `evo2-7b`).
        max_blocks: build only the first N blocks. Anything that reads out
            mid-stack -- an embedding extractor, a LoRA fine-tune with an
            adapter-scoped read-out -- needs no more than `readout_block + 1`,
            and every block not built is weights not loaded and not recomputed.
        causal_lm: wrap in the tied LM head. Needed for the next-token objective;
            a read-out head does not use it.
        random_init: skip checkpoint I/O entirely. Real architecture, random
            weights -- for structure checks and CPU smoke tests in seconds.
            Applies `stabilize_random_init`, without which the modal filters
            diverge to `nan` within three blocks. It is a plumbing harness, not
            a model: never quote a number off it.

    Weights are Arc's own (`arcinstitute/<model>`), never a third-party
    re-upload. fp32 by default. Set `EVO2_CHECKPOINT` to load a local file.
    """
    import torch

    dtype = torch.float32 if dtype is None else dtype
    config = Evo2ArcConfig.for_alias(alias)

    if causal_lm:
        model = Evo2ArcForCausalLM(config, max_blocks=max_blocks)
        target = model.backbone
    else:
        model = StripedHyena(config, max_blocks=max_blocks)
        target = model

    if random_init:
        stabilize_random_init(target)
    else:
        path = checkpoint or default_checkpoint_path(config.get("huggingface_id", "evo2_7b_base"))
        load_into(
            target, path, dtype=dtype, max_blocks=max_blocks,
            column_split=config.get("column_split", True),
        )

    return model.to(dtype).to(torch.device(device))
