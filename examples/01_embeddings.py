#!/usr/bin/env python
"""
Get DNA embeddings out of Evo 2, on CPU or Trainium.

    python examples/01_embeddings.py --device cpu --random-backbone   # seconds
    NEURON_RT_VISIBLE_CORES=0 python -u examples/01_embeddings.py --device neuron

No LoRA, no head, no training — just the backbone read out mid-stack. This is the
cheapest thing in the repo and the right first check that your environment works.

Why block 10 by default: the read-out depth was measured, not guessed. Below it
the representation is still mostly local sequence statistics; the plateau in
representation growth starts at block 10-11 on evo2-7b. Building 11 of 32 blocks
skips two thirds of the weights.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo2_finetune import neuron_env  # noqa: E402  before torch

neuron_env.configure()

import torch  # noqa: E402

from evo2_finetune.backbone import Evo2Backbone  # noqa: E402
from evo2_neuron import ByteTokenizer  # noqa: E402

SEQUENCES = [
    "ATGGCGCCCAAGATTGACTATACCGATCGGCTGATCTTAGAAGCTGGTCTGGCCTCTTAG",
    "TTTATATAAAGGGCGCGCACGTGCCCGGGCTAGCTAGCTAGCATCGATCGATCGGATCCA",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "neuron"])
    parser.add_argument("--backbone", default="evo2-7b", choices=["evo2-1b", "evo2-7b"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--readout-block", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--random-backbone", action="store_true",
                        help="skip the checkpoint; structure only, never for numbers")
    parser.add_argument("--max-blocks", type=int, default=None,
                        help="build fewer blocks; for a fast CPU smoke test")
    args = parser.parse_args(argv)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    readout = args.readout_block

    if args.max_blocks is not None:
        readout = min(args.max_blocks - 1, readout if readout is not None else 10)

    started = time.time()
    backbone = Evo2Backbone(
        args.backbone, dtype=dtype, random_init=args.random_backbone,
        readout_block=readout,
    ).to(neuron_env.resolve_device(args.device))
    print(f"built in {time.time() - started:.1f}s: {backbone.module.num_built_blocks} blocks, "
          f"read-out {backbone.readout_name}, hidden {backbone.hidden_size}")

    # An explicit `max_length` keeps the shape static. A varying length
    # recompiles the graph on Neuron every time it changes.
    ids, mask = ByteTokenizer().encode_batch(SEQUENCES, max_length=args.seq_len)
    target = neuron_env.resolve_device(args.device)

    started = time.time()

    with torch.no_grad():
        hidden = backbone(ids.to(target), mask.to(target))

    neuron_env.synchronize()
    print(f"forward in {time.time() - started:.1f}s (first call includes compilation)")

    hidden = hidden.float().cpu()
    print(f"\nper-position embeddings {tuple(hidden.shape)}")

    # Mask-aware mean over real positions only — padding would drag it to zero.
    weights = mask.unsqueeze(-1).float()
    pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    print(f"per-sequence embeddings {tuple(pooled.shape)}")

    normed = torch.nn.functional.normalize(pooled, dim=-1)
    print(f"\ncosine between the two sequences: {float(normed[0] @ normed[1]):+.4f}")
    print(f"fallback_ops: {neuron_env.fallback_ops() or 'none'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
