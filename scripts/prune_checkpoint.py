#!/usr/bin/env python
"""
Drop the blocks a mid-stack read-out never builds, from the checkpoint file.

Arc ships one `.pt` per model — 13 GB for evo2-7b. A read-out at
`blocks.10.mlp.l3` builds 11 of 32 blocks, so 21 blocks' worth of tensors are
read off disk, deserialised, and discarded on every load. At 64 ranks that is 64
concurrent readers of the same 13 GB file.

    python scripts/prune_checkpoint.py evo2_7b_base.pt evo2_7b_blocks0_10.pt --max-blocks 11
    export EVO2_CHECKPOINT=$PWD/evo2_7b_blocks0_10.pt

The retained tensors are **bit-identical** — nothing is cast, re-ordered or
recomputed. `load_into(strict=True)` still passes, because it filters expected
keys by the same `max_blocks`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from evo2_neuron.loading import load_arc_state_dict  # noqa: E402

BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--max-blocks", type=int, required=True,
                        help="keep blocks 0..max_blocks-1")
    args = parser.parse_args(argv)

    print(f"loading {args.source} ...", flush=True)
    state = load_arc_state_dict(args.source)

    kept, dropped = {}, 0

    for key, value in state.items():
        match = BLOCK_RE.match(key)

        if match and int(match.group(1)) >= args.max_blocks:
            dropped += 1
            continue

        kept[key] = value

    source_bytes = args.source.stat().st_size
    torch.save(kept, args.target)
    target_bytes = args.target.stat().st_size

    print(f"kept    {len(kept)} tensors, {sum(t.numel() for t in kept.values()):,} params")
    print(f"dropped {dropped} tensors (blocks {args.max_blocks}..)")
    print(f"{source_bytes / 1e9:.2f} GB -> {target_bytes / 1e9:.2f} GB "
          f"({100 * target_bytes / source_bytes:.0f}%)")
    print(f"\nexport EVO2_CHECKPOINT={args.target.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
