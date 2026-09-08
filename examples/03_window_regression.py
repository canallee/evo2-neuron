#!/usr/bin/env python
"""
Window-level regression: one prediction per 256 bp bin.

    python examples/03_window_regression.py --device cpu --random-backbone --max-blocks 3
    NEURON_RT_VISIBLE_CORES=0 python -u examples/03_window_regression.py --device neuron

This is the shape most genome-track tasks want — accessibility, coverage,
conservation, any signal measured in bins rather than per base or per sequence.
`WindowTrackHead` mean-pools each bin of the residual stream and runs a shared
MLP over it, so `(B, L, 4096) -> (B, L/256, out_dim)`.

Only LoRA adapters and the head train; the backbone is frozen. `weights` marks
which bins were actually observed, so an unmeasured bin contributes nothing
rather than being trained toward zero.

The synthetic labels are motif density per bin — learnable on purpose. If the
loss does not fall here, the harness is broken, not the task.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo2_finetune import neuron_env  # noqa: E402

neuron_env.configure()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from evo2_finetune.data import SyntheticDNADataset, WindowTrackCollator  # noqa: E402
from evo2_finetune.model import build_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "neuron"])
    parser.add_argument("--backbone", default="evo2-7b")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--random-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = neuron_env.resolve_device(args.device)

    model = build_model(
        objective="window-regression", alias=args.backbone, out_dim=1,
        device="cpu", dtype=dtype, lora_rank=args.lora_rank,
        head_kwargs={"window": args.window},
        random_init=args.random_backbone, max_blocks=args.max_blocks,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    print(f"trainable: {model.trainable_summary()}")
    print(f"read-out {model.backbone.readout_name}, "
          f"{args.seq_len // args.window} windows of {args.window} bp")

    dataset = SyntheticDNADataset(size=64, length=args.seq_len, window=args.window, seed=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=WindowTrackCollator(length=args.seq_len,
                                                       window=args.window))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, foreach=False)

    model.train()
    batches = iter(loader)
    history = []
    started = time.time()

    for step in range(args.steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)

        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = batch["labels"].to(dtype)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch)["loss"]
        loss.backward()
        optimizer.step()
        neuron_env.synchronize()

        history.append(float(loss.detach().cpu()))

        if step % 5 == 0 or step == args.steps - 1:
            print(f"  step {step:>3}  loss {history[-1]:.6f}  {time.time() - started:.1f}s")

    print(f"\nloss {history[0]:.6f} -> {history[-1]:.6f} "
          f"({'descending' if history[-1] < history[0] else 'NOT descending'})")
    print(f"fallback_ops: {neuron_env.fallback_ops() or 'none'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
