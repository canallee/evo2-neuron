#!/usr/bin/env python
"""
Per-nucleotide classification: one label per base.

    python examples/04_per_nucleotide_classification.py --device cpu --random-backbone --max-blocks 3
    NEURON_RT_VISIBLE_CORES=0 python -u examples/04_per_nucleotide_classification.py --device neuron

The finest granularity the model supports — annotation tracks, binding-site
calls, exon/intron labelling, anything with a call at every base. `PerPositionHead`
runs a shared MLP at every position: `(B, L, 4096) -> (B, L, n_classes)`.

`weights` is the attention mask, so padded positions are dropped from the mean
rather than scored against whatever label id happens to sit in the padded slots.
That distinction matters more here than anywhere else: at 8192 with short
sequences, padding can be most of the batch.

The synthetic labels mark planted motif positions — a genuinely learnable
per-base signal, so a flat loss curve here means the harness is wrong.
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

from evo2_finetune.data import PerPositionCollator, SyntheticDNADataset  # noqa: E402
from evo2_finetune.model import build_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "neuron"])
    parser.add_argument("--backbone", default="evo2-7b")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--classes", type=int, default=2)
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
        objective="per-position-classification", alias=args.backbone,
        out_dim=args.classes, device="cpu", dtype=dtype, lora_rank=args.lora_rank,
        random_init=args.random_backbone, max_blocks=args.max_blocks,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    print(f"trainable: {model.trainable_summary()}")
    print(f"read-out {model.backbone.readout_name}, {args.classes} classes per base")

    dataset = SyntheticDNADataset(size=64, length=args.seq_len, window=256, seed=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=PerPositionCollator(length=args.seq_len))

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
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        out["loss"].backward()
        optimizer.step()
        neuron_env.synchronize()

        history.append(float(out["loss"].detach().cpu()))

        if step % 5 == 0 or step == args.steps - 1:
            with torch.no_grad():
                predicted = out["prediction"].argmax(-1)
                correct = (predicted == batch["labels"]).float()
                accuracy = float(
                    (correct * batch["weights"]).sum() / batch["weights"].sum().clamp_min(1))

            print(f"  step {step:>3}  loss {history[-1]:.6f}  acc {accuracy:.4f}  "
                  f"{time.time() - started:.1f}s")

    print(f"\nloss {history[0]:.6f} -> {history[-1]:.6f} "
          f"({'descending' if history[-1] < history[0] else 'NOT descending'})")
    print(f"fallback_ops: {neuron_env.fallback_ops() or 'none'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
