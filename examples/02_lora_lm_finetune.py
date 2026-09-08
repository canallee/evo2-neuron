#!/usr/bin/env python
"""
LoRA continued pretraining: adapt Evo 2 to new sequence with its own objective.

    python examples/02_lora_lm_finetune.py --device cpu --random-backbone --max-blocks 2
    NEURON_RT_VISIBLE_CORES=0 python -u examples/02_lora_lm_finetune.py --fasta genome.fa

Next-token cross-entropy over bytes — the objective Evo 2 was pretrained with.
It needs no labels at all, which makes it the objective to reach for when
adapting the backbone to a new clade or assembly before there is any downstream
task to fine-tune on.

**This is the one objective that uses the whole stack.** A read-out head sees
blocks 0..10; the LM head sits on top of block 31, so nothing can be pruned and
the full 13 GB of evo2-7b is loaded. `--max-blocks` truncates it for a smoke
test, which is a *different model* — use it for plumbing, not for numbers.
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

from evo2_finetune.data import (  # noqa: E402
    FastaDataset,
    LanguageModelCollator,
    SyntheticDNADataset,
)
from evo2_finetune.model import build_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "neuron"])
    parser.add_argument("--backbone", default="evo2-7b")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--fasta", default=None, help="else synthetic DNA")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--random-backbone", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = neuron_env.resolve_device(args.device)

    model = build_model(
        objective="next-token-ce", alias=args.backbone, device="cpu", dtype=dtype,
        lora_rank=args.lora_rank, random_init=args.random_backbone,
        max_blocks=args.max_blocks,
    ).to(device)
    print(f"trainable: {model.trainable_summary()}")

    dataset = (FastaDataset(args.fasta, length=args.seq_len) if args.fasta
               else SyntheticDNADataset(size=args.steps * args.batch_size,
                                        length=args.seq_len, window=256, seed=0))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=LanguageModelCollator(length=args.seq_len),
                        drop_last=True)
    print(f"data: {len(dataset)} windows of {args.seq_len} bp"
          f"{' from ' + args.fasta if args.fasta else ' (synthetic)'}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, foreach=False)

    model.train()
    batches = iter(loader)
    started = time.time()

    for step in range(args.steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)

        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, foreach=True)
        optimizer.step()
        neuron_env.synchronize()

        value = float(loss.detach().cpu())
        # Bits per nucleotide is the number to watch: a byte tokenizer over a
        # 512-row vocabulary starts near log(512)/log(2) = 9 bits and real DNA
        # is compressible to roughly 2.
        print(f"  step {step:>3}  loss {value:.6f}  {value / 0.6931:.3f} bits/nt"
              f"  {time.time() - started:.1f}s")

    print(f"\nfallback_ops: {neuron_env.fallback_ops() or 'none'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
