#!/usr/bin/env python
"""
03 — Training parity: N AdamW steps, CPU vs Neuron.

Asserts three things, and the third is the one people forget:

  1. the loss **trajectory** matches step for step,
  2. the **final weights** match after N steps,
  3. the loss actually **descends**.

Without (3) a port that returns a constant passes (1) and (2) perfectly. A
frozen model is bit-identical across devices.

Weights are pinned from a saved `w0.pt` and inputs replayed from a saved
`inputs.pt`, so the gradients are the only free variable. See `reference.py` for
why seeding is not a substitute.

    NEURON_RT_VISIBLE_CORES=0 python -u parity/03_training_parity.py
"""

from __future__ import annotations

# Import order is load-bearing and must not be sorted: `reference` calls
# `neuron_env.configure()` at import, and that has to happen before anything
# pulls in torch -- `import torch` auto-loads `torch_neuronx`, which reads the
# compiler cache directories at that moment.
# ruff: isort: off

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import (  # noqa: E402
    CUDA_SEQ200,
    banner,
    build,
    build_inputs,
    cosine,
    report_fallbacks,
)

import torch  # noqa: E402

from importlib import import_module  # noqa: E402

synth_labels = import_module("02_backprop_parity").synth_labels

LOSS_GATE = 1e-2
COS_GATE = 0.99
ABS_GATE = 1e-3


def train(device: str, objective: str, dtype, steps: int, lr: float,
          init_from: Path, out_dim: int):
    model = build(device, objective=objective, dtype=dtype, out_dim=out_dim,
                  init_from=init_from)
    batch = synth_labels(build_inputs(device, CUDA_SEQ200), objective, out_dim, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    # `foreach=False` forces the per-tensor loop. The grouped path is ~1.4%
    # faster and is where a long-run defect lives; a parity check has no reason
    # to take that trade.
    optimizer = torch.optim.AdamW(trainable, lr=lr, foreach=False)

    model.train()
    losses = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch)["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    final = {
        name: param.detach().cpu().float()
        for name, param in model.named_parameters() if param.requires_grad
    }

    return losses, final, report_fallbacks(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objective", default="window-regression")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--out-dim", type=int, default=2)
    parser.add_argument("--init-from", default=None)
    args = parser.parse_args(argv)

    banner(f"03 — training parity: {args.steps} AdamW steps, {args.objective} ({args.dtype})")

    pin = Path(args.init_from) if args.init_from else (
        Path(__file__).resolve().parents[1] / "artifacts" / "cpu_seq200_window" / "w0.pt")

    if not pin.exists():
        print(f"  SKIP: no w0.pt to pin from ({pin}). Run scripts/dump_artifacts.py "
              "on CPU first.")
        return 1

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    cpu_losses, cpu_final, _ = train(
        "cpu", args.objective, dtype, args.steps, args.lr, pin, args.out_dim)
    dev_losses, dev_final, fallbacks = train(
        "neuron", args.objective, dtype, args.steps, args.lr, pin, args.out_dim)

    print(f"\n  {'step':>4} {'cpu':>16} {'neuron':>16} {'rel':>11}")
    worst_loss = 0.0

    for index, (a, b) in enumerate(zip(cpu_losses, dev_losses, strict=True)):
        relative = abs(b - a) / (abs(a) + 1e-12)
        worst_loss = max(worst_loss, relative)
        print(f"  {index:>4} {a:16.8f} {b:16.8f} {relative:11.3e}")

    descends = cpu_losses[-1] < cpu_losses[0] and dev_losses[-1] < dev_losses[0]
    print(f"\n  loss descends: cpu {cpu_losses[0]:.6f} -> {cpu_losses[-1]:.6f}   "
          f"neuron {dev_losses[0]:.6f} -> {dev_losses[-1]:.6f}   "
          f"{'YES' if descends else 'NO -- a frozen model would pass every other gate'}")

    shared = sorted(set(cpu_final) & set(dev_final))
    scale = max((float(w.abs().max()) for w in cpu_final.values()), default=0.0)
    real = []

    for name in shared:
        ref, test = cpu_final[name], dev_final[name]
        cos = cosine(ref, test)
        relative = float((test.double() - ref.double()).abs().max()) / (scale or 1.0)

        if cos < COS_GATE and relative > ABS_GATE:
            real.append((name, cos, relative))

    print(f"  final weights: {len(shared)} shared, {len(real)} real disagreement(s)")

    for name, cos, relative in real[:10]:
        print(f"     cos={cos:+.6f}  max_abs/scale={relative:.3e}  {name}")

    print(f"  worst loss rel_err {worst_loss:.3e} (gate {LOSS_GATE:g})")
    print(f"  fallback_ops       {fallbacks or 'none'}")

    ok = worst_loss <= LOSS_GATE and not real and descends and not fallbacks
    print(f"\nGATE (loss trajectory, final weights, loss descends, no fallbacks): "
          f"{'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
