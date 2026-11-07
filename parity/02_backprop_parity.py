#!/usr/bin/env python
"""
02 — Backprop parity: one backward, CPU vs Neuron, per objective.

The gate is **magnitude-aware**, not bare cosine. `lora_B` is zero-initialised, so
its gradient is genuinely tiny and a per-tensor relative error manufactures
alarming ratios out of floating-point noise — on this model the `lora_B` grads
show rel_err ten times the head's (2.8e-4 against 2.9e-5, window-regression, fp32),
though the head is the tensor that carries real magnitude. A tensor counts as a
REAL disagreement only if it fails BOTH:

    cosine < 0.99   AND   max_abs / (max |grad| anywhere in the model) > 1e-3

There is a second check here that the numerical gate structurally cannot do.
Because `lora_B` starts at zero, an **absent** gradient is indistinguishable from
a small one by magnitude — and two real bugs hid in exactly that gap: a
non-differentiable custom kernel, and reentrant activation checkpointing dropping
every adapter but the top one. Both produced `None`, not a wrong number. So this
check also asserts that no trainable tensor comes back with `grad is None`.

    NEURON_RT_VISIBLE_CORES=0 python -u parity/02_backprop_parity.py
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

COS_GATE = 0.99
ABS_GATE = 1e-3


def grads(device: str, objective: str, dtype, init_from: Path, out_dim: int):
    model = build(device, objective=objective, dtype=dtype, out_dim=out_dim,
                  init_from=init_from)
    batch = build_inputs(device, CUDA_SEQ200)
    batch = synth_labels(batch, objective, out_dim, device)

    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(**batch)["loss"]
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    collected = {
        name: param.grad.detach().cpu().float()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    fallbacks = report_fallbacks(device)

    return collected, float(loss.detach().cpu()), missing, fallbacks


def synth_labels(batch: dict, objective: str, out_dim: int, device: str) -> dict:
    """Deterministic labels. Their VALUES do not matter — both sides get the same
    tensor — but they must exist, because a `loss` of `None` backpropagates
    nothing and the check would pass vacuously."""
    target = batch["input_ids"].device
    length = batch["input_ids"].shape[1]
    generator = torch.Generator().manual_seed(0)
    batch = dict(batch)

    if objective == "window-regression":
        windows = length // 200
        batch["labels"] = torch.rand((1, windows, out_dim), generator=generator).to(target)
        batch["weights"] = torch.ones((1, windows)).to(target)
    elif objective == "sequence-regression":
        batch["labels"] = torch.rand((1, out_dim), generator=generator).to(target)
    elif objective == "per-position-classification":
        batch["labels"] = torch.randint(0, out_dim, (1, length), generator=generator).to(target)
        batch["weights"] = batch["attention_mask"]

    return batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objectives", nargs="+",
                        default=["window-regression", "per-position-classification"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--out-dim", type=int, default=2)
    parser.add_argument("--init-from", default=None,
                        help="w0.pt to pin trainables from; defaults to the CPU bundle")
    args = parser.parse_args(argv)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    failures = 0

    for objective in args.objectives:
        banner(f"02 — backprop parity: {objective} ({args.dtype})")

        pin = Path(args.init_from) if args.init_from else _default_pin(objective)

        if pin is None or not pin.exists():
            print(f"  SKIP: no w0.pt to pin from (looked for {pin}). "
                  "Run scripts/dump_artifacts.py on CPU first — seeding instead of "
                  "pinning gives the two devices unrelated head weights.")
            failures += 1
            continue

        cpu_grads, cpu_loss, cpu_missing, _ = grads("cpu", objective, dtype, pin, args.out_dim)
        dev_grads, dev_loss, dev_missing, fallbacks = grads(
            "neuron", objective, dtype, pin, args.out_dim)

        print(f"  cpu    loss {cpu_loss:.8f}   {len(cpu_grads)} grad tensors")
        print(f"  neuron loss {dev_loss:.8f}   {len(dev_grads)} grad tensors")
        print(f"  loss rel_err {abs(dev_loss - cpu_loss) / (abs(cpu_loss) + 1e-12):.3e}")

        if cpu_missing or dev_missing:
            print(f"  MISSING GRADIENTS  cpu {len(cpu_missing)}  neuron {len(dev_missing)}")
            for name in (cpu_missing + dev_missing)[:10]:
                print(f"     grad is None: {name}")
            failures += len(cpu_missing) + len(dev_missing)

        shared = sorted(set(cpu_grads) & set(dev_grads))
        key_mismatch = sorted(set(cpu_grads) ^ set(dev_grads))
        scale = max((float(g.abs().max()) for g in cpu_grads.values()), default=0.0)

        real = []

        for name in shared:
            ref, test = cpu_grads[name], dev_grads[name]
            cos = cosine(ref, test)
            relative = float((test.double() - ref.double()).abs().max()) / (scale or 1.0)

            if cos < COS_GATE and relative > ABS_GATE:
                real.append((name, cos, relative))

        print(f"  global grad scale  {scale:.6e}")
        print(f"  shared tensors     {len(shared)}   key mismatches {len(key_mismatch)}")
        print(f"  REAL disagreements {len(real)} "
              f"(cos < {COS_GATE} AND max_abs/scale > {ABS_GATE:g})")

        for name, cos, relative in real[:10]:
            print(f"     cos={cos:+.6f}  max_abs/scale={relative:.3e}  {name}")

        print(f"  fallback_ops       {fallbacks or 'none'}")
        failures += len(real) + len(key_mismatch) + len(fallbacks)

    print(f"\nGATE (0 real disagreements, no key mismatch, no missing grads, "
          f"no fallbacks): {'PASS' if failures == 0 else f'FAIL ({failures})'}")

    return 0 if failures == 0 else 1


def _default_pin(objective: str) -> Path | None:
    root = Path(__file__).resolve().parents[1] / "artifacts"
    suffix = {"window-regression": "window",
              "per-position-classification": "pernt"}.get(objective)

    return root / f"cpu_seq200_{suffix}" / "w0.pt" if suffix else None


if __name__ == "__main__":
    sys.exit(main())
