#!/usr/bin/env python
"""
Census of ops that ran on the host instead of the device.

`torch_neuronx.get_fallback_ops()` after a full step is the standard check. This
is stricter in three ways, each of which matters:

* **Per stage.** `clear_op_tracking()` between forward / backward /
  `optimizer.step` / eval-forward. One cumulative list localises nothing.
* **With a positive control.** `get_executed_ops()` must be **non-empty**. An
  empty fallback list is also what a stage that executed nothing returns, so
  without this the gate passes vacuously.
* **Per dtype.** The fallback set is dtype-dependent — on torch-neuronx 2.14,
  bf16's forward executes 48 distinct ops against fp32's 50 — so a green fp32 run
  says nothing about bf16.

    NEURON_RT_VISIBLE_CORES=0 PYTHONPATH=src python -u scripts/check_fallback_ops.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo2_finetune import neuron_env  # noqa: E402

neuron_env.configure()

import torch  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def tracking():
    import torch_neuronx

    return torch_neuronx


def stage(name: str, fn) -> tuple[str, int, list[str]]:
    tnx = tracking()
    tnx.clear_op_tracking()
    fn()
    neuron_env.synchronize()
    executed = sorted(set(tnx.get_executed_ops()))
    fallback = sorted(set(tnx.get_fallback_ops()))

    return name, len(executed), fallback


def run(args, dtype_name: str) -> int:
    from evo2_finetune.data import SyntheticDNADataset, WindowTrackCollator
    from evo2_finetune.model import build_model

    dtype = DTYPES[dtype_name]
    device = neuron_env.resolve_device("neuron")

    model = build_model(
        objective=args.objective, alias=args.backbone, out_dim=1,
        device="cpu", dtype=dtype, lora_rank=args.lora_rank,
        head_kwargs={"window": args.window},
        random_init=args.random_backbone, max_blocks=args.max_blocks,
        hcl_kernel=args.hcl_kernel,
    ).to(device)

    dataset = SyntheticDNADataset(size=2, length=args.seq_len, window=args.window, seed=0)
    collator = WindowTrackCollator(length=args.seq_len, window=args.window)
    batch = {k: v.to(device) for k, v in collator([dataset[0]]).items()}
    batch["labels"] = batch["labels"].to(dtype)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, foreach=False)

    state: dict = {}

    def forward():
        model.train()
        state["loss"] = model(**batch)["loss"]

    def backward():
        model.zero_grad(set_to_none=True)
        state["loss"].backward()

    def clip():
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, foreach=True)

    def step():
        optimizer.step()

    def eval_forward():
        model.eval()

        with torch.no_grad():
            model(**batch)

    # Warm up first: the very first call compiles, and a compile emits ops that
    # are not part of steady state.
    forward()
    backward()
    step()

    rows = [
        stage("forward", forward),
        stage("backward", backward),
        stage("clip_grad_norm", clip),
        stage("optimizer.step", step),
        stage("eval forward", eval_forward),
    ]

    print(f"\nSTEP-2 GATE  --  {args.backbone}, {dtype_name}, seq {args.seq_len}, eager")
    print(f"{'stage':<16}{'executed':>10}{'fallback':>10}")

    failures = 0

    for name, executed, fallback in rows:
        # No stage is exempt. The earlier private-beta build sent one scalar
        # `aten::_to_copy` in `clip_grad_norm_` to the host and this gate waived
        # it; torch-neuronx 2.14 runs the clip on the device, so it is gated too.
        print(f"{name:<16}{executed:>10}{len(fallback):>10}"
              f"{('  ' + str(fallback)) if fallback else ''}")

        # A stage that executed nothing is not a pass — it is a stage that did
        # not run, and its empty fallback list means nothing.
        if executed == 0:
            print("    ^ executed 0 ops: this stage did not run, so its empty "
                  "fallback list is vacuous")
            failures += 1

        failures += len(fallback)

    print(f"GATE: {'PASS' if failures == 0 else f'FAIL ({failures})'}")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="evo2-7b")
    parser.add_argument("--objective", default="window-regression")
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"], choices=["fp32", "bf16"])
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--hcl-kernel", default="nki-scan")
    parser.add_argument("--random-backbone", action="store_true")
    args = parser.parse_args(argv)

    failures = sum(run(args, name) for name in args.dtypes)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
