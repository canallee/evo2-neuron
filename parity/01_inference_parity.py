#!/usr/bin/env python
"""
01 — Inference parity: the ported forward, against the saved CUDA reference.

Two axes in one run:

  * **port vs CUDA** — does a vortex-free, TransformerEngine-free, flash-attn-free
    re-implementation reproduce what the original produced on an H200?
  * **CPU vs Neuron** — does the device lowering change anything?

Both are gated on the **residual stream**, `blocks0_10.pt`, before the read-out.
On evo2-7b the read-out is a 0.42% correction to the stream by norm, so `hidden0`
reports *that* something diverged and never *where*. And the metric on the stream
is the increment `delta_k = block_out.k - block_out.{k-1}`, not the cumulative
tensor: block 3 contributes 1.3% of `block_out.3`, so a completely broken
attention block reads ~0.013 cumulatively — inside the bf16 noise floor.

**Gate the CUDA axis in bf16.** The reference's `dtype` field says fp32, but every
element of all 22 stream tensors survives a bf16 round-trip unchanged: the CUDA
runtime materialised the residual stream in bf16 at every module boundary
regardless of that flag. An fp32 arm measures a dtype difference on top of
everything else.

    NEURON_RT_VISIBLE_CORES=0 python -u parity/01_inference_parity.py
    python -u parity/01_inference_parity.py --devices cpu     # no device needed
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# `reference` calls `neuron_env.configure()` at import, which must happen before
# anything pulls in torch -- `import torch` auto-loads `torch_neuronx`, and the
# compiler cache directories are read at that moment. Import it FIRST.
from reference import (  # noqa: E402
    CUDA_SEQ200,
    CUDA_SEQ8192,
    banner,
    build,
    build_inputs,
    cosine,
    load_reference,
    rel_err,
    report_fallbacks,
)

import torch  # noqa: E402

from evo2_finetune.compare import _STREAM_COS_FLOOR, _STREAM_FLOOR  # noqa: E402


def capture(model, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One forward; return the read-out and the whole residual stream."""
    stream: dict[str, torch.Tensor] = {}
    handles = []

    def record(name: str):
        def hook(_module, _args, output) -> None:
            tensor = output[0] if isinstance(output, tuple) else output

            if isinstance(tensor, torch.Tensor):
                stream[name] = tensor.detach().to("cpu", torch.float32)

        return hook

    for index, block in enumerate(model.backbone.module.blocks):
        handles.append(block.register_forward_hook(record(f"block_out.{index}")))
        l3 = getattr(getattr(block, "mlp", None), "l3", None)

        if l3 is not None:
            handles.append(l3.register_forward_hook(record(f"mlp_l3.{index}")))

    try:
        model.eval()

        with torch.no_grad():
            hidden = model.backbone(batch["input_ids"], batch.get("attention_mask"))
    finally:
        for handle in handles:
            handle.remove()

    return hidden.detach().to("cpu", torch.float32), stream


def gate_stream(reference: dict, test: dict, label: str) -> int:
    """Per-block increment table. Returns the number of diverging blocks."""
    indices = sorted(
        int(k.split(".")[1]) for k in set(reference) & set(test) if k.startswith("block_out.")
    )
    print(f"\n  {label}: FLOOR={_STREAM_FLOOR:.1e}, direction floor "
          f"{_STREAM_COS_FLOOR:.0e}; a block diverges only if it fails BOTH")
    print(f"  {'k':>3} {'relerr(out)':>12} {'relerr(delta)':>14} {'1-cos(delta)':>13}"
          f" {'relerr(l3)':>11}")

    diverged = []

    for index in indices:
        out_key, prev_key, l3_key = (
            f"block_out.{index}", f"block_out.{index - 1}", f"mlp_l3.{index}")
        ref_out, test_out = reference[out_key], test[out_key]

        if ref_out.shape != test_out.shape:
            print(f"  {index:>3}  SHAPE {list(ref_out.shape)} vs {list(test_out.shape)}")
            diverged.append(index)
            continue

        has_prev = prev_key in reference and prev_key in test

        if has_prev:
            ref_delta = ref_out.double() - reference[prev_key].double()
            test_delta = test_out.double() - test[prev_key].double()
        else:
            ref_delta, test_delta = ref_out.double(), test_out.double()

        delta_err = rel_err(ref_delta, test_delta)
        delta_cos = cosine(ref_delta, test_delta)
        l3_err = (rel_err(reference[l3_key], test[l3_key])
                  if l3_key in reference and l3_key in test else float("nan"))

        # For block 0 the increment is not isolable from this file — `delta_0` is
        # really `block_out.0`, which sits at a ~4x cancellation against the
        # embedding and reads several times the floor even when correct. Gate the
        # undiluted `mlp_l3.0` there instead.
        gated = delta_err if has_prev else l3_err
        bad = gated == gated and gated > _STREAM_FLOOR and (1 - delta_cos) > _STREAM_COS_FLOOR

        if bad:
            diverged.append(index)

        print(f"  {index:>3} {rel_err(ref_out, test_out):12.3e} {delta_err:14.3e} "
              f"{1 - delta_cos:13.3e} {l3_err:11.3e}"
              f"{'   <-- DIVERGES' if bad else ''}")

    if diverged:
        print(f"  FIRST DIVERGENCE at block {diverged[0]}: bisect inside it "
              "(pre_norm -> projections -> filter -> out_filter_dense -> mlp). "
              "Nothing downstream is evidence.")

    return len(diverged)


def run(bundle: Path, devices: list[str], dtype: torch.dtype, seq_label: str) -> int:
    banner(f"01 — inference parity vs {bundle.name} ({seq_label}, {dtype})")

    ref_stream = load_reference(bundle, "blocks0_10.pt")
    ref_hidden = load_reference(bundle, "hidden0.pt")["h"]
    print(f"  reference: {len(ref_stream)} stream tensors, read-out "
          f"{list(ref_hidden.shape)} norm {float(ref_hidden.double().norm()):.6f}")

    failures = 0
    captured: dict[str, tuple] = {}

    for device in devices:
        model = build(device, dtype=dtype, window=200)
        batch = build_inputs(device, bundle)
        hidden, stream = capture(model, batch)
        captured[device] = (hidden, stream)

        print(f"\n  [{device}] read-out {list(hidden.shape)} "
              f"norm {float(hidden.double().norm()):.6f}")
        print(f"  [{device}] hidden0 vs CUDA: rel_err {rel_err(ref_hidden, hidden):.3e}  "
              f"1-cos {1 - cosine(ref_hidden, hidden):.3e}")

        failures += gate_stream(ref_stream, stream, f"[{device}] stream vs CUDA")

        fallbacks = report_fallbacks(device)
        print(f"  [{device}] fallback_ops: {fallbacks or 'none'}")
        failures += len(fallbacks)

        del model

    if len(devices) == 2:
        (h_cpu, s_cpu), (h_dev, s_dev) = captured["cpu"], captured["neuron"]
        print(f"\n  CPU vs Neuron: hidden0 rel_err {rel_err(h_cpu, h_dev):.3e}  "
              f"1-cos {1 - cosine(h_cpu, h_dev):.3e}")
        failures += gate_stream(s_cpu, s_dev, "  stream CPU vs Neuron")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", nargs="+", default=["cpu", "neuron"],
                        choices=["cpu", "neuron"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                        help="bf16 matches the reference's own residual-stream dtype")
    parser.add_argument("--seq", default="200", choices=["200", "8192", "both"])
    args = parser.parse_args(argv)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    failures = 0

    if args.seq in ("200", "both"):
        failures += run(CUDA_SEQ200, args.devices, dtype, "seq 200")

    if args.seq in ("8192", "both"):
        failures += run(CUDA_SEQ8192, args.devices, dtype, "seq 8192")

    print(f"\nGATE (0 diverging blocks, no fallbacks): "
          f"{'PASS' if failures == 0 else f'FAIL ({failures})'}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
