#!/usr/bin/env python
"""
Write, or compare, an alignment artifact bundle.

A bundle is one directory of tensors that pins everything a device port can get
wrong. Dump one per device, then diff them:

    # CPU arm
    PYTHONPATH=src python scripts/dump_artifacts.py --device cpu --dtype bf16 \
        --inputs-from artifacts/cuda_seq200/inputs.pt \
        --out-dir artifacts/cpu_seq200_fwd --forward-only

    # device arm, then the diff
    NEURON_RT_VISIBLE_CORES=0 PYTHONPATH=src python scripts/dump_artifacts.py \
        --device neuron --dtype bf16 --inputs-from artifacts/cuda_seq200/inputs.pt \
        --out-dir artifacts/neuron_seq200_fwd --forward-only
    PYTHONPATH=src python scripts/dump_artifacts.py --compare \
        artifacts/cuda_seq200 artifacts/neuron_seq200_fwd

Files, and what each one catches:

| file            | contents                                             |
|-----------------|------------------------------------------------------|
| `inputs.pt`     | the exact batches, replayed                           |
| `blocks0_10.pt` | the residual stream, `block_out.k` + `mlp_l3.k`       |
| `hidden0.pt`    | the read-out the head consumes                        |
| `w0.pt`         | trainable weights before step 0                       |
| `g1.pt`         | gradients after the first backward, before any step   |
| `wN.pt`         | trainable weights after N steps                       |
| `losses.json`   | the loss at each step                                 |
| `manifest.json` | shapes, param counts, resolved config, provenance     |

**`--init-from` and `--inputs-from` are effectively mandatory for any
cross-device comparison.** `torch.manual_seed` feeds a *per-device* generator, so
seeding alone gives CPU and Neuron unrelated head weights — measured rel_err
1.478, i.e. a comparison of two different models. Pin both and the gradients are
the only free variable.

Gate the stream first. On evo2-7b the read-out is a 0.42% correction to the
residual stream by norm, so `hidden0` reports *that* something diverged and never
*where*.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo2_finetune import neuron_env  # noqa: E402

neuron_env.configure()

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from evo2_finetune.compare import run_compare  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
STREAM_FILE = "blocks0_10.pt"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def enforce_determinism(strict: bool, seed: int) -> None:
    """Seed everything and, under `strict`, refuse non-deterministic kernels.

    `use_deterministic_algorithms(True)` turns a silently non-reproducible op
    into an exception, which is the only way to find out that a bundle you were
    treating as ground truth is not reproducible even against itself. Run the
    reference twice and diff it against itself before trusting any comparison —
    a self-noise floor above zero bounds every conclusion you can draw.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(strict, warn_only=not strict)

    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_first_forward(model) -> dict[str, Any]:
    """Stash the FIRST forward's read-out and the whole residual stream behind it.

    Wraps the backbone's read-out so the capture happens before any optimizer
    step, which is what makes `hidden0` / `blocks0_10` comparable against a
    bundle produced with a different objective: at step 0 `lora_B` is zero, so
    the adapters are exactly the identity and both files are pure backbone.

    `blocks0_10` exists because `hidden0` alone is not debuggable. See the module
    docstring.
    """
    captured: dict[str, Any] = {}
    backbone = model.backbone
    original = backbone.forward

    def wrapped(input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if "h" in captured:
            return original(input_ids, attention_mask)

        stream: dict[str, Tensor] = {}
        handles = []

        def record(name: str):
            def hook(_module, _args, output) -> None:
                tensor = output[0] if isinstance(output, tuple) else output

                if isinstance(tensor, Tensor):
                    stream[name] = tensor.detach().to("cpu", torch.float32)

            return hook

        # Block count is read off the module, never hard-coded.
        for index, block in enumerate(backbone.module.blocks):
            handles.append(block.register_forward_hook(record(f"block_out.{index}")))
            l3 = getattr(getattr(block, "mlp", None), "l3", None)

            if l3 is not None:
                handles.append(l3.register_forward_hook(record(f"mlp_l3.{index}")))

        try:
            hidden = original(input_ids, attention_mask)
        finally:
            for handle in handles:
                handle.remove()

        captured["h"] = hidden.detach().to("cpu", torch.float32)
        captured["blocks"] = stream

        return hidden

    backbone.forward = wrapped

    return captured


def trainable_weights(model) -> dict[str, Tensor]:
    return {
        name: param.detach().to("cpu", torch.float32)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def trainable_grads(model) -> dict[str, Tensor]:
    """`.grad` of every trainable parameter. Call after backward, before step.

    A parameter whose grad is `None` is **omitted**, deliberately: it then shows
    up as a MISSING key in the diff rather than as a zero tensor that quietly
    passes every numerical gate. Zero-initialised `lora_B` makes an absent
    gradient look small; it must not also look present.
    """
    return {
        name: param.grad.detach().to("cpu", torch.float32)
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }


def native_dtypes(model) -> dict[str, int]:
    """Histogram of the trainable params' on-device dtypes, before the fp32 dump."""
    counts: dict[str, int] = {}

    for _name, param in model.named_parameters():
        if param.requires_grad:
            key = str(param.dtype)
            counts[key] = counts.get(key, 0) + 1

    return counts


def manifest_of(tensors: dict[str, Tensor]) -> dict[str, Any]:
    return {
        "num_tensors": len(tensors),
        "num_params": sum(t.numel() for t in tensors.values()),
        "tensors": {
            name: {"shape": list(t.shape), "numel": t.numel()}
            for name, t in sorted(tensors.items())
        },
    }


def provenance() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
    }

    for module_name, key in (("torch_neuronx", "torch_neuronx"), ("neuronxcc", "neuronx_cc"),
                             ("nki", "nki"), ("numpy", "numpy")):
        try:
            record[key] = __import__(module_name).__version__
        except Exception:
            record[key] = None

    return record


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def synth_labels(batch: dict[str, Tensor], args) -> dict[str, Tensor]:
    """Deterministic labels for a replayed batch.

    A saved `inputs.pt` carries tokens, not labels. The label VALUES are
    irrelevant to a parity check — both arms get the same tensor — but they must
    exist, because a `None` loss backpropagates nothing and every gate downstream
    would pass vacuously.

    Drawn from an explicit CPU generator, never `torch.manual_seed`: the global
    RNG is per-device, so seeding it would hand CPU and Neuron different labels.
    """
    length = batch["input_ids"].shape[1]
    rows = batch["input_ids"].shape[0]
    generator = torch.Generator().manual_seed(args.seed)
    batch = dict(batch)

    if args.objective == "window-regression":
        windows = length // args.window
        batch["labels"] = torch.rand((rows, windows, args.out_dim), generator=generator)
        batch["weights"] = torch.ones((rows, windows))
    elif args.objective in ("sequence-regression",):
        batch["labels"] = torch.rand((rows, args.out_dim), generator=generator)
    elif args.objective in ("sequence-classification",):
        batch["labels"] = torch.randint(0, args.out_dim, (rows,), generator=generator)
    elif args.objective == "per-position-classification":
        batch["labels"] = torch.randint(0, args.out_dim, (rows, length), generator=generator)
        batch["weights"] = batch["attention_mask"].cpu()
    elif args.objective == "per-position-regression":
        batch["labels"] = torch.rand((rows, length, args.out_dim), generator=generator)
        batch["weights"] = batch["attention_mask"].cpu()

    return batch


def build_batches(args, device) -> list[dict[str, Tensor]]:
    """Either replay a saved `inputs.pt` or synthesise fresh, reproducible batches."""
    if args.inputs_from:
        saved = torch.load(args.inputs_from, map_location="cpu", weights_only=False)
        batches = []

        for batch in saved["batches"][: max(args.steps, 1)]:
            batch = {k: v.cpu() for k, v in batch.items() if torch.is_tensor(v)}

            if args.objective != "next-token-ce" and "labels" not in batch:
                batch = synth_labels(batch, args)

            batches.append({k: v.to(device) for k, v in batch.items()})

        return batches

    from torch.utils.data import DataLoader

    from evo2_finetune.data import SyntheticDNADataset
    from train_lora_ddp import COLLATORS

    dataset = SyntheticDNADataset(
        size=args.steps * args.batch_size, length=args.seq_len,
        window=args.window, seed=args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=COLLATORS[args.objective](args),
    )

    return [
        {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        for batch in list(loader)[: args.steps]
    ]


def save_inputs(path: Path, batches: list[dict[str, Tensor]]) -> None:
    torch.save({"batches": [{k: v.cpu() for k, v in b.items()} for b in batches]}, path)


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------

def run_dump(args) -> int:
    from evo2_finetune.model import build_model

    enforce_determinism(args.strict_determinism, args.seed)

    device = (neuron_env.resolve_device("neuron") if args.device == "neuron"
              else torch.device(args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        objective=args.objective,
        alias=args.backbone,
        out_dim=args.out_dim,
        device="cpu",
        dtype=DTYPES[args.dtype],
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head=args.head,
        head_kwargs=({"window": args.window} if args.objective == "window-regression" else {}),
        random_init=args.random_backbone,
        readout_block=args.readout_block,
        max_blocks=args.max_blocks,
        checkpoint=args.checkpoint,
        hcl_kernel=args.hcl_kernel,
    )

    if args.init_from:
        state = torch.load(args.init_from, map_location="cpu", weights_only=False)
        loaded = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in state:
                    param.copy_(state[name].to(param.dtype))
                    loaded += 1

        expected = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        print(f"init-from: loaded {loaded}/{expected} trainable tensors from {args.init_from}")

        if loaded < expected:
            raise SystemExit(
                f"init-from covered only {loaded} of {expected} trainable tensors. "
                "A partial pin makes every downstream comparison meaningless — the "
                "unpinned tensors are drawn from a per-device generator."
            )

    dtypes = native_dtypes(model)
    model = model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.compile_blocks:
        model.backbone.compile_blocks(
            backend="neuron" if args.device == "neuron" else "inductor")

    captured = capture_first_forward(model)
    batches = build_batches(args, device)
    save_inputs(out_dir / "inputs.pt", batches)

    torch.save(trainable_weights(model), out_dir / "w0.pt")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, foreach=False,
    )

    losses: list[float] = []
    started = time.time()

    for step in range(args.steps):
        batch = batches[step % len(batches)]
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = model(**batch)
        loss = out["loss"]

        if args.forward_only:
            neuron_env.synchronize()
            losses.append(float(loss.detach().cpu()) if loss is not None else float("nan"))
            break

        loss.backward()

        if step == 0:
            neuron_env.synchronize()
            torch.save(trainable_grads(model), out_dir / "g1.pt")

        optimizer.step()
        neuron_env.synchronize()
        losses.append(float(loss.detach().cpu()))
        print(f"  step {step}: loss {losses[-1]:.8f}")

    if not args.forward_only:
        torch.save(trainable_weights(model), out_dir / "wN.pt")

    torch.save({"h": captured["h"]}, out_dir / "hidden0.pt")

    if not args.no_dump_blocks:
        torch.save(captured["blocks"], out_dir / STREAM_FILE)

    (out_dir / "losses.json").write_text(json.dumps(losses, indent=2))

    weights = torch.load(out_dir / "w0.pt", map_location="cpu", weights_only=True)
    manifest = {
        "config": {
            "backbone": args.backbone,
            "objective": args.objective,
            "head": args.head,
            "device": args.device,
            "dtype": args.dtype,
            "seq_len": args.seq_len,
            "steps": args.steps if not args.forward_only else 0,
            "forward_only": args.forward_only,
            "batch_size": args.batch_size,
            "readout_block": model.backbone.readout_block,
            "readout": model.backbone.readout_name,
            "blocks_built": model.backbone.module.num_built_blocks,
            "hidden_size": model.backbone.hidden_size,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "hcl_kernel": args.hcl_kernel,
            "compile_blocks": args.compile_blocks,
            "gradient_checkpointing": args.gradient_checkpointing,
            "strict_determinism": args.strict_determinism,
            "random_backbone": args.random_backbone,
            "native_trainable_dtypes": dtypes,
        },
        "provenance": provenance(),
        "w0": manifest_of(weights),
        "hidden0": {"shape": list(captured["h"].shape),
                    "norm": float(captured["h"].double().norm())},
        "stream": sorted(captured["blocks"]),
        "wall_clock_s": round(time.time() - started, 3),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    fallbacks = neuron_env.fallback_ops()
    print(f"\nwrote {out_dir}")
    print(f"  read-out {list(captured['h'].shape)}  norm {manifest['hidden0']['norm']:.6f}")
    print(f"  stream   {len(captured['blocks'])} tensors")
    print(f"  losses   {losses}")
    print(f"  fallback_ops {fallbacks or 'none'}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--compare", nargs=2, metavar=("REF", "TEST"),
                        help="diff two bundles instead of writing one")
    parser.add_argument("--gate-stream", type=float, default=None,
                        help="fail the compare if any block increment exceeds this")

    parser.add_argument("--device", default="cpu", choices=["cpu", "neuron"])
    parser.add_argument("--backbone", default="evo2-7b", choices=["evo2-1b", "evo2-7b"])
    parser.add_argument("--objective", default="window-regression")
    parser.add_argument("--head", default=None)
    parser.add_argument("--out-dim", type=int, default=1)
    parser.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--window-width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--readout-block", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--random-backbone", action="store_true")
    parser.add_argument("--hcl-kernel", default=None,
                        choices=["direct", "chunked", "nki-scan"])
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--forward-only", action="store_true",
                        help="one forward, no backward; the 8192 gate that fits in 24 GiB")
    parser.add_argument("--no-dump-blocks", action="store_true",
                        help="skip blocks0_10.pt; it is 2.9 GB at seq 8192")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-from", default=None, help="pin trainable weights from a w0.pt")
    parser.add_argument("--inputs-from", default=None, help="replay batches from an inputs.pt")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.compare:
        return run_compare(Path(args.compare[0]), Path(args.compare[1]), args.gate_stream)

    if not args.out_dir:
        raise SystemExit("--out-dir is required unless --compare is given")

    return run_dump(args)


if __name__ == "__main__":
    sys.exit(main())
