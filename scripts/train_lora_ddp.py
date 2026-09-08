#!/usr/bin/env python
"""
LoRA fine-tuning of Evo 2 on AWS Trainium, data-parallel over every NeuronCore.

    # single core, smoke
    NEURON_RT_VISIBLE_CORES=0 python scripts/train_lora_ddp.py --steps 5

    # all 64 logical cores of a trn2.48xlarge
    NEURON_RT_VISIBLE_CORES=0-63 torchrun --nproc_per_node=64 \
        scripts/train_lora_ddp.py --objective window-regression \
        --seq-len 8192 --dtype bf16 --steps 200 --save-to runs/lora64

Only the LoRA adapters and the read-out head train; the backbone is frozen. On
evo2-7b at r=8 that is ~2.5% of the parameters, which is why a 7B fine-tune fits
on a 24 GiB logical core at all.

The production configuration is `--dtype bf16 --compile-blocks
--gradient-checkpointing --hcl-kernel nki-scan`, batch 1 per rank. Every one of
those is explained in `docs/OPTIMIZATIONS.md`; three of the four are not
optional at seq 8192.

Four things that fail *silently* if you get them wrong, all handled here:

  * `NEURON_RT_VISIBLE_CORES` must be the full range for DDP. Pinned to `0`,
    every rank shares one core and the job is N× slower with no error.
  * The device index must be the **global** core id, not the local rank.
  * `torch._dynamo.config.recompile_limit` defaults to 8. Eleven blocks exceed
    it and dynamo falls back to eager without saying so.
  * `bucket_cap_mb` / `gradient_as_bucket_view` are requested, then **verified**
    from the built DDP object — asking for a bucket configuration and getting it
    are different things.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# MUST run before anything imports torch: `import torch` auto-loads
# `torch_neuronx`, which reads the four cache dirs at import time.
from evo2_finetune import neuron_env  # noqa: E402

# Reads LOCAL_RANK itself and narrows NEURON_RT_VISIBLE_CORES to this rank's
# single core. Must happen before torch is imported, and before the Neuron
# runtime initialises — every rank claiming the same range is what produces
# `Failed to initialize Neuron Runtime: status code 1` on all but one of them.
neuron_env.configure()
# A cold compile of eleven 7B blocks comfortably exceeds the 300 s default, and
# the timeout surfaces as an opaque runtime error rather than "still compiling".
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "600")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from evo2_finetune.data import (  # noqa: E402
    FastaDataset,
    LanguageModelCollator,
    PerPositionCollator,
    SequenceCollator,
    SyntheticDNADataset,
    WindowTrackCollator,
)
from evo2_finetune.lora import get_lora_state_dict, load_lora_state_dict  # noqa: E402
from evo2_finetune.model import build_model  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def init_distributed(device: str) -> tuple[int, int, int]:
    """Return `(rank, world_size, local_rank)`; `(0, 1, 0)` outside torchrun.

    The backend is `"neuron"` — `torch_neuronx` registers it for the `neuron`
    device type. `gloo` initialises fine and then fails at the first collective
    with `No backend type associated with device type neuron`, because gloo has
    no idea what a neuron tensor is.
    """
    if "RANK" not in os.environ:
        return 0, 1, 0

    dist.init_process_group(backend="neuron" if device == "neuron" else "gloo")

    return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))


def resolve_core(local_rank: int) -> tuple[object, int]:
    """This rank's device, and the *global* core id behind it.

    Two separate things, and getting either wrong fails differently:

    * `neuron_env.configure()` narrowed `NEURON_RT_VISIBLE_CORES` to one core for
      this rank. Without that, every rank's runtime claims the whole range and
      all but one die with `Failed to initialize Neuron Runtime: status code 1`.
    * The device is still addressed by its **global** index. After narrowing to
      core 3, `torch.device("neuron", 0)` raises `Device index 0 is out of range.
      Valid range is 3 to 3` — the visible set is narrowed, not renumbered.
    """
    del local_rank  # already consumed by configure()

    core = neuron_env.global_core_id()

    return neuron_env.resolve_device("neuron", core), core


def assert_cores_distinct(core: int, rank: int, world: int) -> None:
    """Fail loudly if two ranks landed on the same core."""
    if world == 1 or not dist.is_initialized():
        return

    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, core))
    seen: dict[int, int] = {}

    for other_rank, other_core in gathered:
        if other_core in seen:
            raise RuntimeError(
                f"ranks {seen[other_core]} and {other_rank} both resolved to core "
                f"{other_core}; the job would run at 1/N throughput"
            )
        seen[other_core] = other_rank


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

COLLATORS = {
    "next-token-ce": lambda a: LanguageModelCollator(length=a.seq_len),
    "sequence-regression": lambda a: SequenceCollator(
        length=a.seq_len, window_width=a.window_width),
    "sequence-classification": lambda a: SequenceCollator(
        length=a.seq_len, key="class_label", window_width=a.window_width),
    "window-regression": lambda a: WindowTrackCollator(length=a.seq_len, window=a.window),
    "per-position-classification": lambda a: PerPositionCollator(length=a.seq_len),
    "per-position-regression": lambda a: PerPositionCollator(length=a.seq_len),
}


def build_loader(args, rank: int, world: int):
    if args.fasta:
        dataset = FastaDataset(args.fasta, length=args.seq_len)
    else:
        dataset = SyntheticDNADataset(
            size=args.dataset_size, length=args.seq_len, window=args.window, seed=args.seed
        )

    sampler = (DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed)
               if world > 1 else None)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=COLLATORS[args.objective](args),
        drop_last=True,
        num_workers=args.num_workers,
    ), sampler


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model, optimizer, scheduler, step: int) -> None:
    """LoRA + head + **optimizer state**.

    The optimizer state is the part that is easy to leave out and expensive to
    discover missing: AdamW's moments restarting at zero on resume shows up as a
    loss spike an order of magnitude above where training left off, which reads
    like a data bug.
    """
    path.mkdir(parents=True, exist_ok=True)
    inner = model.module if isinstance(model, DistributedDataParallel) else model

    lora = {k: v.cpu() for k, v in get_lora_state_dict(inner).items()}
    head = ({k: v.cpu() for k, v in inner.head.state_dict().items()}
            if inner.head is not None else {})

    # Under full fine-tuning the adapters and head are no longer the whole
    # trainable set, and saving only them would silently discard the run.
    extra = {
        name: param.detach().cpu()
        for name, param in inner.named_parameters()
        if param.requires_grad and not name.endswith((".lora_A", ".lora_B"))
        and not name.startswith("head.")
    }

    torch.save(
        {
            "lora": lora,
            "head": head,
            "backbone": extra,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
        },
        path / "trainable.pt",
    )


def load_checkpoint(path: Path, model, optimizer, scheduler) -> int:
    state = torch.load(Path(path) / "trainable.pt", map_location="cpu", weights_only=False)
    inner = model.module if isinstance(model, DistributedDataParallel) else model

    load_lora_state_dict(inner, state["lora"])

    if inner.head is not None and state["head"]:
        inner.head.load_state_dict(state["head"])

    if state.get("backbone"):
        missing, unexpected = inner.load_state_dict(state["backbone"], strict=False)

        if unexpected:
            raise RuntimeError(f"checkpoint has unknown backbone tensors: {unexpected[:5]}")

    if state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])

    return int(state.get("step", 0))


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, world, local_rank = init_distributed(args.device)
    is_main = rank == 0

    def log(message: str) -> None:
        if is_main or args.log_all_ranks:
            print(f"[rank {rank:>2}] {message}", flush=True)

    torch.manual_seed(args.seed + rank)

    if args.device == "neuron":
        device, core = resolve_core(local_rank)
        assert_cores_distinct(core, rank, world)
    else:
        device, core = torch.device(args.device), -1

    log(f"world={world} device={device} core={core} objective={args.objective} "
        f"seq={args.seq_len} dtype={args.dtype}")

    model = build_model(
        objective=args.objective,
        alias=args.backbone,
        out_dim=args.out_dim,
        device=args.device if args.device != "neuron" else "cpu",
        dtype=DTYPES[args.dtype],
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_kwargs=({"window": args.window} if args.objective == "window-regression" else
                     {"width": args.window_width} if args.head == "conv" else {}),
        head=args.head,
        random_init=args.random_backbone,
        readout_block=args.readout_block,
        max_blocks=args.max_blocks,
        checkpoint=args.checkpoint,
        hcl_kernel=args.hcl_kernel,
        full_finetune=args.full_finetune,
    )
    log(f"trainable: {model.trainable_summary()}")

    if args.full_finetune:
        log("FULL FINE-TUNE: the backbone is trainable. Optimizer state is ~8 bytes "
            "per parameter on top of the weights and the gradients — at evo2-7b "
            "blocks 0..10 that is ~27 GB of AdamW moments alone, against 24 GiB per "
            "logical core. Expect this to need a shorter context, a smaller "
            "backbone, or fewer blocks.")

    model = model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.compile_blocks:
        model.backbone.compile_blocks(backend="neuron" if args.device == "neuron" else "inductor")

    trainable = [p for p in model.parameters() if p.requires_grad]

    if world > 1:
        model = DistributedDataParallel(
            model,
            # The GLOBAL core id, not 0 and not the local rank. See `resolve_core`.
            device_ids=None if args.device == "cpu" else [core],
            output_device=None if args.device == "cpu" else core,
            # `find_unused_parameters` costs a full autograd-graph traversal
            # every step and delays bucket readiness, which is exactly what
            # defeats allreduce/backward overlap. Off by default; turn it on only
            # if a trainable tensor genuinely goes unused on some batches.
            find_unused_parameters=args.find_unused_parameters,
            # One bucket for the whole gradient payload: this backend does not
            # benefit from NCCL-style many-small-bucket overlap. The payload
            # depends on the head — 10.3 MB with WindowTrackHead, past 100 MB
            # with a wide ConvReadoutHead — so it is measured and logged below
            # rather than assumed.
            bucket_cap_mb=args.bucket_cap_mb,
            gradient_as_bucket_view=True,
        )
        # Requesting a bucket configuration and getting it are different things,
        # so read it back off the built object rather than echoing the argument.
        payload = sum(p.numel() * p.element_size() for p in trainable)
        actual_cap = getattr(model, "bucket_bytes_cap", None)
        log(f"DDP: gradient payload {payload / 1e6:.1f} MB, bucket cap "
            f"{(actual_cap or 0) / 1e6:.0f} MB "
            f"(requested {args.bucket_cap_mb}), gradient_as_bucket_view=True, "
            f"find_unused_parameters={args.find_unused_parameters}")

        if actual_cap is not None and actual_cap < payload:
            log(f"WARNING: payload {payload / 1e6:.1f} MB exceeds the bucket cap; "
                "gradients will span multiple buckets")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        # `foreach` is auto-selected on this backend and its grouped kernels are
        # where a long-run defect lives. `--adam-foreach false` forces the
        # per-tensor loop: ~1.4% slower, and the job finishes.
        foreach=args.adam_foreach,
    )

    scheduler = None

    if args.lr_scheduler == "cosine":
        warmup = max(int(args.warmup_ratio * args.steps), 1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, 1e-8, 1.0, warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(args.steps - warmup, 1)
                ),
            ],
            milestones=[warmup],
        )

    start_step = 0

    if args.resume_from:
        start_step = load_checkpoint(Path(args.resume_from), model, optimizer, scheduler)
        log(f"resumed from {args.resume_from} at step {start_step}")

    loader, sampler = build_loader(args, rank, world)
    batches = iter(loader)
    epoch = 0

    losses: list[float] = []
    step_times: list[float] = []
    pending: list[torch.Tensor] = []

    log(f"training {args.steps} steps, batch {args.batch_size}/rank, "
        f"grad_accum {args.grad_accum}, effective {args.batch_size * args.grad_accum * world}")

    for step in range(start_step, args.steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for micro in range(args.grad_accum):
            try:
                batch = next(batches)
            except StopIteration:
                epoch += 1

                if sampler is not None:
                    sampler.set_epoch(epoch)

                batches = iter(loader)
                batch = next(batches)

            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

            # All but the last microstep skip the allreduce. Without this every
            # microstep pays the full gradient collective.
            last = micro == args.grad_accum - 1
            context = (model.no_sync() if (world > 1 and not last)
                       else _null_context())

            with context:
                loss = model(**batch)["loss"] / args.grad_accum
                loss.backward()

            pending.append(loss.detach())

        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm, foreach=True)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Pull losses to the host every N steps, not every step. A per-step
        # `float(loss)` is a host fence inside the training loop.
        # The safety rule: no unchecked loss may reach a checkpoint, so the
        # materialisation below always runs before a save and before the end.
        #
        # The first step is always materialised: it is the cold-compile step, it
        # is excluded from every warm number, and reporting it separately is the
        # difference between "3.8 s/step" and "the first step took 20 minutes".
        due = (step == start_step or (step + 1) % args.log_every == 0
               or step + 1 == args.steps)

        if due:
            neuron_env.synchronize()
            values = [float(t.cpu()) * args.grad_accum for t in pending]
            pending.clear()

            if any(v != v for v in values):          # NaN
                raise RuntimeError(f"NaN loss at step {step}; refusing to continue")

            losses.extend(values)

        elapsed = time.perf_counter() - started
        step_times.append(elapsed)

        # Reset the op census once warmup is over, so the fallback list reported
        # at the end describes STEADY STATE. Left cumulative it always contains
        # `aten::_to_copy` — the host-to-device transfer of the weights and the
        # first batch, which is a copy that has to happen, not a fallback of the
        # compute graph. Per-stage attribution lives in check_fallback_ops.py.
        if step == start_step + args.warmup_steps - 1:
            neuron_env.clear_op_tracking()

        if due:
            recent = step_times[-args.log_every:]
            log(f"step {step + 1:>5}/{args.steps}  loss {losses[-1]:.6f}  "
                f"{sum(recent) / len(recent):.4f} s/step  "
                f"lr {optimizer.param_groups[0]['lr']:.3e}")

        if args.save_to and args.save_every and (step + 1) % args.save_every == 0 and is_main:
            save_checkpoint(Path(args.save_to), model, optimizer, scheduler, step + 1)

    neuron_env.synchronize()

    if args.save_to and is_main:
        save_checkpoint(Path(args.save_to), model, optimizer, scheduler, args.steps)
        log(f"saved {args.save_to}/trainable.pt")

    # Warm steady state, excluding the cold-compile step. Reported separately on
    # purpose: conflating compile with execution is how a 0.09 s step gets
    # published as a 92 s one.
    warm = step_times[args.warmup_steps:] or step_times
    warm_sorted = sorted(warm)
    median = warm_sorted[len(warm_sorted) // 2]

    summary = {
        "rank": rank,
        "world_size": world,
        "core": core,
        "objective": args.objective,
        "backbone": args.backbone,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora_rank": args.lora_rank,
        "compile_blocks": args.compile_blocks,
        "gradient_checkpointing": args.gradient_checkpointing,
        "hcl_kernel": args.hcl_kernel,
        "steps": args.steps,
        "cold_step_s": step_times[0] if step_times else None,
        "median_warm_step_s": median,
        "mean_warm_step_s": sum(warm) / len(warm),
        "min_warm_step_s": min(warm),
        "max_warm_step_s": max(warm),
        "samples_per_s": (args.batch_size * args.grad_accum * world) / median,
        "tokens_per_s": (args.batch_size * args.grad_accum * world * args.seq_len) / median,
        "losses": losses,
        "step_times": step_times,
        "fallback_ops": neuron_env.fallback_ops(),
    }

    if args.diag_dir:
        directory = Path(args.diag_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rank{rank:03d}.json").write_text(json.dumps(summary, indent=2))

    log(f"cold {summary['cold_step_s']:.3f} s | warm median {median:.4f} s/step | "
        f"{summary['tokens_per_s']:,.0f} tok/s across {world} rank(s) | "
        f"fallbacks {summary['fallback_ops'] or 'none'}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return 0


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of Evo 2 on Trainium, data-parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_argument_group("model")
    group.add_argument("--backbone", default="evo2-7b", choices=["evo2-1b", "evo2-7b"])
    group.add_argument("--objective", default="window-regression", choices=sorted(COLLATORS))
    group.add_argument("--head", default=None,
                       help="override the objective's default read-out head")
    group.add_argument("--out-dim", type=int, default=1)
    group.add_argument("--readout-block", type=int, default=None)
    group.add_argument("--max-blocks", type=int, default=None,
                       help="build only N blocks; for CPU smoke runs")
    group.add_argument("--checkpoint", default=None,
                       help="local Arc checkpoint (else HF, or $EVO2_CHECKPOINT)")
    group.add_argument("--random-backbone", action="store_true",
                       help="skip checkpoint I/O; structure only, never for numbers")

    group = parser.add_argument_group("lora")
    group.add_argument("--lora-rank", type=int, default=8)
    group.add_argument("--lora-alpha", type=float, default=None, help="defaults to rank")
    group.add_argument("--lora-dropout", type=float, default=0.0,
                       help="benchmark one step before enabling; dropout is not free here")
    group.add_argument("--full-finetune", action="store_true",
                       help="train the backbone too, not just the adapters and head. "
                            "Does NOT fit evo2-7b at seq 8192 on a 24 GiB core: AdamW "
                            "keeps two fp32 moments per parameter, ~27 GB for the 2.23 B "
                            "built parameters, before weights, gradients or activations. "
                            "Usable on evo2-1b, at a short context, or with --max-blocks.")

    group = parser.add_argument_group("data")
    group.add_argument("--fasta", default=None, help="FASTA to window over; else synthetic")
    group.add_argument("--dataset-size", type=int, default=512)
    group.add_argument("--seq-len", type=int, default=8192)
    group.add_argument("--window", type=int, default=256, help="window-regression bin size")
    group.add_argument("--window-width", type=int, default=None,
                       help="contiguous region of interest for the conv head")
    group.add_argument("--num-workers", type=int, default=0)

    group = parser.add_argument_group("optimisation")
    group.add_argument("--steps", type=int, default=100)
    group.add_argument("--batch-size", type=int, default=1,
                       help="per rank; >1 does not fit at 8192 on a 24 GiB core")
    group.add_argument("--grad-accum", type=int, default=1)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--weight-decay", type=float, default=0.0)
    group.add_argument("--warmup-ratio", type=float, default=0.0333)
    group.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "constant"])
    group.add_argument("--clip-grad-norm", type=float, default=1.0)
    group.add_argument("--adam-foreach", type=lambda s: s.lower() != "false", default=False,
                       help="False forces the per-tensor loop (see docs/OPTIMIZATIONS.md)")

    group = parser.add_argument_group("performance")
    group.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    group.add_argument("--compile-blocks", action="store_true", default=True)
    group.add_argument("--no-compile-blocks", dest="compile_blocks", action="store_false")
    group.add_argument("--gradient-checkpointing", action="store_true", default=True)
    group.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                       action="store_false")
    group.add_argument("--hcl-kernel", default="nki-scan",
                       choices=["direct", "chunked", "nki-scan"])
    group.add_argument("--bucket-cap-mb", type=int, default=128)
    group.add_argument("--find-unused-parameters", action="store_true")
    group.add_argument("--warmup-steps", type=int, default=3,
                       help="steps excluded from the warm-median timing")

    group = parser.add_argument_group("io")
    group.add_argument("--device", default="neuron", choices=["neuron", "cpu"])
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--save-to", default=None)
    group.add_argument("--save-every", type=int, default=0)
    group.add_argument("--resume-from", default=None)
    group.add_argument("--diag-dir", default=None)
    group.add_argument("--log-every", type=int, default=10)
    group.add_argument("--log-all-ranks", action="store_true")

    return parser


if __name__ == "__main__":
    sys.exit(main())
