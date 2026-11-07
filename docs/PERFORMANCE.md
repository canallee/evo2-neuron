# Measured performance

Every number here came from `scripts/train_lora_ddp.py` on this repo, summarised by
`scripts/summarize_bench.py` from the per-rank JSON in `runs/`. Nothing is carried
over from elsewhere.

**Configuration** — the production one, unchanged:

```
--objective window-regression --window 256 --seq-len 8192 --dtype bf16
--hcl-kernel nki-scan --compile-blocks --gradient-checkpointing
--lora-rank 8 --batch-size 1 --grad-accum 1 --adam-foreach false
```

evo2-7b, blocks 0..10 of 32, read-out `blocks.10.mlp.l3`. Trainable: **74 tensors,
5,168,129 of 2,234,384,641 parameters (0.23%)** — 66 LoRA tensors at r=8 plus an
8-tensor `WindowTrackHead`.

**Instance** — `trn2.48xlarge`: 16 Trainium2 devices, `logical-neuroncore-config: 2`,
64 logical NeuronCores at 24 GiB each, 2 NUMA nodes.

**Stack** — torch 2.14.0, torch-neuronx 2.14.3 (Neuron SDK 2.33.0, built from the
public source), neuronx-cc 2.28, nki 0.7.0, Neuron runtime 2.35.

## Reporting rules

Two, and both change the number you would otherwise quote:

* **Cold compile and warm steady state are separate.** They are reported in separate
  columns everywhere. Conflating them is how a 3.4 s step gets published as a 930 s one.
* **At N ranks the step is set by the slowest rank**, not the median, because every rank
  waits at the allreduce. Both are printed. If they diverge by more than ~10% you have a
  straggler, not a throughput number, and `summarize_bench.py` says so.
* **Data parallelism scales throughput, not step time.** Each rank still does one full
  step, so a perfect 64-rank run has the same seconds-per-step as one rank and 64× the
  samples. Speedup here is the ratio of tokens/s, never of step times — dividing step
  times would report a perfect run as no speedup at all.

Before any of these runs: `make health`. An orphaned process holding a core fakes a
mid-run throughput collapse.

## Scaling across NeuronCores

| ranks | cold compile | warm step (median rank) | slowest rank | rank spread | tokens/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 766 s | 3.4140 s | 3.4140 s | 0.0% | 2,400 | 1.00x | 100.0% |
| 4 | 51 s | 3.5311 s | 3.5317 s | 0.0% | 9,278 | 3.87x | 96.7% |
| 16 | 54 s | 3.5470 s | 3.5489 s | 0.1% | 36,934 | 15.39x | 96.2% |
| 64 | 74 s | 3.5608 s | 3.5650 s | 0.2% | 147,067 | 61.29x | 95.8% |

Speedup and efficiency are against the single-core baseline, on the **slowest** rank.

**61.3× on 64 logical NeuronCores, 95.8% efficiency, 147,067 tokens/s** at 8192
context. The step time barely moves from 1 rank to 64 — 3.4140 s to 3.5650 s, a
4.4% cost for a 64× wider batch — which is what data-parallel scaling looks like
when the collective is small: the gradient payload is 10.3 MB in bf16 and fits in
one bucket.

Rank spread stays under 0.5% everywhere, so these are throughput numbers and not
straggler numbers.

Cold compile is a one-off per graph shape and is cached. The large one (766 s at 1
rank) is the first graph of the eleven compiled blocks. On torch-neuronx 2.14 the
first DDP graph is cheap — 4 ranks start in **51 s**, 16 in 54 s and 64 in 74 s —
where the private-beta build paid a second full compile (853 s) for it. Warm with
one rank before launching many regardless: concurrent cold compiles of the block
graphs have been enough to kill the compile service.


## Single core, in detail

```
step times (s): 765.614  76.019  3.281  3.414  3.414  3.414  3.414
                  3.414   3.416  5.682  1.151  5.681
```

Three regimes are visible and worth naming:

* **step 0 — 765.6 s.** Cold compile: eleven 7B blocks at 8192, `neuronx-cc`, empty
  NEFF cache. One-off, and cached for every subsequent process.
* **step 1 — 76.0 s.** Still not steady state. Dynamo recompiles for the shapes and
  guards the first step did not cover. This is why `--warmup-steps` defaults to 3 and
  why the warm median excludes them.
* **steps 2–8 — 3.28 to 3.42 s.** Steady state. Median **3.4140 s**, flat to 0.1%
  after step 2.
* **steps 9–11 — 5.68, 1.15, 5.68 s.** Not contention: the driver's host fences.
  Step 10 materialises the logged loss (`--log-every 10`) and step 12 the final one,
  so device work queued behind them is charged to those steps and the step between
  finds an empty queue. Each pair averages 3.42 s, and the median is unaffected.

**2,400 tokens/s on one logical NeuronCore**, at 8192 context, with 0.23% of the
parameters training.

## Host fallbacks

`get_fallback_ops()` per stage, after warmup, with tracking cleared between stages —
both dtypes:

```
stage             executed  fallback
forward                 48         0
backward                33         0
clip_grad_norm          15         0
optimizer.step          11         0
eval forward            35         0
```

(`scripts/check_fallback_ops.py`, bf16, seq 1024; fp32 is equally clean.) Nothing
runs on the host, and every production run above reports `fallbacks none`. Two
things changed against the private-beta build: `clip_grad_norm_`'s scalar
`aten::_to_copy` is gone, and RMSNorm's `norm` backward — which 2.14 sends to the
host — is replaced by a bit-identical custom backward
([`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) §5).

The driver clears tracking after warmup, so its reported list describes steady
state rather than compilation. (On the private-beta build a *cumulative* list also
always held an `aten::_to_copy` for the host-to-device transfer of the weights; on
2.14 that copy is not counted as a fallback.)

The raw per-rank JSON these tables come from is committed as
[`scaling.json`](scaling.json) — `runs/` itself is gitignored, so the evidence
lives here.

## Reproducing

```bash
make health
NEURON_RT_VISIBLE_CORES=0 python -u scripts/train_lora_ddp.py --steps 12 --diag-dir runs/bench_w1
NEURON_RT_VISIBLE_CORES=0-63 torchrun --nproc_per_node=64 \
    scripts/train_lora_ddp.py --steps 12 --diag-dir runs/bench_w64
python scripts/summarize_bench.py runs/bench_w1 runs/bench_w64
```

The first DDP run carries the allreduce and is a different graph from the
single-rank one, so a single-rank warm-up does not cache it. On torch-neuronx 2.14
that costs ~50 s rather than a second full compile, but warm the block graphs with
one rank first all the same.