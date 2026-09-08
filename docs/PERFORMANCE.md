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
| 1 | 932 s | 3.4418 s | 3.4418 s | 0.0% | 2,380 | 1.00x | 100.0% |
| 4 | 853 s | 3.5969 s | 3.5981 s | 0.0% | 9,107 | 3.83x | 95.7% |
| 16 | 63 s | 3.6247 s | 3.6277 s | 0.2% | 36,131 | 15.18x | 94.9% |
| 64 | 81 s | 3.5496 s | 3.5595 s | 0.4% | 147,292 | 61.88x | 96.7% |

Speedup and efficiency are against the single-core baseline, on the **slowest** rank.

**61.9× on 64 logical NeuronCores, 96.7% efficiency, 147,292 tokens/s** at 8192
context. The step time barely moves from 1 rank to 64 — 3.4418 s to 3.5595 s, a
3.4% cost for a 64× wider batch — which is what data-parallel scaling looks like
when the collective is small: the gradient payload is 10.3 MB in bf16 and fits in
one bucket.

Rank spread stays under 0.5% everywhere, so these are throughput numbers and not
straggler numbers.

Cold compile is a one-off per graph shape and is cached. The two large ones (932 s
at 1 rank, 853 s at 4) are the first non-DDP and first DDP graphs; once the DDP
graph is cached, 16 and 64 ranks start in **63 s and 81 s**. That is the whole
argument for warming with one multi-rank job before launching a large one.


## Single core, in detail

```
step times (s): 931.575  140.087  3.443  3.374  3.621  3.271  3.438
                  3.442    3.443  3.618  3.270  3.616
```

Three regimes are visible and worth naming:

* **step 0 — 931.6 s.** Cold compile: eleven 7B blocks at 8192, `neuronx-cc`, empty
  NEFF cache. One-off, and cached for every subsequent process.
* **step 1 — 140.1 s.** Still not steady state. Dynamo recompiles for the shapes and
  guards the first step did not cover. This is why `--warmup-steps` defaults to 3 and
  why the warm median excludes them.
* **steps 2+ — 3.27 to 3.62 s.** Steady state. Median **3.4418 s**, spread 10.7%
  peak-to-peak, which is the compiled/eager boundary alternating rather than contention.

**2,380 tokens/s on one logical NeuronCore**, at 8192 context, with 0.23% of the
parameters training.

## Host fallbacks

`get_fallback_ops()` per stage, after warmup, with tracking cleared between stages —
both dtypes:

```
stage             executed  fallback
forward                 44         0
backward                22         0
clip_grad_norm          15         1   ['aten::_to_copy']
optimizer.step          13         0
eval forward            27         0
```

The model graph is clean. The one entry is `torch.nn.utils.clip_grad_norm_`, which
costs a single scalar round-trip per step. Every formulation was tried —
`foreach=True`, `foreach=False`, the default, and a hand-written clip with the
threshold pre-placed on device — and all four are numerically identical (worst
gradient difference 0.0, norms agreeing to 8 decimals) and all four produce it. Run
with `--clip-grad-norm 0` for a step with nothing on the host.

Note also that a *cumulative* `get_fallback_ops()` always contains `aten::_to_copy`:
it includes the host-to-device transfer of the weights and the first batch, which is
a copy that has to happen. The driver clears tracking after warmup so its reported
list describes steady state.

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

The first DDP run at a given world size recompiles: a DDP graph carries the
allreduce and is not the single-rank graph, so a single-rank warm-up does not cache
it. One multi-rank warm-up does cache it for every larger world size.
