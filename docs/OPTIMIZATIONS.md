# Running Evo 2 7B + LoRA on Trainium: what is implemented

Everything here is in the shipped code. Each entry names the flag that turns it on
and the number it is worth. Rejected candidates are not documented; if it is not
here, the code does not do it.

**Provenance.** Numbers marked *(measured here)* were produced by this repo and
are reproducible with the commands in [`PERFORMANCE.md`](PERFORMANCE.md).
Everything else was measured during the port campaign that produced the CUDA
reference, on the same instance type and the same configuration, and is carried
over rather than re-run. Where the two overlap they agree: that campaign measured
3.825 s/step at seq 8192 and this repo measures **3.4418 s** on the same
configuration with a smaller read-out head.

**Production configuration:**

```bash
--dtype bf16 --compile-blocks --gradient-checkpointing --hcl-kernel nki-scan
# batch 1 per rank, seq 8192, blocks 0..10 of 32, DDP across all cores
```

Three of those four are not optional at seq 8192: without `nki-scan` the long
convolution does not compile, without `--gradient-checkpointing` the step does not
fit in 24 GiB, and without `--compile-blocks` the graph exceeds the compiler's
instruction limit.

---

## The budget you are optimising against

Per-rank step at seq 8192, bf16, per-block compiled, checkpointing on, `nki-scan`:

| phase | time | share | |
|---|---|---|---|
| forward | 1.137 s | 29.7% | carried over |
| **backward** | **2.671 s** | **69.8%** | carried over |
| `optimizer.step` | 0.017 s | 0.44% | carried over |
| **step, one core** | **3.4418 s** | 100% | **measured here** |

The phase split is from the port campaign; the step total is this repo's own
median over warm steps. The split is what the shares are for — treat them as the
shape of the step, not as three numbers that must add to the fourth.

Two things follow. The backward is where the time is, and it is where it stays:
Trainium scales worse than an H200 with context — 40.96× more tokens costs the
GPU 5.03× and costs Trainium 14.77× — and the entire gap is the backward. And the
optimizer is 0.44%, so no amount of optimizer cleverness is worth measuring.

**0.23% of the parameters train** in the shipped configuration — 5,168,129 of
2,234,384,641, being 66 LoRA tensors at r=8 plus an 8-tensor `WindowTrackHead` —
and the backward is still ~70% of the step. A frozen backbone still has to
propagate gradients *through* every block to reach the adapters underneath, so the
trainable fraction buys memory and optimizer time, not backward time. (A wide
`ConvReadoutHead` pushes the same figure to ~2.5%, almost all of it head.)

---

## 1. `nki-scan` for the Hyena long convolution

**Flag:** `--hcl-kernel nki-scan` · **Worth:** the only form that compiles at 8192

Evo 2's `hcl` blocks apply a modal long convolution. Written directly it is a
length-`L` filter materialised per channel and convolved — `O(C·L²)` work, and at
L = 8192 it does not compile.

The filter is a sum of geometric modes, `h[s] = Σⱼ rⱼ aⱼˢ`, which is exactly the
transfer function of a first-order recurrence:

```
sⱼ[t] = aⱼ · sⱼ[t-1] + x[t]
```

That is `nkilib.experimental.scan.linear_scan`'s contract verbatim, so the
convolution becomes a scan: `O(C·S·L)`, with `S` modes instead of `L` taps — 512×
fewer multiply-accumulates at 8192.

Two implementation notes that are load-bearing:

* **`wrap_nki` output has no autograd formula.** Used bare, the kernel silently
  produced gradients missing the convolution's contribution entirely — the
  surrounding skip and gate terms re-attached a `grad_fn`, so nothing raised and
  nothing looked wrong. It is wrapped in `_ModalCausalConvNKI(torch.autograd.
  Function)`, whose backward runs a second scan over the flipped upstream
  gradient. `tests/test_ops.py` asserts the routing structurally, because a
  numerical check cannot catch it: `lora_B` is zero-initialised, so a *missing*
  gradient reads as a small one.
* **`_NKI_SCAN_TILE = 1024`.** The kernel's redundant `decay` buffer is bounded at
  256 MiB by this tile instead of 2 GiB — and 2 GiB is exactly the allocation
  that failed the first 8192 attempt. The kernel tiles internally at 2048, so
  this trades some throughput for headroom on a core already holding ~9 GB of
  weights.

`--hcl-kernel direct` is kept as the CPU-side reference the scan is validated
against, and `chunked` as a linear-time form that needs no device.

## 2. Per-block `torch.compile`

**Flag:** `--compile-blocks` · **Worth:** 6.560 s → 3.825 s at 8192 (**1.72×**), carried over

Whole-model `torch.compile(backend="neuron")` works at short sequences (2.08× at
seq 200) and is **unavailable** at seq ≥ 1024 — not because it is slow, but
because the graph exceeds `neuronx-cc`'s instruction limit: `NCC_EBVF030`,
13,571,848 instructions against a 5 M cap.

The limit is **per graph**. Compiling each block separately produces 11 graphs
that each fit, with every accuracy verdict unchanged.

Two ways to get this wrong:

* Use `block.compile(backend=...)`, **not** `blocks[i] = torch.compile(blocks[i])`.
  The latter wraps the block in an `OptimizedModule`, which inserts `_orig_mod.`
  into every parameter name — renaming most of the trainable set, breaking
  adapter checkpoints and any comparison keyed by name.
* Raise `torch._dynamo.config.recompile_limit` (default **8**). Eleven blocks plus
  warmup variants exceed it, and exceeding it makes dynamo fall back to eager
  **silently**.

`Evo2Backbone.compile_blocks()` does both.

## 3. Activation checkpointing

**Flag:** `--gradient-checkpointing` · **Worth:** required to fit; +3.0% compiled, carried over

At 8192 on a 24 GiB logical NeuronCore this is what makes the step fit at all.
Cost is +16.1% eager and only +3.0% compiled, and it is **bit-identical** —
verified at exactly 0.0 across all 245 tensors.

Ten of the eleven blocks are recomputed; the read-out block is skipped, because
its output is what the head consumes and the graph is already holding it.

Two settings are not preferences:

* **`use_reentrant=False`** is mandatory. The reentrant implementation decides
  what to recompute from whether the block's *inputs* require grad — and with a
  frozen backbone they do not. It silently dropped the LoRA gradients for every
  adapter but the topmost one.
* **`preserve_rng_state=False`** is only sound because nothing on this path
  consumes RNG. `enable_gradient_checkpointing` therefore **refuses** to enable if
  it finds an `nn.Dropout` with `p > 0`, rather than quietly producing a
  recompute that disagrees with the forward. This is why `--lora-dropout` defaults
  to 0 and why turning it on interacts with the memory lever that makes 8192 fit.

**24 GiB per logical core is architectural, not a misconfiguration.** LNC is
already 2, pairing two physical cores onto one HBM stack; `NEURON_LOGICAL_NC_CONFIG`
and `NEURON_RT_VIRTUAL_CORE_SIZE` were both tried and change nothing. Batch 2 per
rank at 8192 compiles and then fails to *load*, on a 9.5 MB allocation — that is
fragmentation, not exhaustion, and it is why batch size stays 1 and scale comes
from ranks.

## 4. Read-out pruning

**Flag:** `--readout-block` (default 10) · **Worth:** 32 blocks → 11

Anything reading out at `blocks.10.mlp.l3` never touches blocks 11..31. Building
11 of 32 skips ~66% of the weights on evo2-7b and all of their activation memory,
and it is done at construction rather than by replacing loaded blocks afterwards.

`scripts/prune_checkpoint.py` applies the same cut to the checkpoint file: 13 GB
→ 4.46 GB, bit-identical on the tensors retained. At 64 ranks that is the
difference between 832 GB and 285 GB of host reads.

## 5. Op rewrites — the model graph runs entirely on device

**Always on**, in `evo2_finetune/ops.py`. Each is an algebraic identity validated
against the naive form on CPU in `tests/test_ops.py`, and each is used on **every**
device — a rewrite active only on device is a rewrite whose CPU validation proves
nothing.

| op | rewrite | failure without it |
|---|---|---|
| `aten::argmax.out` | cumsum / clamp / sum `first_set_index` | **silent** host fallback |
| `aten::logaddexp.out` | max-shift identity | **silent** host fallback |
| `Conv1d` spanning its whole input | one `addmm` | **hard** compile error, `NCC_ITIN902` |
| `aten::pow.Scalar` in RoPE `inv_freq` | recompute in fp32 (fixed in source) | silent host fallback, bf16 only |

**One known exception, and it is not the model.** `torch.nn.utils.clip_grad_norm_`
costs exactly one `aten::_to_copy` per step — a scalar round-trip. Every
formulation was tried: `foreach=True`, `foreach=False`, the default, and a
hand-written clip with the threshold pre-placed on device. All four are
numerically identical (worst gradient difference 0.0, norms agreeing to 8
decimals) and all four produce it. It is reported by
`scripts/check_fallback_ops.py` as its own stage and excluded from the gate,
because the gate is about the model graph. Run with `--clip-grad-norm 0` if you
want a step with nothing on the host at all.

Note the two failure modes. The argmax and logaddexp fallbacks are numerically
*correct* — the run completes with the right answer and a host round-trip per
step — which is exactly why this needs a tool rather than an eyeball. Use
`scripts/check_fallback_ops.py`, which clears op tracking between forward /
backward / `optimizer.step` / eval (a cumulative list localises nothing), asserts
the executed-op list is **non-empty** (an empty fallback list is also what a stage
that ran nothing returns), and runs both dtypes (the fallback set is
dtype-dependent: bf16's forward executes 53 distinct ops against fp32's 50).

The `log1p` trap is worth stating: rewriting `logaddexp` in terms of `log1p`
merely moves the fallback to `aten::log1p.out`. `log(1 + x)` is used instead.

## 6. DDP across every NeuronCore

**Flags:** `NEURON_RT_VISIBLE_CORES=0-63`, `torchrun --nproc_per_node=64`

Measured here at 8192, bf16, batch 1 per rank — full table and method in
[`PERFORMANCE.md`](PERFORMANCE.md):

| ranks | tokens/s | speedup | efficiency |
|---:|---:|---:|---:|
| 1 | 2,380 | 1.00x | 100% |
| 4 | 9,107 | 3.83x | 95.7% |
| 16 | 36,131 | 15.18x | 94.9% |
| **64** | **147,292** | **61.88x** | **96.7%** |

The step time moves 3.4% from 1 rank to 64 (3.4418 s → 3.5595 s) for a 64x wider
batch, and rank spread stays under 0.5%.

Five mechanics, all handled in `scripts/train_lora_ddp.py`, and each one fails
differently:

* **Launch with the full range visible, then narrow it per rank.** Every rank
  inherits the same `NEURON_RT_VISIBLE_CORES`, and every rank's runtime then
  claims the whole range — one wins and the rest die with `Failed to initialize
  Neuron Runtime: status code 1`, naming neither the cause nor the culprit.
  `neuron_env.configure()` reads `LOCAL_RANK` and rewrites the variable to this
  rank's single core before the runtime ever sees it. Leaving it at `0` instead
  is the opposite failure: every rank shares one core, the job runs, the numbers
  are correct, and it is N× slower with no error.
* **The device is still addressed by its global index.** Narrowing the visible set
  does not renumber it: after narrowing to core 3, `torch.device("neuron", 0)`
  raises `Device index 0 is out of range. Valid range is 3 to 3`. The script
  resolves the global id, uses it for `device_ids`, and all-gathers to assert the
  ids are distinct across ranks.
* **The process-group backend is `"neuron"`**, registered by `torch_neuronx` for
  the `neuron` device type. `gloo` initialises without complaint and then fails
  at the first collective with `No backend type associated with device type
  neuron`.
* `bucket_cap_mb=128` with `gradient_as_bucket_view=True`, so the whole gradient
  payload is one bucket: this backend does not benefit from NCCL-style
  many-small-bucket overlap. **Size this against your own head.** With the shipped
  `WindowTrackHead` the payload is only **10.3 MB in bf16** and the default 25 MB
  cap would already have been one bucket; a wide `ConvReadoutHead` is 52 M
  parameters on its own and lands past 100 MB, where the default splits it across
  five or six. The driver reads the cap back off the built DDP object and logs
  the measured payload beside it, and warns if the payload exceeds the cap —
  requesting a bucket configuration and getting one are different things.
* `--adam-foreach false`. `foreach` is auto-selected on this backend, and its
  grouped kernels are where a long-run defect lives: a 64-rank job dies around
  step 225 in `aten::_foreach_div_.ScalarList`, an op that exists only on the
  grouped path. The per-tensor loop costs ~1.4% and finishes the epoch. *(Port
  campaign — the runs here are 12 steps and do not reach it. It is the default
  because the failure it avoids is unrecoverable and the cost is 1.4%.)*

`find_unused_parameters` is **off**. Frozen backbone parameters have
`requires_grad=False` and DDP never buckets them, so the flag buys nothing here —
and it forces a full autograd-graph traversal every step, delaying bucket
readiness, which is precisely what defeats allreduce/backward overlap. Turn it on
only if a *trainable* tensor is genuinely unused on some batches.

`no_sync()` wraps every microstep but the last, so gradient accumulation pays one
collective per optimizer step rather than one per microstep.

## 7. fp32 depthwise convolution inside a bf16 model

**Always on**, in `hyena.causal_depthwise_conv`.

The model runs bf16; this one op is pinned to fp32 compute. Not caution —
measurement: `F.conv1d` with `groups=4096, k=128` is **11–27× slower in bf16 than
in fp32** on this backend, with a 47× cliff between L=1024 and L=2048 and a
non-monotonic curve. The pathology is confined to (bf16, large kernel); the short
k=3 and k=7 filters are fine either way and are left alone.

## 8. Deferred loss materialisation and gradient clipping

`float(loss)` is a **host fence inside the training loop**. The driver
accumulates loss tensors on device and pulls them every `--log-every` steps.

The safety rule is the important half: **no unchecked loss may reach a
checkpoint.** The materialisation therefore always runs before a save and before
the final step, and a NaN raises rather than being written out.

`--clip-grad-norm` uses `foreach=True`. Clipping 74 small tensors elementwise is
real time; the grouped form is the cheap one, and unlike the optimizer path it is
not implicated in the run-length defect.

---

## Precision

**bf16 is the production dtype and fp32 is the source of truth.** They answer
different questions and both are shipped.

bf16 compiled is the fastest configuration measured, and its worst gradient
cosine against fp32 is 0.9919 — it clears a 0.99 gate by less than 1%. Gate a
port in fp32; run it in bf16 once gated.

Two dtype facts that cost time to find:

* Evo 2 emits very large activations in its deeper blocks that the final norm
  absorbs. In bf16 on this backend those collapse to exactly zero. This is why
  `Evo2Backbone` defaults to fp32 and bf16 is an explicit escalation.
* `load_into(dtype=...)` **rounds weight values but does not set the module
  dtype** on the checkpoint path — `load_state_dict` copies into
  already-constructed fp32 parameters. The random-init path did cast. Same call,
  two dtypes, depending on a flag unrelated to dtype. `Evo2Backbone` now casts
  unconditionally afterwards. Check `next(model.parameters()).dtype`, not the
  argument you passed.

## Compiler cache and measurement hygiene

Cold compile and warm steady state are different numbers and are reported
separately everywhere in this repo. Conflating them is how a 0.09 s step gets
published as a 92 s one.

* Set the four `TORCH_NEURONX_*` cache dirs **before importing torch** —
  `import torch` auto-loads `torch_neuronx` through an entry point, and the dirs
  are read at that moment. `evo2_finetune.neuron_env.configure()` does this and
  refuses if it is already too late. They default under `/tmp`, owned by whoever
  ran Neuron on the box first; the resulting `PermissionError` surfaces as a
  generic "Compilation error".
* Warm the NEFF cache with **one** rank before launching many. Four concurrent
  cold compiles have been enough to kill the compile service.
* A NEFF cache will serve kernels compiled under different `NEURON_CC_FLAGS`.
  Clear it before concluding a flag did nothing.
* Before any timing run, check for orphaned device holders:

  ```bash
  for d in $(seq 0 15); do lsof /dev/neuron$d 2>/dev/null | tail -n +2; done | wc -l
  ```

  An orphaned process fakes a mid-run throughput collapse. A bimodal step-time
  trace is a contention signature until proven otherwise.
