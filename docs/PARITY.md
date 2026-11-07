# Parity: does the port reproduce the original?

Every number in the tables below was produced by the scripts and artifacts in this
repo; the command that regenerates it is in the section. Two calibration figures
are marked *(port campaign)* — they come from the campaign that minted the CUDA
reference, on the same hardware, and are cited as controls rather than re-run.

**Environment:** `trn2.48xlarge`, AWS native-PyTorch device — torch 2.14.0,
torch-neuronx 2.14.3 (Neuron SDK 2.33.0), neuronx-cc 2.28, nki 0.7.0, Neuron
runtime 2.35.
**Model:** evo2-7b, blocks 0..10 of 32, read-out `blocks.10.mlp.l3`, 33 LoRA
adapters at r=8, batch 1. **Reference:** an NVIDIA H200 running evo2 0.1.0 /
vortex 0.0.2 / TransformerEngine 1.13.0, bit-reproducible across two physical
GPUs.

---

## What "PASS" means here, and what it does not

This is a **re-implementation** of Evo 2, gated against the original — not the
original. Passing means no block diverges under the stated rule, not that the
port is bit-exact: the read-out differs from the CUDA reference by 1.6e-02 on CPU
and 3.4e-02 on Trainium in bf16. The sections below argue that the residual is
arithmetic rather than structural, and give the controls that support it.

Whether that residual is acceptable is a property of **your** downstream task,
not of this table. A 1.6e-02 shift in a representation can be irrelevant to a
coarse classifier and decisive for a ranked variant-effect score. Only `evo2-7b`
is gated at all; `evo2-1b` cannot be, for the FP8 reason below.

## Verdict

| axis | what it proves | result |
|---|---|---|
| **CUDA → CPU**, seq 200 | the port reproduces vortex + TransformerEngine | **PASS** — 0 diverging blocks, `hidden0` 1.59e-02 |
| **CUDA → Trainium**, seq 200 | end to end, reference to device | **PASS** — 0 diverging blocks, `hidden0` 3.41e-02 |
| **CUDA → CPU**, seq 8192 | the production context | **PASS** — 0 diverging blocks, `hidden0` 1.98e-02 |
| **CPU ↔ Trainium**, forward | the device lowering is faithful | **PASS** — `hidden0` 3.29e-02 bf16, **8.26e-05** fp32 |
| **CPU ↔ Trainium**, backward | gradients agree | **PASS** — 74/74 tensors, 0 real disagreements |
| **CPU ↔ Trainium**, 3 AdamW steps | on-device training agrees | **PASS** — losses to ~6e-08, 74/74 weights |
| `get_fallback_ops()` | nothing ran on the host | **empty**, every configuration |

**evo2-7b is the backbone that can be gated against CUDA at all.** Arc's 7B
config at evo2 0.1.0 sets `use_fp8_input_projections: False`, so a port with no
TransformerEngine has nothing to emulate. The 1B config sets it `True`, and
disabling FP8 there costs 47 accuracy points — evo2-1b is shipped for shape work,
not as a gateable reference.

---

## What is being measured, and why it is not `hidden0`

The primary gate is the **residual stream** (`blocks0_10.pt`: `block_out.k` and
`mlp_l3.k` for every built block), and the metric on it is the **increment**

```
delta_k = block_out.k − block_out.{k−1}
```

not the cumulative tensor. Two reasons, both measured on this reference:

* `hidden0` is a **0.42% correction** to the stream by norm. It tells you *that*
  something diverged and never *where*, and it amplifies any upstream error.
* `block_out.k` is cumulative, so a block's own contribution can be arbitrarily
  diluted by the residual it rides on. Block 3 contributes **1.3%** of
  `block_out.3`. A completely broken attention block would read ~1.3e-02
  cumulatively — inside the bf16 noise floor, and dismissible. On `delta_3` the
  same break is O(1).

A block is flagged only if it fails **both** magnitude and direction:
`relerr(delta_k) > 3e-3` **and** `1 − cos(delta_k) > 1e-2`.

Those two floors are calibrated, not chosen. The magnitude floor is the bf16
quantisation floor: every element of all 22 reference stream tensors survives a
bf16 round-trip unchanged, so the CUDA runtime materialised the stream in bf16 at
every module boundary regardless of what its `dtype` field says — 8 mantissa bits
is ~1.1e-3 relative, and 3e-3 is that with headroom.

The direction floor has a positive and a negative control. Clean CPU-bf16 against
the reference gives `1 − cos(delta)` of 7.4e-06 .. 6.8e-04 across all 11 blocks
(measured here, §A); a deliberately scrambled block 3 gives 9.999e-01 (measured
during the port campaign). 1e-2 sits ~15× above the worst clean value and ~1000×
below the broken one.

**Why magnitude alone cannot be the verdict at this scale.** Two controls, both
arithmetically free:

* Swapping fused SDPA for an explicit-softmax attention — provably the same
  function, differing only in reduction order and bf16 rounding — moves `delta_3`
  by 4.9e-03 and `delta_10` by 9.3e-03. *(Port campaign.)*
* **Upgrading torch moves `hidden0` by 1.24e-02.** The CPU read-out from torch
  2.12.1 and from torch 2.14.0 — same port, same checkpoint, same input, same
  machine, each bit-reproducible on its own — differ by relerr 1.24e-02, with 82%
  of elements changed. *(Measured here.)* Nothing changed but the bf16 kernels
  PyTorch picks.
* **On torch 2.12.1, flipping `requires_grad` on the backbone moved `hidden0` by
  1.244e-02** — same weights, same input, same process; only the autograd metadata
  differed, which was enough for PyTorch's CPU path to pick a different GEMM. On
  torch 2.14.0 the same flip moves it by exactly 0. *(Both measured here.)*

The entire CUDA↔CPU disagreement is 1.59e-02, and changes that provably cannot
alter the mathematics account for 1.24e-02 of that magnitude. **At bf16 on this
stack, the magnitude of the disagreement is mostly reduction order.** Direction is
the half that discriminates, which is why both gates must fail before a block is
flagged.

It is also why the bundles record their provenance. A CPU bundle is
bit-reproducible only on the torch release that wrote it; compared across releases
it reads as a ~1.2e-02 regression that is not there. Freeze before measuring as
well: `Evo2Backbone.apply_lora(freeze=True)` does, every shipped bundle is produced
through it, and on torch 2.12.1 skipping it cost the same 1.2e-02.

**Gate the CUDA axis in bf16.** An fp32 arm measures a dtype difference stacked
on top of everything else.

---

## A — CUDA → CPU, seq 200

```bash
PYTHONPATH=src python scripts/dump_artifacts.py --device cpu --dtype bf16 \
    --seq-len 200 --inputs-from artifacts/cuda_seq200/inputs.pt \
    --forward-only --strict-determinism --out-dir artifacts/cpu_seq200_fwd
PYTHONPATH=src python scripts/dump_artifacts.py --compare \
    artifacts/cuda_seq200 artifacts/cpu_seq200_fwd
```

Read-out norm: reference **15.395389**, CPU port **15.375441**.

| k | relerr(out) | relerr(delta) | 1−cos(delta) | relerr(l3) |
|---:|---:|---:|---:|---:|
| 0 | 3.849e-03 | 3.849e-03 | 7.408e-06 | 1.240e-03 |
| 1 | 5.254e-03 | 5.680e-03 | 1.613e-05 | 5.730e-03 |
| 2 | 4.959e-03 | 4.959e-03 | 1.204e-05 | 4.958e-03 |
| 3 | 4.998e-03 | 8.020e-03 | 3.214e-05 | 8.341e-03 |
| 4 | 2.239e-02 | 4.955e-03 | 1.181e-05 | 9.432e-02 |
| 5 | 2.224e-02 | 3.255e-02 | 5.299e-04 | 2.479e-02 |
| 6 | 2.499e-02 | 4.165e-02 | 6.821e-04 | 1.656e-02 |
| 7 | 2.376e-02 | 1.618e-02 | 1.294e-04 | 1.306e-02 |
| 8 | 2.235e-02 | 1.688e-02 | 1.403e-04 | 1.504e-02 |
| 9 | 8.135e-03 | 8.158e-03 | 1.148e-05 | 7.630e-03 |
| 10 | 8.110e-03 | 1.065e-02 | 5.667e-05 | 1.589e-02 |

**No block diverges.** `hidden0` 1.589e-02, 0 real disagreements, 0 structural
mismatches.

The residual is arithmetic, not structural: every increment cosine agrees to
≤7e-04, and the error-growth column shows no step change — only the reference's
own ~4.5× cancellation at block 4, which is a property of the reference, not of
the port.

## B — CUDA → Trainium, seq 200

Same command with `--device neuron`. Read-out norm **15.366991**.

Worst increment 4.24e-02 at block 6, worst `1 − cos(delta)` **8.97e-04** at block
6. **No block diverges**, `hidden0` 3.414e-02, 0 real disagreements, 0 structural
mismatches, `get_fallback_ops()` empty. Cold compile plus run: **44 s**.

## C — CUDA → CPU, seq 8192 (the production context)

```bash
PYTHONPATH=src python scripts/dump_artifacts.py --device cpu --dtype bf16 \
    --seq-len 8192 --inputs-from artifacts/cuda_seq8192/inputs.pt \
    --forward-only --strict-determinism --hcl-kernel chunked \
    --out-dir /tmp/cpu_seq8192_full
```

Read-out norm: reference **87.708467**, CPU port **87.667095**.

| k | relerr(out) | relerr(delta) | 1−cos(delta) | relerr(l3) |
|---:|---:|---:|---:|---:|
| 0 | 4.830e-03 | 4.830e-03 | 1.166e-05 | 1.525e-03 |
| 1 | 5.636e-03 | 6.126e-03 | 1.877e-05 | 6.181e-03 |
| 2 | 3.133e-03 | 3.128e-03 | 3.848e-06 | 3.120e-03 |
| 3 | 3.238e-03 | 1.434e-02 | 1.028e-04 | 1.671e-02 |
| 4 | 2.286e-02 | 6.953e-03 | 2.410e-05 | 1.182e-02 |
| 5 | 2.250e-02 | 1.542e-02 | 1.188e-04 | 1.347e-02 |
| 6 | 1.573e-02 | 2.733e-02 | 3.719e-04 | 1.112e-02 |
| 7 | 1.337e-02 | 1.256e-02 | 7.883e-05 | 1.302e-02 |
| 8 | 1.406e-02 | 1.529e-02 | 1.168e-04 | 1.120e-02 |
| 9 | 6.370e-03 | 6.389e-03 | 1.948e-05 | 6.690e-03 |
| 10 | 6.382e-03 | 1.584e-02 | 1.255e-04 | 1.978e-02 |

**No block diverges** at 40.96× the tokens; `hidden0` 1.978e-02, essentially the
seq-200 figure. 0 real disagreements, 0 structural mismatches.

Run with `--hcl-kernel chunked`, the linear-time modal form, because `direct`
materialises an L×L filter per channel. `chunked` shifts fp32 numerics by ~1e-7 —
four orders of magnitude below the 3e-3 floor, and the shipped bundle records the
flag.

The shipped `artifacts/cpu_seq8192_fwd` carries `inputs / hidden0 / manifest`;
the 2.9 GB stream tensor is regenerated by the command above rather than shipped
twice.

## D — CPU ↔ Trainium

This is the axis that isolates the device lowering from the port, and it is
tighter than the CUDA axis by two to three orders of magnitude.

**Forward, bf16** (`cpu_seq200_fwd` vs `neuron_seq200_fwd`): no block diverges,
worst `1 − cos(delta)` 8.68e-04, `hidden0` 3.286e-02, 0 real disagreements.

**Forward, fp32** (inside the training bundles): `hidden0` **8.255e-05** —
fp32 reduction-order noise.

**Backward and 3 AdamW steps, fp32.** Weights pinned from a shared `w0.pt`,
inputs replayed from a shared `inputs.pt`, so the gradients are the only free
variable.

```bash
PYTHONPATH=src python scripts/dump_artifacts.py --compare \
    artifacts/cpu_seq200_window artifacts/neuron_seq200_window
```

| artifact | LoRA max / median | head max / median | real disagreements |
|---|---|---|---|
| `w0` (74 tensors) | **0.0 / 0.0** | **0.0 / 0.0** | 0 of 74 |
| `g1` | 2.818e-04 / 3.089e-05 | 2.904e-05 / 3.586e-07 | 0 of 74 |
| `wN` | 1.254e-03 / 4.390e-05 | 7.245e-06 / 2.406e-08 | 0 of 74 |

Losses, window-regression:

```
cpu    [0.7565693259239197, 0.7331119775772095, 0.7134156227111816]
neuron [0.7565693855285645, 0.7331119775772095, 0.7134155631065369]
```

Agreement to ~6e-08, and the loss descends.

The per-position-classification bundle also has 0 real disagreements on all three
artifacts and losses that agree to ~2e-07, but its gradients differ more:
`g1` LoRA max 9.146e-03 / median 1.777e-03, `wN` LoRA max 2.571e-02. The cause is
**one ReLU**. Of the 51,200 pre-activations after the head's first projection,
exactly one changes sign between the CPU and Trainium forward passes; on CPU it is
3.2e-07, inside fp32 reduction-order noise of zero. Feeding the *Trainium*
read-out through the head *on CPU* reproduces the Trainium gradients (`head.project.bias`
to 9.9e-06), so the backward is faithful — the function itself has a kink at that
point. The torch 2.12 build happened to land on the same side of it (`g1` LoRA max
3.209e-04). This is the case the direction rule exists for: a magnitude-only gate
would call it a regression; the gradients still agree in direction.

`w0` at exactly 0.0 is the control that makes the rest meaningful: it says both
devices genuinely started from the same weights. A head *initialised on the
device* from the same seed instead gives rel_err ~1.4 — two unrelated draws —
because `torch.manual_seed` feeds a **per-device** generator. The shipped scripts
build on CPU and then move, which is why seeding alone happens to match here; pin
anyway.

---

## The gradient gate, and the thing it cannot catch

A gradient tensor counts as a REAL disagreement only if it fails **both**

```
cosine < 0.99   AND   max_abs / (max |grad| anywhere in the model) > 1e-3
```

Per-tensor relative error alone is useless here: `lora_B` is zero-initialised, so
its gradient is genuinely tiny and dividing a small absolute difference by a
small norm manufactures alarming ratios out of floating-point noise. Measured
(`parity/02`, window-regression, fp32), the worst `lora_B` gradient is at rel_err
2.8e-04 while the head — which carries real magnitude — sits at 2.9e-05, ten times
lower. Only one of those is worth looking at, and it is not the `lora_B` figure.

**That same zero-init means a *missing* gradient reads as a small one.** No
numerical gate can catch it. Two real bugs lived in exactly that gap: a
non-differentiable custom kernel whose surrounding skip and gate terms
re-attached a `grad_fn` so nothing raised, and reentrant activation checkpointing
silently dropping every adapter's gradient but the topmost one. Both produced
`None`, not a wrong number.

So `parity/02` additionally asserts that **no trainable tensor comes back with
`grad is None`**, and `tests/test_ops.py` asserts structurally that the NKI path
dispatches through its autograd `Function`. Neither is a numerical check, and
neither is optional.

## Self-noise

The floor is **exactly zero**. Two dumps of the identical configuration in two
separate processes produce bit-identical `hidden0`, and both are bit-identical to
the shipped bundle:

```
self-noise (same flags, two processes): equal=True  relerr=0.000e+00
```

That is the control that makes every number above meaningful — a self-noise
floor above zero would bound every conclusion drawn from them.
`--strict-determinism` turns a silently non-reproducible kernel into an exception
rather than a mystery, and the shipped bundles all carry it.

## CPU results across torch releases

**A CPU result is bit-reproducible on the torch release that produced it, and on
no other.** The shipped CPU bundles were written by torch 2.14.0. The same bundles
written by torch 2.12.1 (the private-beta environment) differ — with the same port,
checkpoint, inputs, pinned weights, flags and machine:

| quantity | torch 2.12.1 → 2.14.0 |
|---|---|
| CPU **bf16** read-out `hidden0`, seq 200 (`cpu_seq200_fwd`) | rel-L2 **1.24e-02**, max abs 9.8e-03, 82% of elements changed |
| CPU **bf16** residual stream `blocks0_10`, seq 200 | worst tensor rel-L2 1.24e-02 |
| CPU **bf16** read-out `hidden0`, seq 8192 (`cpu_seq8192_fwd`) | rel-L2 **1.72e-02**; loss 0.39160013 → 0.39159670 |
| CPU **fp32** forward `hidden0` (training bundles) | **bit-identical** |
| CPU **fp32** losses, 3 steps, both objectives | **bit-identical** |
| CPU **fp32** gradients `g1` | rel-L2 5.4e-07 (window) / 7.1e-07 (per-position) |
| CPU **fp32** weights after 3 steps `wN` | rel-L2 1.9e-05 (window) / 6.8e-06 (per-position) |
| CUDA → CPU `hidden0`, seq 200 | 1.482e-02 → 1.589e-02 |
| CUDA → CPU `hidden0`, seq 8192 | 1.887e-02 → 1.978e-02 |
| `requires_grad` flip control | 1.244e-02 → exactly 0 |

**It is torch, not the port.** Running *today's* code on torch 2.12.1 reproduces
every 2.12.1 bundle bit for bit — `cpu_seq200_fwd` `hidden0`, and `hidden0`, `g1`
and `wN` of `cpu_seq200_window` — including the one code change made for 2.14
(`RMSNorm`'s custom norm backward, see
[`VENDORED.md`](../src/evo2_neuron/VENDORED.md)). So every row above is the torch
upgrade alone: PyTorch 2.14 selects different CPU kernels — different GEMM blocking,
hence a different bf16 reduction order — and, in fp32, a backward that differs at
the 1e-07 level.

What it means in practice:

* **No verdict changed.** Every gate passes on both releases with 0 diverging
  blocks; per-block increments against CUDA move by at most a few 1e-04, and
  direction (`1 − cos(delta)`) by less than 1e-05.
* **The size of the shift is the point of the magnitude discussion above.** A
  1.24e-02 move from a library upgrade is most of the 1.59e-02 CUDA↔CPU residual —
  magnitude at bf16 is reduction order.
* **Compare like with like.** `tests/test_port.py::TestAgainstTheRealCheckpoint`
  asserts the CPU read-out is bit-identical to `cpu_seq200_fwd`, and fails on any
  torch other than the one in that bundle's `manifest.json` (`provenance.torch`).
  On a different torch, regenerate the CPU bundles with the commands in §A, §C
  and §D, rather than reading the failure as a regression.
* **Neuron moved too.** Between the private-beta `torch-neuronx` 2.12.3 and 2.14.3,
  the Neuron bf16 read-out moved by 1.20e-02 and the fp32 read-out by 4.3e-05 —
  device numerics, likewise with no port change. The per-position gradient
  difference in §D is that fp32 shift crossing one ReLU's kink.

The 2.12.1 bundles are on the `native-beta` branch (manifests and checksums) and in
the original release archive (tensors).

## Reproducing all of it

```bash
make health                          # no orphaned process is holding a core
make parity-cpu                      # the CUDA axis; no device needed
make parity                          # all three gates on CPU and Neuron
make fallbacks                       # per-stage, per-dtype host-fallback census
```
