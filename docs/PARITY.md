# Parity: does the port reproduce the original?

Every number in the tables below was produced by the scripts and artifacts in this
repo; the command that regenerates it is in the section. Two calibration figures
are marked *(port campaign)* — they come from the campaign that minted the CUDA
reference, on the same hardware, and are cited as controls rather than re-run.

**Environment:** `trn2.48xlarge`, AWS native-PyTorch beta — torch 2.12.1+cpu,
torch-neuronx 2.12.3.0.1636, neuronx-cc 2.27.2878.0, nki 0.6.0.
**Model:** evo2-7b, blocks 0..10 of 32, read-out `blocks.10.mlp.l3`, 33 LoRA
adapters at r=8, batch 1. **Reference:** an NVIDIA H200 running evo2 0.1.0 /
vortex 0.0.2 / TransformerEngine 1.13.0, bit-reproducible across two physical
GPUs.

---

## Verdict

| axis | what it proves | result |
|---|---|---|
| **CUDA → CPU**, seq 200 | the port reproduces vortex + TransformerEngine | **PASS** — 0 diverging blocks, `hidden0` 1.48e-02 |
| **CUDA → Trainium**, seq 200 | end to end, reference to device | **PASS** — 0 diverging blocks, `hidden0` 3.53e-02 |
| **CUDA → CPU**, seq 8192 | the production context | **PASS** — 0 diverging blocks, `hidden0` 1.89e-02 |
| **CPU ↔ Trainium**, forward | the device lowering is faithful | **PASS** — `hidden0` 3.37e-02 bf16, **8.95e-05** fp32 |
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
* **Flipping `requires_grad` on the backbone moves `hidden0` by 1.244e-02.**
  *(Measured here.)*
  Same weights, same input, same device, same process; only the autograd metadata
  differs, which is enough for PyTorch's CPU path to pick a different GEMM and a
  different reduction order in bf16.

That second one is worth sitting with: the entire CUDA↔CPU disagreement is
1.48e-02, and a flag that provably cannot change the mathematics accounts for
1.244e-02 of that magnitude. **At bf16 on this stack, the magnitude of the
disagreement is mostly reduction order.** Direction is the half that
discriminates, which is why both gates must fail before a block is flagged.

It is also a trap. A frozen backbone and an unfrozen one are the same function
and do not give the same bf16 answer. `Evo2Backbone.apply_lora(freeze=True)`
freezes before anything is measured, and every shipped bundle is produced through
it — a comparison against a model that skipped that step reads as a 1.2e-02
regression that is not there.

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

Read-out norm: reference **15.395389**, CPU port **15.381534**.

| k | relerr(out) | relerr(delta) | 1−cos(delta) | relerr(l3) |
|---:|---:|---:|---:|---:|
| 0 | 3.849e-03 | 3.849e-03 | 7.409e-06 | 1.240e-03 |
| 1 | 5.274e-03 | 5.706e-03 | 1.628e-05 | 5.788e-03 |
| 2 | 4.959e-03 | 4.959e-03 | 1.204e-05 | 4.958e-03 |
| 3 | 4.998e-03 | 7.985e-03 | 3.186e-05 | 8.516e-03 |
| 4 | 2.239e-02 | 4.955e-03 | 1.181e-05 | 9.432e-02 |
| 5 | 2.224e-02 | 3.251e-02 | 5.285e-04 | 2.480e-02 |
| 6 | 2.499e-02 | 4.165e-02 | 6.822e-04 | 1.657e-02 |
| 7 | 2.376e-02 | 1.614e-02 | 1.286e-04 | 1.314e-02 |
| 8 | 2.235e-02 | 1.682e-02 | 1.387e-04 | 1.502e-02 |
| 9 | 8.135e-03 | 8.158e-03 | 1.148e-05 | 7.630e-03 |
| 10 | 8.110e-03 | 1.014e-02 | 5.142e-05 | 1.482e-02 |

**No block diverges.** `hidden0` 1.482e-02, 0 real disagreements, 0 structural
mismatches.

The residual is arithmetic, not structural: every increment cosine agrees to
≤7e-04, and the error-growth column shows no step change — only the reference's
own ~4.5× cancellation at block 4, which is a property of the reference, not of
the port.

## B — CUDA → Trainium, seq 200

Same command with `--device neuron`. Read-out norm **15.362006**.

Worst increment 4.24e-02 at block 6, worst `1 − cos(delta)` **8.98e-04** at block
6. **No block diverges**, `hidden0` 3.534e-02, 0 real disagreements, 0 structural
mismatches, `get_fallback_ops()` empty. Cold compile plus run: **44 s**.

## C — CUDA → CPU, seq 8192 (the production context)

```bash
PYTHONPATH=src python scripts/dump_artifacts.py --device cpu --dtype bf16 \
    --seq-len 8192 --inputs-from artifacts/cuda_seq8192/inputs.pt \
    --forward-only --strict-determinism --hcl-kernel chunked \
    --out-dir /tmp/cpu_seq8192_full
```

Read-out norm: reference **87.708467**, CPU port **87.679067**.

| k | relerr(out) | relerr(delta) | 1−cos(delta) | relerr(l3) |
|---:|---:|---:|---:|---:|
| 0 | 4.812e-03 | 4.812e-03 | 1.158e-05 | 1.519e-03 |
| 1 | 5.636e-03 | 6.132e-03 | 1.880e-05 | 6.193e-03 |
| 2 | 3.338e-03 | 3.334e-03 | 4.175e-06 | 3.326e-03 |
| 3 | 3.439e-03 | 1.433e-02 | 1.027e-04 | 1.665e-02 |
| 4 | 2.322e-02 | 6.954e-03 | 2.411e-05 | 1.182e-02 |
| 5 | 2.284e-02 | 1.543e-02 | 1.191e-04 | 1.346e-02 |
| 6 | 1.653e-02 | 2.733e-02 | 3.718e-04 | 1.116e-02 |
| 7 | 1.379e-02 | 1.244e-02 | 7.736e-05 | 1.291e-02 |
| 8 | 1.431e-02 | 1.505e-02 | 1.132e-04 | 1.103e-02 |
| 9 | 6.550e-03 | 6.553e-03 | 1.673e-05 | 6.125e-03 |
| 10 | 6.555e-03 | 1.524e-02 | 1.161e-04 | 1.887e-02 |

**No block diverges** at 40.96× the tokens; `hidden0` 1.887e-02, essentially the
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
worst `1 − cos(delta)` 8.69e-04, `hidden0` 3.365e-02, 0 real disagreements.

**Forward, fp32** (inside the training bundles): `hidden0` **8.953e-05** —
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
| `g1` | 3.114e-04 / 3.142e-05 | 3.329e-05 / 3.615e-07 | 0 of 74 |
| `wN` | 1.288e-03 / 5.130e-05 | 7.568e-06 / 3.476e-08 | 0 of 74 |

Losses, window-regression:

```
cpu    [0.7565693259239197, 0.7331119775772095, 0.7134156227111816]
neuron [0.7565693855285645, 0.7331119775772095, 0.7134155631065369]
```

Agreement to ~6e-08, and the loss descends. The per-position-classification
bundle is the same story (`g1` LoRA max 3.209e-04, `wN` LoRA max 9.473e-04, 0
real disagreements on all three artifacts).

`w0` at exactly 0.0 is the control that makes the rest meaningful: it says both
devices genuinely started from the same weights. Seeding instead of pinning gives
rel_err ~1.478, because `torch.manual_seed` feeds a **per-device** generator.

---

## The gradient gate, and the thing it cannot catch

A gradient tensor counts as a REAL disagreement only if it fails **both**

```
cosine < 0.99   AND   max_abs / (max |grad| anywhere in the model) > 1e-3
```

Per-tensor relative error alone is useless here: `lora_B` is zero-initialised, so
its gradient is genuinely tiny and dividing a small absolute difference by a
small norm manufactures alarming ratios out of floating-point noise. Measured,
the `lora_B` gradients reach rel_err 2e-2 while the head — which carries real
magnitude — sits at 3e-6. Only one of those is worth looking at, and it is not
the 2e-2.

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

## Reproducing all of it

```bash
make health                          # no orphaned process is holding a core
make parity-cpu                      # the CUDA axis; no device needed
make parity                          # all three gates on CPU and Neuron
make fallbacks                       # per-stage, per-dtype host-fallback census
```
