# Artifact bundles

A bundle is one directory of tensors that pins everything a device port can get
wrong. Dump one per device, then diff them.

## Schema

| file | contents | present when |
|---|---|---|
| `inputs.pt` | `{"batches": [{input_ids, attention_mask, window_mask, labels, weights}]}` | always |
| `blocks0_10.pt` | the residual stream — `block_out.k` and `mlp_l3.k` per built block | unless `--no-dump-blocks` |
| `hidden0.pt` | `{"h": ...}` — the mid-stack read-out, first forward | always |
| `w0.pt` | trainable weights before step 0 | unless `--forward-only` |
| `g1.pt` | gradients after the first backward, before any step | unless `--forward-only` |
| `wN.pt` | trainable weights after N steps | unless `--forward-only` |
| `losses.json` | the loss at each step | always |
| `manifest.json` | shapes, param counts, resolved config, provenance | always |
| `SHA256SUMS.json` | per-file checksum and size | imported bundles |

`hidden0` and `blocks0_10` are captured on the **first** forward, before any
optimizer step. `lora_B` is zero at that point, so the adapters are exactly the
identity and both files are **pure backbone activations** — independent of the
head and objective bolted on top. That is what makes a CUDA bundle produced by
one task usable to gate a port trained on another.

## What ships, and what is in the repo

The bundles are **~3.5 GB of tensors and are not in git** — `cuda_seq8192/blocks0_10.pt`
alone is 2.95 GB, past GitHub's 100 MB per-file limit and its 2 GB release-asset
limit. What *is* committed, for every bundle, is `manifest.json` (shapes, config,
provenance, read-out norm) and `SHA256SUMS.json` (per-file checksum and size).

So the repo tells you exactly what each bundle contains and lets you verify one
byte-for-byte once you have it; it just does not carry the payload. Ask the
maintainer for the tensors, or regenerate the CPU and Neuron bundles yourself with
`scripts/dump_artifacts.py` — every command is in [`PARITY.md`](PARITY.md). Only the
two `cuda_*` bundles cannot be regenerated without an H200.

`parity/01` fails with a clear message rather than a stack trace when a bundle is
missing.

**CPU and Neuron bundles are tied to the stack that wrote them.** The shipped ones
were written by torch 2.14.0 / torch-neuronx 2.14.3 (`provenance` in each
`manifest.json`). A CPU bundle is bit-reproducible on that torch release only —
torch 2.12.1 writes a bf16 read-out 1.24e-02 away — so regenerate the CPU and Neuron
bundles when you change torch, and compare like with like. Measured differences:
[`PARITY.md`](PARITY.md#cpu-results-across-torch-releases). The CUDA bundles are
unaffected.

~3.5 GB.

| bundle | device | seq | contents |
|---|---|---|---|
| `cuda_seq200` | H200 | 200 | inputs, hidden0, blocks0_10 — the reference |
| `cuda_seq8192` | H200 | 8192 | inputs, hidden0, blocks0_10 — the production context |
| `cpu_seq200_fwd` | CPU bf16 | 200 | forward-only, with stream |
| `neuron_seq200_fwd` | Trainium bf16 | 200 | forward-only, with stream |
| `cpu_seq8192_fwd` | CPU bf16 | 8192 | inputs, hidden0 (stream regenerated on demand) |
| `cpu_seq200_window` | CPU fp32 | 200 | 3 steps, window-regression — w0/g1/wN |
| `neuron_seq200_window` | Trainium fp32 | 200 | 3 steps, window-regression — w0/g1/wN |
| `cpu_seq200_pernt` | CPU fp32 | 200 | 3 steps, per-position-classification |
| `neuron_seq200_pernt` | Trainium fp32 | 200 | 3 steps, per-position-classification |

The CUDA bundles are **forward-only by design**. Their `w0`/`g1`/`wN` are a
function of the head and objective that produced them, and reproducing those
would require that exact pairing. The activations are not, so they ship; the
gradient and training axes are gated CPU↔Neuron instead, which anyone with this
repo can regenerate.

The 8192 stream tensor is 2.9 GB. It ships once, on the reference side. The CPU
and Neuron 8192 arms regenerate it on demand rather than shipping a second and
third copy — the command is in [`PARITY.md`](PARITY.md) §C.

## Writing a bundle

```bash
PYTHONPATH=src python scripts/dump_artifacts.py \
    --device neuron --dtype bf16 --seq-len 200 \
    --objective window-regression --window 200 --out-dim 2 --steps 3 \
    --inputs-from artifacts/cuda_seq200/inputs.pt \
    --strict-determinism --out-dir artifacts/my_run
```

**`--inputs-from` and `--init-from` are effectively mandatory for anything
cross-device.** `torch.manual_seed` feeds a *per-device* generator, so a head
initialised on each device from the same seed gives CPU and Neuron unrelated
weights — measured rel_err 1.419, i.e. a comparison between two different models.
(`dump_artifacts.py` builds on CPU and then moves, so it is not exposed; code that
initialises on the device is.) Pin both and the gradients are the
only free variable. `--init-from` refuses a partial pin rather than covering some
tensors and leaving the rest random.

Useful flags: `--forward-only` (the 8192 gate that fits in 24 GiB),
`--no-dump-blocks`, `--random-backbone` (structure only, seconds),
`--hcl-kernel {direct,chunked,nki-scan}`, `--compile-blocks`,
`--gradient-checkpointing`, `--strict-determinism`.

## Comparing two bundles

```bash
PYTHONPATH=src python scripts/dump_artifacts.py --compare REF TEST [--gate-stream 3e-3]
```

Prints, in this order:

1. **config and provenance diffs.** Most "why don't these match" questions are
   answered here — a different dtype, a different seed, a reference minted on
   another build. Config differences are marked `!`; provenance differences
   between a CUDA reference and a Trainium run are expected.
2. **`inputs.pt`.** Run first, because if the inputs differ nothing downstream
   means anything — and an input mismatch is otherwise indistinguishable from a
   broken backward.
3. **the residual stream**, per block, on the increment. This is the gate. See
   [`PARITY.md`](PARITY.md) for why the increment and not the tensor.
4. **`hidden0`, `w0`, `g1`, `wN`**, each with a magnitude-aware verdict.

Exit code is non-zero on any structural mismatch, so it works as a CI step.

Read it top-down and **stop at the first divergence**: everything after it is
downstream of a known-wrong input and is not evidence.

## Importing an external reference

`scripts/sanitize_bundle.py` imports a bundle produced elsewhere, keeping only
the task-independent files and rewriting the manifest into this repo's
vocabulary:

```bash
python scripts/sanitize_bundle.py SOURCE artifacts/cuda_seq200 --note "NVIDIA H200, seq 200"
python scripts/sanitize_bundle.py SOURCE artifacts/cuda_seq200 --manifest-only   # re-stamp
```

It keeps `input_ids` / `attention_mask` / `window_mask` and drops everything else
from the batch dict, then writes `SHA256SUMS.json`.

## Self-noise

Before trusting any comparison, dump the same configuration twice and diff it
against itself. A self-noise floor above zero bounds every conclusion you can
draw from the numbers, and `--strict-determinism` turns a silently
non-reproducible kernel into an exception rather than a mystery.
