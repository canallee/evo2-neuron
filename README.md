# Evo 2 on AWS Trainium — native PyTorch, with LoRA fine-tuning

Evo 2 (Arc Institute's StripedHyena 2 DNA language model) running on
`torch.device("neuron")` in plain PyTorch: **no vortex runtime, no
TransformerEngine, no flash-attn, no Triton, no CUDA**. The same package runs
unchanged on CPU and on Trainium.

Two things ship here:

1. **The port, gated against saved CUDA artifacts.** Inference, backprop and
   training parity, with the reference tensors in the box so the gates are
   runnable rather than quotable.
2. **Generic gLM LoRA fine-tuning.** Six objectives over four read-out heads —
   sequence-level, window-level and per-nucleotide — plus a data-parallel driver
   that trains across every NeuronCore on the instance.

Everything needed is here **except the Evo 2 weights**, which are Arc's and are
downloaded from HuggingFace at first use.

> ### Read this before using it for science
>
> **This is a re-implementation, not Arc Institute's code.** The model definition
> here was written against [`ArcInstitute/evo2`](https://github.com/ArcInstitute/evo2)
> and [`Zymrael/vortex`](https://github.com/Zymrael/vortex) and patched to run
> without CUDA, vortex, TransformerEngine, flash-attn or Triton. The weights are
> Arc's, unmodified. Everything around them is not. It is not endorsed by, or
> affiliated with, Arc Institute or AWS.
>
> **Only `evo2-7b` is validated.** `evo2-1b` builds and runs, but Arc's 1B config
> sets `use_fp8_input_projections: True` and this port has no FP8 path — disabling
> it costs ~47 accuracy points, so the 1B numbers cannot be gated against anything
> and should not be trusted. `evo2-40b` is not implemented.
>
> **It is not bit-exact against CUDA, and it is not meant to be.** The read-out
> differs from the H200 reference by **1.5e-02 (CPU)** and **3.5e-02 (Trainium)**
> in bf16 — real, measured, and consistent with reduction-order differences rather
> than a defect. For calibration: flipping `requires_grad` on the backbone, which
> cannot change the mathematics, moves the same tensor by 1.2e-02. See
> [`docs/PARITY.md`](docs/PARITY.md) for the method and the controls.
>
> That is small for a representation and it is **not nothing for a downstream
> decision**. Variant effect sizes, likelihood ratios and ranked predictions can
> move. If a result matters, validate it against the CUDA path for your own task
> before relying on it.
>
> **Not validated here:** generation and decoding (KV cache and Hyena recurrence),
> context beyond 8192, batch > 1 per core, and full fine-tuning at 7B scale.

---

## Status

| | evo2-7b |
|---|---|
| forward, CPU vs the H200 reference | **PASS** — no block diverges, `hidden0` 1.48e-02 |
| forward, Trainium vs the H200 reference | **PASS** — no block diverges, `hidden0` 3.53e-02 |
| forward at seq 8192 vs the H200 reference | **PASS** — see [`docs/PARITY.md`](docs/PARITY.md) |
| backward + 3 AdamW steps, CPU vs Trainium | **PASS** — 74/74 tensors, 0 real disagreements |
| `torch_neuronx.get_fallback_ops()` | **empty** for forward, backward, `optimizer.step` and eval |
| LoRA fine-tune, seq 8192, one NeuronCore | **3.44 s/step** warm, 2,380 tok/s |
| the same across all 64 NeuronCores | **147,292 tok/s** — 61.9x, **96.7%** efficiency |

Numbers, method and the CPU↔Neuron tables: [`docs/PARITY.md`](docs/PARITY.md).
What makes it run and what makes it fast: [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md).
Measured step times and scaling: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

---

## Install

Assumes the AWS native-PyTorch device is already installed — that is where
`torch-neuronx`, `neuronx-cc` and `nki` come from, and this package deliberately
does not pin them. At the time of writing that stack is a pre-release AWS
offering: **contact the AWS Neuron team for access and installation
instructions.** Versions and constraints in
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

Nothing below the device line needs it — `make test` and `make smoke` pass on a
plain CPU install, and the CUDA parity gate runs without a device.

```bash
conda activate torch-neuron          # the environment you were given
pip install -r requirements.txt      # numpy, huggingface_hub, pytest, ruff
export PYTHONPATH=$PWD/src
```

Check it works, with no device and no weights:

```bash
make test                            # 100 unit tests, ~20 s
make smoke                           # every example on CPU, random weights
```

## Weights

```bash
huggingface-cli download arcinstitute/evo2_7b_base evo2_7b_base.pt
```

Downloaded automatically on first use. A read-out head only needs blocks 0..10,
so prune the file once and cut it from 13 GB to 4.5 GB — worth doing before a
multi-rank job, where every rank reads it:

```bash
python scripts/prune_checkpoint.py evo2_7b_base.pt evo2_7b_blocks0_10.pt --max-blocks 11
export EVO2_CHECKPOINT=$PWD/evo2_7b_blocks0_10.pt
```

## Embeddings in six lines

```python
import evo2_finetune.neuron_env as env; env.configure()   # BEFORE importing torch
import torch
from evo2_finetune.backbone import Evo2Backbone
from evo2_neuron import ByteTokenizer

backbone = Evo2Backbone("evo2-7b").to(env.resolve_device("neuron"))
ids, mask = ByteTokenizer().encode_batch(["ACGTACGTACGT"], max_length=8192)
hidden = backbone(ids.to("neuron"))          # (1, 8192, 4096)
```

`neuron_env.configure()` must run before anything imports `torch` — `import
torch` auto-loads `torch_neuronx`, which reads the compiler cache directories at
that moment. It raises rather than letting you find out later.

## LoRA fine-tuning

```python
from evo2_finetune import build_model

model = build_model(
    objective="window-regression",   # one prediction per 256 bp bin
    alias="evo2-7b", out_dim=1, device="neuron", dtype=torch.bfloat16,
    head_kwargs={"window": 256}, gradient_checkpointing=True,
)
out = model(input_ids=ids, attention_mask=mask, labels=y, weights=w)
out["loss"].backward()
```

The backbone is frozen; only the adapters and the head train — 74 tensors,
5.2 M of 2.23 B parameters at r=8.

| objective | head | label |
|---|---|---|
| `next-token-ce` | tied LM head | none — the sequence itself |
| `sequence-regression` | `MLPReadoutHead` / `ConvReadoutHead` | `(B, out_dim)` float |
| `sequence-classification` | `MLPReadoutHead` / `ConvReadoutHead` | `(B,)` class id |
| `window-regression` | `WindowTrackHead` | `(B, n_windows, out_dim)` |
| `per-position-classification` | `PerPositionHead` | `(B, L)` class id |
| `per-position-regression` | `PerPositionHead` | `(B, L, out_dim)` |

Worked examples, each runnable on CPU in seconds with `--random-backbone`:

```bash
python examples/01_embeddings.py --device cpu --random-backbone --max-blocks 3
python examples/02_lora_lm_finetune.py --fasta genome.fa --device neuron
python examples/03_window_regression.py --device neuron
python examples/04_per_nucleotide_classification.py --device neuron
```

## Training across every NeuronCore

Warm the compiler cache with **one** rank first — concurrent cold compiles have
been enough to kill the compile service:

```bash
NEURON_RT_VISIBLE_CORES=0 python -u scripts/train_lora_ddp.py --steps 12

NEURON_RT_VISIBLE_CORES=0-63 torchrun --nproc_per_node=64 \
    scripts/train_lora_ddp.py --objective window-regression --seq-len 8192 \
    --steps 200 --save-to runs/lora64 --diag-dir runs/lora64/diag
```

Defaults are the production configuration: bf16, per-block compile, activation
checkpointing, `nki-scan`, batch 1 per rank. Measured scaling and step times are
in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

`--full-finetune` unfreezes the backbone and trains everything, saving the full
trainable set rather than adapters alone. It does **not** fit evo2-7b at seq 8192
on a 24 GiB logical core — AdamW keeps two fp32 moments per parameter, ~27 GB for
the 2.23 B built parameters, before weights, gradients or activations. Use it on
evo2-1b, at a short context, or with `--max-blocks`.

## Parity gates

```bash
make health                          # no orphaned process is holding a core
make parity                          # the three gates; each exits non-zero on failure
make parity-cpu                      # the CUDA axis only — no device needed
make fallbacks                       # nothing runs on the host, both dtypes
```

`parity/01` gates the port against the saved H200 activations,
`02` gates one backward CPU↔Neuron under a magnitude-aware rule, and `03` gates
five AdamW steps — loss trajectory, final weights, **and** that the loss actually
descends, because a frozen model passes the first two perfectly.

## Layout

```
src/evo2_neuron/     the port — torch only, no fine-tuning dependencies
src/evo2_finetune/   LoRA, heads, objectives, data, ops, comparator
scripts/             training driver, artifact dump/compare, gates, checkpoint pruning
parity/              the three gates
artifacts/           saved CUDA / CPU / Neuron reference bundles
examples/            four runnable recipes
docs/                PARITY · OPTIMIZATIONS · PERFORMANCE · ARTIFACTS · ENVIRONMENT
```

## Licence

Apache-2.0. The model definition is re-implemented from
[`ArcInstitute/evo2`](https://github.com/ArcInstitute/evo2) and
[`Zymrael/vortex`](https://github.com/Zymrael/vortex), both Apache-2.0 — see
[`src/evo2_neuron/VENDORED.md`](src/evo2_neuron/VENDORED.md) for source commits
and the full patch list. Weights are Arc's and are not redistributed here.
