# Environment

This repo targets AWS's **native-PyTorch** path: a real PrivateUse1 device,
`torch.device("neuron")`, with `torch.compile(backend="neuron")`. It is **not**
the `torch_xla` path — `PJRT_DEVICE`, `xm.xla_device()` and `--device xla` do not
apply and will not work.

## Getting the stack

The native-PyTorch device is AWS's **public** `torch-neuronx` backend (2.13 / 2.14,
Neuron SDK 2.33.0), from the Neuron pip and apt repositories and
[github.com/aws-neuron/torch-neuronx](https://github.com/aws-neuron/torch-neuronx).

[`INSTALL.md`](INSTALL.md) is the recipe for a working conda environment. It is not
a single `pip install` on Ubuntu 24.04: the published wheels need glibc 2.43, so
`torch-neuronx` is built from the public source there, and the DLAMI's Neuron
runtime has to be upgraded first.

Nothing else here depends on that: the port, the heads, the objectives, the LoRA
implementation and the whole comparison harness are plain PyTorch and run on CPU.
`make test` and `make smoke` pass with no Neuron install at all, and
`parity/01_inference_parity.py --devices cpu` gates the port against the saved
CUDA reference without a device.

## Versions this was built and measured against

The short form; the full environment is [below](#the-measured-environment-exactly).

| package | version | notes |
|---|---|---|
| python | 3.12 | |
| torch | 2.14.0 | CPU build |
| torch-neuronx | 2.14.3 | provides the `neuron` device and `backend="neuron"`; built from the `v2.33.0` source on Ubuntu 24.04 |
| neuronx-cc | 2.28 | **import name is `neuronxcc`**, not `neuronx_cc` |
| nki | 0.7.0 | **bundles `nkilib`** — no separate kernel source tree needed |
| numpy | 2.5 | |
| Neuron runtime | 2.35 | driver 2.31; `torch-neuronx` 2.14 needs runtime ≥ 2.35 |

Later versions are likely fine — the constraints this code works around are listed
at the bottom of this file, and any of them may have been fixed since.

**Every CPU number in this repo is tied to its torch release.** CPU results are
bit-reproducible on one release, not across releases: moving from torch 2.12.1 to
2.14.0 shifts the CPU bf16 read-out by 1.24e-02 with no code change. Compare a CPU
run only against a bundle whose `manifest.json` records the same `torch`. The
measured differences are in [`PARITY.md`](PARITY.md#cpu-results-across-torch-releases).

## The measured environment, exactly

Every number in [`PARITY.md`](PARITY.md) and [`PERFORMANCE.md`](PERFORMANCE.md) and
every shipped CPU / Neuron bundle comes from this environment. To reproduce a number
bit for bit, match it — above all the `torch` version.

| layer | exact version |
|---|---|
| instance | `trn2.48xlarge`, `logical-neuroncore-config: 2` |
| OS | Ubuntu 24.04.4 LTS (Neuron DLAMI), kernel `6.17.0-1019-aws`, glibc 2.39 |
| Neuron driver | `aws-neuronx-dkms` 2.31.7.0 |
| Neuron runtime | `aws-neuronx-runtime-lib` 2.35.30.0, `aws-neuronx-collectives` 2.35.43.0, `aws-neuronx-tools` 2.33.38.0 |
| conda / Python | conda 26.5.3, Python 3.12.16 from conda-forge |
| torch | `2.14.0+cpu` (`download.pytorch.org/whl/cpu`) |
| torch-neuronx | `2.14.3.0.3+d9d8423.dev` — built with Bazel 8.4.2 from [aws-neuron/torch-neuronx](https://github.com/aws-neuron/torch-neuronx) at `d9d8423` (tag `v2.33.0` plus two test/example commits; no source change) |
| neuronx-cc | `2.28.7631.0+8f7b0cf5` |
| nki | `0.7.0+37713544100.gfede04bb` |
| other Python | numpy 2.5.4, huggingface_hub 2.4.0, pytest 9.2.0, ruff 0.17.4; playwright 1.62.0 for the PDF only |
| checkpoint | `arcinstitute/evo2_7b_base`, pruned to blocks 0..10 with `scripts/prune_checkpoint.py --max-blocks 11` |

Built with the recipe in [`INSTALL.md`](INSTALL.md): the runtime upgrade, a
conda-forge env named `torch-neuron`, the `torch-neuronx` source build, then a
project env cloned from it:

```bash
conda create -n torch-neuron-evo2 --clone torch-neuron --offline -c conda-forge --override-channels
conda activate torch-neuron-evo2
pip install -r requirements.txt
```

Every run was made with the env **activated** and these variables set:

```bash
export PYTHONPATH=$PWD/src
export EVO2_CHECKPOINT=/path/to/evo2_7b_blocks0_10.pt
export NEURON_CACHE_ROOT=$HOME/.cache/evo2-neuron     # one cache per torch-neuronx build
```

Keep one `NEURON_CACHE_ROOT` per `torch-neuronx` build. Whether a kernel cached by
another build is reused is not something to rely on in either direction; a cache
per build keeps cold-compile timings honest and rules out the stale-kernel trap
described under `NEURON_CC_FLAGS` below.

Instance: `trn2.48xlarge` — 16 Trainium2 devices, `logical-neuroncore-config: 2`,
so **64 logical NeuronCores at 24 GiB each**, split across 2 NUMA nodes.

`pip install -r requirements.txt` installs `numpy` and `huggingface_hub` (plus
`pytest` and `ruff` for development). torch and the Neuron stack are deliberately
**not** pinned in `pyproject.toml`: `torch-neuronx` must match the torch release
exactly, comes from the Neuron index rather than PyPI, and on Ubuntu 24.04 is a
local source build that a resolver would happily replace with a wheel that cannot
load.

## Three things that fail without naming their cause

### 1. Compiler cache directories must be set before `torch` is imported

`import torch` auto-loads `torch_neuronx` through an entry point, and
`torch_neuronx` reads the four cache variables **at import time**. Setting them
afterwards has no effect and the run silently uses `/tmp`.

```python
import evo2_finetune.neuron_env as env
env.configure()          # FIRST — before anything that imports torch
import torch
```

`configure()` raises if it is already too late. It sets:

```
TORCH_NEURONX_HLO_CACHE_DIR
TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR
TORCH_NEURONX_NEFF_CACHE_DIR
TORCH_NEURONX_DEBUG_DIR
```

under `$NEURON_CACHE_ROOT`, defaulting to `~/.cache/evo2-neuron`. The defaults
live under `/tmp`, owned by whoever ran Neuron on the box first; the resulting
`PermissionError` surfaces as a generic **"Compilation error"**, which sends you
looking at your graph.

Point the cache somewhere persistent. Cold compile against warm start is the
difference between minutes and seconds.

### 2. The conda env's `bin/` must be on `PATH`

The runtime spawns `neuronx-cc` **by name**. Without it on `PATH`, every compile
dies with `posix_spawn failed`. `configure()` prepends
`Path(sys.executable).parent`.

### 3. Cores and the process group, under DDP

```bash
NEURON_RT_VISIBLE_CORES=0        # one core
NEURON_RT_VISIBLE_CORES=0-63     # DDP across all 64 logical cores
```

Launch with the full range, but **each rank must end up seeing only its own
core**. Every rank inherits the same variable, and every rank's runtime then
claims the whole range: one wins and the rest die with `Failed to initialize
Neuron Runtime: status code 1`. `neuron_env.configure()` reads `LOCAL_RANK` and
narrows the variable before the runtime sees it.

Narrowing does **not** renumber. After narrowing to core 3, `torch.device
("neuron", 0)` raises `Device index 0 is out of range. Valid range is 3 to 3` —
the device is still addressed by its global id, which `neuron_env.global_core_id()`
returns.

Leaving the variable at `0` under torchrun is the opposite failure: every rank
shares one core, the job runs, the numbers are correct, and it is N× slower with
no error. The driver refuses instead, naming the rank and the range.

The process group must use the **`neuron` backend**, which `torch_neuronx`
registers for the `neuron` device type. `gloo` initialises fine and then fails at
the first collective with `No backend type associated with device type neuron`.

## Also worth knowing

* **`NEURON_CC_FLAGS` persists through the NEFF cache.** A kernel compiled under
  different flags will be served to you. `make clean-cache` before A/B-ing flags.
* **`NEURON_RT_EXEC_TIMEOUT`** defaults to 300 s. A cold compile of eleven 7B
  blocks at 8192 exceeds it, and the timeout surfaces as an opaque runtime error
  rather than "still compiling". The driver sets 600.
* **Warm the cache with one rank** before launching many. Four concurrent cold
  compiles have been enough to kill the compile service.
* **Do not pipe a long device run through `tail`.** It buffers until the process
  exits, so a healthy multi-minute compile looks hung. Use `python -u`.

## Runtime constraints that shape this code

Re-checked on torch-neuronx 2.14. Several have loosened since the private-beta
build this port was first developed on; the code keeps the workarounds because
they are measured, validated and keep every stage on the device.

| constraint (2.14) | consequence here |
|---|---|
| `torch.fft.rfft` runs on the device, but the inverse (`aten::_fft_c2r`) falls back to the host | Hyena's FFT convolutions stay rewritten as depthwise causal `conv1d`, which keeps them on the device |
| dynamic shapes unsupported under `torch.compile` (`dynamic=False`) | every collator pads to a fixed length |
| float64 runs as float32 (int64 is now exact) | comparisons cast to float64 on the **host**, after `.cpu()` |
| `torch_neuronx.trace` does not exist | eager first, then `torch.compile(backend="neuron")` |
| only `mode="default"` is supported for `torch.compile` | only the default compile mode is used |
| the built-in backward of `Tensor.norm` sends an integer-zero comparison to the host | `RMSNorm` uses a bit-identical custom backward (`layers._L2Norm`) |

## Pre-flight

```bash
bash scripts/neuron_health_check.sh
```

Reports the instance, any process holding a NeuronCore, and the cache
configuration. Exits non-zero if a core is held. Run it before believing a timing
number: an orphaned process fakes a mid-run throughput collapse, and a bimodal
step-time trace is a contention signature until proven otherwise.
