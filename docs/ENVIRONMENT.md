# Environment

This repo targets the **AWS native-PyTorch beta**: a real PrivateUse1 device,
`torch.device("neuron")`, with `torch.compile(backend="neuron")`. It is **not**
the `torch_xla` path — `PJRT_DEVICE`, `xm.xla_device()` and `--device xla` do not
apply and will not work.

## Versions this was built and measured against

| package | version | source |
|---|---|---|
| python | 3.12.13 | conda |
| torch | 2.12.1+cpu | PyPI CPU wheel |
| torch-neuronx | 2.12.3.0.1636+5c472775.dev | beta, editable install |
| neuronx-cc | 2.27.2878.0+8220f7ac | beta wheel — **import name is `neuronxcc`** |
| nki | 0.6.0 | beta wheel; **bundles `nkilib`**, no separate source tree needed |
| numpy | 2.5.1 | PyPI |

Instance: `trn2.48xlarge` — 16 Trainium2 devices, `logical-neuroncore-config: 2`,
so **64 logical NeuronCores at 24 GiB each**, split across 2 NUMA nodes.

`pip install -r requirements.txt` installs only `numpy` and `huggingface_hub`.
torch and the Neuron stack are deliberately **not** pinned in `pyproject.toml`:
installing them from PyPI would shadow the editable beta install.

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

## Beta constraints that shape this code

| constraint | consequence here |
|---|---|
| complex / FFT ops do not compile (`NCC_EVRF004`) | Hyena's FFT convolutions are rewritten as depthwise causal `conv1d` |
| dynamic shapes unsupported under `torch.compile` | every collator pads to a fixed length |
| float64 → float32 and int64 → int32 auto-downcast | comparisons cast to float64 on the **host**, after `.cpu()` |
| `torch_neuronx.trace` does not exist | eager first, then `torch.compile(backend="neuron")` |
| `reduce-overhead` / `max-autotune` fall back to default | only the default compile mode is used |

## Pre-flight

```bash
bash scripts/neuron_health_check.sh
```

Reports the instance, any process holding a NeuronCore, and the cache
configuration. Exits non-zero if a core is held. Run it before believing a timing
number: an orphaned process fakes a mid-run throughput collapse, and a bimodal
step-time trace is a contention signature until proven otherwise.
