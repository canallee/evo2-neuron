# Building the `torch-neuron` environment

The native-PyTorch device does not arrive as a `pip install`. You assemble an
environment from artifacts AWS gives you, and several steps fail in ways that do
not name their cause. This file is the assembly recipe and the gotchas.

> **It does not tell you how to obtain the artifacts.** At the time of writing the
> stack is a pre-release AWS offering and its distribution is theirs to control.
> **Contact the AWS Neuron team** for access and for the current instructions;
> public entry points are the [Neuron
> documentation](https://awsdocs-neuron.readthedocs-hosted.com/) and the
> [aws-neuron organisation](https://github.com/aws-neuron). What follows assumes
> you already have:
>
> | you need | referred to below as |
> |---|---|
> | the `nki` and `neuronx-cc` wheels | `$NEURON_WHEELS` |
> | the `torch-neuronx` source tree, for an editable install | `$TORCH_NEURONX_SRC` |
>
> Both come out of the release image AWS points you at. The Neuron **runtime**
> (`aws-neuronx-dkms`, `runtime-lib`, `collectives`, `tools`) is already installed
> on a current Neuron AMI — compare versions before installing anything over it,
> and expect that step to be a no-op.

You do **not** need any of this to work on the code: `make test`, `make smoke` and
`parity/01_inference_parity.py --devices cpu` all run on a plain CPU PyTorch
install. It is only the `neuron` device that needs the stack below.

---

## The environment

```bash
export NEURON_WHEELS=/path/to/wheels          # nki-*.whl, neuronx_cc-*.whl
export TORCH_NEURONX_SRC=/path/to/torch-neuronx

# Miniconda, if you do not already have it
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3" && rm miniconda.sh
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda create -n torch-neuron -c conda-forge --override-channels python=3.12 -y
conda activate torch-neuron
conda install -c conda-forge --override-channels -y pip uv
export UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX

uv pip install "$NEURON_WHEELS"/nki-*-cp312-cp312-linux_x86_64.whl
uv pip install "$NEURON_WHEELS"/neuronx_cc-*-cp312-cp312-linux_x86_64.whl
cd "$TORCH_NEURONX_SRC" && uv pip install -e '.[dev]'
```

### Four things that bite

1. **`conda create` blocks on a Terms-of-Service prompt** if it resolves against
   Anaconda's `defaults` channels — `CondaToSNonInteractiveError`, in a
   non-interactive shell, with no obvious fix. `-c conda-forge
   --override-channels` avoids the channel entirely.
2. **conda-forge's `python` does not bundle `pip`.** A bare `pip install` then
   falls through to the *system* pip and dies with
   `externally-managed-environment` (PEP 668) — which reads like a broken conda
   env and is not. Install `pip` and `uv` from conda-forge into the env first.
3. **Install `torch-neuronx` from the tree AWS gave you, not from a GitHub
   clone.** The public clone is source only; it will not produce the build you
   were given, and the failure surfaces much later as missing device support.
4. **The wheels are `cp312`-only.** That is why the env pins Python 3.12; any
   other version fails at wheel-tag resolution.

No compiler toolchain is required. `torch_neuronx/_C.so` ships prebuilt in the
tree, so the editable install reuses it instead of invoking bazel or cmake.

The locally installed `torch-neuronx` may report a different build string than the
image it came from — the version is derived from git metadata and a copied tree
has none. Same code; cosmetic.

### Verify

```bash
python -c "import torch, torch_neuronx, neuronxcc; print(torch.__version__, torch_neuronx.__version__)"
python -c "import torch, torch_neuronx; print(torch.ones(4, device='neuron').sum().item())"   # 4.0
python -c "
import torch, torch_neuronx
d = torch.device('neuron'); x = torch.ones(8, device=d)
print((x + x).sum().item())"                                                                   # 16.0
neuron-ls
```

If the device checks print `4.0` and `16.0`, the stack is working.

---

## A project environment

Clone rather than rebuild, so the editable `torch-neuronx` install is shared and
you cannot end up with two subtly different device stacks:

```bash
conda create -n torch-neuron-evo2 --clone torch-neuron --offline -c conda-forge --override-channels
conda activate torch-neuron-evo2
pip install -r requirements.txt        # numpy, huggingface_hub
export PYTHONPATH=$PWD/src
```

`--offline` and the explicit channel matter for the same Terms-of-Service reason
as above: a plain `--clone` re-resolves against `defaults` and blocks.

---

## Two environment settings that are not optional

Both are covered in [`ENVIRONMENT.md`](ENVIRONMENT.md); repeated here because they
are part of getting a *working* install, not part of tuning one.

**The env's `bin/` must be on `PATH`.** The runtime spawns `neuronx-cc` by name,
so without it every compile dies with `posix_spawn failed` — which looks like a
compiler bug and is a `PATH` bug. `conda activate` normally handles this; a job
launched from cron, systemd or a bare `ssh host command` may not inherit it.

**The compiler cache directories must be set before `torch` is imported.**
`import torch` auto-loads `torch_neuronx`, which reads them at import time.
Setting them afterwards silently has no effect and the run uses `/tmp` — where, on
a shared box, a `PermissionError` surfaces as a generic "Compilation error".

```python
import evo2_finetune.neuron_env as env
env.configure()          # FIRST — before anything that imports torch
import torch
```

`configure()` raises if it is already too late, rather than letting you find out
from a confusing compile failure an hour in.

---

## Smoke test

```bash
conda activate torch-neuron-evo2
export PYTHONPATH=$PWD/src

make health                                   # no orphaned process holding a core
NEURON_RT_VISIBLE_CORES=0 python -u scripts/check_fallback_ops.py \
    --random-backbone --max-blocks 2 --seq-len 512 --hcl-kernel direct
```

That builds a two-block model with random weights — no checkpoint, no download —
runs a forward, a backward, a clip and an optimizer step on the device, and
reports which ops fell back to the host. It should print `GATE: PASS`. First run
includes a compile; expect a minute or two.
