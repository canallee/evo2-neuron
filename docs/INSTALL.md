# Building the Neuron environment

The `neuron` device comes from AWS's **public native-PyTorch backend**,
`torch-neuronx` 2.13 / 2.14 (Neuron SDK 2.33.0). It installs from public pip and
apt repositories — but on **Ubuntu 24.04** the published wheels do not load, and
several steps fail in ways that do not name their cause. This file is the
recipe and the gotchas.

Upstream entry points: the [torch-neuronx repository](https://github.com/aws-neuron/torch-neuronx)
(README and `release-notes/2.33.0.md`) and the
[install guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/get-started/quickstart-torch-neuron.html).

You do **not** need any of this to work on the code: `make test`, `make smoke` and
`parity/01_inference_parity.py --devices cpu` all run on a plain CPU PyTorch
install. It is only the `neuron` device that needs the stack below.

---

## 1. Neuron driver and runtime: runtime ≥ 2.35

A Neuron DLAMI ships older Neuron packages than `torch-neuronx` 2.14 needs, and
holds them so unattended upgrades cannot swap the driver. Stop every Neuron job
first — the DKMS post-install reloads the kernel module.

```bash
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade \
  --allow-change-held-packages \
  aws-neuronx-dkms aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools
sudo apt-mark hold aws-neuronx-dkms aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools
cat /sys/module/neuron/version
```

Measured against: driver 2.31.7, runtime-lib 2.35.30, collectives 2.35.43,
tools 2.33.38.

## 2. The environment

```bash
# Miniconda, if you do not already have it
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3" && rm miniconda.sh
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda create -n torch-neuron -c conda-forge --override-channels python=3.12 pip -y
conda activate torch-neuron

pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install "neuronx-cc==2.*" "nki>=0.6" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

Then `torch-neuronx` itself — which depends on the OS.

**Ubuntu 26.04:** the published wheel works.

```bash
pip install "torch-neuronx==2.14.*" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

**Ubuntu 24.04: build it from source.** Every published 2.13 / 2.14 wheel is built
on Ubuntu 26.04 and links against glibc **2.43**; 24.04 ships 2.39. It installs
cleanly and then fails at import:

```
ImportError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.43' not found
(required by .../torch_neuronx/_C.so)
```

The wheels are tagged plain `linux_x86_64`, so pip cannot see the mismatch. Build
the same release from the public source with Bazel instead:

```bash
pip install "setuptools>=77.0.3" wheel grpcio-tools patchelf
mkdir -p ~/.local/bin
curl -fsSL -o ~/.local/bin/bazelisk \
  https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-amd64
chmod +x ~/.local/bin/bazelisk && ln -sf ~/.local/bin/bazelisk ~/.local/bin/bazel

git clone https://github.com/aws-neuron/torch-neuronx.git ~/torch-neuronx
cd ~/torch-neuronx                                    # tag v2.33.0
export PATH=$HOME/.local/bin:$PATH USE_BAZEL=1 PYTHON_VERSION=3.12
python setup.py bdist_wheel release -d dist          # compiles
python setup.py bdist_wheel release -d dist          # packages the torch_mlir .so files
git checkout -- torch_neuronx/_build_config.py
pip install dist/torch_neuronx-*.whl
```

A few minutes on a `trn2.48xlarge`; much longer on a small instance. The version
string carries a git suffix (`2.14.3.0.3+d9d8423.dev`) instead of the official build
number — same source, cosmetic.

### Things that bite

1. **`conda create` blocks on a Terms-of-Service prompt** if it resolves against
   Anaconda's `defaults` channels — `CondaToSNonInteractiveError`, in a
   non-interactive shell, with no obvious fix. `-c conda-forge
   --override-channels` avoids the channel entirely.
2. **conda-forge's `python` may not bundle `pip`.** A bare `pip install` then
   falls through to the *system* pip and dies with
   `externally-managed-environment` (PEP 668) — which reads like a broken conda
   env and is not. Ask for `pip` in the `conda create` line, as above.
3. **Keep the version pins.** Older `torch-neuronx` versions on the Neuron pip
   index (`2.9.0.2.x`, `2.8.x`, …) are the XLA-based product, not this one.
   `torch-neuronx` also does not depend on `neuronx-cc` or `nki`; install them
   explicitly.
4. **Run `bdist_wheel` twice.** The first pass compiles and copies `_mlir.so`,
   `_stablehlo.so` and `_torchMlir.so` into the tree *after* setuptools has
   collected the package files, so that wheel holds only `_C.so` and the import
   fails with `No module named 'torch_mlir._mlir_libs._mlir'`. Expect four `.so`
   files in `unzip -l dist/*.whl`.
5. **Only `USE_BAZEL=1` builds.** `pip install .` and the default build path fail
   with `setup script specifies an absolute path`; `USE_CMAKE=1` fails on a stale
   source list. `patchelf` is required for the packaging step, and the compile needs
   the runtime ≥ 2.35 headers from step 1 (`'nrt_compute_tensor_t' does not name a
   type` otherwise).

Python 3.12, 3.13 and 3.14 are all supported; this repo was measured on 3.12.

### Verify

```bash
python -m torch_neuronx.collect_env                  # "device usable from torch : yes"
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

Clone rather than rebuild, so every project shares one device stack:

```bash
conda create -n torch-neuron-evo2 --clone torch-neuron --offline -c conda-forge --override-channels
conda activate torch-neuron-evo2
pip install -r requirements.txt        # numpy, huggingface_hub, pytest, ruff
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
launched from cron, systemd, a bare `ssh host command`, or by calling
`envs/<name>/bin/python` directly may not inherit it.

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
