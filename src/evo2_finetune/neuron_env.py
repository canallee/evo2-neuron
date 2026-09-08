"""
Neuron environment setup — **import this before `torch_neuronx`**.

Three of the four things here are load-bearing and all of them fail in ways that
do not name their cause.

1. **Compiler cache directories.** The four `TORCH_NEURONX_*` cache dirs default
   under `/tmp`, owned by whoever ran Neuron on the box first. On a shared
   machine the resulting `PermissionError` surfaces as a generic
   "Compilation error", which sends you looking at your graph. They must be set
   *before* `torch_neuronx` is imported — it reads them at import time.

2. **Visible cores.** `NEURON_RT_VISIBLE_CORES=0` pins one logical core. For DDP
   it must be the full range (`0-63`), because the backend splits that range per
   local rank; leave it at `0` and every rank shares one core, the job runs, and
   it is silently N× slower.

3. **Compiler on PATH.** The runtime spawns `neuronx-cc` by name. If the conda
   env's `bin/` is not on `PATH`, every compile dies with `posix_spawn failed`.

4. **`NEURON_CC_FLAGS`.** Not set here, but worth knowing: a stale value survives
   in the NEFF cache, so a cached kernel compiled under different flags will be
   served to you. Clear the cache before concluding a flag did nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "clear_op_tracking",
    "configure",
    "fallback_ops",
    "global_core_id",
    "resolve_device",
    "synchronize",
    "visible_cores",
]

_CACHE_VARS = (
    "TORCH_NEURONX_HLO_CACHE_DIR",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
    "TORCH_NEURONX_NEFF_CACHE_DIR",
    "TORCH_NEURONX_DEBUG_DIR",
)


def configure(cache_root: str | Path | None = None, *, cores: str | None = None,
              local_rank: int | None = None) -> Path:
    """Point the compiler caches at a user-owned directory and pin cores.

    `cache_root` defaults to `$NEURON_CACHE_ROOT`, else `~/.cache/evo2-neuron`.
    Returns the root. Idempotent, and never overrides a variable the caller has
    already set.

    Sharing one cache root across runs is the difference between a ~100 s cold
    compile and a ~8 s warm start, so point it somewhere persistent.
    """
    root = Path(cache_root or os.environ.get("NEURON_CACHE_ROOT")
                or Path.home() / ".cache" / "evo2-neuron").expanduser()

    unset = [name for name in _CACHE_VARS if name not in os.environ]

    # `import torch` auto-loads `torch_neuronx` through an entry point in this
    # build, so "before torch_neuronx" means "before torch". Setting a cache dir
    # afterwards has no effect, and the failure is silent -- the run just uses
    # /tmp. Refuse only when we would actually have had to set something; a
    # caller whose shell already exported all four is fine either way.
    if unset and "torch_neuronx" in sys.modules:
        raise RuntimeError(
            "torch_neuronx is already imported, so setting "
            f"{', '.join(unset)} now would have no effect. Call "
            "neuron_env.configure() at the top of your script, before any import "
            "that pulls in torch -- or export the four TORCH_NEURONX_* variables "
            "in the shell."
        )

    for index, name in enumerate(_CACHE_VARS):
        target = root / ("debug" if index == 3 else f"cache{index}")
        target.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(target))

    if cores is not None:
        os.environ["NEURON_RT_VISIBLE_CORES"] = cores
    else:
        os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

    # Under torchrun, EVERY rank inherits the same `NEURON_RT_VISIBLE_CORES`
    # range, and every rank's runtime then tries to claim the whole range. Rank 0
    # wins and the rest die with `Failed to initialize Neuron Runtime: status
    # code 1` — which names neither the cause nor the rank that took the cores.
    #
    # So the range is narrowed here to the single core this rank owns, before the
    # runtime ever sees it. Each rank consequently addresses its core as index 0
    # within its own view; `assert_cores_distinct` still compares the GLOBAL ids,
    # which is what proves two ranks did not land on the same silicon.
    if local_rank is None and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank is not None:
        available = visible_cores()

        if local_rank >= len(available):
            raise RuntimeError(
                f"local rank {local_rank} but NEURON_RT_VISIBLE_CORES exposes only "
                f"{len(available)} core(s): {os.environ['NEURON_RT_VISIBLE_CORES']!r}. "
                "For DDP this must be the full range, e.g. 0-63."
            )

        os.environ["NEURON_RT_GLOBAL_CORE_ID"] = str(available[local_rank])
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(available[local_rank])

    # The runtime spawns the compiler by name.
    bin_dir = str(Path(sys.executable).parent)

    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    return root


def visible_cores() -> list[int]:
    """Parse `NEURON_RT_VISIBLE_CORES` (`"0"`, `"0-3"`, `"0,2,5"`) to a list."""
    spec = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    cores: list[int] = []

    for part in spec.split(","):
        if "-" in part:
            low, high = part.split("-")
            cores.extend(range(int(low), int(high) + 1))
        else:
            cores.append(int(part))

    return cores


def global_core_id() -> int:
    """The global NeuronCore id this process owns, as narrowed by `configure`."""
    return int(os.environ.get("NEURON_RT_GLOBAL_CORE_ID", visible_cores()[0]))


def resolve_device(name: str, index: int | None = None):
    """`"cpu"` or `"neuron"`. Importing `torch_neuronx` is what registers the backend.

    Under DDP, `configure(local_rank=...)` has already narrowed
    `NEURON_RT_VISIBLE_CORES` to this rank's single core, so the index is 0 and
    `index` should be left `None`. Use `global_core_id()` for logging and for the
    distinctness assertion — that is the number that says which silicon this rank
    is actually on.
    """
    import torch

    if name == "neuron":
        import torch_neuronx  # noqa: F401

        return torch.device("neuron", index) if index is not None else torch.device("neuron")

    return torch.device(name)


def fallback_ops() -> list[str]:
    """Sorted `torch_neuronx.get_fallback_ops()` — the ops that ran on the host.

    Empty is the goal. Two caveats before reading one:

    * An empty list is also what a stage that executed nothing returns. Pair it
      with `get_executed_ops()` being non-empty, as `check_fallback_ops.py` does.
    * The list is **cumulative from process start**, so it includes the
      host-to-device transfer of the weights and the first batch. Those show up
      as `aten::_to_copy` and are not a fallback of the compute graph — they are
      the copy that has to happen. Call `clear_op_tracking()` after warmup to get
      a list that describes steady state.
    """
    try:
        import torch_neuronx
    except ImportError:
        return []

    return sorted(set(torch_neuronx.get_fallback_ops()))


def clear_op_tracking() -> None:
    """Reset the executed/fallback op census. A no-op off device."""
    try:
        import torch_neuronx

        torch_neuronx.clear_op_tracking()
    except (ImportError, AttributeError):
        pass


def synchronize() -> None:
    """Block until the device is idle. A no-op off device.

    Needed around any timing measurement: execution is asynchronous, so an
    unsynchronised `time.perf_counter()` around a step measures dispatch.
    """
    try:
        import torch_neuronx

        torch_neuronx.synchronize()
    except (ImportError, AttributeError):
        pass
