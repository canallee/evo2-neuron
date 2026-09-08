"""
Shared builder for the parity checks — one model, one input, on every device.

`01`, `02` and `03` all go through here so that a key mismatch cannot masquerade
as a numerical failure. Running two different builders on the two sides and
diffing the results is how a comparison reports every tensor as MISSING and reads
as a catastrophic numerical failure instead of a typo.

Two things that will bite, both measured:

**Pin the weights; do not seed them.** `torch.manual_seed` feeds a *per-device*
generator, so the same seed draws unrelated numbers on CPU and on Neuron. The
backbone comes from a checkpoint and matches anywhere; a randomly initialised
head does not. An earlier version of a check like `01` seeded instead of pinning
and reported a flawless read-out (cosine 0.99999995) beside a loss that was 235%
off. Gate the loss, not just the read-out.

**Do not pipe a long device run through `tail`.** It buffers everything until the
process exits, so a healthy multi-minute compile looks hung. Use `python -u`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo2_finetune import neuron_env  # noqa: E402  MUST precede any torch import

neuron_env.configure()

import torch  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
CUDA_SEQ200 = ARTIFACTS / "cuda_seq200"
CUDA_SEQ8192 = ARTIFACTS / "cuda_seq8192"

#: Defaults for every parity run. Batch 1 at the sequence length the shipped
#: CUDA bundle uses, so `inputs.pt` can be replayed verbatim.
SEQ_LEN = 200
READOUT_BLOCK = 10
LORA_RANK = 8


def resolve_device(name: str) -> torch.device:
    return neuron_env.resolve_device(name)


def build(
    device: str = "cpu",
    *,
    objective: str = "window-regression",
    dtype: torch.dtype = torch.float32,
    out_dim: int = 1,
    window: int = 200,
    lora_rank: int = LORA_RANK,
    init_from: str | Path | None = None,
    checkpoint: str | None = None,
):
    """Build the port + LoRA + a read-out head on `device`.

    `init_from` pins every trainable tensor from a saved `w0.pt`. **Pass it** for
    anything that compares two devices — see the module docstring.
    """
    from evo2_finetune.model import build_model

    model = build_model(
        objective=objective,
        alias="evo2-7b",
        out_dim=out_dim,
        device="cpu",
        dtype=dtype,
        lora_rank=lora_rank,
        lora_alpha=float(lora_rank),
        lora_dropout=0.0,
        head_kwargs={"window": window} if objective == "window-regression" else {},
        readout_block=READOUT_BLOCK,
        checkpoint=checkpoint or os.environ.get("EVO2_CHECKPOINT"),
    )

    if init_from is not None:
        pin_weights(model, init_from)

    return model.to(resolve_device(device))


def pin_weights(model, path: str | Path) -> int:
    """Copy every trainable tensor from a saved `w0.pt`. Raises on a partial pin."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    loaded = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state and name in trainable:
                param.copy_(state[name].to(param.dtype))
                loaded += 1

    if loaded != len(trainable):
        raise RuntimeError(
            f"pinned {loaded} of {len(trainable)} trainable tensors from {path}. "
            "A partial pin leaves the rest drawn from a per-device generator, which "
            "makes every cross-device number meaningless."
        )

    return loaded


def build_inputs(device: str, bundle: Path = CUDA_SEQ200, index: int = 0) -> dict:
    """Replay one batch from a bundle's `inputs.pt`, byte-identical on every device.

    Using the frozen batch rather than re-tokenising keeps every device on
    identical `input_ids`. Re-tokenising would give a perfectly reproducible run
    on the *wrong* input, which is indistinguishable from a numerics bug.
    """
    saved = torch.load(bundle / "inputs.pt", map_location="cpu", weights_only=False)
    target = resolve_device(device)

    return {k: v.to(target) for k, v in saved["batches"][index].items()}


def load_reference(bundle: Path, name: str):
    return torch.load(bundle / name, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.flatten().double(), b.flatten().double()
    denom = a64.norm() * b64.norm()

    return 1.0 if denom == 0 else float(torch.dot(a64, b64) / denom)


def rel_err(reference: torch.Tensor, test: torch.Tensor) -> float:
    ref64 = reference.double()

    return float((test.double() - ref64).norm() / (ref64.norm() + 1e-12))


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def report_fallbacks(device: str) -> list[str]:
    return neuron_env.fallback_ops() if device == "neuron" else []


def banner(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}", flush=True)
