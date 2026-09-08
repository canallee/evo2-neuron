"""
Artifact comparison: does this run reproduce a saved reference?

Reads two bundles written by `scripts/dump_artifacts.py` and prints a per-tensor
diff with a verdict. The metrics are the interesting part, and each exists
because a simpler one produced a wrong answer:

* **Magnitude-aware disagreement.** A tensor is a REAL disagreement only if it
  fails BOTH direction (`cosine < 0.99`) and magnitude
  (`max_abs / global_scale > 1e-3`). Per-tensor relative error alone is useless
  on LoRA: `lora_B` is zero-initialised, so its gradient is genuinely tiny and
  dividing a small absolute difference by a small norm manufactures alarming
  ratios out of floating-point noise.

* **The residual stream is gated on the increment**, `delta_k = block_out.k -
  block_out.{k-1}`, not on `block_out.k`. Block 3 of evo2-7b contributes 1.3% of
  `block_out.3` by norm, so a completely broken attention block reads ~0.013
  cumulatively — indistinguishable from bf16 noise — and O(1) on the increment.

* **Per-block tolerances are derived from the reference**, not hand-set:
  `tol_k = FLOOR * amp_k` where `amp_k = ||ref delta_k|| / ||ref block_out.k||`.
  Blocks that are cancellation points genuinely need a looser cumulative
  tolerance; nothing to maintain, and it retargets if the reference is re-minted.

Every function here is pure I/O + arithmetic — no model, no device.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = ["run_compare", "rel_err", "real_disagreement", "COS_GATE", "ABS_GATE"]

LORA_PREFIX = "lora."
HEAD_PREFIX = "head."

# The magnitude-aware verdict: a tensor is a REAL disagreement only if it fails
# BOTH direction and magnitude. See `_real_disagreement`.
_COS_GATE = 0.99
_ABS_GATE = 1e-3

# Default per-block-increment tolerance for `_compare_stream`, in units of
# relative error on `block_out.k - block_out.{k-1}`.
#
# Set by the reference's dtype, not by taste. Every element of all 22 tensors in
# the seq-200 CUDA bundle's `blocks0_10.pt` is exactly representable in bfloat16
# (verified: a bf16 round-trip is the identity on all of them), i.e. the CUDA
# runtime materialised the residual stream in bf16 at every module boundary
# regardless of what `--dtype` was passed. bf16 has 8 mantissa bits, so the
# quantisation floor is ~1.1e-3 relative; 3e-3 is that with headroom, and a
# SINGLE block does sit there (measured 1.24e-03 on `mlp_l3.0`).
#
# But `relerr(delta_k)` is not a pure per-block quantity: block k also inherits
# whatever error is already in its input. Measured across evo2-7b blocks 0..10,
# CPU bf16 vs the CUDA reference, the increments run 3.8e-03 .. 4.2e-02 with
# every increment cosine agreeing to 1e-05 .. 7e-04. That is accumulation through
# two ~100x norm discontinuities (blocks 2 and 9) and two cancellation points
# (blocks 0 and 4), not a defect -- so magnitude alone cannot be the verdict.
_STREAM_FLOOR = 3e-3

# Direction gate for the stream, and the half of the verdict that actually
# discriminates. Calibrated against a positive and a negative control, both
# measured:
#   * clean CPU-bf16 vs reference, all 11 blocks: 1-cos(delta) = 7.4e-06 .. 7.0e-04
#   * a deliberately scrambled block 3:           1-cos(delta) = 9.999e-01
# 1e-2 sits ~14x above the worst clean value and ~100x below the broken one.
#
# The upper control is worth stating too: merely swapping fused SDPA for an
# explicit-softmax attention -- provably the same function, differing only in
# reduction order and bf16 rounding -- moves delta_3 by 4.9e-03 and delta_10 by
# 9.3e-03. The port-vs-reference disagreement at those same blocks is 1.5e-02 and
# 1.1e-02, i.e. within 3.1x and 1.2x of a change that is known to be
# arithmetically free. Magnitude at that scale carries no information; direction
# does.
_STREAM_COS_FLOOR = 1e-2

# The residual-stream dump. The filename is fixed by the shipped CUDA bundles and
# is deliberately NOT derived from the block count -- renaming it per run would
# make `--compare` stop finding it.
_STREAM_FILE = "blocks0_10.pt"

COS_GATE = _COS_GATE
ABS_GATE = _ABS_GATE


def _group_of(name: str) -> str:
    """Bucket a tensor name as `lora` / `head` / `other`.

    Reported separately because a read-out head can carry far more parameters
    than the adapters do -- `ConvReadoutHead` at width 200 on a 4096-dim stream is
    52 M against the adapters' 4 M -- so a head-only regression otherwise hides
    inside the aggregate norm, and a LoRA-only regression is invisible next to
    the head's magnitude.
    """

    if name.endswith((".lora_A", ".lora_B")):
        return "lora"

    if name.startswith(HEAD_PREFIX):
        return "head"

    # Residual-stream activations (`blocks0_10.pt`). Normally read by
    # `_compare_stream`, which orders them by block; this branch only keeps the
    # grouping sane if they are ever passed through `_compare_file` instead. No
    # LoRA or head tensor starts with these prefixes, so the weight artifacts are
    # unaffected.
    if name.startswith(("block_out.", "mlp_l3.")):
        return "stream"

    return "other"

def _rel_err(reference: Tensor, test: Tensor) -> float:
    """`||test - ref|| / (||ref|| + eps)`."""

    reference = reference.to(torch.float64)
    test = test.to(torch.float64)

    return float(
        torch.linalg.vector_norm(test - reference)
        / (torch.linalg.vector_norm(reference) + 1e-12)
    )

def _compare_file(ref_dir: Path, test_dir: Path, name: str) -> int:
    """Print a per-tensor rel-err table for one artifact. Returns a failure count."""

    ref_path, test_path = ref_dir / name, test_dir / name

    if not ref_path.exists() or not test_path.exists():
        print(f"\n== {name}: SKIP (missing in {'ref' if not ref_path.exists() else 'test'})")
        return 0

    reference = torch.load(ref_path, map_location="cpu", weights_only=True)
    test = torch.load(test_path, map_location="cpu", weights_only=True)

    only_ref = sorted(set(reference) - set(test))
    only_test = sorted(set(test) - set(reference))
    shared = sorted(set(reference) & set(test))

    print(f"\n== {name}: {len(shared)} shared tensors")

    failures = len(only_ref) + len(only_test)

    if only_ref:
        print(f"   MISSING in test ({len(only_ref)}): {only_ref[:5]}")
    if only_test:
        print(f"   EXTRA in test ({len(only_test)}): {only_test[:5]}")

    # Scale every tensor's absolute disagreement against the LARGEST value
    # anywhere in this artifact, not against its own norm. See `_real_disagreements`.
    global_scale = max(
        (float(reference[key].abs().max()) for key in shared if reference[key].numel()),
        default=0.0,
    )

    rows = []
    for key in shared:
        if reference[key].shape != test[key].shape:
            print(f"   SHAPE MISMATCH {key}: {list(reference[key].shape)} vs {list(test[key].shape)}")
            failures += 1
            continue

        rows.append((_rel_err(reference[key], test[key]), key))

    rows.sort(reverse=True)

    for label in ("lora", "head", "stream", "other"):
        group_rows = [r for r in rows if _group_of(r[1]) == label]

        if group_rows:
            worst = group_rows[0]
            print(
                f"   {label:5s}: n={len(group_rows):3d}  max={worst[0]:.3e} ({worst[1]})  "
                f"median={np.median([r[0] for r in group_rows]):.3e}"
            )

    print("   worst 10:")
    for value, key in rows[:10]:
        print(f"     {value:.6e}  {key}")

    _report_real_disagreements(reference, test, shared, global_scale)

    return failures

def _compare_stream(ref_dir: Path, test_dir: Path, gate: float | None) -> int:
    """Per-block table for `blocks0_10.pt`. Returns a failure count.

    Deliberately NOT folded into `_compare_file`. Three things differ, and all
    three matter:

    1. **The primary metric is the increment, not the tensor.** `block_out.k` is
       cumulative, so a block's own contribution can be arbitrarily diluted by
       the residual it rides on. Measured on the evo2-7b reference, block 3 (the
       first attention block) contributes 1.3% of `block_out.3` by norm -- a
       *completely broken* attention block would show up as rel_err ~0.013 on the
       cumulative stream, barely above the bf16 floor, and would be dismissed.
       Against `delta_3 = block_out.3 - block_out.2` the same break is O(1).

    2. **Scale is per tensor, not per file.** `_compare_file` takes one
       `global_scale` across the whole artifact. On this one the `block_out`
       norms span 2.19 -> 3647 (1670x) because the reference has two genuine
       discontinuities (blocks 2 and 9). A single file-wide scale would render a
       total failure in block 0 as ~1e-3 and pass it.

    3. **Order is the signal.** An 11-element sequence wants a table in block
       order and a first-divergence line, not the global "worst 10" list.

    Tolerances are derived from the reference itself rather than hand-set:
    `tol_k = FLOOR * amp_k`, where `amp_k = ||ref delta_k|| / ||ref block_out.k||`
    is how much block k's own contribution is amplified relative to the stream it
    lands in. Blocks 0 and 4 are cancellation points (amp 4.07 and 3.52) and
    genuinely need a looser cumulative tolerance; blocks 2 and 9 do not. Nothing
    to maintain, and it retargets automatically if the reference is re-minted.
    """

    ref_path, test_path = ref_dir / _STREAM_FILE, test_dir / _STREAM_FILE

    if not (ref_path.exists() and test_path.exists()):
        print(f"\n== {_STREAM_FILE}: SKIP (missing on one side)")
        return 0

    reference = torch.load(ref_path, map_location="cpu", weights_only=False)
    test = torch.load(test_path, map_location="cpu", weights_only=False)

    failures = 0
    missing = sorted(set(reference) - set(test))
    extra = sorted(set(test) - set(reference))

    print(f"\n== {_STREAM_FILE}: {len(reference)} vs {len(test)} tensors")

    for key in missing:
        print(f"   MISSING in test: {key}")

    for key in extra:
        print(f"   EXTRA in test:   {key}")

    failures += len(missing) + len(extra)

    indices = sorted(
        int(key.split(".")[1]) for key in set(reference) & set(test) if key.startswith("block_out.")
    )

    if not indices:
        print("   no block_out.* tensors in common -- nothing to gate")
        return failures

    floor = gate if gate is not None else _STREAM_FLOOR

    print(
        f"   FLOOR={floor:.1e}; tol_k = FLOOR * amp_k (amp from the reference). "
        f"'delta' = block_out.k - block_out.k-1, and is the metric that matters -- "
        f"it isolates block k's own contribution from the residual it rides on. "
        f"Gated on relerr(delta) AND 1-cos(delta)>{_STREAM_COS_FLOOR:.0e}; the first "
        f"block gates on relerr(l3) instead (its increment is not isolable here)."
    )
    print(
        f"\n   {'k':>3}  {'relerr(out)':>11} {'growth':>7}  {'relerr(delta)':>13} "
        f"{'1-cos(delta)':>12} {'tol':>9}  {'relerr(l3)':>10}  {'maxabs/scale':>12}"
    )

    first_divergence: int | None = None
    previous_out_err: float | None = None

    for index in indices:
        out_key = f"block_out.{index}"
        ref_out, test_out = reference[out_key], test[out_key]

        if ref_out.shape != test_out.shape:
            print(f"   {index:>3}  SHAPE {list(ref_out.shape)} vs {list(test_out.shape)}")
            failures += 1
            continue

        out_err = _rel_err(ref_out, test_out)

        # delta_0 is the block's output against the embedding, which is not in
        # this file -- so for k=0 the cumulative tensor IS the increment plus a
        # constant, and there is nothing to subtract. Report it as-is.
        previous_key = f"block_out.{index - 1}"
        has_previous = previous_key in reference and previous_key in test

        if has_previous:
            ref_delta = ref_out.to(torch.float64) - reference[previous_key].to(torch.float64)
            test_delta = test_out.to(torch.float64) - test[previous_key].to(torch.float64)
        else:
            ref_delta, test_delta = ref_out.to(torch.float64), test_out.to(torch.float64)

        delta_err = _rel_err(ref_delta, test_delta)
        _, delta_cos, _ = _real_disagreement(
            ref_delta, test_delta, 1.0, cos_gate=0.0, abs_gate=float("inf")
        )

        # `amp_k` needs the previous block's output. For the first block that is
        # the token embedding, which this file does not carry -- so `delta_0` is
        # really `block_out.0` and its amp is 1.0 by construction, not measured.
        # Print n/a rather than a derived-looking number. (On the evo2-7b
        # reference the true amp_0 is 4.07: block 0 very nearly cancels the
        # embedding. `relerr(l3)` is the undiluted probe for that block.)
        ref_out_norm = float(torch.linalg.vector_norm(ref_out.to(torch.float64)))
        amp = float(torch.linalg.vector_norm(ref_delta)) / (ref_out_norm + 1e-12)
        tolerance = f"{floor * max(amp, 1.0):9.1e}" if has_previous else "      n/a"

        growth = out_err / previous_out_err if previous_out_err else float("nan")
        previous_out_err = out_err

        l3_key = f"mlp_l3.{index}"
        l3_err = (
            _rel_err(reference[l3_key], test[l3_key])
            if l3_key in reference and l3_key in test
            else float("nan")
        )

        # Magnitude-aware verdict against THIS tensor's own scale.
        _, _, relative_abs = _real_disagreement(
            ref_out, test_out, float(ref_out.abs().max()),
            cos_gate=_COS_GATE, abs_gate=_ABS_GATE,
        )

        # Gate the ISOLATED quantity for this block. For k>0 that is the
        # increment. For the first block it cannot be isolated from this file at
        # all -- `delta_0` is really `block_out.0`, which on the evo2-7b
        # reference sits at a 4.07x cancellation against the embedding and so
        # reads ~4x the floor even when the block is correct (measured: 3.85e-03
        # cumulative vs 1.24e-03 on `mlp_l3.0`, which is the same computation
        # undiluted). Gating the cumulative tensor there is a guaranteed false
        # alarm, so gate `mlp_l3.k` instead; if that is absent, do not gate.
        gated = delta_err if has_previous else l3_err
        # BOTH magnitude and direction, for the same reason `_real_disagreement`
        # requires both: at this scale magnitude alone flags arithmetic that is
        # provably equivalent (see `_STREAM_COS_FLOOR`), so it produces false
        # alarms on every honest run and trains the reader to ignore the gate.
        diverged = (
            gated == gated                              # NaN-safe
            and gated > floor
            and (1.0 - delta_cos) > _STREAM_COS_FLOOR
        )

        if diverged and first_divergence is None:
            first_divergence = index

        print(
            f"   {index:>3}  {out_err:11.3e} {growth:7.2f}  {delta_err:13.3e} "
            f"{1.0 - delta_cos:12.3e} {tolerance}  {l3_err:10.3e}  {relative_abs:12.3e}"
            f"{'  <-- FIRST DIVERGENCE' if index == first_divergence else ''}"
        )

    if first_divergence is None:
        print(
            f"   no block diverges: every increment is either within FLOOR={floor:.1e} "
            f"or agrees in direction to better than {_STREAM_COS_FLOOR:.0e}"
        )
    else:
        print(
            f"   FIRST DIVERGENCE at block {first_divergence}: bisect inside it "
            f"(pre_norm -> projections -> filter -> out_filter_dense -> mlp). "
            f"Everything before it agrees, so nothing downstream is evidence."
        )

        if gate is not None:
            failures += 1

    return failures

def _real_disagreement(
    reference: Tensor, test: Tensor, global_scale: float, *, cos_gate: float, abs_gate: float
) -> tuple[bool, float, float]:
    """Magnitude-aware verdict for one tensor: `(is_real, cosine, max_abs/scale)`.

    A tensor counts as a REAL disagreement only if it fails BOTH tests:
    direction (`cosine < cos_gate`) and magnitude (`max_abs / global_scale >
    abs_gate`).

    Why not per-tensor relative error alone: `lora_B` is zero-initialised, so its
    gradient is genuinely tiny, and dividing a small absolute difference by a
    small norm manufactures an alarming ratio out of floating-point noise. On the
    Neuron-vs-CPU dump the `lora_B` grads show rel_err up to 2e-2 while the head
    -- which carries real magnitude -- sits at 3e-6. Only one of those is worth
    looking at, and it is not the 2e-2.

    The reason this gate exists is a measured false alarm on a different port:
    all 24 of its "cosine failures" were bias gradients that are analytically
    zero. Direction is meaningless for a tensor with no magnitude.
    """

    ref64 = reference.to(torch.float64).flatten()
    test64 = test.to(torch.float64).flatten()

    ref_norm = float(torch.linalg.vector_norm(ref64))
    test_norm = float(torch.linalg.vector_norm(test64))

    if ref_norm == 0.0 and test_norm == 0.0:
        return False, 1.0, 0.0

    cosine = (
        float(torch.dot(ref64, test64) / (ref_norm * test_norm))
        if ref_norm > 0.0 and test_norm > 0.0
        else 0.0
    )
    relative_abs = float((test64 - ref64).abs().max()) / (global_scale or 1.0)

    return (cosine < cos_gate and relative_abs > abs_gate), cosine, relative_abs

def _report_real_disagreements(
    reference: dict[str, Tensor],
    test: dict[str, Tensor],
    shared: list[str],
    global_scale: float,
    *,
    cos_gate: float = _COS_GATE,
    abs_gate: float = _ABS_GATE,
) -> None:
    """Print the magnitude-aware summary alongside the raw rel-err table."""

    verdicts = [
        (key, *_real_disagreement(reference[key], test[key], global_scale,
                                  cos_gate=cos_gate, abs_gate=abs_gate))
        for key in shared
        if reference[key].shape == test[key].shape
    ]

    real = [v for v in verdicts if v[1]]

    print(
        f"   magnitude-aware (cos>={cos_gate} OR max_abs/scale<={abs_gate:g}; "
        f"scale={global_scale:.3e}): {len(real)} real disagreement(s) of {len(verdicts)}"
    )

    for key, _, cosine, relative_abs in sorted(real, key=lambda v: v[2])[:10]:
        print(f"     cos={cosine:+.6f}  max_abs/scale={relative_abs:.3e}  {key}")

def _compare_manifests(ref_dir: Path, test_dir: Path) -> None:
    """Surface config/provenance differences before any tensor is compared.

    Most "why don't these match" questions are answered here: a different seed,
    a different grad_accum, or a reference minted on another GPU / torch build.
    Purely informational -- differing provenance is expected between a CUDA
    reference and a Trainium run; differing *config* almost never is.
    """

    ref_path, test_path = ref_dir / "manifest.json", test_dir / "manifest.json"

    if not (ref_path.exists() and test_path.exists()):
        print("\n== manifest.json: SKIP (missing on one side)")
        return

    reference = json.loads(ref_path.read_text())
    test = json.loads(test_path.read_text())

    for section in ("config", "provenance"):
        ref_section = reference.get(section, {})
        test_section = test.get(section, {})
        diffs = [
            (key, ref_section.get(key), test_section.get(key))
            for key in sorted(set(ref_section) | set(test_section))
            if ref_section.get(key) != test_section.get(key)
        ]

        print(f"\n== {section}: {len(diffs)} differing key(s)")

        for key, ref_value, test_value in diffs:
            marker = "  " if section == "provenance" else "! "
            print(f"   {marker}{key}: {ref_value!r} -> {test_value!r}")

def _compare_inputs(ref_dir: Path, test_dir: Path) -> int:
    """Check both sides saw the same DNA and the same tokens. Returns a failure count.

    Run before the weight/gradient diffs, because if the inputs differ nothing
    downstream means anything -- and an input mismatch is otherwise indistinguishable
    from a broken backward.
    """

    ref_path, test_path = ref_dir / "inputs.pt", test_dir / "inputs.pt"

    if not (ref_path.exists() and test_path.exists()):
        print("\n== inputs.pt: SKIP (missing on one side)")
        return 0

    reference = torch.load(ref_path, map_location="cpu", weights_only=False)
    test = torch.load(test_path, map_location="cpu", weights_only=False)

    ref_batches, test_batches = reference["batches"], test["batches"]
    shared = min(len(ref_batches), len(test_batches))
    print(f"\n== inputs.pt: {len(ref_batches)} vs {len(test_batches)} batches")

    failures = 0

    if reference.get("sequences") and test.get("sequences"):
        ref_seqs = [s for batch in reference["sequences"][:shared] for s in batch]
        test_seqs = [s for batch in test["sequences"][:shared] for s in batch]

        if ref_seqs == test_seqs:
            total_bp = sum(len(s) for s in ref_seqs)
            print(f"   DNA: identical ({len(ref_seqs)} sequences, {total_bp} bp)")
            print(f"        first 60 bp: {ref_seqs[0][:60]}")
        else:
            mismatched = sum(1 for a, b in zip(ref_seqs, test_seqs, strict=False) if a != b)
            mismatched += abs(len(ref_seqs) - len(test_seqs))
            print(f"   DNA: MISMATCH in {mismatched}/{len(ref_seqs)} sequences")
            failures += mismatched

    for index in range(shared):
        for key in sorted(set(ref_batches[index]) & set(test_batches[index])):
            ref_tensor, test_tensor = ref_batches[index][key], test_batches[index][key]

            if ref_tensor.shape != test_tensor.shape or not torch.equal(ref_tensor, test_tensor):
                print(f"   batch {index} tensor {key}: MISMATCH "
                      f"{list(ref_tensor.shape)} vs {list(test_tensor.shape)}")
                failures += 1

    if not failures:
        keys = sorted(ref_batches[0]) if ref_batches else []
        print(f"   tensors: all identical across {shared} batch(es) -- {keys}")

    return failures

def run_compare(ref_dir: Path, test_dir: Path, gate_stream: float | None = None) -> int:
    print(f"reference: {ref_dir}\ntest:      {test_dir}")

    _compare_manifests(ref_dir, test_dir)

    for name in ("losses.json",):
        ref_path, test_path = ref_dir / name, test_dir / name

        if ref_path.exists() and test_path.exists():
            reference = json.loads(ref_path.read_text())
            test = json.loads(test_path.read_text())
            print(f"\n== {name}\n   ref : {reference}\n   test: {test}")

    failures = _compare_inputs(ref_dir, test_dir)

    # Before the read-out and the weights: the read-out is a 0.42% correction to
    # this stream on evo2-7b, so if the stream is wrong, `hidden0`/`g1`/`wN` only
    # restate it with worse localisation.
    failures += _compare_stream(ref_dir, test_dir, gate_stream)

    for name in ("hidden0.pt", "w0.pt", "g1.pt", "wN.pt"):
        failures += _compare_file(ref_dir, test_dir, name)

    print(f"\nstructural mismatches: {failures}")

    return 1 if failures else 0


# Public aliases.
rel_err = _rel_err
real_disagreement = _real_disagreement
