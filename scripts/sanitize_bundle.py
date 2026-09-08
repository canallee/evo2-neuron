#!/usr/bin/env python
"""
Import a CUDA reference bundle, keeping only what is task-independent.

The shipped CUDA bundles were minted on an NVIDIA H200 by the original port
campaign. Two of their files depend on nothing but the backbone and the input:

  `blocks0_10.pt`  the residual stream, `block_out.k` + `mlp_l3.k`
  `hidden0.pt`     the mid-stack read-out

Both are captured on the **first** forward, where `lora_B` is still zero and the
adapters are therefore exactly the identity. They are pure backbone activations,
so they gate any port regardless of what head or objective is bolted on top —
which is what makes them worth shipping and what makes this script possible.

`w0` / `g1` / `wN` / `losses.json` are **not** imported: they are a function of
the head and objective that produced them, and reproducing them would require
that exact pairing. Regenerate those on CPU and Neuron with `dump_artifacts.py`
and whichever objective you actually train; the CPU↔Neuron axis is the one that
gates a backward, and unlike the CUDA axis it is reproducible by anyone with the
repo.

`inputs.pt` is rewritten to `input_ids` (+ `attention_mask`, + `window_mask` when
the source had a contiguous region marker). Everything else in the source batch
dict is task metadata and is dropped.

    python scripts/sanitize_bundle.py SOURCE_DIR artifacts/cuda_seq200 \
        --note "NVIDIA H200, seq 200"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

#: Batch tensors that survive. Anything else is task metadata.
KEEP = {"input_ids", "attention_mask"}
#: Renamed on the way through: a contiguous region-of-interest marker.
RENAME = {"central_mask": "window_mask"}
#: Manifest config keys that survive. Deliberately narrow: this is a
#: forward-only bundle, so anything describing a trainable set, an optimizer or
#: a task would be a claim the files cannot back.
CONFIG_KEEP = {
    "backbone", "device", "dtype", "batch_size", "seed", "strict_determinism",
}
#: Source manifest keys renamed to this repo's own spelling, so `--compare`
#: reports a genuine config difference rather than a vocabulary difference.
CONFIG_RENAME = {
    "embedding_layer": "readout",
    "embedding_dim": "hidden_size",
    "deterministic": "strict_determinism",
}
#: Provenance keys that survive. The versions are real and worth keeping — they
#: are what the CUDA numbers were produced by. A commit hash points at a repo the
#: reader does not have.
PROVENANCE_DROP = {"git_commit"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sanitize_inputs(source: Path, target: Path) -> dict:
    saved = torch.load(source, map_location="cpu", weights_only=False)
    batches = []

    for batch in saved["batches"]:
        kept = {}

        for key, value in batch.items():
            if not torch.is_tensor(value):
                continue

            name = RENAME.get(key, key)

            if name in KEEP or name == "window_mask":
                kept[name] = value.cpu()

        batches.append(kept)

    torch.save({"batches": batches}, target)

    return {
        "batches": len(batches),
        "tensors": sorted(batches[0]) if batches else [],
        "shape": list(batches[0]["input_ids"].shape) if batches else None,
        "dropped": sorted(
            set(saved["batches"][0]) - set(RENAME) - KEEP
        ) + (["sequences"] if saved.get("sequences") else []),
    }


def sanitize_manifest(source: Path, target: Path, note: str | None,
                      measured: dict) -> dict:
    original = json.loads(source.read_text())
    source_config = original.get("config", {})
    config = {
        CONFIG_RENAME.get(k, k): v
        for k, v in source_config.items()
        if k in CONFIG_KEEP or k in CONFIG_RENAME
    }
    config["forward_only"] = True
    config["seq_len"] = measured["seq_len"]
    config["blocks_built"] = measured["blocks"]
    config["readout_block"] = (
        int(str(config.get("readout", "")).split(".")[1])
        if str(config.get("readout", "")).startswith("blocks.") else None
    )

    provenance = {
        k: v for k, v in original.get("provenance", {}).items() if k not in PROVENANCE_DROP
    }

    manifest = {
        "config": config,
        "provenance": provenance,
        "hidden0": measured["hidden0"],
        "stream": measured["stream"],
        "note": note or "imported reference; forward-only",
        "contents": sorted(measured["contents"]),
    }
    target.write_text(json.dumps(manifest, indent=2))

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--note", default=None)
    parser.add_argument("--no-stream", action="store_true",
                        help="skip blocks0_10.pt (2.9 GB at seq 8192)")
    parser.add_argument("--manifest-only", action="store_true",
                        help="rewrite manifest.json and checksums in place; copy nothing")
    args = parser.parse_args(argv)

    source, target = args.source, args.target
    target.mkdir(parents=True, exist_ok=True)

    if not (source / "inputs.pt").exists():
        raise SystemExit(f"{source}: no inputs.pt — not a bundle")

    if args.manifest_only:
        hidden = torch.load(target / "hidden0.pt", map_location="cpu",
                            weights_only=False)["h"]
        stream_names = (sorted(torch.load(target / "blocks0_10.pt", map_location="cpu",
                                          weights_only=True))
                        if (target / "blocks0_10.pt").exists() else [])
        sanitize_manifest(
            source / "manifest.json", target / "manifest.json", args.note,
            {
                "seq_len": int(hidden.shape[1]),
                "blocks": len([n for n in stream_names
                               if n.startswith("block_out.")]) or None,
                "hidden0": {"shape": list(hidden.shape),
                            "norm": round(float(hidden.double().norm()), 6)},
                "stream": stream_names,
                "contents": sorted(p.name for p in target.iterdir()
                                   if p.name != "SHA256SUMS.json"),
            },
        )
        print(f"{target}/manifest.json rewritten in place")

        return 0

    summary = sanitize_inputs(source / "inputs.pt", target / "inputs.pt")
    print(f"inputs.pt   {summary['batches']} batches {summary['shape']} "
          f"keeping {summary['tensors']}")
    print(f"            dropped {summary['dropped']}")

    shutil.copy2(source / "hidden0.pt", target / "hidden0.pt")
    hidden = torch.load(target / "hidden0.pt", map_location="cpu", weights_only=False)["h"]
    print(f"hidden0.pt  copied — {list(hidden.shape)} norm {float(hidden.double().norm()):.6f}")

    contents = ["inputs.pt", "hidden0.pt", "manifest.json"]
    stream_names: list[str] = []

    if (source / "blocks0_10.pt").exists() and not args.no_stream:
        shutil.copy2(source / "blocks0_10.pt", target / "blocks0_10.pt")
        stream = torch.load(target / "blocks0_10.pt", map_location="cpu", weights_only=True)
        stream_names = sorted(stream)
        contents.append("blocks0_10.pt")
        print(f"blocks0_10.pt copied — {len(stream)} tensors")
    elif args.no_stream:
        print("blocks0_10.pt SKIPPED (--no-stream)")

    sanitize_manifest(
        source / "manifest.json", target / "manifest.json", args.note,
        {
            "seq_len": int(hidden.shape[1]),
            "blocks": len([n for n in stream_names if n.startswith("block_out.")]) or None,
            "hidden0": {"shape": list(hidden.shape),
                        "norm": round(float(hidden.double().norm()), 6)},
            "stream": stream_names,
            "contents": contents,
        },
    )
    print("manifest.json rewritten")

    checksums = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(target.iterdir()) if path.name != "SHA256SUMS.json"
    }
    (target / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2))

    total = sum(v["bytes"] for v in checksums.values())
    print(f"\n{target}: {len(checksums)} files, {total / 1e6:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
