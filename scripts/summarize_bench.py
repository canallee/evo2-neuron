#!/usr/bin/env python
"""
Turn per-rank diagnostic JSON into a scaling table.

    python scripts/summarize_bench.py runs/bench_w1 runs/bench_w4 runs/bench_w64

Reports, per world size, the **median warm** step (cold compile excluded and
reported separately), the slowest rank, throughput, and scaling efficiency
against the 1-rank baseline.

Two reporting rules, both learned expensively:

* Cold compile and warm steady state are different numbers. Conflating them is
  how a 0.09 s step gets published as a 92 s one.
* At N ranks the step is set by the **slowest** rank, not the median one, because
  every rank waits at the allreduce. Both are printed; if they diverge, you have
  a straggler, not a throughput number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load(directory: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("rank*.json"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    args = parser.parse_args(argv)

    rows = []

    for directory in args.dirs:
        ranks = load(directory)

        if not ranks:
            print(f"{directory}: no rank*.json", file=sys.stderr)
            continue

        world = ranks[0]["world_size"]
        medians = [r["median_warm_step_s"] for r in ranks]
        step = max(medians)                    # the allreduce waits for the slowest
        tokens = ranks[0]["seq_len"] * ranks[0]["batch_size"] * world

        rows.append({
            "run": directory.name,
            "world_size": world,
            "seq_len": ranks[0]["seq_len"],
            "dtype": ranks[0]["dtype"],
            "cold_step_s": max(r["cold_step_s"] for r in ranks),
            "median_rank_s": statistics.median(medians),
            "slowest_rank_s": step,
            "spread_pct": 100 * (max(medians) - min(medians)) / min(medians),
            "samples_per_s": world / step,
            "tokens_per_s": tokens / step,
            "fallback_ops": sorted({op for r in ranks for op in r["fallback_ops"]}),
        })

    rows.sort(key=lambda r: r["world_size"])
    baseline = next((r for r in rows if r["world_size"] == 1), rows[0] if rows else None)

    print(f"{'ranks':>6} {'cold s':>9} {'step s':>9} {'slowest':>9} {'spread':>8} "
          f"{'tok/s':>12} {'speedup':>9} {'eff':>7}")

    for row in rows:
        # Data parallelism scales THROUGHPUT, not step time: each rank still does
        # one full step, so a perfect 64-rank run has the same seconds-per-step as
        # one rank and 64x the samples. Dividing step times would report a perfect
        # run as no speedup at all.
        speedup = row["tokens_per_s"] / baseline["tokens_per_s"] if baseline else float("nan")
        scale = row["world_size"] / (baseline["world_size"] if baseline else 1)
        row["speedup"] = speedup
        row["efficiency"] = speedup / scale

        print(f"{row['world_size']:>6} {row['cold_step_s']:>9.1f} "
              f"{row['median_rank_s']:>9.4f} {row['slowest_rank_s']:>9.4f} "
              f"{row['spread_pct']:>7.1f}% {row['tokens_per_s']:>12,.0f} "
              f"{speedup:>8.2f}x {100 * row['efficiency']:>6.1f}%")

    stragglers = [r for r in rows if r["spread_pct"] > 10]

    if stragglers:
        print("\nrank spread above 10% on: "
              + ", ".join(f"{r['world_size']} ranks ({r['spread_pct']:.0f}%)" for r in stragglers))
        print("A bimodal or wide spread is a contention signature. Check for an "
              "orphaned process holding a core before quoting these numbers.")

    fallbacks = sorted({op for r in rows for op in r["fallback_ops"]})
    print(f"\nfallback_ops across all runs: {fallbacks or 'none'}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
