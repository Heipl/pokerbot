#!/usr/bin/env python3
"""Merge parallel MCCFR shards into one policy.

External-sampling MCCFR accumulates additive regret and strategy sums, so
summing shards is equivalent to one longer run -- but only if every shard used
the same abstraction. Shards must therefore agree exactly on score_ranges (the
quantile cut points that decide what an info key means) and on the bucket
configs; train them with a shared --range-seed and a per-shard --seed. Mismatches
are refused rather than merged into a table that looks fine and is meaningless.

  python tools/merge_mccfr.py --out mccfr_bot/mccfr_ckpt.pkl shard_*.pkl
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys


BUCKET_FIELDS = ("hand_buckets", "board_buckets", "pot_buckets",
                 "spr_buckets", "pressure_buckets")


def _canon(obj):
    """Comparable form for nested bucket configs (dicts with int/str keys)."""
    if isinstance(obj, dict):
        return tuple(sorted((str(k), _canon(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_canon(v) for v in obj)
    if isinstance(obj, float):
        return round(obj, 9)
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", help="shard .pkl files (globs allowed)")
    ap.add_argument("--out", required=True, help="merged output path")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="merge even if abstractions differ (produces a "
                         "meaningless table; for debugging only)")
    args = ap.parse_args()

    paths = []
    for pat in args.shards:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else ([pat] if os.path.exists(pat) else []))
    paths = [p for p in dict.fromkeys(paths)]
    if not paths:
        print("ERROR: no shard files matched.", file=sys.stderr)
        return 1

    merged_regrets: dict = {}
    merged_strategy: dict = {}
    total_traversals = 0
    ref = None
    ref_path = None
    used = 0

    for p in paths:
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception as e:
            print(f"  SKIP {os.path.basename(p)}: unreadable ({e})")
            continue

        if ref is None:
            ref, ref_path = d, p
        else:
            bad = []
            if _canon(d.get("score_ranges")) != _canon(ref.get("score_ranges")):
                bad.append("score_ranges")
            for fld in BUCKET_FIELDS:
                if _canon(d.get(fld)) != _canon(ref.get(fld)):
                    bad.append(fld)
            if bad:
                msg = (f"  {os.path.basename(p)} disagrees with "
                       f"{os.path.basename(ref_path)} on: {', '.join(bad)}")
                if not args.allow_mismatch:
                    print(msg, file=sys.stderr)
                    print("ERROR: shards do not share one abstraction, so their "
                          "info keys mean different things and summing them is "
                          "meaningless.\n       Retrain the shards with the same "
                          "--range-seed (and identical bucket flags).",
                          file=sys.stderr)
                    return 2
                print(msg + "   [merging anyway: --allow-mismatch]")

        for key, arr in (d.get("regrets") or {}).items():
            cur = merged_regrets.get(key)
            merged_regrets[key] = arr.copy() if cur is None else cur + arr
        for key, arr in (d.get("strategy_sum") or {}).items():
            cur = merged_strategy.get(key)
            merged_strategy[key] = arr.copy() if cur is None else cur + arr

        t = int(d.get("traversals", 0))
        total_traversals += t
        used += 1
        print(f"  + {os.path.basename(p):<28} traversals={t:>12,} "
              f"infosets={len(d.get('strategy_sum') or {}):>9,}")

    if used == 0:
        print("ERROR: no readable shards.", file=sys.stderr)
        return 1

    out = dict(ref)
    out["regrets"] = merged_regrets
    out["strategy_sum"] = merged_strategy
    out["traversals"] = total_traversals

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    n = len(merged_strategy)
    print(f"\nmerged {used} shard(s) -> {args.out}")
    print(f"  traversals {total_traversals:,}   infosets {n:,}   "
          f"traversals/infoset {total_traversals / max(1, n):.1f}")

    vis = sorted(float(v.sum()) for v in merged_strategy.values())
    if vis:
        med = vis[len(vis) // 2]
        thin = sum(1 for v in vis if v < 100)
        print(f"  median visits/infoset {med:,.1f}   "
              f"{100 * thin / len(vis):.1f}% of infosets seen <100 times")

        # Median visits scale linearly with traversals (~8.5e-6 per traversal,
        # measured 200k to 4.45M), so the target is just arithmetic.
        if total_traversals > 0:
            rate = med / total_traversals
            if rate > 0:
                need = 1000.0 / rate
                print(f"  scaling: {rate:.2e} median visits per traversal "
                      f"-> ~{need:,.0f} traversals for median 1000")
        if med < 100:
            print("  NOTE: still undertrained -- regret matching wants thousands of "
                  "visits per infoset.")
            print("        More traversals is usually the right lever: coarsening "
                  "buys at most ~4x\n        (hand-buckets 2.1x, board-buckets "
                  "1.7x) and costs strategic resolution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
