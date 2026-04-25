#!/usr/bin/env python3
"""
rosetta_precision_at_k.py

Computes two focused precision metrics on a pair_scores.tsv file:
  1. Precision at top-k  (k = 50, 100, 200, 500, 1000)
  2. Precision among non-zero-scoring pairs only

Usage:
    python scripts/rosetta_precision_at_k.py \
        --pair-scores benchmark_results_phase3_tm0.0/pair_scores.tsv
"""

import argparse
import csv
import random


def load_rows(path):
    rows = []
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append((float(row["score"]), row["label"] == "positive"))
    return rows


def precision_at_k(rows_sorted, k):
    top = rows_sorted[:k]
    return sum(1 for _, is_pos in top if is_pos) / k


def run(args):
    rows = load_rows(args.pair_scores)
    n_pos = sum(1 for _, is_pos in rows if is_pos)
    n_neg = sum(1 for _, is_pos in rows if not is_pos)
    total = len(rows)
    prior = n_pos / total  # random baseline for raw precision

    # Shuffle first so ties are broken randomly (matches AUCPR evaluation)
    rng = random.Random(42)
    rng.shuffle(rows)
    rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)

    print("=" * 60)
    print("Pair Scores Summary")
    print("=" * 60)
    print(f"  Total pairs evaluated : {total:,}")
    print(f"  Positives             : {n_pos:,}")
    print(f"  Negatives             : {n_neg:,}")
    print(f"  Prior (random P@k)    : {prior:.4f}  ({prior*100:.2f}%)")
    print()

    # ── Metric 1: Precision at top-k ────────────────────────────────────
    print("Precision at top-k (raw, unweighted)")
    print("-" * 60)
    print(f"{'k':>8}  {'TP in top-k':>12}  {'Precision':>10}  {'vs random':>10}")
    print(f"{'':>8}  {'':>12}  {'':>10}  {'(lift)':>10}")
    ks = [k for k in [50, 100, 200, 500, 1000] if k <= total]
    for k in ks:
        tp = sum(1 for _, is_pos in rows_sorted[:k] if is_pos)
        prec = tp / k
        lift = prec / prior if prior > 0 else float("inf")
        print(f"{k:>8,}  {tp:>12,}  {prec:>10.4f}  {lift:>10.2f}×")
    print()

    # ── Metric 2: Precision among non-zero-scoring pairs ────────────────
    nonzero = [(s, is_pos) for s, is_pos in rows if s > 0.0]
    nz_total = len(nonzero)
    nz_pos = sum(1 for _, is_pos in nonzero if is_pos)
    nz_neg = nz_total - nz_pos
    nz_prec = nz_pos / nz_total if nz_total > 0 else 0.0
    nz_lift = nz_prec / prior if prior > 0 else float("inf")

    zero = [(s, is_pos) for s, is_pos in rows if s == 0.0]
    zero_pos = sum(1 for _, is_pos in zero if is_pos)
    zero_neg = len(zero) - zero_pos

    print("Precision among non-zero-scoring pairs")
    print("-" * 60)
    print(f"  Non-zero pairs        : {nz_total:,}  ({nz_total/total*100:.1f}% of all evaluated)")
    print(f"    Positives (score>0) : {nz_pos:,}  ({nz_pos/n_pos*100:.1f}% of all positives)")
    print(f"    Negatives (score>0) : {nz_neg:,}  ({nz_neg/n_neg*100:.1f}% of all negatives)")
    print(f"  Precision (score>0)   : {nz_prec:.4f}  ({nz_prec*100:.2f}%)")
    print(f"  vs random baseline    : {nz_lift:.2f}×")
    print()
    print(f"  Zero-scoring pairs    : {len(zero):,}")
    print(f"    Positives (score=0) : {zero_pos:,}  (structurally unreachable)")
    print(f"    Negatives (score=0) : {zero_neg:,}")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-scores", required=True, dest="pair_scores",
                   help="pair_scores.tsv produced by benchmark_rosetta_stone.py")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
