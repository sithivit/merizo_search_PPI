#!/usr/bin/env python3
"""
Precision at top-k and among non-zero-scoring pairs for a pair_scores.tsv file.

Usage:
    python scripts/rosetta_precision_at_k.py \\
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


def load_rows_with_proteins(path):
    rows = []
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append((row["protein_a"], row["protein_b"],
                         float(row["score"]), row["label"] == "positive"))
    return rows


def per_query_hits_at_1(rows_with_proteins):
    from collections import defaultdict
    protein_pairs = defaultdict(list)
    for pa, pb, score, is_pos in rows_with_proteins:
        protein_pairs[pa].append((score, is_pos))
        protein_pairs[pb].append((score, is_pos))

    rng = random.Random(42)
    hit = 0
    miss = 0
    miss_unreachable = 0  # miss because all positives score 0
    miss_outranked = 0    # miss because a negative genuinely scored higher
    no_positive = 0

    for protein, pairs in sorted(protein_pairs.items()):
        has_positive = any(is_pos for _, is_pos in pairs)
        if not has_positive:
            no_positive += 1
            continue
        rng.shuffle(pairs)
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_score, top_is_pos = pairs_sorted[0]
        if top_is_pos:
            hit += 1
        else:
            # Distinguish: can the method see any positive pair at all?
            best_positive_score = max(s for s, is_pos in pairs if is_pos)
            if best_positive_score == 0.0:
                miss_unreachable += 1  # positive is structurally invisible
            else:
                miss_outranked += 1    # positive exists but a negative ranked higher
            miss += 1

    return hit, miss, miss_unreachable, miss_outranked, no_positive


def run(args):
    rows_full = load_rows_with_proteins(args.pair_scores)
    rows = [(s, is_pos) for _, _, s, is_pos in rows_full]

    n_pos = sum(1 for _, is_pos in rows if is_pos)
    n_neg = sum(1 for _, is_pos in rows if not is_pos)
    total = len(rows)
    prior = n_pos / total

    # Shuffle first so ties are broken randomly (same as AUCPR evaluation)
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

    print("Global Precision at top-k (raw, unweighted)")
    print("-" * 60)
    print(f"{'k':>8}  {'TP in top-k':>12}  {'Precision':>10}  {'vs random':>10}")
    print(f"{'':>8}  {'':>12}  {'':>10}  {'(lift)':>10}")
    ks = [k for k in [1, 50, 100, 200, 500, 1000] if k <= total]
    for k in ks:
        tp = sum(1 for _, is_pos in rows_sorted[:k] if is_pos)
        prec = tp / k
        lift = prec / prior if prior > 0 else float("inf")
        print(f"{k:>8,}  {tp:>12,}  {prec:>10.4f}  {lift:>10.2f}×")
    print()

    hit, miss, miss_unreachable, miss_outranked, no_positive = per_query_hits_at_1(rows_full)
    n_queries = hit + miss
    hits_at_1 = hit / n_queries if n_queries > 0 else 0.0
    # Restricted: exclude proteins whose positive partners are all structurally unreachable
    n_reachable = hit + miss_outranked
    hits_at_1_reachable = hit / n_reachable if n_reachable > 0 else 0.0

    print("Per-query Hits@1 (is each protein's top-scored partner its true positive?)")
    print("-" * 60)
    print(f"  Query proteins with ≥1 positive pair : {n_queries:,}")
    print(f"  Proteins with no positive pair        : {no_positive:,}  (excluded)")
    print(f"  Hits (top-1 = true positive)          : {hit:,}")
    print(f"  Misses                                : {miss:,}")
    print(f"    — positive unreachable (score=0)    : {miss_unreachable:,}  (method cannot help)")
    print(f"    — positive outranked by negative    : {miss_outranked:,}  (genuine errors)")
    print(f"  Hits@1 (all queries)                  : {hits_at_1:.4f}  ({hits_at_1*100:.2f}%)")
    print(f"  Hits@1 (reachable queries only)       : {hits_at_1_reachable:.4f}  ({hits_at_1_reachable*100:.2f}%)")
    print(f"  — reachable = positive partner has structural evidence (score > 0)")
    print()

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
