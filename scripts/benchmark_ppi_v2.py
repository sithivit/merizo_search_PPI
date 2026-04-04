#!/usr/bin/env python3
"""
benchmark_ppi_v2.py

Structurally-aware PPI benchmark operating within the TED pair_list universe.

Design
------
  Universe : proteins with cached search results (all are in pair_list / pairlist_db)
  Positives: Zhang et al. high-confidence PPIs where BOTH proteins are in the universe
  Negatives: random pairs drawn from the same universe, excluding all known positives
  Coverage : 100% by construction — no zero-score dilution

No new structural searches are needed; all results come from cached TSV files.

Metrics
-------
  AUCPR   — weighted precision-recall AUC (wp=0.01, comparable to Zhang et al.)
  ROC-AUC — standard discriminative metric, easier to explain
  Precision @ Recall 0.1 / 0.2 / 0.5

Usage
-----
    python scripts/benchmark_ppi_v2.py
    python scripts/benchmark_ppi_v2.py --neg-ratio 10 --output-dir benchmark_results_v2
"""

import argparse
import csv
import logging
import os
import random
import re
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CONTROLS        = os.path.join(PROJECT_ROOT, "benchmark_cache", "benchmarks",
                                       "positives_and_negatives.tsv")
DEFAULT_SEARCH_CACHE    = os.path.join(PROJECT_ROOT, "benchmark_cache", "searches")
DEFAULT_OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "benchmark_results_v2")

_AF_RE      = re.compile(r'AF-([^-]+)-F1-model_v4')
_TAXID_RE   = re.compile(r'"taxid":\s*"(\d+)"')
COL_TARGET  = 2
COL_MAX_TM  = 10
COL_META    = 12


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_universe(search_cache_dir: str) -> set:
    """Universe = proteins that have a non-empty cached search TSV (failed searches excluded)."""
    proteins = set()
    n_empty = 0
    for fname in os.listdir(search_cache_dir):
        if not fname.endswith("_search.tsv"):
            continue
        path = os.path.join(search_cache_dir, fname)
        if os.path.getsize(path) == 0:
            n_empty += 1
            continue
        proteins.add(fname[: -len("_search.tsv")])
    if n_empty:
        log.info(f"Excluded {n_empty} failed (empty) search files from universe.")
    return proteins


def load_zhang_positives(controls_path: str) -> list:
    """Return list of (protein_a, protein_b) positive pairs from Zhang et al."""
    positives = []
    with open(controls_path) as fh:
        next(fh)  # header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, cat = line.split("\t")
            if cat == "positive":
                a, b = pair.split("_")
                positives.append((a, b))
    return positives


def parse_search_tsv(tsv_path: str, query_protein: str, taxid: str = None) -> dict:
    """Parse one cached search TSV. Returns {target_protein_id: max_tm}."""
    self_prefix = f"AF-{query_protein}-"
    best: dict = {}
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= COL_MAX_TM:
                continue
            target = parts[COL_TARGET]
            if self_prefix in target:
                continue
            m = _AF_RE.match(target)
            if not m:
                continue
            pid = m.group(1)
            try:
                tm = float(parts[COL_MAX_TM])
            except ValueError:
                continue
            if taxid and len(parts) > COL_META:
                tm_match = _TAXID_RE.search(parts[COL_META])
                if not tm_match or tm_match.group(1) != taxid:
                    continue
            if pid not in best or tm > best[pid]:
                best[pid] = tm
    return best


def load_search_results(proteins: set, search_cache_dir: str, taxid: str = None) -> dict:
    """Load cached TSVs for the given proteins. Returns {protein: {target: tm}}."""
    results = {}
    total = len(proteins)
    for i, pid in enumerate(sorted(proteins), 1):
        tsv = os.path.join(search_cache_dir, f"{pid}_search.tsv")
        results[pid] = parse_search_tsv(tsv, pid, taxid) if os.path.exists(tsv) else {}
        if i % 500 == 0:
            log.info(f"  Loaded {i:,}/{total:,} search result files...")
    return results


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def make_canonical(a: str, b: str):
    return (a, b) if a < b else (b, a)


def sample_negatives(universe: list, positives_canonical: set,
                     n: int, seed: int = 42) -> list:
    """Sample n random pairs from universe excluding known positives."""
    rng = random.Random(seed)
    neg_set: set = set()
    negatives: list = []
    max_attempts = n * 50
    attempts = 0
    while len(negatives) < n and attempts < max_attempts:
        a = rng.choice(universe)
        b = rng.choice(universe)
        attempts += 1
        if a == b:
            continue
        key = make_canonical(a, b)
        if key in positives_canonical or key in neg_set:
            continue
        neg_set.add(key)
        negatives.append((a, b))
    if len(negatives) < n:
        log.warning(f"Could only generate {len(negatives)}/{n} negatives "
                    f"after {max_attempts} attempts (universe may be too small).")
    return negatives


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_pair(a: str, b: str, search_results: dict) -> float:
    return max(
        search_results.get(a, {}).get(b, 0.0),
        search_results.get(b, {}).get(a, 0.0),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_weighted_pr(pos_scores: list, neg_scores: list,
                        wp: float = 0.01) -> tuple:
    """Weighted precision-recall curve (Zhang et al. methodology, wp weights positives)."""
    P = len(pos_scores)
    if P == 0:
        return [], [], 0.0
    all_pairs = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    random.shuffle(all_pairs)
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    precisions, recalls = [], []
    tp = fp = 0
    for score, label in all_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append((tp * wp) / (tp * wp + fp))
        recalls.append(tp / P)
    aucpr = sum(
        0.5 * (precisions[i] + precisions[i - 1]) * (recalls[i] - recalls[i - 1])
        for i in range(1, len(recalls))
    )
    return precisions, recalls, aucpr


def precision_at_recall(precisions: list, recalls: list, target: float) -> float:
    for p, r in zip(precisions, recalls):
        if r >= target:
            return p
    return 0.0


def compute_roc_auc(pos_scores: list, neg_scores: list) -> float:
    """ROC-AUC via Mann-Whitney U (exact, O(P*N))."""
    P, N = len(pos_scores), len(neg_scores)
    if P == 0 or N == 0:
        return 0.5
    u = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in pos_scores
        for n in neg_scores
    )
    return u / (P * N)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_pair_scores(positives, negatives, pos_scores, neg_scores, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["protein_a", "protein_b", "label", "score"])
        for (a, b), s in zip(positives, pos_scores):
            w.writerow([a, b, "positive", f"{s:.4f}"])
        for (a, b), s in zip(negatives, neg_scores):
            w.writerow([a, b, "negative", f"{s:.4f}"])


def write_pr_curve(precisions, recalls, path, max_points=1000):
    step = max(1, len(precisions) // max_points)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["recall", "precision"])
        for i in range(0, len(precisions), step):
            w.writerow([f"{recalls[i]:.5f}", f"{precisions[i]:.5f}"])


def write_summary(positives, negatives, pos_scores, neg_scores,
                  precs, recs, aucpr, roc_auc, args, path,
                  n_zhang_pos: int, universe_size: int):
    P = len(positives)
    N = len(negatives)
    n_pos_hit = sum(1 for s in pos_scores if s > 0)
    n_neg_hit = sum(1 for s in neg_scores if s > 0)
    baseline_aucpr = args.wp / (args.wp + 1)
    total_hits = n_pos_hit + n_neg_hit
    hit_precision = n_pos_hit / total_hits if total_hits > 0 else 0.0

    with open(path, "w") as fh:
        def w(line=""): fh.write(line + "\n")

        w("=" * 60)
        w("PPI Benchmark v2 — Structural Domain Homology Evaluation")
        w("=" * 60)
        w()
        w("--- Benchmark Design ---")
        w("  Universe : proteins with cached structural search results (TED pair_list)")
        w("  Positives: Zhang et al. high-confidence PPIs (both proteins in universe)")
        w("  Negatives: random pairs from universe, excluding all known Zhang et al. positives")
        w("  Coverage : 100% — no zero-score dilution from uncovered proteins")
        w()
        w("--- Dataset ---")
        w(f"  Universe size:              {universe_size:,} proteins")
        w(f"  Zhang et al. positives:     {n_zhang_pos:,} total")
        w(f"  Covered positives:          {P:,}  ({P/n_zhang_pos*100:.1f}% of Zhang et al.)")
        w(f"  Negatives sampled:          {N:,}  ({args.neg_ratio}:1 ratio)")
        w()
        w("--- Results ---")
        w(f"  Positives with score > 0:   {n_pos_hit}/{P}  ({n_pos_hit/P*100:.1f}% recall)")
        w(f"  Negatives with score > 0:   {n_neg_hit}/{N}  ({n_neg_hit/N*100:.1f}% false positive rate)")
        w(f"  Precision on hits:          {hit_precision*100:.1f}%  ({n_pos_hit} TP, {n_neg_hit} FP)")
        w()
        w("--- Metrics ---")
        w(f"  AUCPR:               {aucpr:.4f}  (random baseline: {baseline_aucpr:.4f})")
        w(f"  AUCPR / baseline:    {aucpr/baseline_aucpr:.2f}x")
        w(f"  ROC-AUC:             {roc_auc:.4f}  (random baseline: 0.5000)")
        if precs:
            w(f"  Precision @ R=0.1:   {precision_at_recall(precs, recs, 0.1):.4f}")
            w(f"  Precision @ R=0.2:   {precision_at_recall(precs, recs, 0.2):.4f}")
            w(f"  Precision @ R=0.5:   {precision_at_recall(precs, recs, 0.5):.4f}")
        w()
        w("--- Parameters ---")
        w(f"  taxid_filter:  {args.taxid or 'none (all organisms)'}")
        w(f"  wp:            {args.wp}  (positive weight for P-R; 1:1000 SNR model)")
        w(f"  neg_ratio:     {args.neg_ratio}")
        w(f"  random_seed:   42")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Universe
    log.info("Identifying universe from search cache...")
    universe = get_universe(args.search_cache_dir)
    log.info(f"Universe: {len(universe):,} proteins")
    universe_list = sorted(universe)

    # 2. Positives — Zhang et al. pairs where both proteins are in universe
    log.info("Loading Zhang et al. positive pairs...")
    zhang_pos = load_zhang_positives(args.controls)
    positives = [(a, b) for a, b in zhang_pos if a in universe and b in universe]
    log.info(f"Covered positives: {len(positives):,}/{len(zhang_pos):,}")

    # Canonical set for fast exclusion lookup (includes both orderings)
    zhang_pos_canonical = {make_canonical(a, b) for a, b in zhang_pos}

    # 3. Negatives
    n_neg = len(positives) * args.neg_ratio
    log.info(f"Sampling {n_neg:,} negatives from universe ({args.neg_ratio}:1 ratio)...")
    negatives = sample_negatives(universe_list, zhang_pos_canonical, n_neg, seed=42)
    log.info(f"Negatives generated: {len(negatives):,}")

    # 4. Load search results for all involved proteins
    all_proteins = {p for pair in positives + negatives for p in pair}
    log.info(f"Loading cached search results for {len(all_proteins):,} proteins...")
    search_results = load_search_results(all_proteins, args.search_cache_dir, taxid=args.taxid)

    # 5. Score
    log.info("Scoring pairs...")
    pos_scores = [score_pair(a, b, search_results) for a, b in positives]
    neg_scores = [score_pair(a, b, search_results) for a, b in negatives]

    n_pos_hit = sum(1 for s in pos_scores if s > 0)
    n_neg_hit = sum(1 for s in neg_scores if s > 0)
    log.info(f"Positives hit: {n_pos_hit}/{len(positives)}")
    log.info(f"Negatives hit: {n_neg_hit}/{len(negatives)}")

    # 6. Metrics
    log.info("Computing metrics...")
    precs, recs, aucpr = compute_weighted_pr(pos_scores, neg_scores, args.wp)
    roc_auc = compute_roc_auc(pos_scores, neg_scores)
    baseline = args.wp / (args.wp + 1)

    log.info(f"AUCPR:   {aucpr:.4f}  (baseline {baseline:.4f}, ratio {aucpr/baseline:.2f}x)")
    log.info(f"ROC-AUC: {roc_auc:.4f}  (baseline 0.5000)")

    # 7. Write outputs
    pairs_path   = os.path.join(args.output_dir, "pair_scores.tsv")
    pr_path      = os.path.join(args.output_dir, "pr_curve.tsv")
    summary_path = os.path.join(args.output_dir, "summary.txt")

    write_pair_scores(positives, negatives, pos_scores, neg_scores, pairs_path)
    write_pr_curve(precs, recs, pr_path)
    write_summary(positives, negatives, pos_scores, neg_scores,
                  precs, recs, aucpr, roc_auc, args, summary_path,
                  n_zhang_pos=len(zhang_pos), universe_size=len(universe))

    log.info(f"Done. Results in {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="PPI benchmark v2: structural domain homology evaluation within pair_list universe."
    )
    p.add_argument("--controls", default=DEFAULT_CONTROLS,
                   help=f"Path to positives_and_negatives.tsv (default: {DEFAULT_CONTROLS})")
    p.add_argument("--search-cache-dir", default=DEFAULT_SEARCH_CACHE, dest="search_cache_dir",
                   help=f"Directory of cached search TSVs (default: {DEFAULT_SEARCH_CACHE})")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, dest="output_dir",
                   help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--neg-ratio", type=int, default=5, dest="neg_ratio",
                   help="Negative:positive ratio (default: 5)")
    p.add_argument("--taxid", default="9606",
                   help="Filter hits to this NCBI taxid (default: 9606 = human). "
                        "Empty string to disable.")
    p.add_argument("--wp", type=float, default=0.01,
                   help="Positive weight for weighted P-R (default: 0.01)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
