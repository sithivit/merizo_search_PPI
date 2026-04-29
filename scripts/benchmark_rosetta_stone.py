#!/usr/bin/env python3
"""
Rosetta Stone PPI benchmark: paired-domain homology transfer.

For each protein P, all TED domain hits are aggregated into H[P]. A pair
(P1, P2) is scored by the best bridging template (A, B) where A in H[P1]
and B in H[P2]: score = max over bridges of min(TM_A, TM_B), checked
bidirectionally with an optional self-interaction filter.

Usage:
    python scripts/benchmark_rosetta_stone.py \\
        --search-cache-dir benchmark_cache/searches \\
        --template-index benchmark_cache/zhang_template_index.json \\
        --output-dir benchmark_results_rosetta_phase1

    python scripts/benchmark_rosetta_stone.py --multi-domain \\
        --search-cache-dir benchmark_cache/rosetta_searches \\
        --template-index benchmark_cache/zhang_template_index.json \\
        --output-dir benchmark_results_rosetta_phase2
"""

import argparse
import csv
import json
import logging
import os
import random
import sys

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CONTROLS = os.path.join(PROJECT_ROOT, "benchmark_cache", "benchmarks",
                                "positives_and_negatives.tsv")
DEFAULT_SEARCH_CACHE = os.path.join(PROJECT_ROOT, "benchmark_cache", "searches")
DEFAULT_TEMPLATE_INDEX = os.path.join(PROJECT_ROOT, "benchmark_cache",
                                      "zhang_template_index.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark_results_rosetta")

COL_TARGET = 2
COL_MAX_TM = 10


def get_universe(search_cache_dir: str, multi_domain: bool) -> set:
    """Proteins with at least one non-empty search TSV in the cache directory."""
    proteins = set()
    for fname in os.listdir(search_cache_dir):
        if not fname.endswith("_search.tsv"):
            continue
        if os.path.getsize(os.path.join(search_cache_dir, fname)) == 0:
            continue
        stem = fname[: -len("_search.tsv")]
        if multi_domain and "_dom" in stem:
            pid = stem.rsplit("_dom", 1)[0]
        else:
            pid = stem
        proteins.add(pid)
    return proteins


def load_zhang_positives(controls_path: str) -> list:
    positives = []
    with open(controls_path) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, cat = line.split("\t")
            if cat == "positive":
                a, b = pair.split("_")
                positives.append((a, b))
    return positives


def parse_search_tsv(tsv_path: str, query_protein: str) -> dict:
    """Parse search TSV. Returns {target_domain_name: max_tm}."""
    self_prefix = f"AF-{query_protein}-"
    best = {}
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= COL_MAX_TM:
                continue
            target = parts[COL_TARGET]
            if self_prefix in target:
                continue
            try:
                tm = float(parts[COL_MAX_TM])
            except ValueError:
                continue
            if target not in best or tm > best[target]:
                best[target] = tm
    return best


def load_search_results_per_domain(proteins: set, search_cache_dir: str,
                                   multi_domain: bool) -> dict:
    """Returns {(protein, domain_label): {template_domain: max_tm}}."""
    results = {}
    for pid in sorted(proteins):
        if multi_domain:
            prefix = pid + "_dom"
            for fname in os.listdir(search_cache_dir):
                if not fname.startswith(prefix) or not fname.endswith("_search.tsv"):
                    continue
                path = os.path.join(search_cache_dir, fname)
                if os.path.getsize(path) == 0:
                    continue
                dom_label = fname[len(pid) + 1: -len("_search.tsv")]
                results[(pid, dom_label)] = parse_search_tsv(path, pid)
        else:
            tsv = os.path.join(search_cache_dir, f"{pid}_search.tsv")
            if os.path.exists(tsv) and os.path.getsize(tsv) > 0:
                results[(pid, "dom00")] = parse_search_tsv(tsv, pid)
    return results


def build_protein_hit_map(per_domain_hits: dict) -> dict:
    """Aggregate per-domain hits into per-protein map (max TM per template domain)."""
    hit_map = {}
    for (pid, _label), hits in per_domain_hits.items():
        protein_hits = hit_map.setdefault(pid, {})
        for tmpl, tm in hits.items():
            if tmpl not in protein_hits or tm > protein_hits[tmpl]:
                protein_hits[tmpl] = tm
    return hit_map


def load_template_index(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def score_pair_rosetta(p1: str, p2: str,
                       domain_hits: dict,
                       template_index: dict,
                       min_bridge_tm: float = 0.0,
                       self_filter_tm: float = None) -> float:
    """
    Returns min(tm_A, tm_B) for the best bridging template (A, B), or 0.0.
    Bridges where P1 or P2 already carries both domain types are discarded
    (self-interaction filter; self_filter_tm defaults to min_bridge_tm).
    """
    H1 = domain_hits.get(p1, {})
    H2 = domain_hits.get(p2, {})
    if not H1 or not H2:
        return 0.0

    sft = min_bridge_tm if self_filter_tm is None else self_filter_tm

    best = 0.0
    for dom_a, tm_a in H1.items():
        if tm_a < min_bridge_tm:
            continue
        partners = template_index.get(dom_a)
        if not partners:
            continue
        for dom_b in partners:
            tm_b = H2.get(dom_b)
            if tm_b is None or tm_b < min_bridge_tm:
                continue
            if H1.get(dom_b, -1.0) >= sft:
                continue
            if H2.get(dom_a, -1.0) >= sft:
                continue
            score = min(tm_a, tm_b)
            if score > best:
                best = score
    for dom_b, tm_b in H1.items():
        if tm_b < min_bridge_tm:
            continue
        partners = template_index.get(dom_b)
        if not partners:
            continue
        for dom_a in partners:
            tm_a = H2.get(dom_a)
            if tm_a is None or tm_a < min_bridge_tm:
                continue
            if H1.get(dom_a, -1.0) >= sft:
                continue
            if H2.get(dom_b, -1.0) >= sft:
                continue
            score = min(tm_b, tm_a)
            if score > best:
                best = score
    return best


def make_canonical(a, b):
    return (a, b) if a < b else (b, a)


def sample_negatives(universe_list, positives_canonical, n, seed=42):
    rng = random.Random(seed)
    neg_set, negatives = set(), []
    attempts, max_attempts = 0, n * 50
    while len(negatives) < n and attempts < max_attempts:
        a, b = rng.choice(universe_list), rng.choice(universe_list)
        attempts += 1
        if a == b:
            continue
        key = make_canonical(a, b)
        if key in positives_canonical or key in neg_set:
            continue
        neg_set.add(key)
        negatives.append((a, b))
    return negatives


def compute_weighted_pr(pos_scores, neg_scores, wp=0.01):
    P = len(pos_scores)
    if P == 0:
        return [], [], 0.0
    all_pairs = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    random.seed(42)
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


def precision_at_recall(precisions, recalls, target):
    for p, r in zip(precisions, recalls):
        if r >= target:
            return p
    return 0.0


def compute_roc_auc(pos_scores, neg_scores):
    P, N = len(pos_scores), len(neg_scores)
    if P == 0 or N == 0:
        return 0.5
    u = sum(1.0 if p > n else 0.5 if p == n else 0.0
            for p in pos_scores for n in neg_scores)
    return u / (P * N)


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
                  n_zhang_pos, universe_size):
    P, N = len(positives), len(negatives)
    n_pos_hit = sum(1 for s in pos_scores if s > 0)
    n_neg_hit = sum(1 for s in neg_scores if s > 0)
    baseline_aucpr = args.wp / (args.wp + 1)

    with open(path, "w") as fh:
        def w(line=""): fh.write(line + "\n")
        w("=" * 60)
        w("PPI Benchmark — Rosetta Stone Paired-Domain Transfer")
        w("=" * 60)
        w()
        w("--- Algorithm ---")
        w("  Template: TED intra-protein domain-domain pairs (A, B)")
        w("  Inference: P1 interacts P2 iff P1 carries A-like domain")
        w("             AND P2 carries B-like domain for some template (A,B)")
        w("  Score:    min(TM(P1,A), TM(P2,B)) for best bridging template")
        w()
        w("--- Dataset ---")
        w(f"  Universe size:          {universe_size:,} proteins")
        w(f"  Zhang positives:        {n_zhang_pos:,} total")
        w(f"  Covered positives:      {P:,}  ({P/n_zhang_pos*100:.1f}% of Zhang)")
        w(f"  Negatives sampled:      {N:,}  ({args.neg_ratio}:1 ratio)")
        w()
        w("--- Results ---")
        w(f"  Positives with score > 0:  {n_pos_hit}/{P}  ({n_pos_hit/P*100:.1f}%)")
        w(f"  Negatives with score > 0:  {n_neg_hit}/{N}  ({n_neg_hit/N*100:.1f}%)")
        w()
        w("--- Metrics ---")
        w(f"  AUCPR:              {aucpr:.4f}  (random baseline: {baseline_aucpr:.4f})")
        w(f"  AUCPR / baseline:   {aucpr / baseline_aucpr:.2f}x")
        w(f"  ROC-AUC:            {roc_auc:.4f}")
        if precs:
            w(f"  Precision @ R=0.1:  {precision_at_recall(precs, recs, 0.1):.4f}")
            w(f"  Precision @ R=0.2:  {precision_at_recall(precs, recs, 0.2):.4f}")
            w(f"  Precision @ R=0.5:  {precision_at_recall(precs, recs, 0.5):.4f}")
        w()
        w("--- Parameters ---")
        w(f"  multi_domain:      {args.multi_domain}")
        w(f"  min_bridge_tm:     {args.min_bridge_tm}")
        w(f"  self_filter_tm:    {args.self_filter_tm} "
          f"(None = same as min_bridge_tm)")
        w(f"  wp:                {args.wp}")
        w(f"  neg_ratio:         {args.neg_ratio}")


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Building universe from search cache...")
    universe = get_universe(args.search_cache_dir, multi_domain=args.multi_domain)
    log.info(f"Universe: {len(universe):,} proteins")
    universe_list = sorted(universe)

    log.info("Loading Zhang positives...")
    zhang_pos = load_zhang_positives(args.controls)
    positives = [(a, b) for a, b in zhang_pos if a in universe and b in universe]
    log.info(f"Covered positives: {len(positives):,}/{len(zhang_pos):,}")

    zhang_pos_canonical = {make_canonical(a, b) for a, b in zhang_pos}
    n_neg = len(positives) * args.neg_ratio
    negatives = sample_negatives(universe_list, zhang_pos_canonical, n_neg)
    log.info(f"Negatives: {len(negatives):,}")

    all_proteins = {p for pair in positives + negatives for p in pair}
    log.info(f"Loading search results for {len(all_proteins):,} proteins...")
    per_domain_hits = load_search_results_per_domain(
        all_proteins, args.search_cache_dir, multi_domain=args.multi_domain)
    domain_hits = build_protein_hit_map(per_domain_hits)
    log.info(f"Loaded domain hits for {len(domain_hits):,} proteins")

    log.info("Loading template pair index...")
    template_index = load_template_index(args.template_index)
    log.info(f"Template index: {len(template_index):,} domain entries")

    log.info(f"Scoring pairs (min_bridge_tm={args.min_bridge_tm}, "
             f"self_filter_tm={args.self_filter_tm})...")
    pos_scores = [score_pair_rosetta(a, b, domain_hits, template_index,
                                     args.min_bridge_tm, args.self_filter_tm)
                  for a, b in positives]
    neg_scores = [score_pair_rosetta(a, b, domain_hits, template_index,
                                     args.min_bridge_tm, args.self_filter_tm)
                  for a, b in negatives]

    n_pos_hit = sum(1 for s in pos_scores if s > 0)
    n_neg_hit = sum(1 for s in neg_scores if s > 0)
    log.info(f"Positives hit: {n_pos_hit}/{len(positives)}")
    log.info(f"Negatives hit: {n_neg_hit}/{len(negatives)}")

    log.info("Computing metrics...")
    precs, recs, aucpr = compute_weighted_pr(pos_scores, neg_scores, args.wp)
    roc_auc = compute_roc_auc(pos_scores, neg_scores)
    baseline = args.wp / (args.wp + 1)
    log.info(f"AUCPR: {aucpr:.4f} (baseline {baseline:.4f}, {aucpr / baseline:.2f}x)")
    log.info(f"ROC-AUC: {roc_auc:.4f}")

    write_pair_scores(positives, negatives, pos_scores, neg_scores,
                      os.path.join(args.output_dir, "pair_scores.tsv"))
    write_pr_curve(precs, recs, os.path.join(args.output_dir, "pr_curve.tsv"))
    write_summary(positives, negatives, pos_scores, neg_scores, precs, recs,
                  aucpr, roc_auc, args,
                  os.path.join(args.output_dir, "summary.txt"),
                  len(zhang_pos), len(universe))
    log.info(f"Done. Results in {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Rosetta Stone PPI benchmark: paired-domain homology transfer.")
    p.add_argument("--controls", default=DEFAULT_CONTROLS)
    p.add_argument("--search-cache-dir", default=DEFAULT_SEARCH_CACHE,
                   dest="search_cache_dir")
    p.add_argument("--template-index", default=DEFAULT_TEMPLATE_INDEX,
                   dest="template_index",
                   help="JSON {domain: [domain,...]} template pair index")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, dest="output_dir")
    p.add_argument("--multi-domain", action="store_true", dest="multi_domain",
                   help="Cache has per-domain files ({pid}_domNN_search.tsv)")
    p.add_argument("--min-bridge-tm", type=float, default=0.0,
                   dest="min_bridge_tm",
                   help="Minimum TM score required on both bridge sides (e.g. 0.5)")
    p.add_argument("--self-filter-tm", type=float, default=None,
                   dest="self_filter_tm",
                   help="TM threshold for self-interaction filter: discard bridge (A,B) "
                        "if P1 already has TM>=threshold to both A and B, or P2 does. "
                        "Defaults to min_bridge_tm when not set.")
    p.add_argument("--neg-ratio", type=int, default=5, dest="neg_ratio")
    p.add_argument("--wp", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
