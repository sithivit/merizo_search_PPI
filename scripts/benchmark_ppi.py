#!/usr/bin/env python3
"""
benchmark_ppi.py

Benchmark the PPI prediction pipeline against the Zhang et al. (Science 2025)
curated control set: 3,000 high-confidence positive pairs (intersection of
STRING/BioGRID/UniProt) and 30,000 random negative pairs.

Methodology mirrors the paper:
  - Each pair (A, B) is scored by max_tm from structural search (A→B or B→A)
  - All pairs ranked by score; weighted precision-recall computed with wp=0.01
    to reflect the 1:1000 signal-to-noise ratio of proteome-wide PPI screens
  - Primary metric: AUCPR (area under weighted precision-recall curve)

Data (pre-downloaded):
  benchmark_cache/benchmarks/positives_and_negatives.tsv

Usage:
    python scripts/benchmark_ppi.py [OPTIONS]

Examples:
    python scripts/benchmark_ppi.py --download-only
    python scripts/benchmark_ppi.py --limit 50
    python scripts/benchmark_ppi.py --force-rerun
"""

import argparse
import csv
import logging
import os
import random
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PAIR_LIST = "/mnt/bigstore/ted/pair_list_20250128"
DEFAULT_PAIRLIST_DB = os.path.join(PROJECT_ROOT, "merizo_pairlist_db", "ted_pairlist")
DEFAULT_CONTROLS_PATH = os.path.join(
    PROJECT_ROOT, "benchmark_cache", "benchmarks", "positives_and_negatives.tsv"
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark_results")
DEFAULT_SEARCH_CACHE_DIR = os.path.join(PROJECT_ROOT, "benchmark_cache", "searches")

BENCHMARKS_URL = "https://conglab.swmed.edu/humanPPI/downloads/benchmarks.tar.gz"

# Merizo search output columns (0-indexed, no header by default)
# Format: query, emb_rank, target, emb_score, q_len, t_len, ali_len, seq_id, q_tm, t_tm, max_tm, rmsd, metadata
COL_TARGET = 2
COL_MAX_TM = 10

TM_SWEEP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_AF_PROTEIN_RE = re.compile(r'AF-([^-]+)-F1-model_v4')


# ---------------------------------------------------------------------------
# Step 1: Data download
# ---------------------------------------------------------------------------

def download_zhang_data(cache_dir: str) -> None:
    """Download benchmarks.tar.gz from the Zhang et al. data repository."""
    tar_path = os.path.join(cache_dir, "benchmarks.tar.gz")
    extract_dir = os.path.join(cache_dir, "benchmarks")

    if os.path.exists(extract_dir):
        log.info(f"Benchmark data already present: {extract_dir}")
        return

    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(tar_path):
        log.info(f"Downloading benchmark controls from {BENCHMARKS_URL} ...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(BENCHMARKS_URL, context=ctx) as resp, \
                open(tar_path, 'wb') as fh:
            shutil.copyfileobj(resp, fh)
        log.info(f"Downloaded: {os.path.getsize(tar_path) // 1024} KB")

    log.info("Extracting benchmarks.tar.gz ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(cache_dir)
    log.info(f"Extracted to {extract_dir}")


# ---------------------------------------------------------------------------
# Step 2: Load benchmark controls
# ---------------------------------------------------------------------------

def load_zhang_controls(path: str) -> tuple:
    """Load positives_and_negatives.tsv.

    Returns:
        positives: list of (protein_a, protein_b) tuples
        negatives: list of (protein_a, protein_b) tuples
    """
    positives, negatives = [], []
    with open(path) as fh:
        next(fh)  # skip header: "Protein pairs\tCategory"
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, category = line.split('\t')
            a, b = pair.split('_')
            if category == "positive":
                positives.append((a, b))
            else:
                negatives.append((a, b))
    log.info(f"Loaded {len(positives)} positives, {len(negatives)} negatives")
    return positives, negatives


# ---------------------------------------------------------------------------
# Step 3: Load pair list (to get Domain B for each protein)
# ---------------------------------------------------------------------------

def _extract_protein_id(domain_id: str):
    """Extract UniProt ID from AF domain ID e.g. AF-Q9Y6K1-F1-model_v4_TED01."""
    m = _AF_PROTEIN_RE.match(domain_id)
    return m.group(1) if m else None


def load_pair_list(path: str, target_proteins: set) -> dict:
    """Stream pair_list, return dict: protein_id -> (domain_a, domain_b, pae).

    Only loads proteins in target_proteins to save memory.
    """
    pairs: dict = {}
    line_count = 0

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or ':' not in line:
                continue
            parts = line.split()
            if not parts or ':' not in parts[0]:
                continue

            domain_a, domain_b = parts[0].split(':', 1)
            protein_id = _extract_protein_id(domain_a)
            if protein_id is None or protein_id not in target_proteins:
                continue
            if protein_id in pairs:
                continue  # keep first pair only

            pae = float(parts[2]) if len(parts) >= 3 else 0.0
            pairs[protein_id] = (domain_a, domain_b, pae)

            line_count += 1
            if line_count % 1_000_000 == 0:
                log.info(f"  Scanned {line_count:,} matching lines, found {len(pairs):,}/{len(target_proteins):,} proteins...")
            if len(pairs) == len(target_proteins):
                break  # found everything

    log.info(f"Pair list: found {len(pairs):,}/{len(target_proteins):,} benchmark proteins")
    return pairs


# ---------------------------------------------------------------------------
# Step 4: Structural search (with caching)
# ---------------------------------------------------------------------------

def extract_domain_b_pdb(domain_b: str, pairlist_db: str, output_pdb: str) -> bool:
    """Run extract_domain_pdb.py. Returns True on success."""
    extract_script = os.path.join(SCRIPT_DIR, "extract_domain_pdb.py")
    db_json = pairlist_db + ".json"
    cmd = [sys.executable, extract_script, db_json, domain_b, output_pdb]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and os.path.exists(output_pdb)
    except (subprocess.TimeoutExpired, Exception) as e:
        log.warning(f"  extract_domain_pdb error for {domain_b}: {e}")
        return False


def run_merizo_search(
    domain_b_pdb: str,
    pairlist_db: str,
    output_prefix: str,
    batchsize: int,
    tm_threshold: float,
    topk: int,
) -> str:
    """Run merizo search. Returns path to _search.tsv or None on failure."""
    merizo_py = os.path.join(PROJECT_ROOT, "merizo_search", "merizo.py")
    tmp_dir = output_prefix + "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        sys.executable, merizo_py, "search",
        domain_b_pdb, pairlist_db, output_prefix, tmp_dir,
        "--search_batchsize", str(batchsize),
        "--mintm", str(tm_threshold),
        "--topk", str(topk),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tsv_path = output_prefix + "_search.tsv"
        if result.returncode != 0 or not os.path.exists(tsv_path):
            log.warning(f"  Search failed (rc={result.returncode}): {result.stderr[:200]}")
            return None
        return tsv_path
    except subprocess.TimeoutExpired:
        log.warning("  Search timed out (>10 min)")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    except Exception as e:
        log.warning(f"  Search error: {e}")
        return None


_TAXID_RE = re.compile(r'"taxid":\s*"(\d+)"')


def parse_search_results(tsv_path: str, query_protein_id: str, taxid_filter: str = None) -> dict:
    """Parse search TSV (no header). Returns dict: target_protein_id -> max_tm.

    Self-hits filtered. Deduplicated by UniProt: keep highest max_tm per protein.
    If taxid_filter is set (e.g. '9606'), only keep hits from that organism.
    Taxid is read from the metadata JSON in the last column.
    """
    self_prefix = f"AF-{query_protein_id}-"
    best: dict = {}
    COL_META = 12

    with open(tsv_path) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= COL_MAX_TM:
                continue
            target = parts[COL_TARGET]
            if self_prefix in target:
                continue
            protein_id = _extract_protein_id(target)
            if protein_id is None:
                continue
            try:
                max_tm = float(parts[COL_MAX_TM])
            except ValueError:
                continue
            if taxid_filter and len(parts) > COL_META:
                m = _TAXID_RE.search(parts[COL_META])
                if not m or m.group(1) != taxid_filter:
                    continue
            if protein_id not in best or max_tm > best[protein_id]:
                best[protein_id] = max_tm

    return best


def _search_one_protein(protein_id, domain_b, search_cache_dir, pairlist_db,
                        taxid, topk, tm_threshold, search_batchsize, force_rerun):
    """Process a single protein in a thread. Returns (protein_id, hits_dict, status_msg)."""
    cache_tsv = os.path.join(search_cache_dir, f"{protein_id}_search.tsv")

    if os.path.exists(cache_tsv) and not force_rerun:
        hits = parse_search_results(cache_tsv, protein_id, taxid_filter=taxid)
        return protein_id, hits, f"cached: {len(hits)} hits (taxid={taxid or 'all'})"

    with tempfile.TemporaryDirectory(prefix="bench_pdb_") as tmpdir:
        pdb_path = os.path.join(tmpdir, f"{protein_id}.pdb")
        if not extract_domain_b_pdb(domain_b, pairlist_db, pdb_path):
            return protein_id, {}, "extract_failed"

        log.info(f"  [{protein_id}] extraction OK — running merizo search (topk={topk}, mintm={tm_threshold})")
        search_prefix = os.path.join(search_cache_dir, protein_id)
        tsv_path = run_merizo_search(
            pdb_path, pairlist_db, search_prefix,
            search_batchsize, tm_threshold, topk,
        )
        if tsv_path is None:
            return protein_id, {}, "search_failed"

        if tsv_path != cache_tsv:
            shutil.move(tsv_path, cache_tsv)
        hits = parse_search_results(cache_tsv, protein_id, taxid_filter=taxid)
        return protein_id, hits, f"search OK: {len(hits)} unique protein hits (taxid={taxid or 'all'})"


def run_searches_for_proteins(
    proteins_to_search: list,
    pair_list: dict,
    args,
) -> dict:
    """Run/load searches for all proteins. Returns dict: protein_id -> {target_id: max_tm}."""
    search_results: dict = {}
    total = len(proteins_to_search)
    workers = getattr(args, 'workers', 1)

    def _submit(protein_id):
        domain_a, domain_b, pae = pair_list[protein_id]
        log.info(f"  extracting domain B: {domain_b}")
        return _search_one_protein(
            protein_id, domain_b,
            args.search_cache_dir, args.pairlist_db, args.taxid,
            args.topk, args.tm_threshold, args.search_batchsize, args.force_rerun,
        )

    if workers == 1:
        for i, protein_id in enumerate(proteins_to_search, start=1):
            domain_a, domain_b, pae = pair_list[protein_id]
            log.info(f"[{i}/{total}] {protein_id}  domain_b={domain_b}")
            _, hits, msg = _submit(protein_id)
            log.info(f"  {msg}")
            search_results[protein_id] = hits
    else:
        log.info(f"Running searches with {workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_submit, pid): pid
                for pid in proteins_to_search
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                protein_id, hits, msg = future.result()
                log.info(f"[{completed}/{total}] {protein_id}  {msg}")
                search_results[protein_id] = hits

    return search_results


# ---------------------------------------------------------------------------
# Step 5: Score benchmark pairs and compute weighted P-R
# ---------------------------------------------------------------------------

def score_pairs(pairs: list, search_results: dict) -> list:
    """Score each pair (A, B) by max(A→B max_tm, B→A max_tm).

    Returns list of (score, label) sorted by score descending.
    Only includes pairs where at least one protein was searched.
    Pairs where neither protein was searched get score=0.
    """
    scored = []
    for a, b in pairs:
        score_ab = search_results.get(a, {}).get(b, 0.0)
        score_ba = search_results.get(b, {}).get(a, 0.0)
        scored.append(max(score_ab, score_ba))
    return scored


def compute_weighted_pr(
    pos_scores: list,
    neg_scores: list,
    wp: float = 0.01,
) -> tuple:
    """Compute weighted precision-recall curve.

    Args:
        pos_scores: scores for positive pairs (length P)
        neg_scores: scores for negative pairs (length N)
        wp: weight per positive (default 0.01 for 1:1000 signal-to-noise)

    Returns:
        precisions: list of precision values
        recalls:    list of recall values
        aucpr:      area under the curve (trapezoidal)
    """
    P = len(pos_scores)
    if P == 0:
        return [], [], 0.0

    # Merge, shuffle first to break ties randomly (prevents list-order bias),
    # then sort descending by score
    all_pairs = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    random.shuffle(all_pairs)
    all_pairs.sort(key=lambda x: x[0], reverse=True)

    precisions, recalls = [], []
    tp, fp = 0, 0

    for score, label in all_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        prec = (tp * wp) / (tp * wp + fp)
        rec = tp / P
        precisions.append(prec)
        recalls.append(rec)

    # AUCPR via trapezoidal rule on recall axis
    aucpr = 0.0
    for i in range(1, len(recalls)):
        dr = recalls[i] - recalls[i - 1]
        aucpr += 0.5 * (precisions[i] + precisions[i - 1]) * dr

    return precisions, recalls, aucpr


def precision_at_recall(precisions: list, recalls: list, recall_target: float) -> float:
    """Precision at the first point where recall >= recall_target."""
    for p, r in zip(precisions, recalls):
        if r >= recall_target:
            return p
    return 0.0


def compute_sweep(
    positives: list,
    negatives: list,
    all_scores: dict,
    thresholds: list,
    wp: float,
) -> list:
    """Re-score pairs at each TM threshold (scores below threshold → 0)."""
    rows = []
    for threshold in thresholds:
        pos_scores = [max(0.0, s) if s >= threshold else 0.0
                      for s in all_scores["positives"]]
        neg_scores = [max(0.0, s) if s >= threshold else 0.0
                      for s in all_scores["negatives"]]
        precs, recs, aucpr = compute_weighted_pr(pos_scores, neg_scores, wp)
        n_pairs_with_hit = sum(1 for s in pos_scores + neg_scores if s > 0)
        rows.append({
            "tm_threshold": threshold,
            "n_pairs_with_hit": n_pairs_with_hit,
            "aucpr": round(aucpr, 6),
            "precision_at_recall_0.1": round(precision_at_recall(precs, recs, 0.1), 4),
            "precision_at_recall_0.2": round(precision_at_recall(precs, recs, 0.2), 4),
            "precision_at_recall_0.5": round(precision_at_recall(precs, recs, 0.5), 4),
            "baseline_precision": round(wp / (wp + 1), 6),  # TP*wp/(TP*wp+FP) at random
        })
    return rows


# ---------------------------------------------------------------------------
# Step 6: Output
# ---------------------------------------------------------------------------

def write_pair_scores_tsv(
    positives: list,
    negatives: list,
    pos_scores: list,
    neg_scores: list,
    output_path: str,
) -> None:
    """Write one row per benchmark pair with its score and label."""
    with open(output_path, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(["protein_a", "protein_b", "label", "pair_score"])
        for (a, b), score in zip(positives, pos_scores):
            writer.writerow([a, b, "positive", f"{score:.4f}"])
        for (a, b), score in zip(negatives, neg_scores):
            writer.writerow([a, b, "negative", f"{score:.4f}"])


def write_sweep_tsv(sweep_rows: list, output_path: str) -> None:
    if not sweep_rows:
        return
    with open(output_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()), delimiter='\t')
        writer.writeheader()
        writer.writerows(sweep_rows)


def write_summary_txt(
    positives: list,
    negatives: list,
    pos_scores: list,
    neg_scores: list,
    precs: list,
    recs: list,
    aucpr: float,
    sweep_rows: list,
    args,
    output_path: str,
) -> None:
    P = len(positives)
    N = len(negatives)
    n_pos_found = sum(1 for s in pos_scores if s > 0)
    n_neg_found = sum(1 for s in neg_scores if s > 0)

    with open(output_path, 'w') as fh:
        def w(line=""):
            fh.write(line + "\n")

        w("=" * 60)
        w("PPI Benchmark Summary — Zhang et al. (Science 2025) Controls")
        w("=" * 60)
        w()

        w("--- Run Parameters ---")
        w(f"  pair_list:       {args.pair_list}")
        w(f"  pairlist_db:     {args.pairlist_db}")
        w(f"  controls:        {args.controls_path}")
        w(f"  tm_threshold:    {args.tm_threshold}  (search; also default sweep start)")
        w(f"  taxid_filter:    {args.taxid or 'none (all organisms)'}")
        w(f"  topk:            {args.topk}")
        w(f"  wp:              {args.wp}  (positive pair weight, 1:1000 SNR)")
        w(f"  limit:           {args.limit}")
        w()

        w("--- Benchmark Set ---")
        w(f"  Positive pairs:  {P:,}  (intersection of STRING/BioGRID/UniProt)")
        w(f"  Negative pairs:  {N:,}  (random human protein pairs)")
        w(f"  Positives found in search (score>0):  {n_pos_found}/{P}")
        w(f"  Negatives found in search (score>0):  {n_neg_found}/{N}")
        w()

        w("--- Primary Metric (weighted P-R, wp=0.01) ---")
        w(f"  AUCPR:                   {aucpr:.4f}")
        if precs:
            w(f"  Precision @ Recall 0.1:  {precision_at_recall(precs, recs, 0.1):.4f}")
            w(f"  Precision @ Recall 0.2:  {precision_at_recall(precs, recs, 0.2):.4f}")
            w(f"  Precision @ Recall 0.5:  {precision_at_recall(precs, recs, 0.5):.4f}")
        baseline = args.wp / (args.wp + 1)
        w(f"  Baseline precision (random):  {baseline:.4f}")
        w()

        w("--- TM Threshold Sensitivity ---")
        w(f"  {'threshold':>10}  {'n_pairs_hit':>12}  {'AUCPR':>8}  "
          f"{'P@R=0.1':>9}  {'P@R=0.2':>9}  {'baseline':>10}")
        for row in sweep_rows:
            w(f"  {row['tm_threshold']:>10.1f}  "
              f"{row['n_pairs_with_hit']:>12d}  "
              f"{row['aucpr']:>8.4f}  "
              f"{row['precision_at_recall_0.1']:>9.4f}  "
              f"{row['precision_at_recall_0.2']:>9.4f}  "
              f"{row['baseline_precision']:>10.6f}")
        w()

        w("--- Caveats ---")
        w("  - Score per pair = max(A→B max_tm, B→A max_tm); 0 if neither searched.")
        w("  - Proteins not in pair_list cannot be searched (no Domain B).")
        w("  - Only first domain pair per protein is used as the search query.")
        w("  - Positives are high-confidence (3-database intersection); still incomplete.")
        w("  - wp=0.01 reflects ~1:1000 true PPI signal-to-noise in the human proteome.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_benchmark(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.search_cache_dir, exist_ok=True)

    # Step 1: Download data if missing
    cache_dir = os.path.dirname(args.controls_path)
    if not os.path.exists(args.controls_path):
        download_zhang_data(cache_dir)

    if args.download_only:
        log.info(f"Controls file: {args.controls_path}")
        positives, negatives = load_zhang_controls(args.controls_path)
        log.info("Download-only mode complete.")
        return

    # Step 2: Load benchmark controls
    positives, negatives = load_zhang_controls(args.controls_path)

    # Step 3: Find all unique proteins; load their domain pairs from pair_list
    all_proteins = set()
    for a, b in positives + negatives:
        all_proteins.add(a)
        all_proteins.add(b)
    log.info(f"Unique proteins in benchmark: {len(all_proteins):,}")

    log.info("Loading pair list (streaming, this may take a few minutes)...")
    pair_list = load_pair_list(args.pair_list, all_proteins)

    # Proteins that can be searched (have a domain B in pairlist)
    searchable = sorted(pair_list.keys())
    if args.limit:
        searchable = searchable[:args.limit]
        log.info(f"Limiting to first {args.limit} proteins ({len(searchable)} searchable)")
    else:
        log.info(f"Searchable proteins: {len(searchable):,}")

    # Step 4: Run/load searches for all searchable proteins
    search_results = run_searches_for_proteins(searchable, pair_list, args)

    # Step 5: Score all benchmark pairs
    pos_scores = score_pairs(positives, search_results)
    neg_scores = score_pairs(negatives, search_results)

    # Compute weighted P-R at the run threshold
    pos_scores_filtered = [s if s >= args.tm_threshold else 0.0 for s in pos_scores]
    neg_scores_filtered = [s if s >= args.tm_threshold else 0.0 for s in neg_scores]
    precs, recs, aucpr = compute_weighted_pr(pos_scores_filtered, neg_scores_filtered, args.wp)

    log.info(f"AUCPR at TM≥{args.tm_threshold}: {aucpr:.4f}")

    # TM threshold sweep
    all_raw_scores = {"positives": pos_scores, "negatives": neg_scores}
    sweep_rows = compute_sweep(positives, negatives, all_raw_scores, TM_SWEEP_THRESHOLDS, args.wp)

    # Step 6: Write outputs
    pairs_tsv = os.path.join(args.output_dir, "benchmark_pair_scores.tsv")
    sweep_tsv = os.path.join(args.output_dir, "precision_recall_by_tm.tsv")
    summary_txt = os.path.join(args.output_dir, "benchmark_summary.txt")

    write_pair_scores_tsv(positives, negatives, pos_scores, neg_scores, pairs_tsv)
    write_sweep_tsv(sweep_rows, sweep_tsv)
    write_summary_txt(
        positives, negatives, pos_scores_filtered, neg_scores_filtered,
        precs, recs, aucpr, sweep_rows, args, summary_txt,
    )

    log.info(f"Results written to {args.output_dir}/")
    log.info(f"  {pairs_tsv}")
    log.info(f"  {sweep_tsv}")
    log.info(f"  {summary_txt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PPI predictions against Zhang et al. (Science 2025) controls."
    )
    parser.add_argument("--pair-list", default=DEFAULT_PAIR_LIST, dest="pair_list",
                        help=f"Path to pair_list file (default: {DEFAULT_PAIR_LIST})")
    parser.add_argument("--pairlist-db", default=DEFAULT_PAIRLIST_DB, dest="pairlist_db",
                        help=f"Foldclass database prefix (default: {DEFAULT_PAIRLIST_DB})")
    parser.add_argument("--controls-path", default=DEFAULT_CONTROLS_PATH, dest="controls_path",
                        help=f"Path to positives_and_negatives.tsv (default: {DEFAULT_CONTROLS_PATH})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, dest="output_dir",
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--search-cache-dir", default=DEFAULT_SEARCH_CACHE_DIR, dest="search_cache_dir",
                        help=f"Directory to cache per-protein search results (default: {DEFAULT_SEARCH_CACHE_DIR})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only search first N unique proteins (for testing)")
    parser.add_argument("--tm-threshold", type=float, default=0.3, dest="tm_threshold",
                        help="Minimum TM-score for merizo search (default: 0.3)")
    parser.add_argument("--topk", type=int, default=500,
                        help="Top-K hits per search (default: 500)")
    parser.add_argument("--wp", type=float, default=0.01,
                        help="Positive pair weight for P-R computation (default: 0.01)")
    parser.add_argument("--taxid", type=str, default="9606",
                        help="Filter search results to this NCBI taxid (default: 9606 = human). "
                             "Set to empty string to disable filtering.")
    parser.add_argument("--force-rerun", action="store_true", dest="force_rerun",
                        help="Ignore cached search results and rerun searches")
    parser.add_argument("--download-only", action="store_true", dest="download_only",
                        help="Only download benchmark data and exit")
    parser.add_argument("--search-batchsize", type=int, default=2097152, dest="search_batchsize",
                        help="Merizo search batch size (default: 2097152)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel search workers (default: 1). "
                             "Each worker runs extract+search for one protein concurrently. "
                             "Set to 4-8 depending on available CPU cores and RAM.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
