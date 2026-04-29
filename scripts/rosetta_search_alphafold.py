#!/usr/bin/env python3
"""
rosetta_search_alphafold.py

Download AlphaFold PDB structures for benchmark proteins that are missing
from the Phase 2 search cache (i.e. single-domain proteins absent from the
TED pairlist SQLite DB), then run merizo search against the Zhang sub-DB.

This extends Phase 2 coverage to single-domain proteins without needing
the ted_pairlist_filters.db infrastructure.

For each missing protein:
  1. Download AF-{uid}-F1-model_v4.pdb from the AlphaFold EBI server
  2. Run merizo search against the Zhang sub-DB
  3. Save result as: {output_dir}/{uid}_dom00_search.tsv
     (dom00 suffix keeps it compatible with --multi-domain benchmark mode)
  4. On 404 or search failure, write empty sentinel to skip on re-runs

Usage:
    python scripts/rosetta_search_alphafold.py \\
        --controls benchmark_cache/benchmarks/positives_and_negatives.tsv \\
        --search-cache-dir benchmark_cache/rosetta_searches \\
        --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \\
        --workers 6
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"


def is_canonical_uniprot(uid: str) -> bool:
    return bool(re.fullmatch(
        r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]',
        uid
    ))


def get_all_benchmark_proteins(controls_path: str) -> set:
    proteins = set()
    with open(controls_path) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, _cat = line.split("\t")
            a, b = pair.split("_")
            for uid in (a, b):
                if is_canonical_uniprot(uid):
                    proteins.add(uid)
    return proteins


def get_universe(search_cache_dir: str) -> set:
    """Proteins with at least one non-empty dom*_search.tsv in the cache."""
    proteins = set()
    for fname in os.listdir(search_cache_dir):
        if not fname.endswith("_search.tsv"):
            continue
        if os.path.getsize(os.path.join(search_cache_dir, fname)) == 0:
            continue
        if "_dom" in fname:
            pid = fname.rsplit("_dom", 1)[0]
        else:
            pid = fname[:-len("_search.tsv")]
        proteins.add(pid)
    return proteins


def has_sentinel(uid: str, search_cache_dir: str) -> bool:
    """Return True if an empty sentinel already exists for this protein."""
    sentinel = os.path.join(search_cache_dir, f"{uid}_dom00_search.tsv")
    return os.path.exists(sentinel) and os.path.getsize(sentinel) == 0


def download_alphafold_pdb(uid: str, dest_path: str) -> bool:
    url = ALPHAFOLD_URL.format(uid=uid)
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False  # no AlphaFold model for this protein
        log.warning(f"{uid}: HTTP error {e.code} downloading PDB")
        return False
    except Exception as e:
        log.warning(f"{uid}: download error — {e}")
        return False


def run_search(pdb_path: str, zhang_db: str, output_prefix: str,
               topk: int, batchsize: int):
    merizo = os.path.join(PROJECT_ROOT, "merizo_search", "merizo.py")
    tmp = output_prefix + "_tmp"
    os.makedirs(tmp, exist_ok=True)
    cmd = [sys.executable, merizo, "search",
           pdb_path, zhang_db, output_prefix, tmp,
           "--search_batchsize", str(batchsize),
           "--mintm", "0.0",
           "--topk", str(topk)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        shutil.rmtree(tmp, ignore_errors=True)
        tsv = output_prefix + "_search.tsv"
        if r.returncode != 0 or not os.path.exists(tsv):
            log.warning(f"Search failed (rc={r.returncode}): {r.stderr[-300:]}")
            return None
        return tsv
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        log.warning(f"Search error: {e}")
        return None


def process_protein(uid: str, search_cache_dir: str, zhang_db: str,
                    topk: int, batchsize: int) -> tuple:
    cache_tsv = os.path.join(search_cache_dir, f"{uid}_dom00_search.tsv")

    # Already done (non-empty result exists)
    if os.path.exists(cache_tsv) and os.path.getsize(cache_tsv) > 0:
        return uid, "already_cached"

    # Already failed (empty sentinel exists)
    if os.path.exists(cache_tsv) and os.path.getsize(cache_tsv) == 0:
        return uid, "skipped:sentinel_exists"

    with tempfile.TemporaryDirectory(prefix="afdown_") as tmpdir:
        pdb_path = os.path.join(tmpdir, f"{uid}.pdb")

        if not download_alphafold_pdb(uid, pdb_path):
            open(cache_tsv, "w").close()  # empty sentinel
            return uid, "no_alphafold_model"

        prefix = os.path.join(tmpdir, uid)
        tsv = run_search(pdb_path, zhang_db, prefix, topk, batchsize)

        if tsv:
            shutil.move(tsv, cache_tsv)
            n_hits = sum(1 for _ in open(cache_tsv))
            return uid, f"ok ({n_hits} hits)"
        else:
            open(cache_tsv, "w").close()
            return uid, "search_failed"


def run(args):
    os.makedirs(args.search_cache_dir, exist_ok=True)

    log.info("Loading benchmark proteins...")
    all_proteins = get_all_benchmark_proteins(args.controls)
    log.info(f"Total canonical proteins in benchmark: {len(all_proteins):,}")

    log.info("Loading current search universe...")
    universe = get_universe(args.search_cache_dir)
    log.info(f"Already in universe: {len(universe):,}")

    todo = sorted(all_proteins - universe)
    # Skip proteins that already have an empty sentinel (no AlphaFold model)
    todo = [uid for uid in todo
            if not has_sentinel(uid, args.search_cache_dir)]
    log.info(f"Remaining to process: {len(todo):,}")

    if not todo:
        log.info("Nothing to do.")
        return

    total = len(todo)
    log.info(f"Running with {args.workers} workers, topk={args.topk}...")

    def submit(uid):
        return process_protein(uid, args.search_cache_dir, args.zhang_db,
                               args.topk, args.batchsize)

    counts = {"ok": 0, "no_alphafold_model": 0, "search_failed": 0,
              "already_cached": 0, "skipped:sentinel_exists": 0}

    if args.workers == 1:
        for i, uid in enumerate(todo, 1):
            uid_out, msg = submit(uid)
            status = msg.split(" ")[0]
            counts[status] = counts.get(status, 0) + 1
            log.info(f"[{i}/{total}] {uid_out}: {msg}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(submit, uid): uid for uid in todo}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                uid_out, msg = future.result()
                status = msg.split(" ")[0]
                counts[status] = counts.get(status, 0) + 1
                log.info(f"[{done_count}/{total}] {uid_out}: {msg}")

    log.info("=== Summary ===")
    log.info(f"  Searched successfully:  {counts.get('ok', 0):,}")
    log.info(f"  No AlphaFold model:    {counts.get('no_alphafold_model', 0):,}")
    log.info(f"  Search failed:         {counts.get('search_failed', 0):,}")
    log.info(f"  Already cached:        {counts.get('already_cached', 0):,}")
    log.info("Done. Re-run benchmark_rosetta_stone.py --multi-domain to evaluate.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Download AlphaFold PDBs and search missing benchmark proteins.")
    p.add_argument("--controls",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "benchmarks", "positives_and_negatives.tsv"))
    p.add_argument("--search-cache-dir",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "rosetta_searches"),
                   dest="search_cache_dir")
    p.add_argument("--zhang-db", required=True, dest="zhang_db",
                   help="Zhang sub-database to search against")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--batchsize", type=int, default=2097152)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
