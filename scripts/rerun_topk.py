#!/usr/bin/env python3
"""
rerun_topk.py

Re-run merizo search with a higher topk for a targeted set of proteins
(those involved in covered positive pairs), overwriting their cached TSVs.

This avoids re-running all 6,000+ searches — only the ~1,000 proteins
involved in positive pairs need higher topk to improve recall.

Usage:
    python scripts/rerun_topk.py --topk 5000 --workers 8
"""

import argparse
import csv
import logging
import mmap
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'merizo_search', 'programs'))
from Foldclass.dbutil import (
    read_dbinfo, retrieve_names_by_idx, retrieve_start_end_by_idx,
    retrieve_bytes, coord_conv,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CONTROLS     = os.path.join(PROJECT_ROOT, "benchmark_cache", "benchmarks",
                                    "positives_and_negatives.tsv")
DEFAULT_SEARCH_CACHE = os.path.join(PROJECT_ROOT, "benchmark_cache", "searches")
DEFAULT_PAIR_LIST    = "/mnt/bigstore/ted/pair_list_20250128"
DEFAULT_PAIRLIST_DB  = os.path.join(PROJECT_ROOT, "merizo_pairlist_db", "ted_pairlist")

_AF_RE = re.compile(r'AF-([^-]+)-F1-model_v4')

_AA_1TO3 = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
    'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
    'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL',
}


# ---------------------------------------------------------------------------
# Identify target proteins
# ---------------------------------------------------------------------------

def get_universe(search_cache_dir: str) -> set:
    proteins = set()
    for fname in os.listdir(search_cache_dir):
        if fname.endswith("_search.tsv"):
            path = os.path.join(search_cache_dir, fname)
            if os.path.getsize(path) > 0:
                proteins.add(fname[:-len("_search.tsv")])
    return proteins


def get_covered_positive_proteins(controls_path: str, universe: set) -> set:
    """Return unique proteins from covered positive pairs (both in universe)."""
    proteins = set()
    with open(controls_path) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, cat = line.split("\t")
            if cat != "positive":
                continue
            a, b = pair.split("_")
            if a in universe and b in universe:
                proteins.add(a)
                proteins.add(b)
    return proteins


# ---------------------------------------------------------------------------
# Load pair_list (streaming — only for target proteins)
# ---------------------------------------------------------------------------

def load_pair_list(pair_list_path: str, target_proteins: set) -> dict:
    pairs = {}
    with open(pair_list_path) as fh:
        for line in fh:
            if ":" not in line:
                continue
            parts = line.split()
            if not parts or ":" not in parts[0]:
                continue
            domain_a, domain_b = parts[0].split(":", 1)
            m = _AF_RE.match(domain_a)
            if not m:
                continue
            pid = m.group(1)
            if pid not in target_proteins or pid in pairs:
                continue
            pae = float(parts[2]) if len(parts) >= 3 else 0.0
            pairs[pid] = (domain_a, domain_b, pae)
            if len(pairs) == len(target_proteins):
                break
    log.info(f"Pair list: found {len(pairs)}/{len(target_proteins)} target proteins")
    return pairs


# ---------------------------------------------------------------------------
# Domain extraction
# ---------------------------------------------------------------------------

def build_domain_index(pairlist_db: str, target_names: set) -> dict:
    db_json = pairlist_db + ".json"
    dbinfo  = read_dbinfo(db_json)
    db_dir  = os.path.dirname(os.path.abspath(db_json))

    def resolve(key):
        p = dbinfo[key]
        return p if os.path.isabs(p) else os.path.join(db_dir, p)

    name_to_idx = {}

    if "INDEX_LIST_FILE" in dbinfo:
        index_list_path = os.path.join(db_dir, dbinfo["INDEX_LIST_FILE"])
        with open(index_list_path) as f:
            indices = [int(l.strip()) for l in f]
        with open(resolve("db_names_f"), "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            names = retrieve_names_by_idx(idx=indices, mm=mm)
        for name, idx in zip(names, indices):
            if name in target_names:
                name_to_idx[name] = idx
    else:
        ENTRY_SIZE = 33
        db_size = dbinfo["DB_SIZE"]
        with open(resolve("db_names_f"), "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for idx in range(db_size):
                mm.seek(idx * ENTRY_SIZE)
                raw  = mm.read(ENTRY_SIZE)
                name = raw.decode("utf-8", errors="ignore").strip("\x00").strip()
                if name in target_names:
                    name_to_idx[name] = idx
                    if len(name_to_idx) == len(target_names):
                        break

    log.info(f"Domain index: {len(name_to_idx)}/{len(target_names)} found")
    return name_to_idx


def extract_domain_pdb(domain_b: str, pairlist_db: str, output_pdb: str,
                       name_to_idx: dict) -> bool:
    idx = name_to_idx.get(domain_b)
    if idx is None:
        return False
    try:
        db_json = pairlist_db + ".json"
        dbinfo  = read_dbinfo(db_json)
        db_dir  = os.path.dirname(os.path.abspath(db_json))

        def resolve(key):
            p = dbinfo[key]
            return p if os.path.isabs(p) else os.path.join(db_dir, p)

        with open(resolve("sif"), "rb") as sif, open(resolve("sdf"), "rb") as sdf:
            si_mm = mmap.mmap(sif.fileno(), 0, access=mmap.ACCESS_READ)
            sd_mm = mmap.mmap(sdf.fileno(), 0, access=mmap.ACCESS_READ)
            se    = retrieve_start_end_by_idx(idx=[idx], mm=si_mm)
            seq   = retrieve_bytes(se[0][0], se[0][1], mm=sd_mm,
                                   typeconv=lambda x: x.decode("ascii"))

        with open(resolve("cif"), "rb") as cif, open(resolve("cdf"), "rb") as cdf:
            ci_mm = mmap.mmap(cif.fileno(), 0, access=mmap.ACCESS_READ)
            cd_mm = mmap.mmap(cdf.fileno(), 0, access=mmap.ACCESS_READ)
            ce    = retrieve_start_end_by_idx(idx=[idx], mm=ci_mm)
            coords = retrieve_bytes(ce[0][0], ce[0][1], mm=cd_mm, typeconv=coord_conv)

        with open(output_pdb, "w") as f:
            for i, (aa, coord) in enumerate(zip(seq, coords), start=1):
                resname = _AA_1TO3.get(aa, "UNK")
                f.write(f"ATOM  {i:>5}  CA  {resname:>3} A{i:>4}    "
                        f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
                        f"  1.00  0.00\n")
            f.write("END\n")
        return True
    except Exception as e:
        log.warning(f"Extract failed for {domain_b}: {e}")
        return False


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def run_search(pdb_path: str, pairlist_db: str, output_prefix: str,
               topk: int, batchsize: int) -> str:
    merizo = os.path.join(PROJECT_ROOT, "merizo_search", "merizo.py")
    tmp    = output_prefix + "_tmp"
    os.makedirs(tmp, exist_ok=True)
    import subprocess
    cmd = [
        sys.executable, merizo, "search",
        pdb_path, pairlist_db, output_prefix, tmp,
        "--search_batchsize", str(batchsize),
        "--mintm", "0.3",
        "--topk", str(topk),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        shutil.rmtree(tmp, ignore_errors=True)
        tsv = output_prefix + "_search.tsv"
        return tsv if r.returncode == 0 and os.path.exists(tsv) else None
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        log.warning(f"Search error: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-protein worker
# ---------------------------------------------------------------------------

def process_protein(pid, domain_b, pairlist_db, search_cache_dir,
                    topk, batchsize, name_to_idx):
    cache_tsv = os.path.join(search_cache_dir, f"{pid}_search.tsv")
    with tempfile.TemporaryDirectory(prefix="rerun_pdb_") as tmpdir:
        pdb = os.path.join(tmpdir, f"{pid}.pdb")
        if not extract_domain_pdb(domain_b, pairlist_db, pdb, name_to_idx):
            return pid, "extract_failed"
        prefix = os.path.join(tmpdir, pid)
        tsv = run_search(pdb, pairlist_db, prefix, topk, batchsize)
        if tsv is None:
            return pid, "search_failed"
        shutil.move(tsv, cache_tsv)
        n = sum(1 for _ in open(cache_tsv))
        return pid, f"ok ({n} hits)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    # 1. Target proteins — from covered positive pairs
    # If search cache exists, restrict to proteins already in the universe.
    # If cache is empty (e.g. after accidental deletion), use all positive-pair proteins.
    os.makedirs(args.search_cache_dir, exist_ok=True)
    universe = get_universe(args.search_cache_dir)

    if universe:
        log.info(f"Search cache found: {len(universe):,} proteins in universe")
        targets = get_covered_positive_proteins(args.controls, universe)
    else:
        log.info("Search cache is empty — loading all positive-pair proteins from controls")
        targets = set()
        with open(args.controls) as fh:
            next(fh)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                pair, cat = line.split("\t")
                if cat == "positive":
                    a, b = pair.split("_")
                    targets.add(a)
                    targets.add(b)

    log.info(f"Target proteins: {len(targets):,}")

    # 2. Load pair_list for domain info
    log.info("Loading pair_list (streaming)...")
    pair_list = load_pair_list(args.pair_list, targets)

    searchable = [pid for pid in targets if pid in pair_list]
    log.info(f"Searchable targets: {len(searchable):,}")

    # 3. Build domain index once
    needed_domains = {pair_list[pid][1] for pid in searchable}
    log.info(f"Building domain index for {len(needed_domains):,} domains...")
    name_to_idx = build_domain_index(args.pairlist_db, needed_domains)

    # 4. Re-run searches
    total = len(searchable)
    log.info(f"Re-running {total} searches with topk={args.topk}, workers={args.workers}...")

    def submit(pid):
        _, domain_b, _ = pair_list[pid]
        return process_protein(pid, domain_b, args.pairlist_db,
                               args.search_cache_dir, args.topk,
                               args.batchsize, name_to_idx)

    if args.workers == 1:
        for i, pid in enumerate(searchable, 1):
            pid_out, msg = submit(pid)
            log.info(f"[{i}/{total}] {pid_out}: {msg}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(submit, pid): pid for pid in searchable}
            done = 0
            for future in as_completed(futures):
                done += 1
                pid_out, msg = future.result()
                log.info(f"[{done}/{total}] {pid_out}: {msg}")

    log.info("Done. Re-run benchmark_ppi_v2.py to get updated metrics.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-run merizo search with higher topk for covered positive-pair proteins only."
    )
    p.add_argument("--controls",        default=DEFAULT_CONTROLS)
    p.add_argument("--search-cache-dir", default=DEFAULT_SEARCH_CACHE, dest="search_cache_dir")
    p.add_argument("--pair-list",        default=DEFAULT_PAIR_LIST,    dest="pair_list")
    p.add_argument("--pairlist-db",      default=DEFAULT_PAIRLIST_DB,  dest="pairlist_db")
    p.add_argument("--topk",    type=int, default=5000,
                   help="New topk for searches (default: 5000)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers (default: 4)")
    p.add_argument("--batchsize", type=int, default=2097152, dest="batchsize")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
