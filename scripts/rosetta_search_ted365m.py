#!/usr/bin/env python3
"""
rosetta_search_ted365m.py

Extend Phase 2 coverage to single-domain benchmark proteins by querying
the full ted_365M database instead of the filtered ted_pairlist subset.

The ted_pairlist_filters.db only indexes multi-domain proteins (those that
appear in at least one intra-protein domain-domain pair). Single-domain
proteins are absent. The full ted_365M database contains ALL TED-segmented
AlphaFold domains, including single-domain proteins.

This script:
  1. Determines which benchmark proteins are still missing from the cache
  2. Scans the ted_365M names file ONCE to find domain indices for all
     missing proteins (33-byte fixed-width entries, linear scan)
  3. Extracts each domain PDB from ted_365M using its integer index
  4. Runs merizo search against the Zhang sub-DB
  5. Saves results as: {output_dir}/{uid}_dom{NN}_search.tsv
     (dom-suffixed to stay compatible with --multi-domain benchmark mode)

Usage:
    python scripts/rosetta_search_ted365m.py \\
        --ted365m-db /mnt/bigstore/foldclass-db-ted/ted_365M.json \\
        --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \\
        --workers 6
"""

import argparse
import logging
import mmap
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'merizo_search', 'programs'))
from Foldclass.dbutil import (
    read_dbinfo, retrieve_start_end_by_idx, retrieve_bytes, coord_conv,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ENTRY_SIZE = 33  # fixed-width bytes per name entry in the TED names file

_AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


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
    """Proteins with at least one non-empty *_dom*_search.tsv in the cache."""
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
    """True if an empty sentinel already exists for dom00 — means no model found."""
    p = os.path.join(search_cache_dir, f"{uid}_dom00_search.tsv")
    return os.path.exists(p) and os.path.getsize(p) == 0


def scan_ted365m_for_proteins(target_proteins: set, db_json: str) -> dict:
    """
    Single linear scan of the ted_365M names file.

    Domain names follow: AF-{uid}-F1-model_v4_TED{NN}
    uid starts at byte 3, length 6. We extract it from each 33-byte entry
    and check against target_proteins.

    Returns: {uid: [(domain_name, domain_idx), ...]}
    """
    dbinfo = read_dbinfo(db_json)
    db_dir = os.path.dirname(os.path.abspath(db_json))

    def resolve(key):
        p = dbinfo[key]
        return p if os.path.isabs(p) else os.path.join(db_dir, p)

    names_path = resolve("db_names_f")
    db_size = dbinfo.get("DB_SIZE", 0)

    log.info(f"Scanning {db_size:,} entries in ted_365M names file...")
    log.info(f"Looking for {len(target_proteins):,} proteins...")
    start = time.time()

    found = {uid: [] for uid in target_proteins}
    n_found_proteins = 0

    with open(names_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for idx in range(db_size):
            raw = mm[idx * ENTRY_SIZE: idx * ENTRY_SIZE + ENTRY_SIZE]
            name = raw.decode("utf-8", errors="ignore").rstrip("\x00").strip()
            # Domain name format: AF-XXXXXX-F1-model_v4_TED01
            # UniProt ID is at position 3..9
            if len(name) >= 9 and name.startswith("AF-"):
                uid = name[3:9]
                if uid in found:
                    if not found[uid]:  # first domain for this protein
                        n_found_proteins += 1
                    found[uid].append((name, idx))

            if idx % 10_000_000 == 0 and idx > 0:
                elapsed = time.time() - start
                pct = idx / db_size * 100
                log.info(f"  {idx/1e6:.0f}M / {db_size/1e6:.0f}M ({pct:.0f}%) "
                         f"— {n_found_proteins} proteins found so far "
                         f"({elapsed:.0f}s elapsed)")

        mm.close()

    elapsed = time.time() - start
    log.info(f"Scan complete in {elapsed:.0f}s. "
             f"Found domains for {n_found_proteins:,} / {len(target_proteins):,} proteins.")

    # Remove proteins with no domains found
    return {uid: domains for uid, domains in found.items() if domains}


def extract_domain_pdb(domain_idx: int, db_json: str, output_pdb: str) -> bool:
    try:
        dbinfo = read_dbinfo(db_json)
        db_dir = os.path.dirname(os.path.abspath(db_json))

        def resolve(key):
            p = dbinfo[key]
            return p if os.path.isabs(p) else os.path.join(db_dir, p)

        with open(resolve("sif"), "rb") as sif, open(resolve("sdf"), "rb") as sdf:
            si_mm = mmap.mmap(sif.fileno(), 0, access=mmap.ACCESS_READ)
            sd_mm = mmap.mmap(sdf.fileno(), 0, access=mmap.ACCESS_READ)
            se = retrieve_start_end_by_idx(idx=[domain_idx], mm=si_mm)
            seq = retrieve_bytes(se[0][0], se[0][1], mm=sd_mm,
                                 typeconv=lambda x: x.decode("ascii"))

        with open(resolve("cif"), "rb") as cif, open(resolve("cdf"), "rb") as cdf:
            ci_mm = mmap.mmap(cif.fileno(), 0, access=mmap.ACCESS_READ)
            cd_mm = mmap.mmap(cdf.fileno(), 0, access=mmap.ACCESS_READ)
            ce = retrieve_start_end_by_idx(idx=[domain_idx], mm=ci_mm)
            coords = retrieve_bytes(ce[0][0], ce[0][1], mm=cd_mm,
                                    typeconv=coord_conv)

        with open(output_pdb, "w") as f:
            for i, (aa, coord) in enumerate(zip(seq, coords), start=1):
                resname = _AA_1TO3.get(aa, "UNK")
                f.write(f"ATOM  {i:>5}  CA  {resname:>3} A{i:>4}    "
                        f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
                        f"  1.00  0.00\n")
            f.write("END\n")
        return True
    except Exception as e:
        log.warning(f"Extract failed (idx={domain_idx}): {e}")
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


def process_protein(uid: str, domains: list, db_json: str, zhang_db: str,
                    search_cache_dir: str, topk: int, batchsize: int) -> tuple:
    status_parts = []
    with tempfile.TemporaryDirectory(prefix="ted365m_") as tmpdir:
        for n, (domain_name, domain_idx) in enumerate(sorted(domains)):
            cache_tsv = os.path.join(search_cache_dir, f"{uid}_dom{n:02d}_search.tsv")
            if os.path.exists(cache_tsv):
                status_parts.append(f"dom{n:02d}:already_cached")
                continue

            pdb = os.path.join(tmpdir, f"{uid}_dom{n:02d}.pdb")
            if not extract_domain_pdb(domain_idx, db_json, pdb):
                open(cache_tsv, "w").close()
                status_parts.append(f"dom{n:02d}:extract_failed")
                continue

            prefix = os.path.join(tmpdir, f"{uid}_dom{n:02d}")
            tsv = run_search(pdb, zhang_db, prefix, topk, batchsize)
            if tsv:
                shutil.move(tsv, cache_tsv)
                n_hits = sum(1 for _ in open(cache_tsv))
                status_parts.append(f"dom{n:02d}:ok({n_hits}hits)")
            else:
                open(cache_tsv, "w").close()
                status_parts.append(f"dom{n:02d}:search_failed")

    return uid, " | ".join(status_parts) if status_parts else "no_domains_processed"


def run(args):
    os.makedirs(args.search_cache_dir, exist_ok=True)

    log.info("Loading benchmark proteins...")
    all_proteins = get_all_benchmark_proteins(args.controls)
    log.info(f"Total canonical proteins in benchmark: {len(all_proteins):,}")

    log.info("Loading current search universe...")
    universe = get_universe(args.search_cache_dir)
    log.info(f"Already in universe: {len(universe):,}")

    # Missing = not in universe AND not already marked as having no domains
    missing = sorted(
        uid for uid in all_proteins - universe
        if not has_sentinel(uid, args.search_cache_dir)
    )
    log.info(f"Proteins still to process: {len(missing):,}")

    if not missing:
        log.info("Nothing to do.")
        return

    # Single scan of ted_365M to find all domain indices at once
    domain_map = scan_ted365m_for_proteins(set(missing), args.ted365m_db)
    log.info(f"Domains found for {len(domain_map):,} proteins in ted_365M")

    # Write sentinels for proteins not found in ted_365M
    not_in_ted = [uid for uid in missing if uid not in domain_map]
    log.info(f"Not found in ted_365M (writing sentinels): {len(not_in_ted):,}")
    for uid in not_in_ted:
        open(os.path.join(args.search_cache_dir, f"{uid}_dom00_search.tsv"), "w").close()

    todo = sorted(domain_map.items())  # [(uid, [(name, idx), ...]), ...]
    total = len(todo)
    log.info(f"Running searches for {total:,} proteins with {args.workers} workers...")

    def submit(item):
        uid, domains = item
        return process_protein(uid, domains, args.ted365m_db, args.zhang_db,
                               args.search_cache_dir, args.topk, args.batchsize)

    if args.workers == 1:
        for i, item in enumerate(todo, 1):
            uid_out, msg = submit(item)
            log.info(f"[{i}/{total}] {uid_out}: {msg}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(submit, item): item[0] for item in todo}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                uid_out, msg = future.result()
                log.info(f"[{done_count}/{total}] {uid_out}: {msg}")

    log.info("Done. Re-run benchmark_rosetta_stone.py --multi-domain to evaluate.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Search missing benchmark proteins using the full ted_365M database.")
    p.add_argument("--ted365m-db",
                   default="/mnt/bigstore/foldclass-db-ted/ted_365M.json",
                   dest="ted365m_db",
                   help="Path to ted_365M.json (full TED database)")
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
