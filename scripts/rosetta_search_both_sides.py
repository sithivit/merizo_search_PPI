#!/usr/bin/env python3
"""
Search all TED domains of all Zhang benchmark proteins against the Zhang sub-database.

Results are cached as {output_dir}/{P}_dom{N:02d}_search.tsv. Failed extractions
write empty sentinel files so re-runs skip them.

Usage:
    python scripts/rosetta_search_both_sides.py \\
        --controls benchmark_cache/benchmarks/positives_and_negatives.tsv \\
        --filter-db merizo_pairlist_db/ted_pairlist_filters.db \\
        --pairlist-db merizo_pairlist_db/ted_pairlist \\
        --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \\
        --output-dir benchmark_cache/rosetta_searches \\
        --workers 8
"""

import argparse
import logging
import mmap
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
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

_AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def is_canonical_uniprot(uid: str) -> bool:
    """True only for canonical Swiss-Prot accessions (reviewed; have AlphaFold structures in TED)."""
    import re
    return bool(re.fullmatch(
        r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]',
        uid
    ))


def get_all_benchmark_proteins(controls_path: str) -> set:
    """All unique canonical proteins from both sides of all Zhang pairs."""
    proteins = set()
    skipped = 0
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
                else:
                    skipped += 1
    if skipped:
        log.info(f"Skipped {skipped:,} non-canonical accessions (no AlphaFold in TED)")
    return proteins


def get_protein_domains(uid: str, filter_db_path: str) -> list:
    """Return [(domain_id, domain_idx), ...] for all TED domains of a protein."""
    conn = sqlite3.connect(filter_db_path)
    cursor = conn.cursor()
    pattern = f"AF-{uid}-F1-model_v4_TED%"
    cursor.execute(
        "SELECT domain_id, domain_idx FROM domains WHERE domain_id LIKE ?",
        (pattern,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows  # [(domain_id_str, int_idx), ...]


def extract_domain_pdb(domain_idx: int, pairlist_db: str, output_pdb: str) -> bool:
    try:
        db_json = pairlist_db + ".json"
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


def run_search(pdb_path: str, target_db: str, output_prefix: str,
               topk: int, batchsize: int) -> str:
    merizo = os.path.join(PROJECT_ROOT, "merizo_search", "merizo.py")
    tmp = output_prefix + "_tmp"
    os.makedirs(tmp, exist_ok=True)
    cmd = [sys.executable, merizo, "search",
           pdb_path, target_db, output_prefix, tmp,
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


def process_protein(uid: str, filter_db: str, pairlist_db: str,
                    zhang_db: str, output_dir: str,
                    topk: int, batchsize: int) -> tuple:
    """Extract and search all TED domains for one protein."""
    domains = get_protein_domains(uid, filter_db)
    if not domains:
        return uid, "no_domains_in_sqlite"

    status_parts = []
    with tempfile.TemporaryDirectory(prefix="rsearch_") as tmpdir:
        for n, (domain_id, domain_idx) in enumerate(sorted(domains)):
            cache_tsv = os.path.join(output_dir, f"{uid}_dom{n:02d}_search.tsv")
            if os.path.exists(cache_tsv):
                status_parts.append(f"dom{n:02d}:already_cached")
                continue

            pdb = os.path.join(tmpdir, f"{uid}_dom{n:02d}.pdb")
            if not extract_domain_pdb(domain_idx, pairlist_db, pdb):
                open(cache_tsv, "w").close()  # empty sentinel to skip retries
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

    return uid, " | ".join(status_parts)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Collecting all unique canonical proteins from Zhang benchmark (both sides)...")
    all_proteins = get_all_benchmark_proteins(args.controls)
    log.info(f"Total canonical proteins: {len(all_proteins):,}")

    done = {fname.split("_dom")[0]
            for fname in os.listdir(args.output_dir)
            if fname.endswith("_search.tsv")
            and os.path.getsize(os.path.join(args.output_dir, fname)) > 0}
    todo = sorted(all_proteins - done)
    log.info(f"Already done (have results): {len(done):,}, remaining: {len(todo):,}")

    total = len(todo)
    log.info(f"Running with {args.workers} workers, topk={args.topk}...")

    def submit(uid):
        return process_protein(uid, args.filter_db, args.pairlist_db,
                               args.zhang_db, args.output_dir,
                               args.topk, args.batchsize)

    if args.workers == 1:
        for i, uid in enumerate(todo, 1):
            uid_out, msg = submit(uid)
            log.info(f"[{i}/{total}] {uid_out}: {msg}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(submit, uid): uid for uid in todo}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                uid_out, msg = future.result()
                log.info(f"[{done_count}/{total}] {uid_out}: {msg}")

    log.info("Done. Run benchmark_rosetta_stone.py --multi-domain to evaluate.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Rosetta Stone: search all TED domains of all benchmark proteins.")
    p.add_argument("--controls",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "benchmarks", "positives_and_negatives.tsv"))
    p.add_argument("--filter-db", required=True, dest="filter_db",
                   help="SQLite metadata DB: ted_pairlist_filters.db")
    p.add_argument("--pairlist-db", required=True, dest="pairlist_db",
                   help="Foldclass binary DB for domain PDB extraction")
    p.add_argument("--zhang-db", required=True, dest="zhang_db",
                   help="Zhang sub-database to search against")
    p.add_argument("--output-dir",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "rosetta_searches"),
                   dest="output_dir")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batchsize", type=int, default=2097152)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
