#!/usr/bin/env python3
"""
Build a merizo search sub-database restricted to Zhang et al. benchmark proteins.

Restricting to ~500-800 Zhang proteins ensures every topk slot is a verifiable
hit, maximising recall within the benchmark universe.

Usage:
    python scripts/build_zhang_subdb.py \\
        --controls benchmark_cache/benchmarks/positives_and_negatives.tsv \\
        --pair-list /mnt/bigstore/ted/pair_list_20250128 \\
        --source-db /mnt/bigstore/foldclass-db-ted/ted_365M.json \\
        --output-dir zhang_pairlist_db/
"""

import argparse
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transform_pairlist_to_database

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CONTROLS  = os.path.join(PROJECT_ROOT, "benchmark_cache", "benchmarks",
                                 "positives_and_negatives.tsv")
DEFAULT_PAIR_LIST = "/mnt/bigstore/ted/pair_list_20250128"
DEFAULT_SOURCE_DB = "/mnt/bigstore/foldclass-db-ted/ted_365M.json"
DEFAULT_OUTPUT    = os.path.join(PROJECT_ROOT, "zhang_pairlist_db")

_AF_RE = re.compile(r'AF-([^-]+)-F1-model_v4')


def extract_zhang_proteins(controls_path: str) -> set:
    proteins = set()
    with open(controls_path) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, cat = line.split("\t")
            if cat == "positive":
                a, b = pair.split("_")
                proteins.add(a)
                proteins.add(b)
    return proteins


def filter_pair_list(pair_list_path: str, zhang_proteins: set,
                     output_path: str) -> int:
    """Filter pair_list to entries whose Domain A protein is in zhang_proteins. Returns line count."""
    written = 0
    with open(pair_list_path) as src, open(output_path, "w") as dst:
        for line in src:
            parts = line.split()
            if not parts or ":" not in parts[0]:
                continue
            domain_a = parts[0].split(":")[0]
            m = _AF_RE.match(domain_a)
            if not m:
                continue
            if m.group(1) in zhang_proteins:
                dst.write(line)
                written += 1
    return written


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Extracting Zhang protein IDs from controls file...")
    zhang_proteins = extract_zhang_proteins(args.controls)
    log.info(f"  Unique proteins in Zhang positives: {len(zhang_proteins):,}")

    filtered_pair_list = os.path.join(args.output_dir, "zhang_pair_list.txt")
    log.info(f"Filtering pair_list to Zhang proteins → {filtered_pair_list}")
    n_lines = filter_pair_list(args.pair_list, zhang_proteins, filtered_pair_list)
    log.info(f"  Wrote {n_lines:,} pair_list entries")

    if n_lines == 0:
        log.error("No entries written — check that pair_list protein IDs match Zhang controls.")
        sys.exit(1)

    subdb_dir = os.path.join(args.output_dir, "zhang_pairlist_db")
    log.info(f"Building Zhang sub-database → {subdb_dir}")
    transform_pairlist_to_database.main(
        filtered_pair_list,
        args.source_db,
        subdb_dir,
    )

    db_path = os.path.join(subdb_dir, "ted_pairlist")
    log.info("")
    log.info("Done. Pass the following to rerun_topk.py:")
    log.info(f"  --zhang-db {db_path}")
    log.info("  (topk will be set automatically to cover the full Zhang universe)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a merizo search sub-database restricted to Zhang et al. proteins."
    )
    p.add_argument("--controls",   default=DEFAULT_CONTROLS,
                   help=f"positives_and_negatives.tsv (default: {DEFAULT_CONTROLS})")
    p.add_argument("--pair-list",  default=DEFAULT_PAIR_LIST, dest="pair_list",
                   help=f"Full TED pair_list file (default: {DEFAULT_PAIR_LIST})")
    p.add_argument("--source-db",  default=DEFAULT_SOURCE_DB, dest="source_db",
                   help=f"Source TED database JSON (default: {DEFAULT_SOURCE_DB})")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT, dest="output_dir",
                   help=f"Output directory (default: {DEFAULT_OUTPUT})")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
