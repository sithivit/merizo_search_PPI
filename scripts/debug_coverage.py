#!/usr/bin/env python3
"""
debug_coverage.py

Diagnose why covered positives is stuck at 595/3000 despite Phase 2.

Checks each uncovered positive pair and classifies why it's missing:
  1. protein not in Phase 2 universe (no search file)
     a. no_domains_in_sqlite (not in TED pair list DB)
     b. extract_failed (in DB but structure extraction failed)
     c. never_searched (not processed at all)
  2. both proteins in universe but still not scoring
"""

import argparse
import os
import sqlite3
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def get_universe(search_cache_dir):
    proteins = set()
    for fname in os.listdir(search_cache_dir):
        if not fname.endswith("_search.tsv"):
            continue
        path = os.path.join(search_cache_dir, fname)
        stem = fname[:-len("_search.tsv")]
        if "_dom" in stem:
            pid = stem.rsplit("_dom", 1)[0]
        else:
            pid = stem
        if os.path.getsize(path) > 0:
            proteins.add(pid)
    return proteins


def get_all_proteins_in_positives(controls_path):
    pairs = []
    with open(controls_path) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, cat = line.split("\t")
            if cat == "positive":
                a, b = pair.split("_")
                pairs.append((a, b))
    return pairs


def check_sqlite(uid, filter_db):
    conn = sqlite3.connect(filter_db)
    cursor = conn.cursor()
    pattern = f"AF-{uid}-F1-model_v4_TED%"
    cursor.execute("SELECT domain_id FROM domains WHERE domain_id LIKE ? LIMIT 1", (pattern,))
    row = cursor.fetchone()
    conn.close()
    return row is not None


def has_empty_sentinel(uid, search_cache_dir):
    """Protein was processed but all its domain files are empty (extract/search failed)."""
    prefix = uid + "_dom"
    found_any = False
    all_empty = True
    for fname in os.listdir(search_cache_dir):
        if fname.startswith(prefix) and fname.endswith("_search.tsv"):
            found_any = True
            path = os.path.join(search_cache_dir, fname)
            if os.path.getsize(path) > 0:
                all_empty = False
    return found_any and all_empty


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controls",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "benchmarks", "positives_and_negatives.tsv"))
    p.add_argument("--search-cache-dir",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "rosetta_searches"),
                   dest="search_cache_dir")
    p.add_argument("--filter-db",
                   default=os.path.join(PROJECT_ROOT, "merizo_pairlist_db",
                                        "ted_pairlist_filters.db"),
                   dest="filter_db")
    p.add_argument("--show-missing", type=int, default=20, dest="show_missing",
                   help="Show this many example missing proteins")
    args = p.parse_args()

    print("Loading universe...")
    universe = get_universe(args.search_cache_dir)
    print(f"Universe: {len(universe):,} proteins with non-empty search results\n")

    print("Loading positive pairs...")
    pairs = get_all_proteins_in_positives(args.controls)
    print(f"Total positive pairs: {len(pairs):,}")

    # Classify each pair
    covered = []
    missing_pairs = []
    for a, b in pairs:
        if a in universe and b in universe:
            covered.append((a, b))
        else:
            missing_pairs.append((a, b, a in universe, b in universe))

    print(f"Covered: {len(covered):,}")
    print(f"Missing: {len(missing_pairs):,}\n")

    # Collect all missing proteins
    missing_proteins = set()
    for a, b, a_in, b_in in missing_pairs:
        if not a_in:
            missing_proteins.add(a)
        if not b_in:
            missing_proteins.add(b)

    print(f"Unique proteins causing missing pairs: {len(missing_proteins):,}\n")

    # Classify each missing protein
    cats = {"no_sqlite": [], "empty_sentinel": [], "never_processed": []}
    print("Classifying missing proteins (this may take a moment)...")

    # Build a set of all protein IDs that have ANY file (empty or not) in cache
    any_file = set()
    for fname in os.listdir(args.search_cache_dir):
        if fname.endswith("_search.tsv") and "_dom" in fname:
            pid = fname.rsplit("_dom", 1)[0]
            any_file.add(pid)

    for uid in sorted(missing_proteins):
        in_sqlite = check_sqlite(uid, args.filter_db)
        if not in_sqlite:
            cats["no_sqlite"].append(uid)
        elif uid in any_file:
            cats["empty_sentinel"].append(uid)
        else:
            cats["never_processed"].append(uid)

    print("\n=== Classification of missing proteins ===\n")
    print(f"  Not in TED SQLite DB (no AlphaFold model or single-domain): "
          f"{len(cats['no_sqlite']):,}")
    print(f"  In SQLite but all extractions/searches failed (empty sentinels): "
          f"{len(cats['empty_sentinel']):,}")
    print(f"  Never processed (bug / skipped): "
          f"{len(cats['never_processed']):,}")

    if cats["no_sqlite"]:
        print(f"\n  Example 'no_sqlite' proteins (first {args.show_missing}):")
        for uid in cats["no_sqlite"][:args.show_missing]:
            print(f"    {uid}")

    if cats["empty_sentinel"]:
        print(f"\n  Example 'empty_sentinel' proteins (first {args.show_missing}):")
        for uid in cats["empty_sentinel"][:args.show_missing]:
            print(f"    {uid}")

    if cats["never_processed"]:
        print(f"\n  Example 'never_processed' proteins (first {args.show_missing}):")
        for uid in cats["never_processed"][:args.show_missing]:
            print(f"    {uid}")

    print("\n=== Summary ===")
    print(f"  Covered pairs:        {len(covered):,} / {len(pairs):,} "
          f"({len(covered)/len(pairs)*100:.1f}%)")
    print(f"  Missing pairs:        {len(missing_pairs):,}")
    print(f"  Missing proteins:     {len(missing_proteins):,}")
    total_classified = sum(len(v) for v in cats.values())
    print(f"  → no_sqlite:          {len(cats['no_sqlite']):,} "
          f"({len(cats['no_sqlite'])/max(total_classified,1)*100:.1f}% of missing)")
    print(f"  → empty_sentinel:     {len(cats['empty_sentinel']):,} "
          f"({len(cats['empty_sentinel'])/max(total_classified,1)*100:.1f}% of missing)")
    print(f"  → never_processed:    {len(cats['never_processed']):,} "
          f"({len(cats['never_processed'])/max(total_classified,1)*100:.1f}% of missing)")


if __name__ == "__main__":
    main()
