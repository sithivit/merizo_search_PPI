#!/usr/bin/env python3
"""
Predict whether two proteins interact using Rosetta Stone paired-domain transfer.

Usage:
    python scripts/predict_ppi.py \\
        --pdb1 /path/to/proteinA.pdb \\
        --pdb2 /path/to/proteinB.pdb \\
        --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \\
        --template-index benchmark_cache/zhang_template_index.json

    python scripts/predict_ppi.py \\
        --uid1 P07948 --uid2 P12931 \\
        --search-cache benchmark_cache/searches \\
        --template-index benchmark_cache/zhang_template_index.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COL_TARGET = 2
COL_MAX_TM = 10


def run_merizo_search(pdb_path: str, zhang_db: str, topk: int,
                      batchsize: int) -> dict:
    """Run merizo search for one PDB. Returns {target_domain_name: max_tm}."""
    merizo = os.path.join(PROJECT_ROOT, "merizo_search", "merizo.py")
    with tempfile.TemporaryDirectory(prefix="predict_ppi_") as tmpdir:
        prefix = os.path.join(tmpdir, "search")
        tmp_sub = os.path.join(tmpdir, "tmp")
        os.makedirs(tmp_sub)
        cmd = [sys.executable, merizo, "search",
               pdb_path, zhang_db, prefix, tmp_sub,
               "--search_batchsize", str(batchsize),
               "--mintm", "0.0",
               "--topk", str(topk)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        tsv = prefix + "_search.tsv"
        if r.returncode != 0 or not os.path.exists(tsv):
            print(f"  [ERROR] merizo search failed:\n{r.stderr[-500:]}")
            return {}
        return _parse_tsv(tsv, query_hint=None)


def _parse_tsv(tsv_path: str, query_hint: Optional[str]) -> dict:
    best = {}
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= COL_MAX_TM:
                continue
            target = parts[COL_TARGET]
            if query_hint and f"AF-{query_hint}-" in target:
                continue
            try:
                tm = float(parts[COL_MAX_TM])
            except ValueError:
                continue
            if target not in best or tm > best[target]:
                best[target] = tm
    return best


def load_from_cache(uid: str, cache_dir: str) -> dict:
    tsv = os.path.join(cache_dir, f"{uid}_search.tsv")
    if not os.path.exists(tsv) or os.path.getsize(tsv) == 0:
        return {}
    return _parse_tsv(tsv, query_hint=uid)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def find_bridges(H1: dict, H2: dict, template_index: dict,
                 min_bridge_tm: float,
                 self_filter_tm: float = None) -> list:
    """
    Bridges where P1 or P2 already carries both domain types are discarded
    (self_filter_tm defaults to min_bridge_tm).
    """
    sft = min_bridge_tm if self_filter_tm is None else self_filter_tm
    bridges = []
    seen = set()

    for dom_a, tm_a in H1.items():
        if tm_a < min_bridge_tm:
            continue
        for dom_b in (template_index.get(dom_a) or []):
            tm_b = H2.get(dom_b)
            if tm_b is None or tm_b < min_bridge_tm:
                continue
            if H1.get(dom_b, -1.0) >= sft:
                continue
            if H2.get(dom_a, -1.0) >= sft:
                continue
            key = (dom_a, dom_b)
            if key not in seen:
                seen.add(key)
                bridges.append((tm_a, dom_a, dom_b, tm_b, min(tm_a, tm_b)))

    for dom_b, tm_b in H1.items():
        if tm_b < min_bridge_tm:
            continue
        for dom_a in (template_index.get(dom_b) or []):
            tm_a = H2.get(dom_a)
            if tm_a is None or tm_a < min_bridge_tm:
                continue
            if H1.get(dom_a, -1.0) >= sft:
                continue
            if H2.get(dom_b, -1.0) >= sft:
                continue
            key = (dom_b, dom_a)
            if key not in seen:
                seen.add(key)
                bridges.append((tm_b, dom_b, dom_a, tm_a, min(tm_b, tm_a)))

    bridges.sort(key=lambda x: x[4], reverse=True)
    return bridges


def print_report(name1: str, name2: str, H1: dict, H2: dict,
                 bridges: list, min_bridge_tm: float, top_n: int = 5):
    score = bridges[0][4] if bridges else 0.0

    print()
    print("=" * 65)
    print(f"  Rosetta Stone PPI Prediction")
    print("=" * 65)
    print(f"  Protein 1 : {name1}  ({len(H1)} template domain hits)")
    print(f"  Protein 2 : {name2}  ({len(H2)} template domain hits)")
    print(f"  Threshold : min_bridge_tm = {min_bridge_tm}")
    print()

    if not bridges:
        print("  SCORE: 0.0  →  No bridge found — cannot predict interaction")
        print()
        print("  This could mean:")
        print("  - Neither protein resembles any Zhang template domain above threshold")
        print("  - No TED fusion protein contains both an A-like and B-like domain")
        print("  - These proteins may not interact, or are outside the template coverage")
        return

    print(f"  SCORE: {score:.4f}  ({len(bridges)} bridges found)")
    print()

    # Rough confidence bracket
    if score >= 0.7:
        verdict = "HIGH confidence — strong structural evidence for interaction"
    elif score >= 0.5:
        verdict = "MODERATE confidence — reasonable structural evidence"
    else:
        verdict = "LOW confidence — weak structural evidence"
    print(f"  Verdict: {verdict}")
    print()
    print(f"  Top {min(top_n, len(bridges))} bridging template pairs:")
    print()
    print(f"  {'TM(P1)':>7}  {'Domain A (P1 resembles)':^45}  "
          f"{'Domain B (P2 resembles)':^45}  {'TM(P2)':>7}  {'Score':>7}")
    print("  " + "-" * 120)
    for tm_a, dom_a, dom_b, tm_b, s in bridges[:top_n]:
        marker = " ←" if s == score else ""
        print(f"  {tm_a:>7.4f}  {dom_a:^45}  {dom_b:^45}  {tm_b:>7.4f}  {s:>7.4f}{marker}")

    print()
    best = bridges[0]
    print("  How to read the best bridge:")
    print(f"  - {name1} structurally resembles  {best[1]}")
    print(f"  - {name2} structurally resembles  {best[2]}")
    fusion = best[1].split("-")[1] if "-" in best[1] else best[1]
    print(f"  - Both domains co-occur in fusion protein  {fusion}  in TED")
    print(f"  - Evidence: since {fusion} carries both domain types naturally fused,")
    print(f"    any protein resembling one domain is predicted to interact with")
    print(f"    any protein resembling the other.")
    print()


def main():
    p = argparse.ArgumentParser(
        description="Predict PPI between two proteins using Rosetta Stone "
                    "paired-domain transfer.")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pdb1", help="PDB file for protein 1 (runs merizo search)")
    grp.add_argument("--uid1", help="UniProt ID for protein 1 (uses search cache)")

    grp2 = p.add_mutually_exclusive_group(required=True)
    grp2.add_argument("--pdb2", help="PDB file for protein 2 (runs merizo search)")
    grp2.add_argument("--uid2", help="UniProt ID for protein 2 (uses search cache)")

    p.add_argument("--zhang-db",
                   default=os.path.join(PROJECT_ROOT, "zhang_pairlist_db",
                                        "zhang_pairlist_db", "ted_pairlist"),
                   dest="zhang_db",
                   help="Merizo binary Zhang sub-database")
    p.add_argument("--search-cache",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache", "searches"),
                   dest="search_cache",
                   help="Directory with pre-computed {uid}_search.tsv files")
    p.add_argument("--template-index",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "zhang_template_index.json"),
                   dest="template_index")
    p.add_argument("--min-bridge-tm", type=float, default=0.5,
                   dest="min_bridge_tm",
                   help="Minimum TM score required on both bridge sides (default 0.5)")
    p.add_argument("--self-filter-tm", type=float, default=None,
                   dest="self_filter_tm",
                   help="TM threshold for self-interaction filter (default: same as "
                        "--min-bridge-tm). Bridges where P1 or P2 already carries both "
                        "domain types are discarded.")
    p.add_argument("--topk", type=int, default=50,
                   help="Top-k hits to retrieve from merizo search")
    p.add_argument("--batchsize", type=int, default=2097152)
    p.add_argument("--top-bridges", type=int, default=5, dest="top_bridges")
    args = p.parse_args()

    if args.uid1 and args.pdb2:
        p.error("--uid1 must be paired with --uid2, not --pdb2")
    if args.pdb1 and args.uid2:
        p.error("--pdb1 must be paired with --pdb2, not --uid2")

    print(f"Loading template index from {args.template_index}...")
    with open(args.template_index) as fh:
        template_index = json.load(fh)
    print(f"  {len(template_index):,} domain entries loaded.")

    if args.uid1:
        name1, name2 = args.uid1, args.uid2
        print(f"\nLoading cached search results for {name1} and {name2}...")
        H1 = load_from_cache(args.uid1, args.search_cache)
        H2 = load_from_cache(args.uid2, args.search_cache)
        if not H1:
            print(f"  [ERROR] No cache found for {args.uid1}. "
                  f"Run rosetta_search_both_sides.py first.")
            sys.exit(1)
        if not H2:
            print(f"  [ERROR] No cache found for {args.uid2}. "
                  f"Run rosetta_search_both_sides.py first.")
            sys.exit(1)
    else:
        name1 = os.path.basename(args.pdb1)
        name2 = os.path.basename(args.pdb2)
        print(f"\nSearching {name1} against Zhang sub-DB...")
        H1 = run_merizo_search(args.pdb1, args.zhang_db, args.topk, args.batchsize)
        print(f"  {len(H1)} template domain hits.")
        print(f"Searching {name2} against Zhang sub-DB...")
        H2 = run_merizo_search(args.pdb2, args.zhang_db, args.topk, args.batchsize)
        print(f"  {len(H2)} template domain hits.")

    _sft = args.min_bridge_tm if args.self_filter_tm is None else args.self_filter_tm
    print(f"\nFinding bridges (min_bridge_tm={args.min_bridge_tm}, "
          f"self_filter_tm={_sft})...")
    bridges = find_bridges(H1, H2, template_index,
                           args.min_bridge_tm, args.self_filter_tm)
    print_report(name1, name2, H1, H2, bridges, args.min_bridge_tm, args.top_bridges)


if __name__ == "__main__":
    main()
