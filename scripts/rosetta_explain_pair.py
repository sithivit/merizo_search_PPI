#!/usr/bin/env python3
"""
rosetta_explain_pair.py

Trace the Rosetta Stone scoring step-by-step for one protein pair.

For the given P1 and P2, this script:
  1. Shows the top search hits for each protein (their closest Zhang template domains)
  2. Finds every bridging template pair (A in H[P1], B in H[P2] or vice versa)
  3. Prints each bridge with its two TM scores and the min() score
  4. Reports the final score = max over all bridges

This confirms the algorithm is exactly what the professor described:
  "A domain-domain association (A,B) observed in a fused protein transfers
   evidence that any protein with an A-like domain interacts with any protein
   with a B-like domain."

Usage:
    python scripts/rosetta_explain_pair.py \\
        --p1 P13501 --p2 P51681 \\
        --search-cache-dir benchmark_cache/searches \\
        --template-index benchmark_cache/zhang_template_index.json

    # To find a good example pair (top-scoring positives in the cache):
    python scripts/rosetta_explain_pair.py \\
        --find-examples \\
        --controls benchmark_cache/benchmarks/positives_and_negatives.tsv \\
        --search-cache-dir benchmark_cache/searches \\
        --template-index benchmark_cache/zhang_template_index.json \\
        --min-bridge-tm 0.5
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COL_TARGET = 2
COL_MAX_TM = 10


def parse_search_tsv(tsv_path: str, query_protein: str) -> dict:
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


def load_hits(pid: str, cache_dir: str) -> dict:
    tsv = os.path.join(cache_dir, f"{pid}_search.tsv")
    if not os.path.exists(tsv) or os.path.getsize(tsv) == 0:
        return {}
    return parse_search_tsv(tsv, pid)


def find_bridges(H1: dict, H2: dict, template_index: dict,
                 min_bridge_tm: float = 0.0) -> list:
    """
    Return all (tm_a, domain_a, domain_b, tm_b, score, direction) tuples,
    sorted by score descending.
    direction = "forward" means A in H[P1], B in H[P2].
    """
    bridges = []
    seen = set()

    # Forward: A in H1, B in H2
    for dom_a, tm_a in H1.items():
        if tm_a < min_bridge_tm:
            continue
        partners = template_index.get(dom_a)
        if not partners:
            continue
        for dom_b in partners:
            tm_b = H2.get(dom_b)
            if tm_b is not None and tm_b >= min_bridge_tm:
                key = (dom_a, dom_b)
                if key not in seen:
                    seen.add(key)
                    bridges.append((tm_a, dom_a, dom_b, tm_b,
                                    min(tm_a, tm_b), "forward"))

    # Reverse: B in H1, A in H2
    for dom_b, tm_b in H1.items():
        if tm_b < min_bridge_tm:
            continue
        partners = template_index.get(dom_b)
        if not partners:
            continue
        for dom_a in partners:
            tm_a = H2.get(dom_a)
            if tm_a is not None and tm_a >= min_bridge_tm:
                key = (dom_b, dom_a)
                if key not in seen:
                    seen.add(key)
                    bridges.append((tm_b, dom_b, dom_a, tm_a,
                                    min(tm_b, tm_a), "reverse"))

    bridges.sort(key=lambda x: x[4], reverse=True)
    return bridges


def explain_pair(p1: str, p2: str, cache_dir: str, template_index: dict,
                 min_bridge_tm: float, top_hits: int, top_bridges: int):
    H1 = load_hits(p1, cache_dir)
    H2 = load_hits(p2, cache_dir)

    print("=" * 70)
    print(f"Rosetta Stone trace: {p1}  ×  {p2}")
    print("=" * 70)

    if not H1:
        print(f"ERROR: No search results found for {p1} in cache.")
        return
    if not H2:
        print(f"ERROR: No search results found for {p2} in cache.")
        return

    print(f"\n--- Step 1: Top {top_hits} template domain hits ---")
    print(f"\nP1 = {p1}  ({len(H1)} total hits)")
    for dom, tm in sorted(H1.items(), key=lambda x: -x[1])[:top_hits]:
        in_idx = "✓ in template index" if dom in template_index else "  (not in index)"
        print(f"  TM={tm:.4f}  {dom}  {in_idx}")

    print(f"\nP2 = {p2}  ({len(H2)} total hits)")
    for dom, tm in sorted(H2.items(), key=lambda x: -x[1])[:top_hits]:
        in_idx = "✓ in template index" if dom in template_index else "  (not in index)"
        print(f"  TM={tm:.4f}  {dom}  {in_idx}")

    print(f"\n--- Step 2: Bridge search (min_bridge_tm={min_bridge_tm}) ---")
    print("Looking for template pairs (A, B) where:")
    print("  A ∈ hits(P1)  AND  B ∈ hits(P2)   [or vice versa]")
    print("  Both TM scores ≥", min_bridge_tm)
    print("  Evidence: A and B co-occur in a TED fusion protein\n")

    bridges = find_bridges(H1, H2, template_index, min_bridge_tm)

    if not bridges:
        print("  No bridges found — score = 0.0")
        return

    best_score = bridges[0][4]

    print(f"  Found {len(bridges)} bridge(s). Top {min(top_bridges, len(bridges))}:\n")
    print(f"  {'TM(P1→A)':>10}  {'Domain A':^50}  {'Domain B':^50}  {'TM(P2→B)':>10}  {'min()':>7}  Dir")
    print("  " + "-" * 140)
    for tm_a, dom_a, dom_b, tm_b, score, direction in bridges[:top_bridges]:
        marker = " ← BEST" if score == best_score else ""
        print(f"  {tm_a:>10.4f}  {dom_a:^50}  {dom_b:^50}  {tm_b:>10.4f}  {score:>7.4f}  {direction}{marker}")

    print(f"\n--- Step 3: Final score ---")
    print(f"  score({p1}, {p2}) = max over all bridges = {best_score:.4f}")
    print()
    print("Interpretation:")
    print(f"  P1 ({p1}) structurally resembles domain  {bridges[0][1]}")
    print(f"  P2 ({p2}) structurally resembles domain  {bridges[0][2]}")
    print(f"  Those two domains co-occur in the same TED fusion protein →")
    print(f"  evidence that P1 and P2 interact.")


def find_examples(controls: str, cache_dir: str, template_index: dict,
                  min_bridge_tm: float, n: int = 10):
    """Score all covered positives and show the top-scoring ones."""
    proteins_in_cache = set()
    for fname in os.listdir(cache_dir):
        if fname.endswith("_search.tsv") and os.path.getsize(
                os.path.join(cache_dir, fname)) > 0:
            proteins_in_cache.add(fname[:-len("_search.tsv")])

    results = []
    with open(controls) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, cat = line.split("\t")
            if cat != "positive":
                continue
            p1, p2 = pair.split("_")
            if p1 not in proteins_in_cache or p2 not in proteins_in_cache:
                continue
            H1 = load_hits(p1, cache_dir)
            H2 = load_hits(p2, cache_dir)
            bridges = find_bridges(H1, H2, template_index, min_bridge_tm)
            score = bridges[0][4] if bridges else 0.0
            results.append((score, p1, p2, len(bridges)))

    results.sort(reverse=True)
    print(f"Top {n} scoring positive pairs (min_bridge_tm={min_bridge_tm}):\n")
    print(f"  {'Score':>7}  {'P1':>12}  {'P2':>12}  {'#bridges':>10}")
    print("  " + "-" * 50)
    for score, p1, p2, nb in results[:n]:
        print(f"  {score:>7.4f}  {p1:>12}  {p2:>12}  {nb:>10}")
    print(f"\nRun with --p1 <P1> --p2 <P2> to trace the top pair.")


def main():
    p = argparse.ArgumentParser(
        description="Trace Rosetta Stone scoring for a specific protein pair.")
    p.add_argument("--p1", help="UniProt accession for protein 1")
    p.add_argument("--p2", help="UniProt accession for protein 2")
    p.add_argument("--find-examples", action="store_true", dest="find_examples",
                   help="List top-scoring positive pairs in the cache")
    p.add_argument("--controls",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "benchmarks", "positives_and_negatives.tsv"))
    p.add_argument("--search-cache-dir",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache", "searches"),
                   dest="search_cache_dir")
    p.add_argument("--template-index",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "zhang_template_index.json"),
                   dest="template_index")
    p.add_argument("--min-bridge-tm", type=float, default=0.0,
                   dest="min_bridge_tm")
    p.add_argument("--top-hits", type=int, default=10, dest="top_hits",
                   help="Number of top search hits to display per protein")
    p.add_argument("--top-bridges", type=int, default=10, dest="top_bridges",
                   help="Number of top bridges to display")
    args = p.parse_args()

    with open(args.template_index) as fh:
        template_index = json.load(fh)

    if args.find_examples:
        find_examples(args.controls, args.search_cache_dir,
                      template_index, args.min_bridge_tm)
    elif args.p1 and args.p2:
        explain_pair(args.p1, args.p2, args.search_cache_dir,
                     template_index, args.min_bridge_tm,
                     args.top_hits, args.top_bridges)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
