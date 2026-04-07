#!/usr/bin/env python3
"""
find_best_example.py

Find the best example protein to showcase the PPI prediction pipeline in a report.

Criteria for a good example:
  1. Both proteins in the positive pair are in the search cache (covered)
  2. The true positive partner appears in the query protein's search results
     (i.e., the search actually found the true interactor — a demonstrable hit)
  3. The TM-score for that true hit is high (>= 0.5 ideally)
  4. The query protein has a moderate hit count (10-300) — not trivial, not noisy
  5. The query protein is well-known (filter by a curated list of famous proteins)

Usage:
    python scripts/find_best_example.py
    python scripts/find_best_example.py --top 20 --min-tm 0.4
"""

import argparse
import os
import re
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CONTROLS     = os.path.join(PROJECT_ROOT, "benchmark_cache", "benchmarks",
                                    "positives_and_negatives.tsv")
DEFAULT_SEARCH_CACHE = os.path.join(PROJECT_ROOT, "benchmark_cache", "searches")

_AF_RE     = re.compile(r'AF-([^-]+)-F1-model_v4')
COL_TARGET = 2
COL_MAX_TM = 10

# Well-known human proteins — good for a report narrative.
# Expand this list freely; proteins NOT in this set are still shown but ranked lower.
FAMOUS_PROTEINS = {
    # Cell cycle / cancer
    "P04637",  # TP53
    "P38936",  # CDKN1A (p21)
    "P24941",  # CDK2
    "P06493",  # CDK1
    "P11802",  # CDK4
    "P14635",  # CCNB1 (Cyclin B1)
    "P22674",  # CCND1
    "P30279",  # CCND2 -- in logs!
    "P51959",  # CCNG1
    "Q00534",  # CDK6
    # DNA replication / repair
    "P12004",  # PCNA
    "P28340",  # POLD1
    "P26358",  # DNMT1
    # Kinase signalling
    "P00533",  # EGFR
    "P06241",  # FYN  -- in logs!
    "P12931",  # SRC   -- in logs!
    "P43403",  # ZAP70
    "P27361",  # MAPK3 (ERK1) -- in logs!
    "P28482",  # MAPK1 (ERK2)
    "Q02750",  # MAP2K1 (MEK1) -- in logs!
    "P45983",  # MAPK8 (JNK1) -- in logs!
    "P15056",  # BRAF
    "P15336",  # ATF2
    # Ubiquitin / proteasome
    "P0CG48",  # UBC (ubiquitin) -- in logs!
    "P62333",  # PSMD10 -- in logs!
    "P17980",  # PSMC3 -- in logs!
    # Chaperones
    "P07900",  # HSP90AA1
    "P08238",  # HSP90AB1
    "P11142",  # HSPA8 -- in logs!
    "P10809",  # HSPD1 (HSP60) -- in logs!
    # Transcription
    "P19838",  # NF1
    "P04792",  # HSPB1
    # Apoptosis
    "P55211",  # CASP9 -- in logs!
    "P55210",  # CASP7
    "P10415",  # BCL2
    "Q07812",  # BAX
    # GTPases / signalling adaptors
    "P01112",  # HRAS
    "P62993",  # GRB2 -- in logs!
    "P09619",  # PDGFRB -- in logs!
    # RNA / splicing
    "Q07955",  # SRSF1 -- in logs!
    "P38919",  # EIF4A3 -- in logs!
    # Adhesion / cytoskeleton
    "P35613",  # BSG (Basigin/CD147) -- in logs!
    "P49023",  # PXN (Paxillin) -- in logs!
}


def load_positives(controls_path):
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


def get_cached_proteins(search_cache_dir):
    proteins = set()
    for fname in os.listdir(search_cache_dir):
        if fname.endswith("_search.tsv"):
            path = os.path.join(search_cache_dir, fname)
            if os.path.getsize(path) > 0:
                proteins.add(fname[:-len("_search.tsv")])
    return proteins


def parse_search_results(tsv_path, query_protein):
    """Return {target_protein: max_tm} from a search TSV."""
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
            m = _AF_RE.match(target)
            if not m:
                continue
            pid = m.group(1)
            try:
                tm = float(parts[COL_MAX_TM])
            except ValueError:
                continue
            if pid not in best or tm > best[pid]:
                best[pid] = tm
    return best


def count_hits(tsv_path):
    try:
        with open(tsv_path) as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def run(args):
    # 1. Load positives and filter to covered pairs
    print("Loading Zhang positives...")
    all_positives = load_positives(args.controls)
    cached = get_cached_proteins(args.search_cache_dir)

    covered = [(a, b) for a, b in all_positives if a in cached and b in cached]
    print(f"Covered positive pairs: {len(covered)} / {len(all_positives)}")

    # 2. For each covered pair, check if the true partner appears in the search results
    candidates = []
    checked = 0
    for a, b in covered:
        for query, partner in [(a, b), (b, a)]:
            tsv = os.path.join(args.search_cache_dir, f"{query}_search.tsv")
            if not os.path.exists(tsv):
                continue
            results = parse_search_results(tsv, query)
            if partner not in results:
                continue
            tm = results[partner]
            if tm < args.min_tm:
                continue
            hit_count = count_hits(tsv)
            if not (args.min_hits <= hit_count <= args.max_hits):
                continue
            is_famous = query in FAMOUS_PROTEINS
            candidates.append({
                "query":     query,
                "partner":   partner,
                "tm":        tm,
                "hit_count": hit_count,
                "famous":    is_famous,
            })
        checked += 1
        if checked % 100 == 0:
            print(f"  Checked {checked}/{len(covered)} pairs...")

    print(f"\nFound {len(candidates)} candidate example proteins (partner in results, TM >= {args.min_tm})\n")

    if not candidates:
        print("No candidates found. Try lowering --min-tm or broadening --min-hits / --max-hits.")
        return

    # 3. Rank: famous first, then by TM score descending
    candidates.sort(key=lambda x: (-int(x["famous"]), -x["tm"]))

    # 4. Print top N
    print(f"{'Rank':<5} {'Query':<12} {'Partner':<12} {'TM':>6} {'Hits':>6}  {'Famous?'}")
    print("-" * 55)
    seen_queries = set()
    rank = 0
    for c in candidates:
        if c["query"] in seen_queries:
            continue
        seen_queries.add(c["query"])
        rank += 1
        famous_str = "*** FAMOUS ***" if c["famous"] else ""
        print(f"{rank:<5} {c['query']:<12} {c['partner']:<12} {c['tm']:>6.3f} {c['hit_count']:>6}  {famous_str}")
        if rank >= args.top:
            break

    print()
    print("Recommendation: pick a protein marked '*** FAMOUS ***' with high TM and moderate hit count.")
    print("That gives you a well-known protein where the pipeline demonstrably finds the true partner.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Find best example protein for report pipeline demonstration."
    )
    p.add_argument("--controls",         default=DEFAULT_CONTROLS)
    p.add_argument("--search-cache-dir", default=DEFAULT_SEARCH_CACHE, dest="search_cache_dir")
    p.add_argument("--top",    type=int, default=30,
                   help="Number of top candidates to show (default: 30)")
    p.add_argument("--min-tm", type=float, default=0.4, dest="min_tm",
                   help="Minimum TM-score for the true partner hit (default: 0.4)")
    p.add_argument("--min-hits", type=int, default=10, dest="min_hits",
                   help="Minimum total hit count (default: 10)")
    p.add_argument("--max-hits", type=int, default=400, dest="max_hits",
                   help="Maximum total hit count (default: 400)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
