#!/usr/bin/env python3
"""
rosetta_make_mini_sample.py

Create a small sample controls file (N positive pairs + 5*N negatives) from
the full Zhang controls, restricted to proteins that already have cached
search results. Used to validate the Rosetta Stone pipeline before running
the full benchmark.

Usage:
    python scripts/rosetta_make_mini_sample.py \
        --controls benchmark_cache/benchmarks/positives_and_negatives.tsv \
        --search-cache benchmark_cache/searches \
        --n-positives 30 \
        --output benchmark_cache/benchmarks/mini_controls.tsv
"""

import argparse
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def run(args):
    # Get proteins with cached search results
    cached = set()
    if os.path.isdir(args.search_cache):
        for fname in os.listdir(args.search_cache):
            if fname.endswith("_search.tsv"):
                path = os.path.join(args.search_cache, fname)
                if os.path.getsize(path) > 0:
                    cached.add(fname[:-len("_search.tsv")])
    print(f"Proteins with cached search results: {len(cached)}")

    # Load positives and negatives
    positives, negatives = [], []
    with open(args.controls) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, cat = line.split("\t")
            a, b = pair.split("_")
            if cat == "positive" and a in cached and b in cached:
                positives.append((a, b))
            elif cat == "negative" and a in cached and b in cached:
                negatives.append((a, b))

    print(f"Covered positives: {len(positives)}, covered negatives: {len(negatives)}")

    rng = random.Random(42)
    sample_pos = rng.sample(positives, min(args.n_positives, len(positives)))
    sample_neg = rng.sample(negatives, min(args.n_positives * 5, len(negatives)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write("Protein pairs\tCategory\n")
        for a, b in sample_pos:
            fh.write(f"{a}_{b}\tpositive\n")
        for a, b in sample_neg:
            fh.write(f"{a}_{b}\tnegative\n")

    print(f"Wrote {len(sample_pos)} positives + {len(sample_neg)} negatives to {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--controls",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "benchmarks", "positives_and_negatives.tsv"))
    p.add_argument("--search-cache",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache", "searches"),
                   dest="search_cache")
    p.add_argument("--n-positives", type=int, default=30, dest="n_positives")
    p.add_argument("--output",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "benchmarks", "mini_controls.tsv"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
