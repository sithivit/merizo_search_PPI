#!/usr/bin/env python3
"""
Run on the server to check how many benchmark pairs are 'covered'
(both proteins present in pair_list), without running any searches.

Usage:
    python check_coverage.py \
        --pair-list /mnt/bigstore/ted/pair_list_20250128 \
        --controls benchmark_cache/benchmarks/positives_and_negatives.tsv
"""
import argparse, re, sys
from collections import defaultdict

_AF_RE = re.compile(r'AF-([^-]+)-F1-model_v4')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair-list', required=True)
    ap.add_argument('--controls', required=True)
    args = ap.parse_args()

    # 1. Load benchmark pairs
    positives, negatives = [], []
    with open(args.controls) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line: continue
            pair, cat = line.split('\t')
            a, b = pair.split('_')
            if cat == 'positive': positives.append((a, b))
            else: negatives.append((a, b))

    all_proteins = set()
    for a, b in positives + negatives:
        all_proteins.add(a); all_proteins.add(b)
    print(f"Benchmark proteins (unique): {len(all_proteins):,}")
    print(f"Positive pairs: {len(positives):,}")
    print(f"Negative pairs: {len(negatives):,}")

    # 2. Stream pair_list to find which benchmark proteins are present
    found = set()
    n_lines = 0
    with open(args.pair_list) as f:
        for line in f:
            if ':' not in line: continue
            parts = line.split()
            if not parts or ':' not in parts[0]: continue
            domain_a = parts[0].split(':')[0]
            m = _AF_RE.match(domain_a)
            if m:
                pid = m.group(1)
                if pid in all_proteins:
                    found.add(pid)
            n_lines += 1
            if n_lines % 5_000_000 == 0:
                print(f"  ... scanned {n_lines:,} lines, found {len(found):,}/{len(all_proteins):,} proteins")
            if len(found) == len(all_proteins):
                break

    print(f"\nPair-list coverage:")
    print(f"  Benchmark proteins found in pair_list: {len(found):,}/{len(all_proteins):,} "
          f"({len(found)/len(all_proteins)*100:.1f}%)")

    # 3. Covered pairs (both proteins in pair_list)
    pos_covered = [(a,b) for a,b in positives if a in found and b in found]
    neg_covered = [(a,b) for a,b in negatives if a in found and b in found]
    pos_partial = [(a,b) for a,b in positives if (a in found) != (b in found)]

    print(f"\n  Positive pairs — both in pair_list:    {len(pos_covered):,}/{len(positives):,} "
          f"({len(pos_covered)/len(positives)*100:.1f}%)")
    print(f"  Positive pairs — one in pair_list:     {len(pos_partial):,}/{len(positives):,}")
    print(f"  Negative pairs — both in pair_list:    {len(neg_covered):,}/{len(negatives):,} "
          f"({len(neg_covered)/len(negatives)*100:.1f}%)")

    # 4. If we have the search results, how many covered pairs scored > 0?
    print(f"\n  => If all covered positive pairs scored, max possible recall = "
          f"{len(pos_covered)/len(positives)*100:.1f}%")
    print(f"  => Actual hits so far: 23  ({23/len(pos_covered)*100:.1f}% of covered positives)"
          if pos_covered else "")

if __name__ == '__main__':
    main()
