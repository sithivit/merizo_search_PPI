#!/usr/bin/env python3
"""
Build a bidirectional template pair index from the TED pair list, restricted to
pairs where at least one domain belongs to a Zhang benchmark protein.
Saves as JSON: {domain_name: [co-occurring domain names]}.

Usage:
    python scripts/rosetta_build_template_index.py \\
        --pair-list /mnt/bigstore/ted/pair_list_20250128 \\
        --zhang-domain-names benchmark_cache/zhang_domain_names.txt \\
        --output benchmark_cache/zhang_template_index.json
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def load_zhang_domain_names(path: str) -> set:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def build_index(pair_list_path: str, zhang_domains: set) -> dict:
    """Stream pair list; retain pairs where at least one domain is a Zhang domain. Returns bidirectional index."""
    index = {}
    n_included = 0
    n_total = 0

    with open(pair_list_path) as fh:
        for line in fh:
            n_total += 1
            if n_total % 5_000_000 == 0:
                log.info(f"  {n_total:,} lines read, {n_included:,} included...")
            parts = line.split()
            if not parts or ":" not in parts[0]:
                continue
            domain_a, domain_b = parts[0].split(":", 1)
            if domain_a not in zhang_domains and domain_b not in zhang_domains:
                continue
            n_included += 1
            index.setdefault(domain_a, set()).add(domain_b)
            index.setdefault(domain_b, set()).add(domain_a)

    log.info(f"Pair list: {n_total:,} lines, {n_included:,} pairs retained")
    log.info(f"Index: {len(index):,} unique domain entries")
    return {k: sorted(v) for k, v in index.items()}


def run(args):
    log.info("Loading Zhang domain names...")
    zhang_domains = load_zhang_domain_names(args.zhang_domain_names)
    log.info(f"Zhang domains: {len(zhang_domains):,}")

    log.info("Streaming pair list and building index (may take 20-30 min)...")
    index = build_index(args.pair_list, zhang_domains)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(index, fh)
    log.info(f"Index saved to {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-list", required=True, dest="pair_list")
    p.add_argument("--zhang-domain-names", required=True, dest="zhang_domain_names")
    p.add_argument("--output",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "zhang_template_index.json"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
