#!/usr/bin/env python
"""
Extract unique domain IDs from pair_list_20250128
"""
import sys
from typing import Set

def extract_domain_ids(pair_list_path: str) -> Set[str]:
    """
    Parse pair_list file and extract all unique domain IDs.

    Input: pair_list_20250128
    Output: Set of domain IDs
    """
    domain_ids = set()

    try:
        with open(pair_list_path, 'r') as f:
            for line in f:
                # Parse: DOMAIN1:DOMAIN2 SCORE1:SCORE2 PAE
                parts = line.strip().split()
                if len(parts) >= 1:
                    pair = parts[0]
                    # Check if the line is a valid pair (contains :)
                    if ':' in pair:
                        domain1, domain2 = pair.split(':')
                        domain_ids.add(domain1)
                        domain_ids.add(domain2)
    except FileNotFoundError:
        print(f"Error: File not found: {pair_list_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing pair list: {e}")
        sys.exit(1)

    return domain_ids

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract unique domain IDs from a TED pair list file.")
    parser.add_argument("pair_list", help="Path to pair_list file (e.g. pair_list_20250128)")
    args = parser.parse_args()
    domain_ids = extract_domain_ids(args.pair_list)
    print(f"Unique domains in pair list: {len(domain_ids)}")
