#!/usr/bin/env python
"""
Complete pipeline to create indexed database from pair list.

Usage:
    python transform_pairlist_to_database.py \\
        /mnt/bigstore/ted/pair_list_20250128 \\
        /mnt/bigstore/foldclass-db-ted/ted_365M.json \\
        output_dir/
"""

import sys
import os
import json

# Import local modules
# Ensure we can import from the same directory
sys.path.append(os.path.dirname(__file__))

import extract_pairlist_domains
import map_domains_to_indices
import create_pairlist_filter_db
import create_pairlist_json

def create_index_list(domain_to_idx, output_path):
    """
    Save sorted list of indices.

    Output format: One index per line
    """
    indices = sorted(domain_to_idx.values())

    with open(output_path, 'w') as f:
        for idx in indices:
            f.write(f"{idx}\n")

    print(f"Created index list: {len(indices)} domains -> {output_path}")
    return indices

def main(pair_list_path, db_config_path, output_dir):
    """
    Complete transformation pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Starting Transformation Pipeline ===")
    print(f"Pair List: {pair_list_path}")
    print(f"Source DB: {db_config_path}")
    print(f"Output:    {output_dir}")

    print("\nStep 1: Extracting domain IDs from pair list...")
    domain_ids = extract_pairlist_domains.extract_domain_ids(pair_list_path)
    print(f"  Found {len(domain_ids)} unique domains")

    print("\nStep 2: Mapping domain IDs to database indices...")
    domain_to_idx = map_domains_to_indices.map_ids_to_indices(domain_ids, db_config_path)
    print(f"  Mapped {len(domain_to_idx)} domains")
    
    if len(domain_to_idx) == 0:
        print("Warning: No domains were mapped. Check if pair list IDs match database names.")
        # Proceeding anyway? Maybe allow creating empty DB?
        # But indices list will be empty.

    print("\nStep 3: Creating index list file...")
    indices = create_index_list(
        domain_to_idx,
        os.path.join(output_dir, 'ted_pairlist.indices')
    )

    print("\nStep 4: Creating filter database...")
    create_pairlist_filter_db.create_pairlist_filter_db(
        domain_to_idx,
        db_config_path,
        os.path.join(output_dir, 'ted_pairlist_filters.db')
    )

    print("\nStep 5: Creating JSON configuration...")
    create_pairlist_json.create_pairlist_json(
        db_config_path,
        indices,
        os.path.join(output_dir, 'ted_pairlist.json')
    )

    print("\n=== Transformation complete! ===")
    print(f"  Output directory: {output_dir}")
    print(f"  Database size: {len(indices)} domains")
    
    # Calculate reduction if possible
    try:
        with open(db_config_path, 'r') as f:
            full_size = json.load(f).get('DB_SIZE', 1)
        if full_size > 0:
            print(f"  Reduction: {100 * (1 - len(indices) / full_size):.1f}%")
    except:
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Build an indexed Foldclass-compatible database from the TED domain pair list.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pair_list", help="Path to TED pair list file (e.g. pair_list_20250128)")
    parser.add_argument("db_config", help="Path to ted_365M.json Foldclass database config")
    parser.add_argument("output_dir", help="Output directory for the filtered database")
    args = parser.parse_args()
    main(args.pair_list, args.db_config, args.output_dir)
