#!/usr/bin/env python
"""
Create JSON config for pair-list indexed database
"""
import json
import os
from typing import List

def create_pairlist_json(original_json_path: str, indices: List[int], output_json_path: str):
    """
    Create new JSON config that references same data files
    but with restricted index set.

    Input:
    - original_json_path: ted_365M.json
    - indices: List of database indices to include
    - output_json_path: ted_pairlist.json
    """

    # Load original config
    with open(original_json_path, 'r') as f:
        config = json.load(f)

    # Create new config with same file references
    # but different DB_SIZE and additional index file
    new_config = config.copy()

    # FIX: Ensure all file paths are ABSOLUTE.
    # The original config might contain relative paths (e.g. "ted_365M_raw_128d_norm.db")
    # which assumes the code is run from that directory.
    # Since our new json is in a new folder, we must point back to the original files explicitly.
    original_db_dir = os.path.dirname(os.path.abspath(original_json_path))
    
    # List of keys that are file paths in 'ted_365M.json'
    # Based on checking the file, usually: 'db', 'db_names_f', 'mif', 'mdf'
    # We iterate and fix any string value that looks like a file and exists in source dir
    for key, value in new_config.items():
        if isinstance(value, str) and not key.endswith('_FILE') and not key == 'DESCRIPTION':
             # Check if it matches a file in the original dir
             potential_path = os.path.join(original_db_dir, value)
             if os.path.exists(potential_path):
                 new_config[key] = potential_path

    # Do NOT overwrite DB_SIZE: dbsearch uses it to shape the memmap over the
    # original embeddings file.  The actual search restriction is done via
    # INDEX_LIST_FILE (filtered_iterator reads only those indices).
    new_config['PAIRLIST_SIZE'] = len(indices)
    new_config['DB_SIZE_ORIGINAL'] = config.get('DB_SIZE')
    new_config['INDEX_LIST_FILE'] = 'ted_pairlist.indices'
    new_config['FILTER_DB_FILE'] = 'ted_pairlist_filters.db'
    new_config['DESCRIPTION'] = 'Filtered database containing only domains from pair_list_20250128'

    # Save new JSON config
    with open(output_json_path, 'w') as f:
        json.dump(new_config, f, indent=2)

    print(f"Created JSON config: {output_json_path}")
    print(f"  Original DB size: {config.get('DB_SIZE')}")
    print(f"  Filtered DB size: {len(indices)}")

if __name__ == "__main__":
    pass
