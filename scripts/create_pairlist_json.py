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
    new_config['DB_SIZE'] = len(indices)
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
