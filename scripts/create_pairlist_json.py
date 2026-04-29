#!/usr/bin/env python
"""
Create JSON config for pair-list indexed database
"""
import json
import os
from typing import List

def create_pairlist_json(original_json_path: str, indices: List[int], output_json_path: str):
    """Create a new JSON config referencing the same data files but with a restricted index set."""
    with open(original_json_path, 'r') as f:
        config = json.load(f)

    new_config = config.copy()

    # Absolutise relative file paths so the new JSON works from a different directory.
    original_db_dir = os.path.dirname(os.path.abspath(original_json_path))
    for key, value in new_config.items():
        if isinstance(value, str) and not key.endswith('_FILE') and not key == 'DESCRIPTION':
            potential_path = os.path.join(original_db_dir, value)
            if os.path.exists(potential_path):
                new_config[key] = potential_path

    # DB_SIZE is intentionally not overwritten — dbsearch uses it to shape the memmap.
    # Search restriction is handled via INDEX_LIST_FILE.
    new_config['PAIRLIST_SIZE'] = len(indices)
    new_config['DB_SIZE_ORIGINAL'] = config.get('DB_SIZE')
    new_config['INDEX_LIST_FILE'] = 'ted_pairlist.indices'
    new_config['FILTER_DB_FILE'] = 'ted_pairlist_filters.db'
    new_config['DESCRIPTION'] = 'Filtered database containing only domains from pair_list_20250128'

    with open(output_json_path, 'w') as f:
        json.dump(new_config, f, indent=2)

    print(f"Created JSON config: {output_json_path}")
    print(f"  Original DB size: {config.get('DB_SIZE')}")
    print(f"  Filtered DB size: {len(indices)}")

if __name__ == "__main__":
    print("This module is a library. Use transform_pairlist_to_database.py to run the full pipeline.")
    import sys; sys.exit(0)
