#!/usr/bin/env python
"""
Map domain IDs to their indices in the 365M database
"""
import os
import sys
import json
import mmap
import time
from typing import Set, Dict

def map_ids_to_indices(domain_ids: Set[str], db_config_path: str) -> Dict[str, int]:
    """Return {domain_id: database_index} for all domain IDs found in the 365M database."""
    try:
        with open(db_config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Database config not found: {db_config_path}")
        sys.exit(1)

    db_dir = os.path.dirname(db_config_path)
    db_size = config.get('DB_SIZE', 0)
    names_file_rel = config.get('db_names_f')
    
    if not names_file_rel:
        print("Error: 'db_names_f' not found in database config")
        sys.exit(1)

    names_file = os.path.join(db_dir, names_file_rel)
    
    if not os.path.exists(names_file):
        print(f"Error: Names file not found: {names_file}")
        sys.exit(1)

    domain_to_idx = {}
    ENTRY_SIZE = 33

    print(f"Searching {db_size} domains in {names_file}...")
    start_time = time.time()

    with open(names_file, 'rb') as f:
        if db_size == 0 or os.path.getsize(names_file) == 0:
            return {}

        try:
            names_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            print("Error creating mmap. File might be empty.")
            return {}

        mm_read = names_mm.read
        mm_seek = names_mm.seek

        for idx in range(db_size):
            mm_seek(idx * ENTRY_SIZE)
            raw_data = mm_read(ENTRY_SIZE)
            domain_name = raw_data.decode('utf-8', errors='ignore').strip('\x00').strip()

            if domain_name in domain_ids:
                domain_to_idx[domain_name] = idx
                if len(domain_to_idx) % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Found {len(domain_to_idx)} domains... ({elapsed:.1f}s)")
                if len(domain_to_idx) == len(domain_ids):
                    print("All domains found. Stopping early.")
                    break
        
        names_mm.close()

    elapsed = time.time() - start_time
    print(f"Mapped {len(domain_to_idx)} / {len(domain_ids)} domains in {elapsed:.1f}s")

    return domain_to_idx

if __name__ == "__main__":
    print("This module is a library. Use transform_pairlist_to_database.py to run the full pipeline.")
    import sys; sys.exit(0)
