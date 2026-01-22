#!/usr/bin/env python
"""
Debug tool to fetch raw metadata for specific indices.
Usage: python debug_fetch_metadata.py <config.json> <index1> [index2 ...]
"""
import sys
import os
import mmap
import json

# Add path to import Foldclass package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../merizo_search/programs')))

try:
    from Foldclass import metadata_extractor
    from Foldclass import dbutil
except ImportError as e:
    print(f"Error importing Foldclass modules: {e}")
    sys.exit(1)

def inspect_indices(db_config_path, indices):
    try:
        dbinfo = dbutil.read_dbinfo(db_config_path)
    except Exception as e:
        print(f"Error reading DB config: {e}")
        sys.exit(1)

    db_dir = os.path.dirname(db_config_path)
    metadata_index_path = os.path.join(db_dir, dbinfo['mif'])
    metadata_db_path = os.path.join(db_dir, dbinfo['mdf'])

    print(f"Inspecting {len(indices)} indices from {os.path.basename(db_config_path)}")

    with open(metadata_index_path, 'rb') as mif, \
         open(metadata_db_path, 'rb') as mdf:

        mi_mm = mmap.mmap(mif.fileno(), 0, access=mmap.ACCESS_READ)
        md_mm = mmap.mmap(mdf.fileno(), 0, access=mmap.ACCESS_READ)

        for idx in indices:
            print(f"\n--- Index {idx} ---")
            try:
                # Read raw bytes first
                offset = idx * 16
                if offset + 16 > len(mi_mm):
                    print("Index out of bounds")
                    continue
                    
                start_pos = int.from_bytes(mi_mm[offset:offset+8], byteorder='little')
                end_pos = int.from_bytes(mi_mm[offset+8:offset+16], byteorder='little')
                
                print(f"Position: {start_pos} - {end_pos} (Length: {end_pos - start_pos})")
                
                metadata_bytes = md_mm[start_pos:end_pos]
                print(f"Raw Bytes: {metadata_bytes}")
                
                metadata_str = metadata_bytes.decode('utf-8')
                print(f"String: {metadata_str}")
                
                import ast
                parsed = ast.literal_eval(metadata_str)
                print(f"Parsed: {parsed}")
                
            except Exception as e:
                print(f"ERROR: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
        
    config = sys.argv[1]
    indices = [int(x) for x in sys.argv[2:]]
    inspect_indices(config, indices)
