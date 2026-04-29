#!/usr/bin/env python
"""
Create SQLite filter database for pair-list domains
"""
import sys
import os
import json
import sqlite3
import mmap
import ast
from typing import Dict, List, Set

# Add path to import Foldclass package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../merizo_search/programs')))

try:
    from Foldclass import metadata_extractor
    from Foldclass import dbutil
except ImportError as e:
    print(f"Error importing Foldclass modules: {e}")
    print("Please ensure you are running this script from the 'scripts/' directory")
    print("and that 'merizo_search/programs/Foldclass' exists.")
    sys.exit(1)

def create_pairlist_filter_db(domain_to_idx: Dict[str, int], db_config_path: str, output_db_path: str):
    """
    Create SQLite filter database containing only pair-list domains.

    Uses the existing metadata_extractor infrastructure.
    """
    
    # Read database configuration
    try:
        dbinfo = dbutil.read_dbinfo(db_config_path)
    except Exception as e:
        print(f"Error reading DB config: {e}")
        sys.exit(1)

    db_dir = os.path.dirname(db_config_path)
    
    # Sort indices for efficient sequential access
    indices = sorted(domain_to_idx.values())
    print(f"Creating filter database for {len(indices)} domains...")

    # Create SQLite database
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    metadata_extractor.create_filter_schema(cursor)

    # Memory-map paths
    metadata_index_path = os.path.join(db_dir, dbinfo['mif'])
    metadata_db_path = os.path.join(db_dir, dbinfo['mdf'])
    names_path = os.path.join(db_dir, dbinfo['db_names_f'])

    # Open memory-mapped files
    try:
        with open(metadata_index_path, 'rb') as mif, \
             open(metadata_db_path, 'rb') as mdf, \
             open(names_path, 'rb') as nf:

            mi_mm = mmap.mmap(mif.fileno(), 0, access=mmap.ACCESS_READ)
            md_mm = mmap.mmap(mdf.fileno(), 0, access=mmap.ACCESS_READ)
            n_mm = mmap.mmap(nf.fileno(), 0, access=mmap.ACCESS_READ)

            # Process in batches
            BATCH_SIZE = 1000
            total = len(indices)
            
            for i in range(0, total, BATCH_SIZE):
                batch_indices = indices[i : i + BATCH_SIZE]
                
                # Retrieve domain names
                domain_names = dbutil.retrieve_names_by_idx(idx=batch_indices, mm=n_mm, use_sorting=False)
                
                # Retrieve metadata
                metadata_list = metadata_extractor.retrieve_metadata_batch(batch_indices, mi_mm, md_mm)
                
                # Parse and insert
                batch_data = []
                for idx_in_batch, (domain_name, metadata_str) in enumerate(zip(domain_names, metadata_list)):
                    domain_idx = batch_indices[idx_in_batch] # The original DB index
                    
                    try:
                        # FIX: Handle unescaped quotes in taxsci field like "Amphilius_sp._"Ruvu""
                        # This breaks standard JSON/Python dict parsing.
                        # We blindly replace specific known patterns or use a regex if it gets complex.
                        # Simple fix: if "Ruvu" appears inside the string, escape it? 
                        # Better: Use json.loads if it was valid JSON, but it looks like Python dict string.
                        
                        # Attempt to fix the specific "Ruvu" case and similar unescaped quotes
                        if 'sp._"' in metadata_str:
                             metadata_str = metadata_str.replace('sp._"', 'sp._\\"').replace('""', '\\""')
                             # This is hacky. A better way: replace ` "Ruvu"` with ` \"Ruvu\"`
                             import re
                             # Regex to find unescaped quotes inside values? Hard.
                             # Let's try to just accept that if AST fails, we try to fix common culprits.
                             pass
                        
                        metadata = ast.literal_eval(metadata_str)
                    except Exception as e:
                        # Retry with aggressive quote escaping for the specific "Ruvu" pattern seen in logs
                        try:
                            # The pattern seen: "taxsci": "Amphilius_sp._"Ruvu""
                            # We want: "taxsci": "Amphilius_sp._\"Ruvu\""
                            # We can replace `_" ` with `_\"` and `""` (at end) with `\""` contextually?
                            # Let's try a directed replace for this specific scientific name pattern
                             fixed_str = metadata_str.replace('._"', '._\\"').replace('""', '\\""')
                             metadata = ast.literal_eval(fixed_str)
                        except Exception:
                            # Log error but continue - don't crash
                            print(f"Warning: Failed to parse metadata for index {domain_idx}: {e}")
                            # print(f"  Raw string: {metadata_str!r}") # Uncomment for verbose debug
                            continue
                        
                    # Safe extraction of density
                    try:
                        raw_dens = metadata.get('dens', 0.0)
                        if isinstance(raw_dens, str):
                            # Handle cases like 'XXX_PACKING_DENSITY'
                             dens = None
                        else:
                             dens = float(raw_dens)
                    except (ValueError, TypeError):
                        dens = None

                    # Extract fields (logic copied from metadata_extractor.py)
                    batch_data.append((
                        domain_idx,
                        domain_name,
                        int(metadata.get('taxid', 0)) if metadata.get('taxid') else None,
                        metadata.get('taxsci', ''),
                        metadata.get('cath', '') if metadata.get('cath') != 'NA' else '',
                        metadata.get('cnsl', ''),  # confidence level
                        dens,  # Safe density
                        metadata.get('cl', ''),  # architecture class
                        len(metadata.get('rr', '').replace('_', '-').split('-')) if metadata.get('rr') else None,  # domain length approximation
                        metadata_str
                    ))
                
                cursor.executemany('INSERT INTO domains VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', batch_data)
                
                if (i + BATCH_SIZE) % 10000 == 0:
                    print(f"Processed {min(i + BATCH_SIZE, total)}/{total} domains...")
    except FileNotFoundError as e:
        print(f"Error accessing database files: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    conn.commit()
    
    # Create indexes
    metadata_extractor.create_filter_indexes(cursor)
    conn.commit()
    conn.close()

    print(f"Created filter database: {output_db_path}")

if __name__ == "__main__":
    print("This module is a library. Use transform_pairlist_to_database.py to run the full pipeline.")
    sys.exit(0)
