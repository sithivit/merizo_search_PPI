#!/usr/bin/env python
"""
Extract metadata from Merizo database files and populate SQLite index.

This module provides functionality to create a searchable SQLite database
from Merizo's metadata files, enabling fast filtering by taxonomy, CATH fold,
confidence level, and other domain properties.
"""

import sqlite3
import ast
import os
import mmap
import sys
from .dbutil import read_dbinfo, retrieve_names_by_idx


def extract_metadata_from_database(db_path, output_db_path):
    """
    Extract all metadata from a Merizo database and store in SQLite.

    Args:
        db_path: Path to Merizo database JSON config (e.g., 'ted100_9606_small.json')
        output_db_path: Path for output SQLite database (e.g., 'ted100_9606_small_filters.db')

    Returns:
        Number of domains processed
    """
    # Read database configuration
    dbinfo = read_dbinfo(db_path)
    db_dir = os.path.dirname(db_path)
    db_size = dbinfo['DB_SIZE']

    print(f"Reading database: {db_path}")
    print(f"Total domains: {db_size}")

    # Create SQLite database
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()

    # Create schema
    create_filter_schema(cursor)

    # Memory-map metadata file
    metadata_index_path = os.path.join(db_dir, dbinfo['mif'])
    metadata_db_path = os.path.join(db_dir, dbinfo['mdf'])

    # Memory-map domain names file
    names_path = os.path.join(db_dir, dbinfo['db_names_f'])

    # Open memory-mapped files
    with open(metadata_index_path, 'rb') as mif, \
         open(metadata_db_path, 'rb') as mdf, \
         open(names_path, 'rb') as nf:

        mi_mm = mmap.mmap(mif.fileno(), 0, access=mmap.ACCESS_READ)
        md_mm = mmap.mmap(mdf.fileno(), 0, access=mmap.ACCESS_READ)
        n_mm = mmap.mmap(nf.fileno(), 0, access=mmap.ACCESS_READ)

        # Process in batches
        batch_size = 1000
        for batch_start in range(0, db_size, batch_size):
            batch_end = min(batch_start + batch_size, db_size)
            indices = list(range(batch_start, batch_end))

            # Retrieve domain names (33 bytes each, fixed width)
            domain_names = retrieve_names_by_idx(idx=indices, mm=n_mm, use_sorting=False)

            # Retrieve metadata
            metadata_list = retrieve_metadata_batch(indices, mi_mm, md_mm)

            # Parse and insert into SQLite
            batch_data = []
            for idx, (domain_name, metadata_str) in enumerate(zip(domain_names, metadata_list)):
                domain_idx = batch_start + idx

                # Parse metadata (using ast.literal_eval since it's Python dict format)
                try:
                    metadata = ast.literal_eval(metadata_str)
                except Exception as e:
                    print(f"Warning: Failed to parse metadata for index {domain_idx}: {e}")
                    continue

                # Extract fields with defaults for missing values
                batch_data.append((
                    domain_idx,
                    domain_name,
                    int(metadata.get('taxid', 0)) if metadata.get('taxid') else None,
                    metadata.get('taxsci', ''),
                    metadata.get('cath', '') if metadata.get('cath') != 'NA' else '',
                    metadata.get('cnsl', ''),  # confidence level
                    float(metadata.get('dens', 0.0)) if metadata.get('dens') else None,  # globularity/density
                    metadata.get('cl', ''),  # architecture class
                    len(metadata.get('rr', '').replace('_', '-').split('-')) if metadata.get('rr') else None,  # domain length approximation
                    metadata_str  # Store full metadata for future extensibility
                ))

            # Batch insert
            cursor.executemany('''
                INSERT INTO domains VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch_data)

            conn.commit()
            print(f"Processed {batch_end}/{db_size} domains...")

    # Create indexes after all data inserted (faster)
    create_filter_indexes(cursor)
    conn.commit()

    print(f"\nSuccessfully created filter database: {output_db_path}")
    print(f"Total domains indexed: {db_size}")

    # Get file size
    file_size_mb = os.path.getsize(output_db_path) / 1024 / 1024
    print(f"Index size: {file_size_mb:.2f} MB")

    conn.close()
    return db_size


def create_filter_schema(cursor):
    """Create SQLite schema for filter database."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS domains (
            domain_idx INTEGER PRIMARY KEY,
            domain_id TEXT NOT NULL,
            taxonomy_id INTEGER,
            species TEXT,
            cath_fold TEXT,
            confidence TEXT,
            globularity_score REAL,
            architecture_class TEXT,
            domain_length INTEGER,
            metadata_raw TEXT
        )
    ''')


def create_filter_indexes(cursor):
    """Create indexes for fast filtering."""
    print("\nCreating indexes...")

    # Inverted index for taxonomy (most common filter)
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_taxonomy ON domains(taxonomy_id)')

    # Inverted index for CATH fold
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cath ON domains(cath_fold)')

    # Inverted index for confidence level
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON domains(confidence)')

    # Composite index for common filter combinations
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_tax_cath
        ON domains(taxonomy_id, cath_fold)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_tax_conf
        ON domains(taxonomy_id, confidence)
    ''')

    # Index for domain_id lookups (hash table behavior)
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain_id ON domains(domain_id)')

    print("Indexes created successfully")


def retrieve_metadata_batch(indices, index_mm, data_mm):
    """
    Retrieve batch of metadata strings.
    Adapted from dbutil.py retrieve_bytes pattern.

    Args:
        indices: List of domain indices
        index_mm: Memory-mapped index file
        data_mm: Memory-mapped data file

    Returns:
        List of metadata strings
    """
    metadata_list = []

    for idx in indices:
        # Read start/end positions from index file (2 int64 values)
        offset = idx * 16  # 2 * 8 bytes
        start_pos = int.from_bytes(index_mm[offset:offset+8], byteorder='little')
        end_pos = int.from_bytes(index_mm[offset+8:offset+16], byteorder='little')

        # Read metadata string from data file
        metadata_bytes = data_mm[start_pos:end_pos]
        metadata_str = metadata_bytes.decode('utf-8')
        metadata_list.append(metadata_str)

    return metadata_list


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python metadata_extractor.py <db_config.json> <output_filters.db>")
        print("Example: python metadata_extractor.py ted100_9606_small.json ted100_9606_small_filters.db")
        sys.exit(1)

    db_path = sys.argv[1]
    output_path = sys.argv[2]

    extract_metadata_from_database(db_path, output_path)
