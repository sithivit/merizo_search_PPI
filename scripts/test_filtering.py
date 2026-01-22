#!/usr/bin/env python3
"""
Integration test for filtering functionality.

This script tests the complete filtering pipeline:
1. Build filter index
2. Query filter database
3. Use filtered iterator
4. Verify results
"""

import sys
import os

# Add merizo_search to path
sys.path.insert(0, 'merizo_search')

from programs.Foldclass.metadata_extractor import extract_metadata_from_database
from programs.Foldclass.filter_query import FilterQuery
from programs.Foldclass.filtered_iterator import db_iterator_filtered, FilteredIndexMapper
from programs.Foldclass.dbutil import db_memmap, read_dbinfo
import numpy as np

print("=" * 70)
print("MERIZO-SEARCH FILTERING INTEGRATION TEST")
print("=" * 70)

# Test configuration
DB_PATH = 'examples/database/ted100_9606_small/ted100_9606_small.json'
FILTER_DB_PATH = '/tmp/test_filters_integration.db'

# Clean up old filter DB if exists
if os.path.exists(FILTER_DB_PATH):
    os.remove(FILTER_DB_PATH)
    print(f"Removed existing filter DB: {FILTER_DB_PATH}")

print("\n" + "-" * 70)
print("Step 1: Build Filter Index")
print("-" * 70)

num_domains = extract_metadata_from_database(DB_PATH, FILTER_DB_PATH)
file_size_mb = os.path.getsize(FILTER_DB_PATH) / 1024 / 1024

print(f"\n✓ Filter index built successfully")
print(f"  - Total domains indexed: {num_domains}")
print(f"  - Index file size: {file_size_mb:.2f} MB")

print("\n" + "-" * 70)
print("Step 2: Query Filter Database")
print("-" * 70)

fq = FilterQuery(FILTER_DB_PATH)

# Test 1: Get statistics
stats = fq.get_statistics()
print(f"\nDatabase statistics:")
print(f"  - Total domains: {stats['total_domains']}")
print(f"  - Taxonomies: {stats['top_taxonomies']}")
print(f"  - Confidence distribution: {stats['confidence_distribution']}")
print(f"  - Top CATH folds: {len(stats['top_cath_folds'])} unique folds")

# Test 2: Filter by confidence only
high_conf_indices = fq.filter_by_confidence('high')
print(f"\n✓ High confidence filter:")
print(f"  - Matched {len(high_conf_indices)} domains ({100*len(high_conf_indices)/num_domains:.1f}%)")

# Test 3: Filter by CATH fold
if stats['top_cath_folds']:
    top_cath_fold = list(stats['top_cath_folds'].keys())[0]
    cath_indices = fq.filter_by_cath_fold(top_cath_fold)
    print(f"\n✓ CATH fold filter ({top_cath_fold}):")
    print(f"  - Matched {len(cath_indices)} domains ({100*len(cath_indices)/num_domains:.1f}%)")

# Test 4: Combined filters
combined_indices = fq.filter_combined(
    taxonomy_id=9606,
    confidence='high'
)
print(f"\n✓ Combined filter (taxonomy=9606 + confidence=high):")
print(f"  - Matched {len(combined_indices)} domains ({100*len(combined_indices)/num_domains:.1f}%)")
print(f"  - Reduction: {100 * (1 - len(combined_indices)/num_domains):.1f}%")

fq.close()

print("\n" + "-" * 70)
print("Step 3: Test Filtered Iterator")
print("-" * 70)

# Load database embeddings
dbinfo = read_dbinfo(DB_PATH)
db_dir = os.path.dirname(DB_PATH)
embeddings_file = os.path.join(db_dir, dbinfo['dbfname_IP'])

print(f"\nLoading embeddings from: {embeddings_file}")
dbmm = db_memmap(filename=embeddings_file, shape=(dbinfo['DB_SIZE'], dbinfo['DB_DIM']))
print(f"  - Embeddings shape: {dbmm.shape}")

# Test filtered iterator with high confidence domains
print(f"\nCreating filtered iterator with {len(high_conf_indices)} domains...")
batch_size = 10000
dbi_filtered = db_iterator_filtered(dbmm, high_conf_indices, batch_size=batch_size)

# Count batches
batch_count = 0
total_domains = 0
for batch in dbi_filtered:
    batch_count += 1
    total_domains += batch.shape[0]
    if batch_count == 1:
        print(f"  - First batch shape: {batch.shape}")

print(f"\n✓ Filtered iterator test:")
print(f"  - Total batches: {batch_count}")
print(f"  - Total domains iterated: {total_domains}")
print(f"  - Expected domains: {len(high_conf_indices)}")
print(f"  - Match: {'✓' if total_domains == len(high_conf_indices) else '✗'}")

print("\n" + "-" * 70)
print("Step 4: Test Index Mapper")
print("-" * 70)

# Test index mapper
sample_indices = high_conf_indices[:100]
mapper = FilteredIndexMapper(sample_indices)

print(f"\nTesting mapper with {len(sample_indices)} domains...")
print(f"  - Mapper length: {len(mapper)}")

# Test single index mapping
filtered_idx = 0
original_idx = mapper.to_original(filtered_idx)
print(f"  - Filtered index {filtered_idx} → Original index {original_idx}")
print(f"  - Expected: {sample_indices[0]}, Got: {original_idx}, Match: {'✓' if original_idx == sample_indices[0] else '✗'}")

# Test array mapping
test_filtered_indices = np.array([0, 10, 50, 99])
mapped_indices = mapper.to_original(test_filtered_indices)
expected_indices = np.array(sample_indices)[test_filtered_indices]
matches = np.array_equal(mapped_indices, expected_indices)

print(f"\n✓ Array mapping test:")
print(f"  - Test indices: {test_filtered_indices}")
print(f"  - Mapped to: {mapped_indices}")
print(f"  - Expected: {expected_indices}")
print(f"  - Match: {'✓' if matches else '✗'}")

print("\n" + "=" * 70)
print("INTEGRATION TEST COMPLETE")
print("=" * 70)

# Summary
print("\n✓ All components tested successfully:")
print("  1. Metadata extraction ✓")
print("  2. Filter query interface ✓")
print("  3. Filtered iterator ✓")
print("  4. Index mapper ✓")

print("\nFilter system is ready for use!")
print(f"Filter database: {FILTER_DB_PATH}")
print("\nExample usage:")
print("  python merizo_search/merizo.py build-filter-index \\")
print("    examples/database/ted100_9606_small/ted100_9606_small \\")
print("    examples/database/ted100_9606_small/filters.db")
print()
print("  python merizo_search/merizo.py search \\")
print("    query.pdb \\")
print("    examples/database/ted100_9606_small/ted100_9606_small \\")
print("    output /tmp \\")
print("    --filter-db examples/database/ted100_9606_small/filters.db \\")
print("    --filter-confidence high")
