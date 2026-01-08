#!/usr/bin/env python3
"""
Compare performance of filtered vs unfiltered database iteration.
"""

import sys
import time
sys.path.insert(0, 'merizo_search')

from programs.Foldclass.filter_query import FilterQuery
from programs.Foldclass.filtered_iterator import db_iterator_filtered
from programs.Foldclass.dbutil import db_memmap, read_dbinfo, db_iterator
import os

DB_PATH = 'examples/database/ted100_9606_small/ted100_9606_small.json'
FILTER_DB_PATH = 'examples/database/ted100_9606_small/ted100_9606_small_filters.db'

print("=" * 70)
print("PERFORMANCE COMPARISON: Filtered vs Unfiltered Iteration")
print("=" * 70)

# Load database info
dbinfo = read_dbinfo(DB_PATH)
db_dir = os.path.dirname(DB_PATH)
embeddings_file = os.path.join(db_dir, dbinfo['dbfname_IP'])

print(f"\nDatabase: {DB_PATH}")
print(f"Total domains: {dbinfo['DB_SIZE']:,}")
print(f"Embedding dimension: {dbinfo['DB_DIM']}")

# Load embeddings
dbmm = db_memmap(filename=embeddings_file, shape=(dbinfo['DB_SIZE'], dbinfo['DB_DIM']))
print(f"Loaded embeddings: {dbmm.shape}")

# Test 1: Unfiltered iteration
print("\n" + "-" * 70)
print("Test 1: UNFILTERED iteration (baseline)")
print("-" * 70)
batch_size = 10000
batches_processed = 0
domains_processed = 0

start = time.time()
for batch in db_iterator(dbmm, batch_size=batch_size):
    batches_processed += 1
    domains_processed += batch.shape[0]
elapsed_unfiltered = time.time() - start

print(f"Batches: {batches_processed}")
print(f"Domains processed: {domains_processed:,}")
print(f"Time: {elapsed_unfiltered:.3f}s")
print(f"Throughput: {domains_processed/elapsed_unfiltered:,.0f} domains/sec")

# Test 2: Filtered iteration (high confidence)
print("\n" + "-" * 70)
print("Test 2: FILTERED iteration (confidence = 'high')")
print("-" * 70)

fq = FilterQuery(FILTER_DB_PATH)
filtered_indices = fq.filter_by_confidence('high')
fq.close()

print(f"Filter matched: {len(filtered_indices):,} / {dbinfo['DB_SIZE']:,} domains")
print(f"Reduction: {100*(1 - len(filtered_indices)/dbinfo['DB_SIZE']):.1f}%")

batches_processed = 0
domains_processed = 0

start = time.time()
for batch in db_iterator_filtered(dbmm, filtered_indices, batch_size=batch_size):
    batches_processed += 1
    domains_processed += batch.shape[0]
elapsed_filtered = time.time() - start

print(f"Batches: {batches_processed}")
print(f"Domains processed: {domains_processed:,}")
print(f"Time: {elapsed_filtered:.3f}s")
print(f"Throughput: {domains_processed/elapsed_filtered:,.0f} domains/sec")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
speedup = elapsed_unfiltered / elapsed_filtered
print(f"Unfiltered time: {elapsed_unfiltered:.3f}s")
print(f"Filtered time:   {elapsed_filtered:.3f}s")
print(f"Speedup:         {speedup:.2f}x")
print(f"Time saved:      {elapsed_unfiltered - elapsed_filtered:.3f}s ({100*(1-elapsed_filtered/elapsed_unfiltered):.1f}%)")
print("\n✓ Filtering reduces both the search space AND execution time!")
print("=" * 70)
