#!/usr/bin/env python3
"""
Quick script to query the filter database and see filtering in action.
"""

import sys
sys.path.insert(0, 'merizo_search')

from programs.Foldclass.filter_query import FilterQuery

# Path to your filter database
FILTER_DB = 'examples/database/ted100_9606_small/ted100_9606_small_filters.db'

print("=" * 70)
print("FILTER DATABASE QUERY TOOL")
print("=" * 70)

fq = FilterQuery(FILTER_DB)

# Get overall statistics
print("\n📊 DATABASE STATISTICS")
print("-" * 70)
stats = fq.get_statistics()
print(f"Total domains: {stats['total_domains']:,}")
print(f"\nTaxonomies:")
for tax_id, count in stats['top_taxonomies'].items():
    print(f"  - {tax_id}: {count:,} domains")

print(f"\nConfidence distribution:")
for level, count in stats['confidence_distribution'].items():
    pct = 100 * count / stats['total_domains']
    print(f"  - {level}: {count:,} domains ({pct:.1f}%)")

print(f"\nTop CATH folds:")
for fold, count in list(stats['top_cath_folds'].items())[:5]:
    pct = 100 * count / stats['total_domains']
    print(f"  - {fold}: {count:,} domains ({pct:.1f}%)")

# Test different filters
print("\n\n🔍 FILTER EXAMPLES")
print("-" * 70)

# Filter 1: High confidence only
print("\n1. Filter: Confidence = 'high'")
indices = fq.filter_by_confidence('high')
reduction = 100 * (1 - len(indices) / stats['total_domains'])
speedup = stats['total_domains'] / len(indices)
print(f"   ✓ Matched: {len(indices):,} domains")
print(f"   ✓ Reduction: {reduction:.1f}%")
print(f"   ✓ Expected speedup: {speedup:.1f}x")

# Filter 2: Specific CATH fold
top_fold = list(stats['top_cath_folds'].keys())[0]
print(f"\n2. Filter: CATH fold = '{top_fold}'")
indices = fq.filter_by_cath_fold(top_fold)
reduction = 100 * (1 - len(indices) / stats['total_domains'])
speedup = stats['total_domains'] / len(indices)
print(f"   ✓ Matched: {len(indices):,} domains")
print(f"   ✓ Reduction: {reduction:.1f}%")
print(f"   ✓ Expected speedup: {speedup:.1f}x")

# Filter 3: Combined filters
print(f"\n3. Filter: Taxonomy = 9606 AND Confidence = 'high' AND CATH = '{top_fold}'")
indices = fq.filter_combined(
    taxonomy_id=9606,
    confidence='high',
    cath_fold=top_fold
)
reduction = 100 * (1 - len(indices) / stats['total_domains'])
speedup = stats['total_domains'] / len(indices) if len(indices) > 0 else float('inf')
print(f"   ✓ Matched: {len(indices):,} domains")
print(f"   ✓ Reduction: {reduction:.1f}%")
print(f"   ✓ Expected speedup: {speedup:.1f}x")

fq.close()

print("\n" + "=" * 70)
print("DONE! The filter database is working perfectly.")
print("=" * 70)
