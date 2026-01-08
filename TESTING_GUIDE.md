# Testing Guide: Filtering System with Larger Databases

## Current Status
- ✓ Filtering system implemented and tested on `ted100_9606_small` (66,943 domains)
- ✓ All filter types working correctly
- ✓ Performance: < 10ms for filter queries

## Testing Strategy for Larger Databases

### Quick Performance Test (No New Data Needed)

You can benchmark the current database with timing tests:

```python
# test_performance.py
import sys
import time
sys.path.insert(0, 'merizo_search')

from programs.Foldclass.filter_query import FilterQuery

filter_db = 'examples/database/ted100_9606_small/filters.db'
fq = FilterQuery(filter_db)

# Benchmark different filter types
filters_to_test = [
    ("Taxonomy only", lambda: fq.filter_by_taxonomy(9606)),
    ("Confidence only", lambda: fq.filter_by_confidence('high')),
    ("CATH fold only", lambda: fq.filter_by_cath_fold('2.60.40.10')),
    ("Combined filters", lambda: fq.filter_combined(
        taxonomy_id=9606,
        confidence='high',
        cath_fold='2.60.40.10'
    ))
]

for name, filter_func in filters_to_test:
    times = []
    for _ in range(100):  # Run 100 times
        start = time.time()
        result = filter_func()
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"{name}:")
    print(f"  - Avg: {avg_time:.2f}ms")
    print(f"  - Min: {min_time:.2f}ms")
    print(f"  - Max: {max_time:.2f}ms")
    print(f"  - Results: {len(result)} domains")
    print()

fq.close()
```

### Option 1: Find or Create a Larger Database

#### Check for TED-365M Database

The project documentation mentions a 365-million domain dataset. Check if it exists:

```bash
# Check various possible locations
ls -lh /mnt/bigstore/ted/ted_365* 2>/dev/null
find /mnt -name "ted_365*" 2>/dev/null
find /data -name "ted_365*" 2>/dev/null
find ~ -name "ted_365*" 2>/dev/null
```

If found, you need to create a Merizo database from it (requires the Merizo database build tool).

#### Required Directory Structure for Large Database

```
examples/database/ted_365/
├── ted_365.json                          # Database configuration
├── ted_365_raw_128d_norm.db              # Embeddings (187 GB for 365M domains)
├── ted_365_raw_128d.index_names          # Domain names
├── ted_365_metadata.db                   # Metadata (binary)
├── ted_365_metadata.index                # Metadata index
├── ted_365_ca.db                         # CA coordinates
├── ted_365_ca.index                      # CA coordinate index
├── ted_365_seq.db                        # Sequences
├── ted_365_seq.index                     # Sequence index
└── filters.db                            # Filter index (NEW - will be ~5-10 GB)
```

**JSON configuration (ted_365.json):**
```json
{
  "dbfname_IP": "ted_365_raw_128d_norm.db",
  "DB_SIZE": 365000000,
  "DB_DIM": 128,
  "db_names_f": "ted_365_raw_128d.index_names",
  "sif": "ted_365_seq.index",
  "sdf": "ted_365_seq.db",
  "cif": "ted_365_ca.index",
  "cdf": "ted_365_ca.db",
  "mif": "ted_365_metadata.index",
  "mdf": "ted_365_metadata.db"
}
```

### Option 2: Create Medium-Sized Test Database

If TED-365M is not available, you can create intermediate-sized databases:

#### Multi-Taxonomy Database (Recommended)

Combine multiple organism databases to test diversity:

```bash
# Create directory
mkdir -p examples/database/ted100_multi/

# If you have access to other organism databases:
# - Human (9606): Already have
# - Mouse (10090): If available
# - E. coli (562): If available
# - Yeast (4932): If available
```

This tests filter effectiveness across different taxonomies.

### Testing Checklist

Once you have a larger database:

#### 1. Build Filter Index
```bash
python merizo_search/merizo.py build-filter-index \
    examples/database/ted_365/ted_365 \
    examples/database/ted_365/filters.db
```

**Expected time:**
- 66K domains: ~30 seconds
- 1M domains: ~7 minutes
- 10M domains: ~70 minutes
- 365M domains: ~7 hours

#### 2. Test Filter Performance

```bash
# Run performance test
python test_performance.py

# Expected results for 365M domains:
# - Single filter: < 50ms
# - Combined filters: < 200ms
# - Database query overhead: < 100ms
```

#### 3. Test Memory Usage

```bash
# Monitor memory during filter index build
/usr/bin/time -l python merizo_search/merizo.py build-filter-index \
    examples/database/ted_365/ted_365 \
    examples/database/ted_365/filters.db

# Expected memory:
# - Batch processing: ~500 MB - 1 GB
# - SQLite index: Scales with data size
```

#### 4. Test Filter Accuracy

```python
# Verify filter results are correct
import sys
sys.path.insert(0, 'merizo_search')

from programs.Foldclass.filter_query import FilterQuery
from programs.Foldclass.dbutil import read_dbinfo
import os

db_path = 'examples/database/ted_365/ted_365.json'
filter_db = 'examples/database/ted_365/filters.db'

# Get database statistics
fq = FilterQuery(filter_db)
stats = fq.get_statistics()

print(f"Total domains: {stats['total_domains']}")
print(f"Taxonomies: {len(stats['top_taxonomies'])}")
print(f"CATH folds: {len(stats['top_cath_folds'])}")
print(f"Confidence distribution: {stats['confidence_distribution']}")

# Verify filters return expected counts
high_conf = fq.filter_by_confidence('high')
print(f"\nHigh confidence domains: {len(high_conf)}")
print(f"Percentage: {100*len(high_conf)/stats['total_domains']:.1f}%")

fq.close()
```

## Performance Expectations

### Small Database (66K domains) - Current
| Metric | Value |
|--------|-------|
| Filter index build | 30 seconds |
| Filter index size | 27 MB |
| Single filter query | < 10 ms |
| Combined filter query | < 20 ms |
| Filtering overhead | Negligible |

### Medium Database (1M domains) - Estimated
| Metric | Value |
|--------|-------|
| Filter index build | 7 minutes |
| Filter index size | ~400 MB |
| Single filter query | < 20 ms |
| Combined filter query | < 50 ms |
| Memory usage | ~1 GB |

### Large Database (365M domains) - Estimated
| Metric | Value |
|--------|-------|
| Filter index build | 6-8 hours |
| Filter index size | ~14 GB |
| Single filter query | < 50 ms |
| Combined filter query | < 200 ms |
| Memory usage | ~2-3 GB |
| **Speedup benefit** | **10x - 1000x+** |

## Recommendations

### For Your Testing:

1. **Start with performance baseline:**
   - Run `test_performance.py` on current database
   - Document current performance metrics

2. **Find larger database:**
   - Check if TED-365M or similar exists on your system
   - Check with your supervisor about access

3. **If large database not available:**
   - Use current database for demonstration
   - Create synthetic multi-taxonomy test if needed
   - Document expected scaling behavior

4. **For final year project presentation:**
   - Current 66K database is sufficient to demonstrate functionality
   - Show scaling calculations/estimates for larger databases
   - Emphasize filter reduction percentages (46% - 99%+)
   - Demonstrate that overhead is negligible (< 50ms)

### Key Metrics to Report:

- **Filter index build time:** One-time cost
- **Filter query time:** < 10ms (current), < 50ms (projected for 365M)
- **Filtering reduction:** 46% - 99%+ depending on filters
- **Search speedup:** 2x - 1000x+ for filtered searches
- **Memory efficiency:** Batch processing keeps memory constant

## Questions to Ask Your Supervisor

1. Do you have access to the TED-365M database mentioned in the domain summary file?
2. Are there other intermediate-sized databases (100K - 10M domains) available?
3. Would you like me to create synthetic test databases for scaling demonstrations?
4. What database size would be most relevant for your research use case?
