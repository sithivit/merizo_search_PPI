# Rosetta Stone Benchmark System

This module provides comprehensive benchmarking for the Rosetta Stone PPI prediction system.

## Quick Start

### 1. Speed Benchmark

Measures query search performance and throughput.

```bash
# Basic speed benchmark
python benchmark_rosetta.py \
    --mode speed \
    --fusion-db fusion_db/ \
    --test-queries test_queries/ \
    --device cuda

# Limit to first 10 queries for quick test
python benchmark_rosetta.py \
    --mode speed \
    --fusion-db fusion_db/ \
    --test-queries test_queries/ \
    --num-queries 10 \
    --device cuda
```

**Output:**
- `benchmark_results/speed_benchmark_results.json` - Detailed metrics
- `benchmark_results/speed_benchmark_per_query.csv` - Per-query timing data
- `benchmark_results/speed_benchmark_report.md` - Human-readable report
- `benchmark.log` - Detailed execution log

### 2. Sensitivity Benchmark (Coming Soon)

Measures recall and coverage of known interactions.

### 3. Accuracy Benchmark (Coming Soon)

Measures precision and false discovery rate.

## Preparing Test Data

### Test Queries

Create a directory with test query PDB files:

```bash
# Option 1: Use a subset from your database
mkdir -p test_queries
cp examples/database/AF-*.pdb test_queries/

# Option 2: Download specific test proteins
# (Add your specific proteins here)
```

### Gold Standard Dataset (for sensitivity/accuracy)

Download and prepare gold standard PPI data:

```bash
# Example: STRING Database
# 1. Visit https://string-db.org/
# 2. Download high-confidence interactions for your organism
# 3. Filter for physical interactions (score > 0.7)
# 4. Save as TSV with columns: protein_a, protein_b, confidence
```

## Output Files

### Speed Benchmark Results

**JSON Results** (`speed_benchmark_results.json`):
```json
{
  "benchmark_info": {
    "timestamp": "2025-01-16T10:30:00",
    "fusion_db": "fusion_db/",
    "device": "cuda"
  },
  "metrics": {
    "total_queries": 100,
    "successful_queries": 98,
    "failed_queries": 2,
    "avg_query_time": 25.3,
    "median_query_time": 24.1,
    "queries_per_minute": 2.37
  }
}
```

**CSV Per-Query Data** (`speed_benchmark_per_query.csv`):
```csv
query_name,time_seconds
AF-Q14686.pdb,23.456
AF-P12345.pdb,27.891
...
```

**Markdown Report** (`speed_benchmark_report.md`):
- Summary statistics
- Pass/fail evaluation
- Performance recommendations
- Slowest queries analysis

## Expected Performance Targets

### Speed Benchmark

| Metric | Target | Hardware |
|--------|--------|----------|
| Average query time | ≤ 30 seconds | RTX 3060 6GB |
| Queries per minute | ≥ 2.0 | RTX 3060 6GB |
| Success rate | ≥ 95% | - |

### Sensitivity Benchmark (Coming Soon)

| Metric | Target |
|--------|--------|
| Overall recall | ≥ 40% |
| Recall at conf ≥ 0.7 | ≥ 25% |
| Coverage | ≥ 60% |

### Accuracy Benchmark (Coming Soon)

| Metric | Target |
|--------|--------|
| Overall precision | ≥ 50% |
| Precision at conf ≥ 0.7 | ≥ 65% |
| F1 Score | ≥ 0.40 |

## Command Line Options

```
usage: benchmark_rosetta.py [-h] --mode {speed,sensitivity,accuracy,all}
                            --fusion-db FUSION_DB --test-queries TEST_QUERIES
                            [--output-dir OUTPUT_DIR] [--device {cpu,cuda,mps}]
                            [--num-queries NUM_QUERIES]
                            [--gold-standard GOLD_STANDARD] [--verbose]

Rosetta Stone PPI Benchmark Suite

optional arguments:
  -h, --help            show this help message and exit
  --mode {speed,sensitivity,accuracy,all}
                        Benchmark mode to run (default: speed)
  --fusion-db FUSION_DB
                        Path to fusion database directory
  --test-queries TEST_QUERIES
                        Path to directory containing test query PDB files
  --output-dir OUTPUT_DIR
                        Output directory for benchmark results (default:
                        benchmark_results/)
  --device {cpu,cuda,mps}
                        Device to use for inference (default: cuda)
  --num-queries NUM_QUERIES
                        Limit number of test queries (default: use all)
  --gold-standard GOLD_STANDARD
                        Path to gold standard interactions TSV (required for
                        sensitivity/accuracy)
  --verbose             Enable verbose logging
```

## Examples

### Example 1: Quick Speed Test (10 queries)

```bash
python benchmark_rosetta.py \
    --mode speed \
    --fusion-db fusion_db/ \
    --test-queries test_queries/ \
    --num-queries 10 \
    --output-dir benchmark_quick/ \
    --device cuda \
    --verbose
```

### Example 2: Full Speed Benchmark (All queries)

```bash
python benchmark_rosetta.py \
    --mode speed \
    --fusion-db fusion_db/ \
    --test-queries test_queries/ \
    --output-dir benchmark_full/ \
    --device cuda
```

### Example 3: CPU Benchmark (No GPU)

```bash
python benchmark_rosetta.py \
    --mode speed \
    --fusion-db fusion_db/ \
    --test-queries test_queries/ \
    --device cpu \
    --output-dir benchmark_cpu/
```

## Interpreting Results

### Speed Benchmark

**Good Performance:**
- Average query time < 30 seconds
- Success rate > 95%
- Low standard deviation (consistent timing)

**Poor Performance:**
- Average query time > 60 seconds
- High failure rate
- Check:
  - GPU is being used (`--device cuda`)
  - Fusion database is properly built
  - No memory issues (watch `nvidia-smi`)

### Troubleshooting

**Issue: Queries timing out**
- Check GPU memory usage
- Reduce batch size in fusion database building
- Verify query proteins are valid PDB files

**Issue: High failure rate**
- Check `benchmark.log` for error messages
- Verify fusion database integrity
- Ensure merizo.py script is accessible

**Issue: Very slow queries**
- Profile individual components
- Check if FAISS is using GPU
- Verify database size is reasonable

## Advanced Usage

### Custom Output Directory

```bash
python benchmark_rosetta.py \
    --mode speed \
    --fusion-db fusion_db/ \
    --test-queries test_queries/ \
    --output-dir my_benchmark_$(date +%Y%m%d_%H%M%S)/
```

### Benchmark Specific Queries

```bash
# Create a subset
mkdir test_queries_subset
cp test_queries/AF-Q14686.pdb test_queries_subset/
cp test_queries/AF-P12345.pdb test_queries_subset/

# Run benchmark
python benchmark_rosetta.py \
    --mode speed \
    --fusion-db fusion_db/ \
    --test-queries test_queries_subset/
```

### Comparing CPU vs GPU Performance

```bash
# GPU benchmark
python benchmark_rosetta.py --mode speed --fusion-db fusion_db/ \
    --test-queries test_queries/ --device cuda --output-dir benchmark_gpu/

# CPU benchmark
python benchmark_rosetta.py --mode speed --fusion-db fusion_db/ \
    --test-queries test_queries/ --device cpu --output-dir benchmark_cpu/

# Compare results
cat benchmark_gpu/speed_benchmark_results.json
cat benchmark_cpu/speed_benchmark_results.json
```

## Analyzing Results

### Load and Analyze with Python

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results
with open('benchmark_results/speed_benchmark_results.json', 'r') as f:
    results = json.load(f)

# Load per-query data
df = pd.read_csv('benchmark_results/speed_benchmark_per_query.csv')

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(df['time_seconds'], bins=20, edgecolor='black')
plt.xlabel('Query Time (seconds)')
plt.ylabel('Frequency')
plt.title('Query Time Distribution')
plt.axvline(results['metrics']['avg_query_time'], color='r',
            linestyle='--', label='Average')
plt.legend()
plt.savefig('query_time_distribution.png', dpi=300)
plt.show()

# Summary statistics
print(df['time_seconds'].describe())
```

## Contributing

To add new benchmark modes:

1. Create a new benchmark class (e.g., `SensitivityBenchmark`)
2. Implement `run_benchmark()` method
3. Add metrics dataclass
4. Add to main() function in `benchmark_rosetta.py`

## Future Enhancements

- [ ] Sensitivity/Recall benchmarking
- [ ] Accuracy/Precision benchmarking
- [ ] Database build speed benchmarking
- [ ] Memory profiling
- [ ] Component-level timing breakdown
- [ ] Visualization dashboard
- [ ] Comparison to baseline methods
- [ ] Automated report generation (PDF)

## Contact

For issues or questions about the benchmark system, please refer to the main GUIDE.md
or contact the development team.
