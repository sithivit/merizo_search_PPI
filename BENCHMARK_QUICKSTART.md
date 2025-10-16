# Benchmark Quick Start Guide

**Get your Rosetta Stone benchmark running in 3 commands!**

---

## Prerequisites

1. ✓ Fusion database built (`fusion_db/`)
2. ✓ Test proteins available (`examples/database/`)
3. ✓ Python environment with dependencies

---

## Quick Commands

### 1. Prepare Test Queries (Random Sample)

```bash
python prepare_test_queries.py --source examples/database/ --output test_queries/ --num 10
```

**What this does:** Randomly selects 10 proteins for testing

---

### 2. Run Speed Benchmark

```bash
python benchmark_rosetta.py --mode speed --fusion-db fusion_db/ --test-queries test_queries/ --device cuda
```

**What this does:** Measures query search performance
**Time:** ~5 minutes for 10 queries
**Output:** Results in `benchmark_results/`

---

### 3. Visualize Results

```bash
python visualize_benchmark.py benchmark_results/
```

**What this does:** Generates performance charts
**Output:** Plots in `benchmark_results/plots/`

---

## One-Line Quick Test

```bash
# Windows
run_benchmark_example.bat

# Linux/Mac
bash run_benchmark_example.sh
```

This runs steps 1-2 automatically!

---

## Understanding Results

### Speed Benchmark Output

```
benchmark_results/
├── speed_benchmark_results.json      ← Detailed metrics
├── speed_benchmark_per_query.csv     ← Per-query times
├── speed_benchmark_report.md         ← Human-readable report
└── plots/                             ← Visualizations (after step 3)
    ├── query_time_histogram.png
    ├── query_time_boxplot.png
    ├── cumulative_time.png
    ├── slowest_queries.png
    └── performance_summary.png
```

### Key Metrics

| Metric | Good | Needs Work |
|--------|------|------------|
| **Average Query Time** | < 30s | > 60s |
| **Success Rate** | > 95% | < 90% |
| **Throughput** | > 2 queries/min | < 1.5 queries/min |

### Reading the Report

```bash
# View markdown report
cat benchmark_results/speed_benchmark_report.md

# Or open in browser (Linux)
markdown benchmark_results/speed_benchmark_report.md > report.html
firefox report.html
```

---

## Common Usage Patterns

### Quick Test (Small Sample)

```bash
# Just 5 queries for quick validation
python prepare_test_queries.py --source examples/database/ --output test_queries_small/ --num 5
python benchmark_rosetta.py --mode speed --fusion-db fusion_db/ --test-queries test_queries_small/
```

### Full Benchmark (All Queries)

```bash
# Use all available proteins
python prepare_test_queries.py --source examples/database/ --output test_queries_full/ --all
python benchmark_rosetta.py --mode speed --fusion-db fusion_db/ --test-queries test_queries_full/
```

### Specific Proteins

```bash
# Edit sample_protein_list.txt with your protein IDs
python prepare_test_queries.py --source examples/database/ --output test_queries_custom/ \
       --select sample_protein_list.txt
python benchmark_rosetta.py --mode speed --fusion-db fusion_db/ --test-queries test_queries_custom/
```

### CPU Benchmark (No GPU)

```bash
python benchmark_rosetta.py --mode speed --fusion-db fusion_db/ --test-queries test_queries/ \
       --device cpu --output-dir benchmark_cpu/
```

---

## Troubleshooting

### Issue: "Fusion database not found"

**Solution:**
```bash
python merizo_search/merizo.py rosetta build examples/database/ fusion_db/ -d cuda
```

### Issue: "No PDB files found"

**Solution:** Check your source directory has `.pdb` files:
```bash
ls examples/database/*.pdb
```

### Issue: Benchmark taking too long

**Solution:** Use fewer queries:
```bash
python benchmark_rosetta.py --num-queries 5 ...
```

### Issue: Out of memory errors

**Solution:** Use CPU device or build smaller database:
```bash
python benchmark_rosetta.py --device cpu ...
```

---

## Next Steps

### After Running Speed Benchmark

1. **Review Results**
   - Check `speed_benchmark_report.md`
   - Look at performance summary
   - Identify slow queries

2. **Generate Plots**
   ```bash
   python visualize_benchmark.py benchmark_results/
   ```

3. **Optimize if Needed**
   - Profile slow queries
   - Check GPU utilization
   - Verify database size

4. **Run Sensitivity/Accuracy** *(Coming Soon)*
   ```bash
   python benchmark_rosetta.py --mode all --fusion-db fusion_db/ \
          --test-queries test_queries/ --gold-standard interactions.tsv
   ```

---

## File Reference

| File | Purpose |
|------|---------|
| `benchmark_rosetta.py` | Main benchmark script |
| `prepare_test_queries.py` | Helper to create test sets |
| `visualize_benchmark.py` | Generate plots |
| `run_benchmark_example.sh` | Automated workflow (Linux/Mac) |
| `run_benchmark_example.bat` | Automated workflow (Windows) |
| `sample_protein_list.txt` | Template for custom protein selection |
| `BENCHMARK_README.md` | Detailed documentation |

---

## Tips

- **Start small:** Use 5-10 queries for quick validation
- **Use verbose mode:** Add `--verbose` to see detailed progress
- **Save logs:** Output is saved to `benchmark.log`
- **Compare runs:** Use different output directories to compare
- **Monitor GPU:** Run `nvidia-smi` in another terminal

---

## Example Output

```
=======================================================================
  ROSETTA STONE PPI BENCHMARK SUITE
=======================================================================
Mode: SPEED
Date: 2025-01-16 10:30:00
=======================================================================

[1/10] Processing: AF-Q14686.pdb
  ✓ Success in 23.45s
  → 15 predictions

[2/10] Processing: AF-P12345.pdb
  ✓ Success in 27.89s
  → 12 predictions

...

=======================================================================
QUERY SPEED BENCHMARK SUMMARY
=======================================================================
Total queries:      10
Successful:         10
Failed:             0

Average time:       25.30s
Median time:        24.10s
Throughput:         2.37 queries/minute

✓ PASS: System meets speed target
=======================================================================
```

---

**Need help?** See [BENCHMARK_README.md](BENCHMARK_README.md) for full documentation.
