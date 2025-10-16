#!/bin/bash
# Quick start example for running Rosetta Stone benchmarks

echo "======================================================================="
echo "  ROSETTA STONE BENCHMARK - QUICK START EXAMPLE"
echo "======================================================================="
echo ""

# Configuration
FUSION_DB="fusion_db"
SOURCE_DATABASE="examples/database"
TEST_QUERIES="test_queries_benchmark"
NUM_QUERIES=10
DEVICE="cuda"

# Step 1: Check if fusion database exists
echo "[1/3] Checking fusion database..."
if [ ! -d "$FUSION_DB" ]; then
    echo "  ✗ Fusion database not found at: $FUSION_DB"
    echo "  → Please build the fusion database first:"
    echo "    python merizo_search/merizo.py rosetta build $SOURCE_DATABASE $FUSION_DB -d $DEVICE"
    exit 1
else
    echo "  ✓ Fusion database found"
fi

# Step 2: Prepare test queries
echo ""
echo "[2/3] Preparing test queries..."
if [ ! -d "$TEST_QUERIES" ]; then
    echo "  Creating test query set (randomly sampling $NUM_QUERIES proteins)..."
    python prepare_test_queries.py \
        --source "$SOURCE_DATABASE" \
        --output "$TEST_QUERIES" \
        --num $NUM_QUERIES

    if [ $? -ne 0 ]; then
        echo "  ✗ Failed to prepare test queries"
        exit 1
    fi
else
    echo "  ✓ Test queries already exist at: $TEST_QUERIES"
    count=$(ls -1 "$TEST_QUERIES"/*.pdb 2>/dev/null | wc -l)
    echo "    Found $count PDB files"
fi

# Step 3: Run speed benchmark
echo ""
echo "[3/3] Running speed benchmark..."
echo "  This will take approximately $((NUM_QUERIES * 30 / 60)) minutes..."
echo ""

python benchmark_rosetta.py \
    --mode speed \
    --fusion-db "$FUSION_DB" \
    --test-queries "$TEST_QUERIES" \
    --device "$DEVICE" \
    --output-dir benchmark_results \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "  BENCHMARK COMPLETE!"
    echo "======================================================================="
    echo ""
    echo "Results saved to: benchmark_results/"
    echo ""
    echo "View results:"
    echo "  - Report:     cat benchmark_results/speed_benchmark_report.md"
    echo "  - JSON:       cat benchmark_results/speed_benchmark_results.json"
    echo "  - Per-query:  cat benchmark_results/speed_benchmark_per_query.csv"
    echo ""
else
    echo ""
    echo "✗ Benchmark failed. Check benchmark.log for details."
    exit 1
fi
