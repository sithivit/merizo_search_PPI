#!/usr/bin/env python3
"""
Rosetta Stone PPI Benchmark Suite

This module provides comprehensive benchmarking for the Rosetta Stone PPI prediction system,
measuring speed, sensitivity (recall), and accuracy (precision).

Usage:
    # Run speed benchmark only
    python benchmark_rosetta.py --mode speed --fusion-db fusion_db/ --test-queries test_queries/

    # Run all benchmarks
    python benchmark_rosetta.py --mode all --fusion-db fusion_db/ --test-queries test_queries/ \
                                --gold-standard gold_standard.tsv

Author: Merizo-Search Team
"""

import argparse
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sys
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SpeedMetrics:
    """Speed performance metrics for Rosetta Stone system"""
    # Query search metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    # Timing statistics (seconds)
    avg_query_time: float = 0.0
    median_query_time: float = 0.0
    std_query_time: float = 0.0
    min_query_time: float = 0.0
    max_query_time: float = 0.0

    # Component breakdown (average per query)
    avg_segmentation_time: float = 0.0
    avg_embedding_time: float = 0.0
    avg_search_time: float = 0.0
    avg_ranking_time: float = 0.0

    # Throughput
    queries_per_minute: float = 0.0
    total_benchmark_time: float = 0.0

    # Per-query timing data
    query_times: List[float] = None
    query_names: List[str] = None

    def __post_init__(self):
        if self.query_times is None:
            self.query_times = []
        if self.query_names is None:
            self.query_names = []


@dataclass
class DatabaseBuildMetrics:
    """Speed metrics for database building"""
    total_proteins: int = 0
    processed_proteins: int = 0
    failed_proteins: int = 0
    multi_domain_proteins: int = 0

    # Timing
    total_time_hours: float = 0.0
    avg_time_per_protein: float = 0.0
    proteins_per_minute: float = 0.0

    # Memory
    peak_gpu_memory_gb: float = 0.0
    avg_gpu_memory_gb: float = 0.0

    # Component breakdown
    avg_segmentation_time: float = 0.0
    avg_embedding_time: float = 0.0
    avg_fusion_time: float = 0.0


class SpeedBenchmark:
    """Speed benchmarking for Rosetta Stone PPI prediction"""

    def __init__(
        self,
        fusion_db_path: Path,
        test_queries_path: Path,
        output_dir: Path,
        device: str = 'cuda',
        num_queries: Optional[int] = None
    ):
        self.fusion_db_path = Path(fusion_db_path)
        self.test_queries_path = Path(test_queries_path)
        self.output_dir = Path(output_dir)
        self.device = device
        self.num_queries = num_queries

        # Verify paths
        if not self.fusion_db_path.exists():
            raise FileNotFoundError(f"Fusion database not found: {self.fusion_db_path}")
        if not self.test_queries_path.exists():
            raise FileNotFoundError(f"Test queries directory not found: {self.test_queries_path}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find merizo.py script
        self.merizo_script = Path('merizo_search/merizo.py')
        if not self.merizo_script.exists():
            raise FileNotFoundError(f"Merizo script not found: {self.merizo_script}")

        logger.info(f"Speed Benchmark initialized")
        logger.info(f"  Fusion DB: {self.fusion_db_path}")
        logger.info(f"  Test queries: {self.test_queries_path}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Device: {self.device}")

    def get_test_queries(self) -> List[Path]:
        """Get list of test query PDB files"""
        # Find all PDB files
        pdb_files = list(self.test_queries_path.glob('*.pdb'))

        if not pdb_files:
            raise ValueError(f"No PDB files found in {self.test_queries_path}")

        # Limit number of queries if specified
        if self.num_queries:
            pdb_files = pdb_files[:self.num_queries]

        logger.info(f"Found {len(pdb_files)} test queries")
        return pdb_files

    def run_single_query(
        self,
        query_pdb: Path,
        output_prefix: str
    ) -> Tuple[float, bool, Optional[Dict]]:
        """
        Run Rosetta Stone search for a single query

        Returns:
            Tuple of (query_time, success, result_dict)
        """
        output_path = self.output_dir / f"{output_prefix}_rosetta.json"

        # Build command
        cmd = [
            sys.executable,  # Use current Python interpreter
            str(self.merizo_script),
            'rosetta',
            'search',
            str(query_pdb),
            str(self.fusion_db_path),
            str(self.output_dir / output_prefix),
            '-d', self.device,
            '--output-headers'
        ]

        logger.debug(f"Running command: {' '.join(cmd)}")

        # Run and time
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=False
            )
            query_time = time.time() - start_time

            # Check if successful
            if result.returncode != 0:
                logger.warning(f"Query failed: {query_pdb.name}")
                logger.debug(f"  stderr: {result.stderr}")
                return query_time, False, None

            # Load results if available
            result_dict = None
            if output_path.exists():
                try:
                    with open(output_path, 'r') as f:
                        result_dict = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON output: {output_path}")

            return query_time, True, result_dict

        except subprocess.TimeoutExpired:
            query_time = time.time() - start_time
            logger.warning(f"Query timed out after {query_time:.1f}s: {query_pdb.name}")
            return query_time, False, None

        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"Error running query {query_pdb.name}: {e}")
            return query_time, False, None

    def run_query_benchmark(self) -> SpeedMetrics:
        """
        Run speed benchmark on query searches

        Returns:
            SpeedMetrics object with results
        """
        logger.info("=" * 80)
        logger.info("RUNNING QUERY SPEED BENCHMARK")
        logger.info("=" * 80)

        # Get test queries
        test_queries = self.get_test_queries()

        metrics = SpeedMetrics()
        metrics.total_queries = len(test_queries)

        # Benchmark start
        benchmark_start = time.time()

        # Run each query
        for i, query_pdb in enumerate(test_queries, 1):
            logger.info(f"\n[{i}/{len(test_queries)}] Processing: {query_pdb.name}")

            output_prefix = f"query_{i:04d}_{query_pdb.stem}"
            query_time, success, result_dict = self.run_single_query(
                query_pdb,
                output_prefix
            )

            # Record results
            metrics.query_times.append(query_time)
            metrics.query_names.append(query_pdb.name)

            if success:
                metrics.successful_queries += 1
                logger.info(f"  ✓ Success in {query_time:.2f}s")

                # Log predictions if available
                if result_dict and 'num_predictions' in result_dict:
                    num_preds = result_dict['num_predictions']
                    logger.info(f"  → {num_preds} predictions")
            else:
                metrics.failed_queries += 1
                logger.info(f"  ✗ Failed after {query_time:.2f}s")

        # Calculate statistics
        benchmark_time = time.time() - benchmark_start
        metrics.total_benchmark_time = benchmark_time

        if metrics.query_times:
            metrics.avg_query_time = float(np.mean(metrics.query_times))
            metrics.median_query_time = float(np.median(metrics.query_times))
            metrics.std_query_time = float(np.std(metrics.query_times))
            metrics.min_query_time = float(np.min(metrics.query_times))
            metrics.max_query_time = float(np.max(metrics.query_times))
            metrics.queries_per_minute = 60.0 / metrics.avg_query_time if metrics.avg_query_time > 0 else 0

        # Log summary
        logger.info("\n" + "=" * 80)
        logger.info("QUERY SPEED BENCHMARK SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total queries:      {metrics.total_queries}")
        logger.info(f"Successful:         {metrics.successful_queries}")
        logger.info(f"Failed:             {metrics.failed_queries}")
        logger.info(f"")
        logger.info(f"Average time:       {metrics.avg_query_time:.2f}s")
        logger.info(f"Median time:        {metrics.median_query_time:.2f}s")
        logger.info(f"Std deviation:      {metrics.std_query_time:.2f}s")
        logger.info(f"Min time:           {metrics.min_query_time:.2f}s")
        logger.info(f"Max time:           {metrics.max_query_time:.2f}s")
        logger.info(f"")
        logger.info(f"Throughput:         {metrics.queries_per_minute:.2f} queries/minute")
        logger.info(f"Total time:         {benchmark_time:.2f}s ({benchmark_time/60:.1f} min)")
        logger.info("=" * 80)

        return metrics

    def save_results(self, metrics: SpeedMetrics, filename: str = "speed_benchmark_results.json"):
        """Save benchmark results to JSON file"""
        output_path = self.output_dir / filename

        # Convert to dict (handle numpy types)
        results = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'fusion_db': str(self.fusion_db_path),
                'test_queries_path': str(self.test_queries_path),
                'device': self.device,
                'num_queries': self.num_queries
            },
            'metrics': {
                'total_queries': metrics.total_queries,
                'successful_queries': metrics.successful_queries,
                'failed_queries': metrics.failed_queries,
                'avg_query_time': metrics.avg_query_time,
                'median_query_time': metrics.median_query_time,
                'std_query_time': metrics.std_query_time,
                'min_query_time': metrics.min_query_time,
                'max_query_time': metrics.max_query_time,
                'queries_per_minute': metrics.queries_per_minute,
                'total_benchmark_time': metrics.total_benchmark_time
            },
            'per_query_data': {
                'query_names': metrics.query_names,
                'query_times': metrics.query_times
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

        # Also save a simple CSV for easy analysis
        csv_path = self.output_dir / "speed_benchmark_per_query.csv"
        with open(csv_path, 'w') as f:
            f.write("query_name,time_seconds\n")
            for name, time_val in zip(metrics.query_names, metrics.query_times):
                f.write(f"{name},{time_val:.3f}\n")

        logger.info(f"Per-query data saved to: {csv_path}")

    def generate_report(self, metrics: SpeedMetrics):
        """Generate markdown report"""
        report_path = self.output_dir / "speed_benchmark_report.md"

        # Calculate pass/fail
        target_avg_time = 30.0  # seconds
        passed = metrics.avg_query_time <= target_avg_time

        report = f"""# Rosetta Stone Speed Benchmark Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Configuration

- **Fusion Database:** `{self.fusion_db_path}`
- **Test Queries:** `{self.test_queries_path}`
- **Device:** `{self.device}`
- **Number of Queries:** {metrics.total_queries}

---

## Results Summary

### Query Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Average Query Time** | {metrics.avg_query_time:.2f}s | ≤ 30s | {'✓ PASS' if passed else '✗ FAIL'} |
| **Median Query Time** | {metrics.median_query_time:.2f}s | - | - |
| **Std Deviation** | {metrics.std_query_time:.2f}s | - | - |
| **Min Query Time** | {metrics.min_query_time:.2f}s | - | - |
| **Max Query Time** | {metrics.max_query_time:.2f}s | - | - |

### Throughput

- **Queries per minute:** {metrics.queries_per_minute:.2f}
- **Total benchmark time:** {metrics.total_benchmark_time:.2f}s ({metrics.total_benchmark_time/60:.1f} minutes)

### Success Rate

- **Total queries:** {metrics.total_queries}
- **Successful:** {metrics.successful_queries} ({100*metrics.successful_queries/metrics.total_queries:.1f}%)
- **Failed:** {metrics.failed_queries} ({100*metrics.failed_queries/metrics.total_queries:.1f}%)

---

## Performance Analysis

### Query Time Distribution

```
Min:    {metrics.min_query_time:6.2f}s  ▕{'█' * 5}
Q1:     {np.percentile(metrics.query_times, 25):6.2f}s  ▕{'█' * 10}
Median: {metrics.median_query_time:6.2f}s  ▕{'█' * 15}
Q3:     {np.percentile(metrics.query_times, 75):6.2f}s  ▕{'█' * 20}
Max:    {metrics.max_query_time:6.2f}s  ▕{'█' * 25}
```

### Interpretation

{"✓ **PASS**: The system meets the target average query time of ≤30 seconds." if passed else "✗ **FAIL**: The system exceeds the target average query time of 30 seconds."}

The speed benchmark evaluates the throughput and latency of the Rosetta Stone search system.
A well-performing system should process queries in under 30 seconds on average.

---

## Recommendations

"""

        # Add recommendations
        if passed:
            report += """
**Performance is Good:**
- The system meets speed targets
- Query times are within acceptable range
- No optimization needed at this time
"""
        else:
            report += """
**Performance Needs Improvement:**
- Consider the following optimizations:
  1. Verify GPU is being utilized (check device configuration)
  2. Check if FAISS index can be moved to GPU for faster search
  3. Profile slow queries to identify bottlenecks
  4. Consider batch processing for multiple queries
"""

        # Add slowest queries
        if metrics.query_times:
            sorted_indices = np.argsort(metrics.query_times)[::-1][:5]
            report += "\n\n### Slowest Queries\n\n"
            report += "| Rank | Query | Time |\n"
            report += "|------|-------|------|\n"
            for rank, idx in enumerate(sorted_indices, 1):
                report += f"| {rank} | `{metrics.query_names[idx]}` | {metrics.query_times[idx]:.2f}s |\n"

        report += "\n---\n\n**End of Report**\n"

        # Save report
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to: {report_path}")

        # Print to console
        print("\n" + report)


def main():
    """Main entry point for benchmark script"""
    parser = argparse.ArgumentParser(
        description="Rosetta Stone PPI Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['speed', 'sensitivity', 'accuracy', 'all'],
        default='speed',
        help='Benchmark mode to run (default: speed)'
    )

    # Required paths
    parser.add_argument(
        '--fusion-db',
        required=True,
        type=Path,
        help='Path to fusion database directory'
    )

    parser.add_argument(
        '--test-queries',
        required=True,
        type=Path,
        help='Path to directory containing test query PDB files'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('benchmark_results'),
        help='Output directory for benchmark results (default: benchmark_results/)'
    )

    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'mps'],
        default='cuda',
        help='Device to use for inference (default: cuda)'
    )

    parser.add_argument(
        '--num-queries',
        type=int,
        help='Limit number of test queries (default: use all)'
    )

    parser.add_argument(
        '--gold-standard',
        type=Path,
        help='Path to gold standard interactions TSV (required for sensitivity/accuracy)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print header
    print("\n" + "=" * 80)
    print("  ROSETTA STONE PPI BENCHMARK SUITE")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # Run speed benchmark
    if args.mode in ['speed', 'all']:
        try:
            benchmark = SpeedBenchmark(
                fusion_db_path=args.fusion_db,
                test_queries_path=args.test_queries,
                output_dir=args.output_dir,
                device=args.device,
                num_queries=args.num_queries
            )

            metrics = benchmark.run_query_benchmark()
            benchmark.save_results(metrics)
            benchmark.generate_report(metrics)

        except Exception as e:
            logger.error(f"Speed benchmark failed: {e}", exc_info=True)
            return 1

    # Placeholder for other modes
    if args.mode in ['sensitivity', 'all']:
        logger.warning("Sensitivity benchmark not yet implemented")

    if args.mode in ['accuracy', 'all']:
        logger.warning("Accuracy benchmark not yet implemented")

    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
