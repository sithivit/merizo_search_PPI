#!/usr/bin/env python3
"""
Visualize Benchmark Results

Creates visualizations from benchmark results including:
- Query time distribution histogram
- Query time boxplot
- Cumulative time plot
- Performance comparison charts

Usage:
    python visualize_benchmark.py benchmark_results/
    python visualize_benchmark.py benchmark_results/ --output plots/
"""

import argparse
import json
from pathlib import Path
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip install pandas matplotlib seaborn")
    sys.exit(1)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_benchmark_data(results_dir: Path):
    """Load benchmark results from JSON and CSV files"""

    # Load JSON results
    json_path = results_dir / "speed_benchmark_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Benchmark results not found: {json_path}")

    with open(json_path, 'r') as f:
        results = json.load(f)

    # Load CSV data
    csv_path = results_dir / "speed_benchmark_per_query.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Per-query data not found: {csv_path}")

    df = pd.read_csv(csv_path)

    return results, df


def plot_histogram(df: pd.DataFrame, output_dir: Path, results: dict):
    """Plot query time distribution histogram"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(df['time_seconds'], bins=20, edgecolor='black', alpha=0.7)

    # Add statistics lines
    mean_time = results['metrics']['avg_query_time']
    median_time = results['metrics']['median_query_time']

    ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f}s')
    ax.axvline(median_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_time:.1f}s')

    # Add target line
    ax.axvline(30, color='orange', linestyle=':', linewidth=2, label='Target: 30s')

    ax.set_xlabel('Query Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Query Time Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'query_time_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_boxplot(df: pd.DataFrame, output_dir: Path, results: dict):
    """Plot query time boxplot"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Boxplot
    box = ax.boxplot(df['time_seconds'], vert=True, patch_artist=True, widths=0.5)
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][0].set_alpha(0.7)

    # Add target line
    ax.axhline(30, color='orange', linestyle=':', linewidth=2, label='Target: 30s')

    # Add statistics
    stats_text = (
        f"Min: {results['metrics']['min_query_time']:.1f}s\n"
        f"Q1: {np.percentile(df['time_seconds'], 25):.1f}s\n"
        f"Median: {results['metrics']['median_query_time']:.1f}s\n"
        f"Q3: {np.percentile(df['time_seconds'], 75):.1f}s\n"
        f"Max: {results['metrics']['max_query_time']:.1f}s"
    )
    ax.text(1.15, 0.5, stats_text, transform=ax.transData,
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('Query Time (seconds)', fontsize=12)
    ax.set_title('Query Time Distribution (Boxplot)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['All Queries'])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'query_time_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_cumulative(df: pd.DataFrame, output_dir: Path):
    """Plot cumulative query time"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by time and calculate cumulative
    df_sorted = df.sort_values('time_seconds')
    cumulative_time = np.cumsum(df_sorted['time_seconds'])

    ax.plot(range(1, len(cumulative_time) + 1), cumulative_time / 60,
            linewidth=2, color='steelblue', marker='o', markersize=3)

    ax.set_xlabel('Number of Queries Processed', fontsize=12)
    ax.set_ylabel('Cumulative Time (minutes)', fontsize=12)
    ax.set_title('Cumulative Query Processing Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add total time annotation
    total_time = cumulative_time.iloc[-1] / 60
    ax.text(len(cumulative_time) * 0.95, total_time * 0.95,
            f'Total: {total_time:.1f} min',
            fontsize=11, ha='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    output_path = output_dir / 'cumulative_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_slowest_queries(df: pd.DataFrame, output_dir: Path, top_n: int = 10):
    """Plot slowest queries bar chart"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get top N slowest
    slowest = df.nlargest(top_n, 'time_seconds').sort_values('time_seconds')

    # Shorten names for display
    labels = [name[:30] + '...' if len(name) > 30 else name for name in slowest['query_name']]

    bars = ax.barh(range(len(slowest)), slowest['time_seconds'], color='coral', edgecolor='black')

    # Add target line
    ax.axvline(30, color='orange', linestyle=':', linewidth=2, label='Target: 30s')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, slowest['time_seconds'])):
        ax.text(val + 0.5, i, f'{val:.1f}s', va='center', fontsize=9)

    ax.set_yticks(range(len(slowest)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Query Time (seconds)', fontsize=12)
    ax.set_title(f'Top {top_n} Slowest Queries', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'slowest_queries.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_performance_summary(results: dict, output_dir: Path):
    """Create performance summary dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Summary Dashboard', fontsize=16, fontweight='bold')

    metrics = results['metrics']

    # 1. Success Rate Pie Chart
    ax = axes[0, 0]
    sizes = [metrics['successful_queries'], metrics['failed_queries']]
    labels = ['Successful', 'Failed']
    colors = ['lightgreen', 'lightcoral']
    explode = (0.05, 0)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax.set_title('Query Success Rate', fontweight='bold')

    # 2. Timing Statistics Bar Chart
    ax = axes[0, 1]
    stats = ['Min', 'Median', 'Mean', 'Max', 'Target']
    values = [
        metrics['min_query_time'],
        metrics['median_query_time'],
        metrics['avg_query_time'],
        metrics['max_query_time'],
        30
    ]
    colors_bars = ['green', 'blue', 'orange', 'red', 'gray']

    bars = ax.bar(stats, values, color=colors_bars, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Query Time Statistics', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)

    # 3. Throughput Gauge
    ax = axes[1, 0]
    throughput = metrics['queries_per_minute']
    target_throughput = 2.0

    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)

    ax.plot(theta, r, 'k-', linewidth=2)
    ax.fill_between(theta[:33], 0, r[:33], color='red', alpha=0.3, label='Poor (<1.5)')
    ax.fill_between(theta[33:66], 0, r[33:66], color='yellow', alpha=0.3, label='Fair (1.5-2.0)')
    ax.fill_between(theta[66:], 0, r[66:], color='green', alpha=0.3, label='Good (>2.0)')

    # Add needle
    angle = np.pi * min(throughput / 4, 1)  # Scale to 0-π
    ax.plot([0, np.cos(angle)], [0, np.sin(angle)], 'b-', linewidth=3)
    ax.plot(0, 0, 'bo', markersize=10)

    ax.text(0, -0.3, f'{throughput:.2f}\nqueries/min', ha='center', fontsize=12, fontweight='bold')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.2)
    ax.axis('off')
    ax.set_title('Throughput Gauge', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # 4. Key Metrics Table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Total Queries', f"{metrics['total_queries']}"],
        ['Successful', f"{metrics['successful_queries']} ({100*metrics['successful_queries']/metrics['total_queries']:.1f}%)"],
        ['Avg Time', f"{metrics['avg_query_time']:.2f}s"],
        ['Throughput', f"{metrics['queries_per_minute']:.2f} q/min"],
        ['Total Time', f"{metrics['total_benchmark_time']/60:.1f} min"],
        ['Status', '✓ PASS' if metrics['avg_query_time'] <= 30 else '✗ FAIL']
    ]

    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style status row
    status_color = '#90EE90' if metrics['avg_query_time'] <= 30 else '#FFB6C1'
    table[(6, 1)].set_facecolor(status_color)
    table[(6, 1)].set_text_props(weight='bold')

    ax.set_title('Key Metrics', fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = output_dir / 'performance_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Rosetta Stone benchmark results"
    )

    parser.add_argument(
        'results_dir',
        type=Path,
        help='Directory containing benchmark results'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output directory for plots (default: results_dir/plots/)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of slowest queries to show (default: 10)'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.results_dir / 'plots'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("  BENCHMARK VISUALIZATION")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory:  {output_dir}")
    print("="*70 + "\n")

    # Load data
    print("Loading benchmark data...")
    try:
        results, df = load_benchmark_data(args.results_dir)
        print(f"  ✓ Loaded {len(df)} query results\n")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return 1

    # Generate plots
    print("Generating visualizations...")

    try:
        plot_histogram(df, output_dir, results)
        plot_boxplot(df, output_dir, results)
        plot_cumulative(df, output_dir)
        plot_slowest_queries(df, output_dir, args.top_n)
        plot_performance_summary(results, output_dir)

        print("\n" + "="*70)
        print("  VISUALIZATION COMPLETE")
        print("="*70)
        print(f"\nPlots saved to: {output_dir}/")
        print("\nGenerated plots:")
        print("  1. query_time_histogram.png   - Distribution of query times")
        print("  2. query_time_boxplot.png     - Boxplot with statistics")
        print("  3. cumulative_time.png        - Cumulative processing time")
        print("  4. slowest_queries.png        - Top slowest queries")
        print("  5. performance_summary.png    - Dashboard overview")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
