#!/usr/bin/env python3
"""
Prepare Test Queries for Benchmarking

This script helps prepare a test query set from your protein database.
It can randomly sample proteins or select specific proteins for benchmarking.

Usage:
    # Random sample
    python prepare_test_queries.py --source examples/database/ --output test_queries/ --num 50

    # Select specific proteins
    python prepare_test_queries.py --source examples/database/ --output test_queries/ \
                                   --select protein_list.txt

    # Copy all
    python prepare_test_queries.py --source examples/database/ --output test_queries/ --all
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_pdb_files(source_dir: Path) -> List[Path]:
    """Get all PDB files from source directory"""
    pdb_files = list(source_dir.glob('*.pdb'))
    logger.info(f"Found {len(pdb_files)} PDB files in {source_dir}")
    return pdb_files


def random_sample(pdb_files: List[Path], num: int, seed: int = 42) -> List[Path]:
    """Randomly sample PDB files"""
    if num > len(pdb_files):
        logger.warning(f"Requested {num} files but only {len(pdb_files)} available")
        num = len(pdb_files)

    random.seed(seed)
    sampled = random.sample(pdb_files, num)
    logger.info(f"Randomly sampled {len(sampled)} files")
    return sampled


def select_from_list(pdb_files: List[Path], list_file: Path, source_dir: Path) -> List[Path]:
    """Select specific PDB files from a list"""
    # Read list
    with open(list_file, 'r') as f:
        selected_names = [line.strip() for line in f if line.strip()]

    # Find matching files
    pdb_dict = {pdb.stem: pdb for pdb in pdb_files}

    selected = []
    missing = []
    for name in selected_names:
        # Try exact match first
        if name in pdb_dict:
            selected.append(pdb_dict[name])
        # Try with .pdb extension
        elif name.endswith('.pdb') and name[:-4] in pdb_dict:
            selected.append(pdb_dict[name[:-4]])
        else:
            missing.append(name)

    logger.info(f"Selected {len(selected)} files from list")
    if missing:
        logger.warning(f"Could not find {len(missing)} files: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return selected


def copy_files(files: List[Path], output_dir: Path, dry_run: bool = False):
    """Copy PDB files to output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying {len(files)} files to {output_dir}")

    if dry_run:
        logger.info("DRY RUN - No files will be copied")
        for pdb in files[:10]:  # Show first 10
            logger.info(f"  Would copy: {pdb.name}")
        if len(files) > 10:
            logger.info(f"  ... and {len(files) - 10} more")
        return

    # Copy files
    for i, pdb in enumerate(files, 1):
        dest = output_dir / pdb.name
        shutil.copy2(pdb, dest)
        if i % 10 == 0:
            logger.info(f"  Copied {i}/{len(files)} files...")

    logger.info(f"✓ Successfully copied {len(files)} files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare test queries for Rosetta Stone benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--source',
        type=Path,
        required=True,
        help='Source directory containing PDB files'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for test queries'
    )

    # Selection mode (mutually exclusive)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        '--num',
        type=int,
        help='Number of random files to sample'
    )
    selection.add_argument(
        '--select',
        type=Path,
        help='Text file with list of protein IDs to select (one per line)'
    )
    selection.add_argument(
        '--all',
        action='store_true',
        help='Copy all PDB files'
    )

    # Optional arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be copied without actually copying'
    )

    args = parser.parse_args()

    # Validate source directory
    if not args.source.exists():
        logger.error(f"Source directory not found: {args.source}")
        return 1

    if not args.source.is_dir():
        logger.error(f"Source is not a directory: {args.source}")
        return 1

    # Get all PDB files
    pdb_files = get_pdb_files(args.source)

    if not pdb_files:
        logger.error("No PDB files found in source directory")
        return 1

    # Select files based on mode
    if args.num:
        selected_files = random_sample(pdb_files, args.num, args.seed)
    elif args.select:
        if not args.select.exists():
            logger.error(f"Selection list file not found: {args.select}")
            return 1
        selected_files = select_from_list(pdb_files, args.select, args.source)
    elif args.all:
        selected_files = pdb_files
        logger.info(f"Selected all {len(selected_files)} files")
    else:
        logger.error("Must specify --num, --select, or --all")
        return 1

    if not selected_files:
        logger.error("No files selected")
        return 1

    # Copy files
    copy_files(selected_files, args.output, args.dry_run)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Source directory:    {args.source}")
    print(f"Output directory:    {args.output}")
    print(f"Files selected:      {len(selected_files)}")
    print(f"Mode:                ", end="")
    if args.num:
        print(f"Random sample ({args.num} files, seed={args.seed})")
    elif args.select:
        print(f"From list ({args.select})")
    elif args.all:
        print("All files")
    print("="*60)

    if not args.dry_run:
        print(f"\n✓ Test queries ready at: {args.output}")
        print(f"\nRun benchmark with:")
        print(f"  python benchmark_rosetta.py --mode speed \\")
        print(f"      --fusion-db fusion_db/ \\")
        print(f"      --test-queries {args.output}/")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
