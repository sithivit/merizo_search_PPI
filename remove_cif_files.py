#!/usr/bin/env python3
"""
Remove all .cif files in the EColi directory.
"""

import os
from pathlib import Path


def remove_cif_files(directory):
    """
    Remove all .cif files in the specified directory.

    Args:
        directory: Path to the directory containing .cif files
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    # Find all .cif files (including .cif.gz)
    cif_files = list(directory.glob("*.cif*"))

    if not cif_files:
        print(f"No .cif files found in {directory}")
        return

    print(f"Found {len(cif_files)} .cif files to remove")

    # Ask for confirmation
    response = input(f"Are you sure you want to delete {len(cif_files)} files? (y/N): ")

    if response.lower() != 'y':
        print("Operation cancelled")
        return

    removed_count = 0
    failed_count = 0

    for cif_file in cif_files:
        try:
            cif_file.unlink()
            removed_count += 1

            # Print progress every 100 files
            if removed_count % 100 == 0:
                print(f"Removed {removed_count}/{len(cif_files)} files...")

        except Exception as e:
            print(f"Error removing {cif_file}: {e}")
            failed_count += 1

    print(f"\nRemoval complete!")
    print(f"Successfully removed: {removed_count} files")
    if failed_count > 0:
        print(f"Failed: {failed_count} files")


if __name__ == "__main__":
    # Path to EColi directory
    ecoli_dir = Path(__file__).parent / "EColi"

    print(f"Starting removal in: {ecoli_dir}")
    remove_cif_files(ecoli_dir)
