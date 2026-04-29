#!/usr/bin/env python3
"""
Extract all .gz files in the EColi directory and remove the compressed files.
"""

import gzip
import os
import shutil
from pathlib import Path


def extract_gz_files(directory):
    """
    Extract all .gz files in the specified directory and delete them after extraction.

    Args:
        directory: Path to the directory containing .gz files
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    # Find all .gz files
    gz_files = list(directory.glob("*.gz"))

    if not gz_files:
        print(f"No .gz files found in {directory}")
        return

    print(f"Found {len(gz_files)} .gz files to extract")

    extracted_count = 0
    failed_count = 0

    for gz_file in gz_files:
        try:
            # Output filename (remove .gz extension)
            output_file = gz_file.with_suffix('')

            # Extract the file
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Delete the .gz file
            gz_file.unlink()

            extracted_count += 1

            # Print progress every 100 files
            if extracted_count % 100 == 0:
                print(f"Extracted {extracted_count}/{len(gz_files)} files...")

        except Exception as e:
            print(f"Error extracting {gz_file}: {e}")
            failed_count += 1

    print(f"\nExtraction complete!")
    print(f"Successfully extracted: {extracted_count} files")
    if failed_count > 0:
        print(f"Failed: {failed_count} files")


if __name__ == "__main__":
    # Path to EColi directory
    ecoli_dir = Path(__file__).parent / "EColi"

    print(f"Starting extraction in: {ecoli_dir}")
    extract_gz_files(ecoli_dir)
