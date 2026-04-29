#!/usr/bin/env python3
"""
Extract domain(s) from the TED365M database as CA-only PDB files.
Used to create ground-truth queries for self-hit validation of the search pipeline.

Usage:
    # Extract one domain by name:
    python extract_domain_pdb.py <db_json> <domain_name> <output.pdb>

    # Extract the first N domains listed in the pairlist index:
    python extract_domain_pdb.py <db_json> --batch <N> <output_dir>

Examples:
    python extract_domain_pdb.py merizo_pairlist_db/ted_pairlist.json \
        AF-A0A174DPU5-F1-model_v4_TED01 test_output/self_hit.pdb

    python extract_domain_pdb.py merizo_pairlist_db/ted_pairlist.json \
        --batch 5 test_output/extracted/
"""

import sys
import os
import mmap
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'merizo_search', 'programs'))

from Foldclass.dbutil import (
    read_dbinfo,
    retrieve_names_by_idx,
    retrieve_start_end_by_idx,
    retrieve_bytes,
    coord_conv,
)

AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def resolve(dbinfo, db_dir, key):
    """Resolve a file-path value from dbinfo (handles absolute or relative)."""
    p = dbinfo[key]
    return p if os.path.isabs(p) else os.path.join(db_dir, p)


def extract_domain(dbinfo, db_dir, domain_idx):
    """Return (name, sequence, CA_coords) for one domain by original-DB index."""
    idx_list = [domain_idx]

    with open(resolve(dbinfo, db_dir, 'db_names_f'), 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        names = retrieve_names_by_idx(idx=idx_list, mm=mm)
    name = names[0]

    with open(resolve(dbinfo, db_dir, 'sif'), 'rb') as sif, \
         open(resolve(dbinfo, db_dir, 'sdf'), 'rb') as sdf:
        si_mm = mmap.mmap(sif.fileno(), 0, access=mmap.ACCESS_READ)
        sd_mm = mmap.mmap(sdf.fileno(), 0, access=mmap.ACCESS_READ)
        startend = retrieve_start_end_by_idx(idx=idx_list, mm=si_mm)
        sequence = retrieve_bytes(startend[0][0], startend[0][1],
                                  mm=sd_mm, typeconv=lambda x: x.decode('ascii'))

    with open(resolve(dbinfo, db_dir, 'cif'), 'rb') as cif, \
         open(resolve(dbinfo, db_dir, 'cdf'), 'rb') as cdf:
        ci_mm = mmap.mmap(cif.fileno(), 0, access=mmap.ACCESS_READ)
        cd_mm = mmap.mmap(cdf.fileno(), 0, access=mmap.ACCESS_READ)
        startend = retrieve_start_end_by_idx(idx=idx_list, mm=ci_mm)
        coords = retrieve_bytes(startend[0][0], startend[0][1],
                                mm=cd_mm, typeconv=coord_conv)

    return name, sequence, coords


def write_ca_pdb(coords, sequence, output_path):
    """Write a CA-only PDB in Foldclass-compatible format."""
    assert len(coords) == len(sequence), \
        f"Coord/seq length mismatch: {len(coords)} vs {len(sequence)}"

    with open(output_path, 'w') as f:
        for i, (aa, coord) in enumerate(zip(sequence, coords), start=1):
            resname = AA_1TO3.get(aa, 'UNK')
            f.write(f"ATOM  {i: >5}  CA  {resname: >3} A{i: >4}    "
                    f"{coord[0]: >8.3f}{coord[1]: >8.3f}{coord[2]: >8.3f}"
                    f"  1.00  0.00\n")
        f.write("END\n")


def find_index_by_name(target_name, dbinfo, db_dir):
    """Find a domain's original-DB index by name, checking INDEX_LIST_FILE first if present."""
    names_path = resolve(dbinfo, db_dir, 'db_names_f')

    if 'INDEX_LIST_FILE' in dbinfo:
        index_list_path = os.path.join(db_dir, dbinfo['INDEX_LIST_FILE'])
        with open(index_list_path, 'r') as f:
            indices = [int(line.strip()) for line in f]
        print(f"  Scanning {len(indices)} pairlist indices for '{target_name}'...")
        with open(names_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            names = retrieve_names_by_idx(idx=indices, mm=mm)
        for name, orig_idx in zip(names, indices):
            if name == target_name:
                return orig_idx
        return None
    else:
        db_size = dbinfo['DB_SIZE']
        ENTRY_SIZE = 33
        print(f"  Linear scan of {db_size} entries...")
        with open(names_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for idx in range(db_size):
                mm.seek(idx * ENTRY_SIZE)
                raw = mm.read(ENTRY_SIZE)
                name = raw.decode('utf-8', errors='ignore').strip('\x00').strip()
                if name == target_name:
                    return idx
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract domain(s) from the TED365M database as CA-only PDB files.",
    )
    parser.add_argument("db_json", help="Path to Foldclass database JSON config (e.g. ted_pairlist.json)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single", help="Extract one domain by name")
    single.add_argument("domain_name", help="TED domain identifier (e.g. AF-P12345-F1-model_v4_TED01)")
    single.add_argument("output", help="Output PDB file path")

    batch = subparsers.add_parser("batch", help="Extract the first N domains from the pairlist index")
    batch.add_argument("n", type=int, help="Number of domains to extract")
    batch.add_argument("output_dir", help="Output directory for extracted PDB files")

    args = parser.parse_args()

    dbinfo = read_dbinfo(args.db_json)
    db_dir = os.path.dirname(os.path.abspath(args.db_json))

    if args.mode == "batch":
        os.makedirs(args.output_dir, exist_ok=True)
        if 'INDEX_LIST_FILE' not in dbinfo:
            print("Error: batch mode requires INDEX_LIST_FILE in the database JSON.")
            sys.exit(1)
        index_list_path = os.path.join(db_dir, dbinfo['INDEX_LIST_FILE'])
        with open(index_list_path, 'r') as f:
            all_indices = [int(line.strip()) for line in f]
        print(f"Pairlist has {len(all_indices)} domains. Extracting first {args.n}...")
        for i, idx in enumerate(all_indices[:args.n]):
            name, seq, coords = extract_domain(dbinfo, db_dir, idx)
            out_pdb = os.path.join(args.output_dir, f"{name}.pdb")
            write_ca_pdb(coords, seq, out_pdb)
            print(f"  [{i+1}/{args.n}] idx={idx:>12d}  len={len(seq):>4d}  {name}")
    else:
        print(f"Looking up '{args.domain_name}'...")
        domain_idx = find_index_by_name(args.domain_name, dbinfo, db_dir)
        if domain_idx is None:
            print(f"Error: '{args.domain_name}' not found.")
            sys.exit(1)
        print(f"  Found at original DB index: {domain_idx}")
        name, seq, coords = extract_domain(dbinfo, db_dir, domain_idx)
        write_ca_pdb(coords, seq, args.output)
        print(f"  Wrote {len(seq)}-residue CA-only PDB: {args.output}")


if __name__ == '__main__':
    main()
