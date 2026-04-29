#!/usr/bin/env python3
"""
Dump all domain names from the Zhang sub-database JSON config to a text file.
One domain name per line — used as input to rosetta_build_template_index.py.

Usage:
    python scripts/rosetta_dump_zhang_domains.py \\
        --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \\
        --output benchmark_cache/zhang_domain_names.txt
"""
import argparse
import mmap
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'merizo_search', 'programs'))
from Foldclass.dbutil import read_dbinfo, retrieve_names_by_idx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def run(args):
    db_json = args.zhang_db + ".json"
    dbinfo = read_dbinfo(db_json)
    db_dir = os.path.dirname(os.path.abspath(db_json))

    def resolve(key):
        p = dbinfo[key]
        return p if os.path.isabs(p) else os.path.join(db_dir, p)

    if "INDEX_LIST_FILE" in dbinfo:
        index_list_path = os.path.join(db_dir, dbinfo["INDEX_LIST_FILE"])
        with open(index_list_path) as f:
            indices = [int(line.strip()) for line in f]
        with open(resolve("db_names_f"), "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            names = retrieve_names_by_idx(idx=indices, mm=mm)
    else:
        ENTRY_SIZE = 33
        db_size = dbinfo["DB_SIZE"]
        names = []
        with open(resolve("db_names_f"), "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for idx in range(db_size):
                mm.seek(idx * ENTRY_SIZE)
                raw = mm.read(ENTRY_SIZE)
                name = raw.decode("utf-8", errors="ignore").strip("\x00").strip()
                if name:
                    names.append(name)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        for name in names:
            fh.write(name + "\n")
    print(f"Wrote {len(names):,} domain names to {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zhang-db", required=True, dest="zhang_db")
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
