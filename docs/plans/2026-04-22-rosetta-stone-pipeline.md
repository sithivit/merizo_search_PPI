# Rosetta Stone PPI Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current one-sided domain B search with a true paired-domain homology
transfer (Rosetta Stone) algorithm, where a template pair (A, B) co-occurring in a TED
protein predicts that any protein with an A-like domain interacts with any protein with a
B-like domain — eliminating the requirement for the query protein to be multi-domain.

**Architecture:** Two new search scripts search ALL domains of ALL benchmark proteins against
the Zhang sub-database. A template pair index derived from the TED pair list enables
cross-join scoring: a pair (P1, P2) scores positively only when some template (A, B) has
P1 carrying an A-like domain AND P2 carrying a B-like domain simultaneously. Phase 2 extends
the Zhang sub-database to cover single-domain proteins (currently excluded), increasing coverage.

**Tech Stack:** Python 3.9+, SQLite3 (domain metadata), Foldclass binary DB (mmap), TED
pair list (streaming), existing `merizo_search/merizo.py search` command, same weighted-PR
benchmark framework.

---

## Background: What Changes and Why

### Current (one-sided) algorithm
```
For protein P (must be in pair list / multi-domain):
  domain_B = pair_list[P].domain_B   # interface domain
  H(P) = search(domain_B, zhang_subdb)
  candidates = {protein owning each domain in H(P)}
  score(P, candidate) = max_tm of best hit
```

### Rosetta Stone (paired-domain transfer)
```
Template library: TED pair list = {(A, B) : A and B co-occur in same protein}

For each protein P in benchmark (both sides):
  for each TED domain d of P:
    H[P][d] = search(d, zhang_subdb)  # hits to Zhang template domains

score(P1, P2) = max over all template pairs (A, B) from TED where:
                  A in H[P1] AND B in H[P2]:  min(H[P1][A], H[P2][B])
                  OR B in H[P1] AND A in H[P2]: min(H[P1][B], H[P2][A])
```

### Why coverage improves
- Current: BOTH P1 and P2 must be in the pair list (multi-domain). Only 19.8% of positive pairs qualify.
- Rosetta Stone: P1/P2 only need TED domains (even one domain is sufficient). Coverage jumps substantially.

---

## Phase 1: Algorithm + Scoring (Days 1-2, on your laptop)

These tasks implement the Rosetta Stone scoring in pure Python against the existing
search cache. No new searches needed yet. This proves the algorithm is correct.

---

### Task 1: Create Branch

**Step 1: Branch from filter**
```bash
git checkout filter
git pull origin filter
git checkout -b rosetta_stone_v2
```

**Step 2: Create tests directory**
```bash
mkdir -p tests
```

**Step 3: Commit**
```bash
git commit --allow-empty -m "chore: start rosetta_stone_v2 branch"
```

---

### Task 2: Write Template Pair Index Builder

Reads the TED pair list and builds `{domain_name: [co-occurring domain names]}`,
restricted to pairs involving Zhang benchmark domains. Saves as JSON.

**Files:**
- Create: `scripts/rosetta_dump_zhang_domains.py`
- Create: `scripts/rosetta_build_template_index.py`
- Create: `tests/test_rosetta_template_index.py`

**Step 1: Write `scripts/rosetta_dump_zhang_domains.py`**

```python
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
```

**Step 2: Write `scripts/rosetta_build_template_index.py`**

```python
#!/usr/bin/env python3
"""
rosetta_build_template_index.py

Build a bidirectional template pair index from the TED pair list, restricted
to pairs where at least one domain belongs to a Zhang benchmark protein.
Saves as JSON: {domain_name: [co-occurring domain names]}.

Usage:
    python scripts/rosetta_build_template_index.py \\
        --pair-list /mnt/bigstore/ted/pair_list_20250128 \\
        --zhang-domain-names benchmark_cache/zhang_domain_names.txt \\
        --output benchmark_cache/zhang_template_index.json
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def load_zhang_domain_names(path: str) -> set:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def build_index(pair_list_path: str, zhang_domains: set) -> dict:
    """
    Stream the TED pair list. For each pair (A, B) where at least one of A, B
    is a Zhang benchmark domain, add A->B and B->A to the index.
    Returns {domain_name: list_of_co_occurring_domains}.
    """
    index = {}  # domain -> set (converted to list when saving)
    n_included = 0
    n_total = 0

    with open(pair_list_path) as fh:
        for line in fh:
            n_total += 1
            if n_total % 5_000_000 == 0:
                log.info(f"  {n_total:,} lines read, {n_included:,} included...")
            parts = line.split()
            if not parts or ":" not in parts[0]:
                continue
            domain_a, domain_b = parts[0].split(":", 1)
            if domain_a not in zhang_domains and domain_b not in zhang_domains:
                continue
            n_included += 1
            index.setdefault(domain_a, set()).add(domain_b)
            index.setdefault(domain_b, set()).add(domain_a)

    log.info(f"Pair list: {n_total:,} lines, {n_included:,} pairs retained")
    log.info(f"Index: {len(index):,} unique domain entries")
    return {k: sorted(v) for k, v in index.items()}  # sets -> sorted lists for JSON


def run(args):
    log.info("Loading Zhang domain names...")
    zhang_domains = load_zhang_domain_names(args.zhang_domain_names)
    log.info(f"Zhang domains: {len(zhang_domains):,}")

    log.info("Streaming pair list and building index (may take 20-30 min)...")
    index = build_index(args.pair_list, zhang_domains)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(index, fh)
    log.info(f"Index saved to {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-list", required=True, dest="pair_list")
    p.add_argument("--zhang-domain-names", required=True, dest="zhang_domain_names")
    p.add_argument("--output",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "zhang_template_index.json"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
```

**Step 3: Write unit tests `tests/test_rosetta_template_index.py`**

```python
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def _write_pair_list(path, pairs):
    with open(path, 'w') as f:
        for a, b in pairs:
            f.write(f"{a}:{b} X Y 1.0\n")


def _write_zhang_names(path, names):
    with open(path, 'w') as f:
        for n in names:
            f.write(n + "\n")


def test_bidirectional_index():
    from rosetta_build_template_index import build_index, load_zhang_domain_names
    with tempfile.TemporaryDirectory() as tmp:
        pair_list = os.path.join(tmp, "pairs.txt")
        names_file = os.path.join(tmp, "names.txt")
        _write_pair_list(pair_list, [("domA", "domB"), ("domC", "domD")])
        _write_zhang_names(names_file, ["domA", "domB"])

        zhang_domains = load_zhang_domain_names(names_file)
        index = build_index(pair_list, zhang_domains)

        assert "domA" in index
        assert "domB" in index["domA"]
        assert "domA" in index["domB"]  # bidirectional
        assert "domC" not in index      # not a Zhang domain
        assert "domD" not in index


def test_non_zhang_pair_excluded():
    from rosetta_build_template_index import build_index, load_zhang_domain_names
    with tempfile.TemporaryDirectory() as tmp:
        pair_list = os.path.join(tmp, "pairs.txt")
        names_file = os.path.join(tmp, "names.txt")
        _write_pair_list(pair_list, [("domX", "domY")])
        _write_zhang_names(names_file, ["domA"])  # neither X nor Y is Zhang

        zhang_domains = load_zhang_domain_names(names_file)
        index = build_index(pair_list, zhang_domains)
        assert len(index) == 0


def test_saves_and_loads_json():
    from rosetta_build_template_index import build_index, load_zhang_domain_names
    with tempfile.TemporaryDirectory() as tmp:
        pair_list = os.path.join(tmp, "pairs.txt")
        names_file = os.path.join(tmp, "names.txt")
        out_json = os.path.join(tmp, "index.json")
        _write_pair_list(pair_list, [("domA", "domB")])
        _write_zhang_names(names_file, ["domA"])

        zhang_domains = load_zhang_domain_names(names_file)
        index = build_index(pair_list, zhang_domains)

        with open(out_json, "w") as fh:
            json.dump(index, fh)
        with open(out_json) as fh:
            loaded = json.load(fh)

        assert loaded["domA"] == ["domB"]
        assert loaded["domB"] == ["domA"]
```

**Step 4: Run tests to verify they pass**
```bash
python -m pytest tests/test_rosetta_template_index.py -v
```
Expected: 3 PASSED

**Step 5: Commit**
```bash
git add scripts/rosetta_dump_zhang_domains.py \
        scripts/rosetta_build_template_index.py \
        tests/test_rosetta_template_index.py
git commit -m "feat: add Rosetta Stone template pair index builder"
```

---

### Task 3: Write the Rosetta Stone Benchmark Script

The core new script. Replaces one-sided `score_pair` with bidirectional cross-join.

**Files:**
- Create: `scripts/benchmark_rosetta_stone.py`
- Create: `tests/test_benchmark_rosetta_stone.py`

**Step 1: Write the failing tests first**

Create `tests/test_benchmark_rosetta_stone.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def test_score_pair_bridge_found():
    """P1 has A-like domain, P2 has B-like domain — bridge via template (A,B)."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9, "domX": 0.7},
        "P2": {"domB": 0.8, "domY": 0.6},
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    assert abs(score - 0.8) < 1e-6  # min(0.9, 0.8) = 0.8


def test_score_pair_no_bridge():
    """No template pair bridges P1 hits to P2 hits."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9},
        "P2": {"domC": 0.8},  # domC not paired with domA
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    assert score == 0.0


def test_score_pair_reverse_bridge():
    """Template index is bidirectional: B in H[P1], A in H[P2] also works."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domB": 0.75},
        "P2": {"domA": 0.85},
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    assert abs(score - 0.75) < 1e-6  # min(0.85, 0.75) = 0.75


def test_score_pair_best_bridge_wins():
    """With multiple bridging templates, best score is returned."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9, "domC": 0.6},
        "P2": {"domB": 0.5, "domD": 0.95},
    }
    template_index = {
        "domA": ["domB"], "domB": ["domA"],
        "domC": ["domD"], "domD": ["domC"],
    }
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    # Bridge 1: min(0.9, 0.5) = 0.5 ; Bridge 2: min(0.6, 0.95) = 0.6
    assert abs(score - 0.6) < 1e-6


def test_build_protein_hit_map_max_per_template():
    """Multiple domains in P1 — max TM per template domain is taken."""
    from benchmark_rosetta_stone import build_protein_hit_map
    per_domain_hits = {
        ("P1", "dom00"): {"tmplX": 0.7, "tmplY": 0.5},
        ("P1", "dom01"): {"tmplX": 0.9, "tmplZ": 0.4},
    }
    hit_map = build_protein_hit_map(per_domain_hits)
    assert hit_map["P1"]["tmplX"] == 0.9  # max over domains
    assert hit_map["P1"]["tmplY"] == 0.5
    assert hit_map["P1"]["tmplZ"] == 0.4


def test_score_missing_protein_returns_zero():
    """If a protein has no search results, score is 0."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {"P1": {"domA": 0.9}}
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "MISSING", domain_hits, template_index)
    assert score == 0.0
```

**Step 2: Run tests — expect ImportError (module not written yet)**
```bash
python -m pytest tests/test_benchmark_rosetta_stone.py -v 2>&1 | head -20
```
Expected: `ImportError: No module named 'benchmark_rosetta_stone'`

**Step 3: Write `scripts/benchmark_rosetta_stone.py`**

```python
#!/usr/bin/env python3
"""
benchmark_rosetta_stone.py

Rosetta Stone PPI benchmark: paired-domain homology transfer.

Algorithm
---------
For each benchmark protein P, all its TED domains are searched against the
Zhang sub-database. The per-domain hits are aggregated into:
  H[P] = {template_domain: max_tm_across_all_domains_of_P}

For a candidate pair (P1, P2), we look for the best "bridging" template pair
(A, B) from the TED pair list such that A in H[P1] and B in H[P2]:
  score = min(H[P1][A], H[P2][B])
The final score is the maximum over all valid bridging templates. This is
checked bidirectionally (A in H[P1] and B in H[P2], OR B in H[P1] and A in H[P2]).

Usage (Phase 1 — existing cache, Domain B only):
    python scripts/benchmark_rosetta_stone.py \\
        --search-cache-dir benchmark_cache/searches \\
        --template-index benchmark_cache/zhang_template_index.json \\
        --output-dir benchmark_results_rosetta_phase1

Usage (Phase 2 — new per-domain cache, all domains both sides):
    python scripts/benchmark_rosetta_stone.py \\
        --search-cache-dir benchmark_cache/rosetta_searches \\
        --template-index benchmark_cache/zhang_template_index.json \\
        --output-dir benchmark_results_rosetta_phase2 \\
        --multi-domain
"""

import argparse
import csv
import json
import logging
import os
import random
import sys

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CONTROLS = os.path.join(PROJECT_ROOT, "benchmark_cache", "benchmarks",
                                "positives_and_negatives.tsv")
DEFAULT_SEARCH_CACHE = os.path.join(PROJECT_ROOT, "benchmark_cache", "searches")
DEFAULT_TEMPLATE_INDEX = os.path.join(PROJECT_ROOT, "benchmark_cache",
                                      "zhang_template_index.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark_results_rosetta")

COL_TARGET = 2
COL_MAX_TM = 10


# ---------------------------------------------------------------------------
# Universe and input loading
# ---------------------------------------------------------------------------

def get_universe(search_cache_dir: str, multi_domain: bool) -> set:
    """Proteins with at least one non-empty search TSV in the cache directory."""
    proteins = set()
    for fname in os.listdir(search_cache_dir):
        if not fname.endswith("_search.tsv"):
            continue
        if os.path.getsize(os.path.join(search_cache_dir, fname)) == 0:
            continue
        stem = fname[: -len("_search.tsv")]
        if multi_domain and "_dom" in stem:
            pid = stem.rsplit("_dom", 1)[0]
        else:
            pid = stem
        proteins.add(pid)
    return proteins


def load_zhang_positives(controls_path: str) -> list:
    positives = []
    with open(controls_path) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, cat = line.split("\t")
            if cat == "positive":
                a, b = pair.split("_")
                positives.append((a, b))
    return positives


def parse_search_tsv(tsv_path: str, query_protein: str) -> dict:
    """Parse search TSV. Returns {target_domain_name: max_tm}."""
    self_prefix = f"AF-{query_protein}-"
    best = {}
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= COL_MAX_TM:
                continue
            target = parts[COL_TARGET]
            if self_prefix in target:
                continue
            try:
                tm = float(parts[COL_MAX_TM])
            except ValueError:
                continue
            if target not in best or tm > best[target]:
                best[target] = tm
    return best


def load_search_results_per_domain(proteins: set, search_cache_dir: str,
                                   multi_domain: bool) -> dict:
    """
    Returns {(protein, domain_label): {template_domain_name: max_tm}}.

    single-domain mode (multi_domain=False):
      reads {pid}_search.tsv, label = "dom00"

    multi-domain mode (multi_domain=True):
      reads {pid}_dom{NN}_search.tsv, label = "domNN"
    """
    results = {}
    for pid in sorted(proteins):
        if multi_domain:
            prefix = pid + "_dom"
            for fname in os.listdir(search_cache_dir):
                if not fname.startswith(prefix) or not fname.endswith("_search.tsv"):
                    continue
                path = os.path.join(search_cache_dir, fname)
                if os.path.getsize(path) == 0:
                    continue
                dom_label = fname[len(pid) + 1 : -len("_search.tsv")]
                results[(pid, dom_label)] = parse_search_tsv(path, pid)
        else:
            tsv = os.path.join(search_cache_dir, f"{pid}_search.tsv")
            if os.path.exists(tsv) and os.path.getsize(tsv) > 0:
                results[(pid, "dom00")] = parse_search_tsv(tsv, pid)
    return results


def build_protein_hit_map(per_domain_hits: dict) -> dict:
    """
    Aggregate per-domain hits into per-protein map (max TM per template domain).

    Input:  {(protein, domain_label): {template_domain: tm}}
    Output: {protein: {template_domain: max_tm_across_all_domains}}
    """
    hit_map = {}
    for (pid, _label), hits in per_domain_hits.items():
        protein_hits = hit_map.setdefault(pid, {})
        for tmpl, tm in hits.items():
            if tmpl not in protein_hits or tm > protein_hits[tmpl]:
                protein_hits[tmpl] = tm
    return hit_map


def load_template_index(path: str) -> dict:
    """Load JSON {domain_name: [co-occurring domain names]} template pair index."""
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Rosetta Stone scoring
# ---------------------------------------------------------------------------

def score_pair_rosetta(p1: str, p2: str,
                       domain_hits: dict,
                       template_index: dict) -> float:
    """
    Rosetta Stone score for a candidate protein pair.

    Finds the best bridging template pair (A, B) in the TED pair list where
    P1 has an A-like domain and P2 has a B-like domain (or vice versa).
    Returns min(tm_A, tm_B) for the best bridge, 0.0 if none exists.
    """
    H1 = domain_hits.get(p1, {})
    H2 = domain_hits.get(p2, {})
    if not H1 or not H2:
        return 0.0

    best = 0.0
    # Forward: A in H1, B in H2
    for dom_a, tm_a in H1.items():
        partners = template_index.get(dom_a)
        if not partners:
            continue
        for dom_b in partners:
            tm_b = H2.get(dom_b)
            if tm_b is not None:
                score = min(tm_a, tm_b)
                if score > best:
                    best = score
    # Reverse: B in H1, A in H2
    # (template_index is already bidirectional, but we still need this loop
    # because the pairs stored in H1 and H2 may be on different sides)
    for dom_b, tm_b in H1.items():
        partners = template_index.get(dom_b)
        if not partners:
            continue
        for dom_a in partners:
            tm_a = H2.get(dom_a)
            if tm_a is not None:
                score = min(tm_b, tm_a)
                if score > best:
                    best = score
    return best


# ---------------------------------------------------------------------------
# Negative sampling and metrics (same as benchmark_ppi_v2.py)
# ---------------------------------------------------------------------------

def make_canonical(a, b):
    return (a, b) if a < b else (b, a)


def sample_negatives(universe_list, positives_canonical, n, seed=42):
    rng = random.Random(seed)
    neg_set, negatives = set(), []
    attempts, max_attempts = 0, n * 50
    while len(negatives) < n and attempts < max_attempts:
        a, b = rng.choice(universe_list), rng.choice(universe_list)
        attempts += 1
        if a == b:
            continue
        key = make_canonical(a, b)
        if key in positives_canonical or key in neg_set:
            continue
        neg_set.add(key)
        negatives.append((a, b))
    return negatives


def compute_weighted_pr(pos_scores, neg_scores, wp=0.01):
    P = len(pos_scores)
    if P == 0:
        return [], [], 0.0
    all_pairs = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    random.seed(42)
    random.shuffle(all_pairs)
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    precisions, recalls = [], []
    tp = fp = 0
    for score, label in all_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append((tp * wp) / (tp * wp + fp))
        recalls.append(tp / P)
    aucpr = sum(
        0.5 * (precisions[i] + precisions[i - 1]) * (recalls[i] - recalls[i - 1])
        for i in range(1, len(recalls))
    )
    return precisions, recalls, aucpr


def precision_at_recall(precisions, recalls, target):
    for p, r in zip(precisions, recalls):
        if r >= target:
            return p
    return 0.0


def compute_roc_auc(pos_scores, neg_scores):
    P, N = len(pos_scores), len(neg_scores)
    if P == 0 or N == 0:
        return 0.5
    u = sum(1.0 if p > n else 0.5 if p == n else 0.0
            for p in pos_scores for n in neg_scores)
    return u / (P * N)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_pair_scores(positives, negatives, pos_scores, neg_scores, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["protein_a", "protein_b", "label", "score"])
        for (a, b), s in zip(positives, pos_scores):
            w.writerow([a, b, "positive", f"{s:.4f}"])
        for (a, b), s in zip(negatives, neg_scores):
            w.writerow([a, b, "negative", f"{s:.4f}"])


def write_pr_curve(precisions, recalls, path, max_points=1000):
    step = max(1, len(precisions) // max_points)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["recall", "precision"])
        for i in range(0, len(precisions), step):
            w.writerow([f"{recalls[i]:.5f}", f"{precisions[i]:.5f}"])


def write_summary(positives, negatives, pos_scores, neg_scores,
                  precs, recs, aucpr, roc_auc, args, path,
                  n_zhang_pos, universe_size):
    P, N = len(positives), len(negatives)
    n_pos_hit = sum(1 for s in pos_scores if s > 0)
    n_neg_hit = sum(1 for s in neg_scores if s > 0)
    baseline_aucpr = args.wp / (args.wp + 1)

    with open(path, "w") as fh:
        def w(line=""): fh.write(line + "\n")
        w("=" * 60)
        w("PPI Benchmark — Rosetta Stone Paired-Domain Transfer")
        w("=" * 60)
        w()
        w("--- Algorithm ---")
        w("  Template: TED intra-protein domain-domain pairs (A, B)")
        w("  Inference: P1 interacts P2 iff P1 carries A-like domain")
        w("             AND P2 carries B-like domain for some template (A,B)")
        w("  Score:    min(TM(P1,A), TM(P2,B)) for best bridging template")
        w()
        w("--- Dataset ---")
        w(f"  Universe size:          {universe_size:,} proteins")
        w(f"  Zhang positives:        {n_zhang_pos:,} total")
        w(f"  Covered positives:      {P:,}  ({P/n_zhang_pos*100:.1f}% of Zhang)")
        w(f"  Negatives sampled:      {N:,}  ({args.neg_ratio}:1 ratio)")
        w()
        w("--- Results ---")
        w(f"  Positives with score > 0:  {n_pos_hit}/{P}  ({n_pos_hit/P*100:.1f}%)")
        w(f"  Negatives with score > 0:  {n_neg_hit}/{N}  ({n_neg_hit/N*100:.1f}%)")
        w()
        w("--- Metrics ---")
        w(f"  AUCPR:              {aucpr:.4f}  (random baseline: {baseline_aucpr:.4f})")
        w(f"  AUCPR / baseline:   {aucpr / baseline_aucpr:.2f}x")
        w(f"  ROC-AUC:            {roc_auc:.4f}")
        if precs:
            w(f"  Precision @ R=0.1:  {precision_at_recall(precs, recs, 0.1):.4f}")
            w(f"  Precision @ R=0.2:  {precision_at_recall(precs, recs, 0.2):.4f}")
            w(f"  Precision @ R=0.5:  {precision_at_recall(precs, recs, 0.5):.4f}")
        w()
        w("--- Parameters ---")
        w(f"  multi_domain:  {args.multi_domain}")
        w(f"  wp:            {args.wp}")
        w(f"  neg_ratio:     {args.neg_ratio}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Building universe from search cache...")
    universe = get_universe(args.search_cache_dir, multi_domain=args.multi_domain)
    log.info(f"Universe: {len(universe):,} proteins")
    universe_list = sorted(universe)

    log.info("Loading Zhang positives...")
    zhang_pos = load_zhang_positives(args.controls)
    positives = [(a, b) for a, b in zhang_pos if a in universe and b in universe]
    log.info(f"Covered positives: {len(positives):,}/{len(zhang_pos):,}")

    zhang_pos_canonical = {make_canonical(a, b) for a, b in zhang_pos}
    n_neg = len(positives) * args.neg_ratio
    negatives = sample_negatives(universe_list, zhang_pos_canonical, n_neg)
    log.info(f"Negatives: {len(negatives):,}")

    all_proteins = {p for pair in positives + negatives for p in pair}
    log.info(f"Loading search results for {len(all_proteins):,} proteins...")
    per_domain_hits = load_search_results_per_domain(
        all_proteins, args.search_cache_dir, multi_domain=args.multi_domain)
    domain_hits = build_protein_hit_map(per_domain_hits)
    log.info(f"Loaded domain hits for {len(domain_hits):,} proteins")

    log.info("Loading template pair index...")
    template_index = load_template_index(args.template_index)
    log.info(f"Template index: {len(template_index):,} domain entries")

    log.info("Scoring pairs...")
    pos_scores = [score_pair_rosetta(a, b, domain_hits, template_index)
                  for a, b in positives]
    neg_scores = [score_pair_rosetta(a, b, domain_hits, template_index)
                  for a, b in negatives]

    n_pos_hit = sum(1 for s in pos_scores if s > 0)
    n_neg_hit = sum(1 for s in neg_scores if s > 0)
    log.info(f"Positives hit: {n_pos_hit}/{len(positives)}")
    log.info(f"Negatives hit: {n_neg_hit}/{len(negatives)}")

    log.info("Computing metrics...")
    precs, recs, aucpr = compute_weighted_pr(pos_scores, neg_scores, args.wp)
    roc_auc = compute_roc_auc(pos_scores, neg_scores)
    baseline = args.wp / (args.wp + 1)
    log.info(f"AUCPR: {aucpr:.4f} (baseline {baseline:.4f}, {aucpr / baseline:.2f}x)")
    log.info(f"ROC-AUC: {roc_auc:.4f}")

    write_pair_scores(positives, negatives, pos_scores, neg_scores,
                      os.path.join(args.output_dir, "pair_scores.tsv"))
    write_pr_curve(precs, recs, os.path.join(args.output_dir, "pr_curve.tsv"))
    write_summary(positives, negatives, pos_scores, neg_scores, precs, recs,
                  aucpr, roc_auc, args,
                  os.path.join(args.output_dir, "summary.txt"),
                  len(zhang_pos), len(universe))
    log.info(f"Done. Results in {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Rosetta Stone PPI benchmark: paired-domain homology transfer.")
    p.add_argument("--controls", default=DEFAULT_CONTROLS)
    p.add_argument("--search-cache-dir", default=DEFAULT_SEARCH_CACHE,
                   dest="search_cache_dir")
    p.add_argument("--template-index", default=DEFAULT_TEMPLATE_INDEX,
                   dest="template_index",
                   help="JSON {domain: [domain,...]} template pair index")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, dest="output_dir")
    p.add_argument("--multi-domain", action="store_true", dest="multi_domain",
                   help="Cache has per-domain files ({pid}_domNN_search.tsv)")
    p.add_argument("--neg-ratio", type=int, default=5, dest="neg_ratio")
    p.add_argument("--wp", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
```

**Step 4: Run the tests**
```bash
python -m pytest tests/test_benchmark_rosetta_stone.py -v
```
Expected: 6 PASSED

**Step 5: Commit**
```bash
git add scripts/benchmark_rosetta_stone.py tests/test_benchmark_rosetta_stone.py
git commit -m "feat: add Rosetta Stone paired-domain benchmark script with tests"
```

---

### Task 4: Write the Extended Search Script (Both Sides, All Domains)

For Phase 2 coverage improvement: search ALL TED domains of ALL benchmark
proteins against the Zhang sub-database. This enables single-domain proteins
to participate in the cross-join.

**Files:**
- Create: `scripts/rosetta_search_both_sides.py`

The SQLite `domains` table schema (from `filter_query.py`):
- `domain_id` TEXT — full name e.g. `AF-P15056-F1-model_v4TED01`
- `domain_idx` INTEGER — position in Foldclass binary
- `taxonomy_id`, `species`, `cath_fold`, `confidence`, `globularity_score`

```python
#!/usr/bin/env python3
"""
rosetta_search_both_sides.py

Search ALL TED domains of ALL Zhang benchmark proteins against the Zhang
sub-database. Needed for Phase 2 of the Rosetta Stone pipeline to include
single-domain proteins (not in the existing pair-list cache).

For each protein P:
  1. Query SQLite for all TED domain IDs and their binary indices.
  2. Extract each domain PDB from the pairlist_db binary by integer index.
  3. Search that domain against the Zhang sub-database.
  4. Cache as: benchmark_cache/rosetta_searches/{P}_dom{N:02d}_search.tsv

If a domain's index is not in the pairlist_db (e.g. a single-domain protein
absent from the pair list), extraction will fail gracefully with a warning and
an empty cache file is written (so re-runs skip it).

Usage (on UCL cluster, in a screen session):
    python scripts/rosetta_search_both_sides.py \\
        --controls benchmark_cache/benchmarks/positives_and_negatives.tsv \\
        --filter-db merizo_pairlist_db/ted_pairlist_filters.db \\
        --pairlist-db merizo_pairlist_db/ted_pairlist \\
        --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \\
        --output-dir benchmark_cache/rosetta_searches \\
        --workers 8
"""

import argparse
import logging
import mmap
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'merizo_search', 'programs'))
from Foldclass.dbutil import (
    read_dbinfo, retrieve_start_end_by_idx, retrieve_bytes, coord_conv,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

_AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def get_all_benchmark_proteins(controls_path: str) -> set:
    """All unique proteins from both sides of ALL Zhang pairs (positive + negative)."""
    proteins = set()
    with open(controls_path) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair, _cat = line.split("\t")
            a, b = pair.split("_")
            proteins.add(a)
            proteins.add(b)
    return proteins


def get_protein_domains(uid: str, filter_db_path: str) -> list:
    """
    Return [(domain_id, domain_idx), ...] for all TED domains of protein uid.
    Queries the SQLite metadata database by UniProt accession pattern.
    """
    conn = sqlite3.connect(filter_db_path)
    cursor = conn.cursor()
    pattern = f"AF-{uid}-F1-model_v4TED%"
    cursor.execute(
        "SELECT domain_id, domain_idx FROM domains WHERE domain_id LIKE ?",
        (pattern,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows  # [(domain_id_str, int_idx), ...]


def extract_domain_pdb(domain_idx: int, pairlist_db: str, output_pdb: str) -> bool:
    """Extract a domain structure by integer index from the Foldclass binary DB."""
    try:
        db_json = pairlist_db + ".json"
        dbinfo = read_dbinfo(db_json)
        db_dir = os.path.dirname(os.path.abspath(db_json))

        def resolve(key):
            p = dbinfo[key]
            return p if os.path.isabs(p) else os.path.join(db_dir, p)

        with open(resolve("sif"), "rb") as sif, open(resolve("sdf"), "rb") as sdf:
            si_mm = mmap.mmap(sif.fileno(), 0, access=mmap.ACCESS_READ)
            sd_mm = mmap.mmap(sdf.fileno(), 0, access=mmap.ACCESS_READ)
            se = retrieve_start_end_by_idx(idx=[domain_idx], mm=si_mm)
            seq = retrieve_bytes(se[0][0], se[0][1], mm=sd_mm,
                                 typeconv=lambda x: x.decode("ascii"))

        with open(resolve("cif"), "rb") as cif, open(resolve("cdf"), "rb") as cdf:
            ci_mm = mmap.mmap(cif.fileno(), 0, access=mmap.ACCESS_READ)
            cd_mm = mmap.mmap(cdf.fileno(), 0, access=mmap.ACCESS_READ)
            ce = retrieve_start_end_by_idx(idx=[domain_idx], mm=ci_mm)
            coords = retrieve_bytes(ce[0][0], ce[0][1], mm=cd_mm,
                                    typeconv=coord_conv)

        with open(output_pdb, "w") as f:
            for i, (aa, coord) in enumerate(zip(seq, coords), start=1):
                resname = _AA_1TO3.get(aa, "UNK")
                f.write(f"ATOM  {i:>5}  CA  {resname:>3} A{i:>4}    "
                        f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
                        f"  1.00  0.00\n")
            f.write("END\n")
        return True
    except Exception as e:
        log.warning(f"Extract failed (idx={domain_idx}): {e}")
        return False


def run_search(pdb_path: str, target_db: str, output_prefix: str,
               topk: int, batchsize: int) -> str:
    merizo = os.path.join(PROJECT_ROOT, "merizo_search", "merizo.py")
    tmp = output_prefix + "_tmp"
    os.makedirs(tmp, exist_ok=True)
    cmd = [sys.executable, merizo, "search",
           pdb_path, target_db, output_prefix, tmp,
           "--search_batchsize", str(batchsize),
           "--mintm", "0.0",
           "--topk", str(topk)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        shutil.rmtree(tmp, ignore_errors=True)
        tsv = output_prefix + "_search.tsv"
        if r.returncode != 0 or not os.path.exists(tsv):
            log.warning(f"Search failed (rc={r.returncode}): {r.stderr[-300:]}")
            return None
        return tsv
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        log.warning(f"Search error: {e}")
        return None


def process_protein(uid: str, filter_db: str, pairlist_db: str,
                    zhang_db: str, output_dir: str,
                    topk: int, batchsize: int) -> tuple:
    """Extract and search all TED domains for one protein."""
    domains = get_protein_domains(uid, filter_db)
    if not domains:
        return uid, "no_domains_in_sqlite"

    status_parts = []
    with tempfile.TemporaryDirectory(prefix="rsearch_") as tmpdir:
        for n, (domain_id, domain_idx) in enumerate(sorted(domains)):
            cache_tsv = os.path.join(output_dir, f"{uid}_dom{n:02d}_search.tsv")
            if os.path.exists(cache_tsv):
                status_parts.append(f"dom{n:02d}:already_cached")
                continue

            pdb = os.path.join(tmpdir, f"{uid}_dom{n:02d}.pdb")
            if not extract_domain_pdb(domain_idx, pairlist_db, pdb):
                open(cache_tsv, "w").close()  # empty sentinel to skip retries
                status_parts.append(f"dom{n:02d}:extract_failed")
                continue

            prefix = os.path.join(tmpdir, f"{uid}_dom{n:02d}")
            tsv = run_search(pdb, zhang_db, prefix, topk, batchsize)
            if tsv:
                shutil.move(tsv, cache_tsv)
                n_hits = sum(1 for _ in open(cache_tsv))
                status_parts.append(f"dom{n:02d}:ok({n_hits}hits)")
            else:
                open(cache_tsv, "w").close()
                status_parts.append(f"dom{n:02d}:search_failed")

    return uid, " | ".join(status_parts)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Collecting all unique proteins from Zhang benchmark (both sides)...")
    all_proteins = get_all_benchmark_proteins(args.controls)
    log.info(f"Total unique proteins: {len(all_proteins):,}")

    # Skip proteins that already have any cached domain search
    done = {fname.split("_dom")[0]
            for fname in os.listdir(args.output_dir)
            if fname.endswith("_search.tsv")}
    todo = sorted(all_proteins - done)
    log.info(f"Already done: {len(done):,}, remaining: {len(todo):,}")

    total = len(todo)
    log.info(f"Running with {args.workers} workers, topk={args.topk}...")

    def submit(uid):
        return process_protein(uid, args.filter_db, args.pairlist_db,
                               args.zhang_db, args.output_dir,
                               args.topk, args.batchsize)

    if args.workers == 1:
        for i, uid in enumerate(todo, 1):
            uid_out, msg = submit(uid)
            log.info(f"[{i}/{total}] {uid_out}: {msg}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(submit, uid): uid for uid in todo}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                uid_out, msg = future.result()
                log.info(f"[{done_count}/{total}] {uid_out}: {msg}")

    log.info("Done. Run benchmark_rosetta_stone.py --multi-domain to evaluate.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Rosetta Stone: search all TED domains of all benchmark proteins.")
    p.add_argument("--controls",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "benchmarks", "positives_and_negatives.tsv"))
    p.add_argument("--filter-db", required=True, dest="filter_db",
                   help="SQLite metadata DB: ted_pairlist_filters.db")
    p.add_argument("--pairlist-db", required=True, dest="pairlist_db",
                   help="Foldclass binary DB for domain PDB extraction")
    p.add_argument("--zhang-db", required=True, dest="zhang_db",
                   help="Zhang sub-database to search against")
    p.add_argument("--output-dir",
                   default=os.path.join(PROJECT_ROOT, "benchmark_cache",
                                        "rosetta_searches"),
                   dest="output_dir")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batchsize", type=int, default=2097152)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
```

**Step 2: Commit**
```bash
git add scripts/rosetta_search_both_sides.py
git commit -m "feat: add extended both-sides all-domain search script for Rosetta Stone"
```

---

## Phase 3: Run on UCL Cluster (Days 2-4)

SSH to UCL cluster. Run inside `screen` or `tmux` so jobs survive disconnection.

### Task 5: Build template index (~30 min)

```bash
# Step 1: dump Zhang domain names (~1 min)
python scripts/rosetta_dump_zhang_domains.py \
    --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \
    --output benchmark_cache/zhang_domain_names.txt
# Expected: "Wrote ~2172 domain names..."

# Step 2: build template index (streams 186 GB — ~20-30 min)
python scripts/rosetta_build_template_index.py \
    --pair-list /mnt/bigstore/ted/pair_list_20250128 \
    --zhang-domain-names benchmark_cache/zhang_domain_names.txt \
    --output benchmark_cache/zhang_template_index.json
# Expected: "Index saved..." with some thousands of domain entries
```

### Task 6: Phase 1 benchmark — validate algorithm (~5 min)

Uses existing search cache (Domain B only). Proves the cross-join works
before running expensive new searches.

```bash
python scripts/benchmark_rosetta_stone.py \
    --search-cache-dir benchmark_cache/searches \
    --template-index benchmark_cache/zhang_template_index.json \
    --output-dir benchmark_results_rosetta_phase1 \
    --neg-ratio 5 --wp 0.01

cat benchmark_results_rosetta_phase1/summary.txt
```

Compare covered positives and AUCPR against `benchmark_results_v2/summary.txt`.

### Task 7: Extended search — both sides, all domains (6-18 hours)

```bash
screen -S rosetta_search

python scripts/rosetta_search_both_sides.py \
    --controls benchmark_cache/benchmarks/positives_and_negatives.tsv \
    --filter-db merizo_pairlist_db/ted_pairlist_filters.db \
    --pairlist-db merizo_pairlist_db/ted_pairlist \
    --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \
    --output-dir benchmark_cache/rosetta_searches \
    --topk 50 \
    --workers 8

# Detach: Ctrl-A D   |   Reattach: screen -r rosetta_search
```

Log messages show `dom00:ok(47hits)` per domain. Proteins absent from SQLite
log `no_domains_in_sqlite` — these are genuinely absent from TED and cannot
be included (fundamental limitation, document in report).

### Task 8: Phase 2 benchmark (~10 min)

```bash
python scripts/benchmark_rosetta_stone.py \
    --search-cache-dir benchmark_cache/rosetta_searches \
    --template-index benchmark_cache/zhang_template_index.json \
    --output-dir benchmark_results_rosetta_phase2 \
    --multi-domain \
    --neg-ratio 5 --wp 0.01

cat benchmark_results_rosetta_phase2/summary.txt
```

Key numbers to record: covered positives %, AUCPR, ROC-AUC, precision@recall.

### Task 9: Generate figures

```bash
# Rosetta Stone figures
python scripts/plot_benchmark_curves.py \
    --pair-scores benchmark_results_rosetta_phase2/pair_scores.tsv \
    --output-dir figures/rosetta/

# Old method figures (for comparison in report)
python scripts/plot_benchmark_curves.py \
    --pair-scores benchmark_results_v2/pair_scores.tsv \
    --output-dir figures/old_method/
```

---

## Phase 4: Report Rewrites (Days 5-7)

### What to rewrite in each section

**Abstract:**
- Remove: "both proteins must possess an intra-protein domain-domain contact"
- Add: domain pair (A, B) from TED is the transferred unit of evidence (Rosetta Stone template)
- Update: coverage % and AUCPR to Phase 2 results

**Section 1.3 Project Aim / 1.5 Project Approach:**
Replace one-sided description. New paragraph:

> Following the Rosetta Stone principle from evolutionary genomics, the approach
> uses domain-domain associations observed within fused proteins as transferable
> interaction evidence. If domains A and B co-occur in the same protein (the Rosetta
> Stone template), any separate protein carrying an A-like domain is predicted to
> interact with any protein carrying a B-like domain. Crucially, neither query
> protein needs to be multi-domain — the template pair, not the query architecture,
> is the unit of transferred evidence.

**Section 3 System Design:**
- Update Figure 3.1 to show bidirectional pipeline (P1 domains → H1, P2 domains → H2, cross-join)
- Add new subsections:
  - 3.3.1 Template Pair Index — how TED pair list becomes a lookup dict
  - 3.3.2 Bidirectional Domain Search — why both proteins' all domains are searched
  - 3.3.3 Cross-Join Scoring — the min(tm_A, tm_B) function and why min is used

**Section 4 Results:**
- 4.1 BRAF Case Study: rewrite to show BRAF-RAF1 via the Rosetta Stone path
  (which template pair bridges them? both have similar kinase domains — show the template)
- 4.2.1 Coverage: new coverage % with full explanation of what improved and why
- 4.2.2 Discrimination: new AUCPR/ROC-AUC table comparing old vs new
- Add figure showing old PR curve vs new Rosetta Stone PR curve on same axes

**Section 5 Discussion:**
- Add "Method Revision" subsection: acknowledge that the original pipeline was
  one-sided, explain what changed, compare results
- Explain residual coverage gap (proteins genuinely absent from TED / SQLite)

---

## File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `scripts/rosetta_dump_zhang_domains.py` | New | Dump domain names from Zhang sub-DB |
| `scripts/rosetta_build_template_index.py` | New | Build TED template pair index (JSON) |
| `scripts/rosetta_search_both_sides.py` | New | Search all domains of all benchmark proteins |
| `scripts/benchmark_rosetta_stone.py` | New | Rosetta Stone benchmark evaluation |
| `tests/test_rosetta_template_index.py` | New | Unit tests for index builder |
| `tests/test_benchmark_rosetta_stone.py` | New | Unit tests for scoring functions |
| `benchmark_cache/zhang_domain_names.txt` | Generated on cluster | Domain names in Zhang sub-DB |
| `benchmark_cache/zhang_template_index.json` | Generated on cluster | Template pair index |
| `benchmark_cache/rosetta_searches/` | Generated on cluster | Per-domain search results |
| `benchmark_results_rosetta_phase1/` | Generated | Phase 1 results (existing cache) |
| `benchmark_results_rosetta_phase2/` | Generated | Phase 2 results (all domains) |
| `figures/rosetta/` | Generated | PR and ROC curve figures |

## Timeline

| Day | Tasks |
|-----|-------|
| 1 | Tasks 1-4 (code all 4 scripts, both test suites pass) |
| 2 | Task 5-6 (build template index + Phase 1 benchmark on cluster) |
| 3-4 | Task 7 (extended searches running, 8 workers, 6-18h) |
| 4 | Tasks 8-9 (Phase 2 benchmark + figures) |
| 5-7 | Tasks 10-14 (all report section rewrites) |
