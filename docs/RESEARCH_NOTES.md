# Research Notes — Rosetta Stone PPI Pipeline

> Running record of decisions, findings, and numbers for the final report.
> Updated throughout the experiment on 2026-04-22.

---

## Context: Why the Pipeline Was Redesigned

The original submitted pipeline was a **one-sided structural similarity search**:
- Take the query protein's interface domain (Domain B from TED pair list)
- Search for structurally similar domains in other proteins
- Nominate those proteins as candidate partners

**Professor's feedback:** This does not implement the Rosetta Stone principle. The query
protein was required to be multi-domain and present in the TED pair list, which excluded
80.2% of the Zhang benchmark positive pairs (single-domain proteins or proteins absent
from TED).

**What was expected:** A paired-domain homology transfer model where:
- A template pair (A, B) co-occurring in any fused protein is the unit of transferred evidence
- Any protein carrying an A-like domain is predicted to interact with any protein carrying a B-like domain
- The query protein does not need to be multi-domain at all

---

## New Algorithm: Rosetta Stone Paired-Domain Transfer

### Core scoring function

For a candidate pair (P1, P2):

    H[P] = {template_domain: max_TM_score}  for all TED domains of P searched against Zhang sub-DB

    score(P1, P2) = max over all template pairs (A, B) in TED where:
                      A in H[P1] AND B in H[P2] -> min(H[P1][A], H[P2][B])
                      OR B in H[P1] AND A in H[P2] -> min(H[P1][B], H[P2][A])

The min() ensures **both sides** of the bridge must be structurally supported. Both P1 and
P2 must carry domains that resemble the two sides of a known co-occurring pair.

### Two-phase implementation

| Phase | Search target | Coverage |
|-------|--------------|----------|
| Phase 1 | Existing Domain B cache (old search results reused) | Same proteins as old method |
| Phase 2 | All TED domains of ALL benchmark proteins (both sides) | Substantially wider |

---

## Implementation: New Scripts

| Script | Purpose |
|--------|---------|
| scripts/rosetta_dump_zhang_domains.py | Dump domain names from Zhang sub-DB |
| scripts/rosetta_build_template_index.py | Build TED template pair index (JSON) |
| scripts/rosetta_search_both_sides.py | Search all domains of all benchmark proteins |
| scripts/benchmark_rosetta_stone.py | Rosetta Stone cross-join benchmark |
| scripts/rosetta_make_mini_sample.py | Create small sample for pipeline validation |
| tests/test_rosetta_template_index.py | Unit tests: index builder (3 tests) |
| tests/test_benchmark_rosetta_stone.py | Unit tests: scoring functions (6 tests) |

All 9 unit tests pass.

---

## Experiment Log

### 2026-04-22 — Template Index Build

**Step 1: Dump Zhang sub-DB domain names**

    python scripts/rosetta_dump_zhang_domains.py \
        --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \
        --output benchmark_cache/zhang_domain_names.txt

Result: Wrote 2,753 domain names to benchmark_cache/zhang_domain_names.txt

---

**Step 2: Build template pair index**

    python scripts/rosetta_build_template_index.py \
        --pair-list /mnt/bigstore/ted/pair_list_20250128 \
        --zhang-domain-names benchmark_cache/zhang_domain_names.txt \
        --output benchmark_cache/zhang_template_index.json

Result:
    Pair list: 129,440,391 lines, 1,756 pairs retained
    Index: 2,753 unique domain entries
    Index saved to benchmark_cache/zhang_template_index.json

Interpretation:
- The TED pair list contains 129 million intra-protein domain-domain contacts
- Only 1,756 template pairs involve at least one Zhang benchmark domain
- This is because the template library is restricted to Zhang sub-DB proteins
- A fuller implementation would use templates from all organisms (see Limitations)

---

## Key Numbers for Report

| Metric | Value | Source |
|--------|-------|--------|
| Zhang sub-DB domain names | 2,753 | rosetta_dump_zhang_domains.py |
| TED pair list total lines | 129,440,391 | rosetta_build_template_index.py |
| Template pairs retained | 1,756 | rosetta_build_template_index.py |
| Existing search cache | 1,086 files | ls benchmark_cache/searches/ |
| Old method covered positives | 595 / 3,000 (19.8%) | Original report |
| Old method AUCPR | 0.0384 (3.87x baseline) | Original report |
| Random AUCPR baseline | 0.0099 | Derived from wp=0.01 |

Phase 1 results: (to be filled in)
Phase 2 results: (to be filled in)

---

## Limitations to Acknowledge in Report

1. **Template library restricted to Zhang proteins:** The template index only contains
   co-occurring domain pairs from Zhang benchmark proteins (because we search against the
   Zhang sub-DB). A true cross-organism Rosetta Stone would use templates from all 129M+
   TED pairs, dramatically expanding coverage. This is a computational scope limitation,
   not a conceptual flaw in the algorithm.

2. **Proteins absent from TED entirely:** Proteins with no AlphaFold2 structure in TED
   cannot be covered by any domain-based method. This is a fundamental ceiling.

3. **Single structural signal:** No co-expression, co-localisation, or evolutionary
   co-variation signals are used. Real PPI prediction systems combine multiple signals.

---

## Decisions Made

- Dropped comparison with old method in report — old method was conceptually wrong
- Using JSON (not binary serialisation) for template index — safer and portable
- Phase 1 validates algorithm on existing cache before expensive re-searches
- Using tmux for all long-running cluster jobs
