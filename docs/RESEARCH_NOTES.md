# Research Notes — Rosetta Stone PPI Pipeline

> Reference document for the final report. All numbers, interpretations, and
> "what to write" blocks reflect the final implemented system.

---

## Algorithm: Rosetta Stone Paired-Domain Transfer

### Core scoring function

For a candidate pair (P1, P2):

    H[P] = {template_domain: max_TM_score}  for all TED domains of P searched against Zhang sub-DB

    score(P1, P2) = max over all valid template pairs (A, B) where:
                      A in H[P1] AND B in H[P2] → min(H[P1][A], H[P2][B])
                      OR B in H[P1] AND A in H[P2] → min(H[P1][B], H[P2][A])

The `min()` ensures both sides of the bridge must be structurally supported.

### Self-interaction filter (validity condition)

A template pair (A, B) constitutes a valid Rosetta Stone bridge only if the two
domains are **split across the two query proteins** — i.e. P1 carries an A-like domain
but not a B-like domain, and P2 carries a B-like domain but not an A-like domain
(or vice versa).

If P1 already carries both an A-like **and** a B-like domain (TM ≥ `self_filter_tm`),
the A–B co-occurrence in the template is trivially explained by P1's own intrachain
contacts and provides no interchain evidence. Likewise for P2. Such bridges are
discarded before scoring.

    Valid bridge (A, B): NOT (A in H[P1] AND B in H[P1])
                         AND NOT (A in H[P2] AND B in H[P2])

`self_filter_tm` defaults to `min_bridge_tm`. For benchmarks run at `min_bridge_tm=0.0`,
setting `--self-filter-tm 0.3` avoids over-rejection from very weak residual hits.

### Domain discovery — two complementary paths

The pipeline searches all TED domains of every benchmark protein against the Zhang sub-DB.
Domain discovery uses two paths depending on the protein type:

- **Multi-domain proteins**: looked up via `ted_pairlist_filters.db` SQLite index
- **Single-domain proteins**: discovered via linear scan of the full `ted_365M` database
  (365M TED-segmented AlphaFold domains, 33-byte fixed-width entries, uid at bytes 3–9)

Together these cover **79.0% of the Zhang benchmark (2,369/3,000 positive pairs)**. The
remaining 20.1% have no AlphaFold model and are unreachable by any structure-based method.

---

## Implementation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/rosetta_dump_zhang_domains.py` | Dump domain names from Zhang sub-DB |
| `scripts/rosetta_build_template_index.py` | Build TED template pair index (JSON) |
| `scripts/rosetta_search_both_sides.py` | Search all domains of multi-domain benchmark proteins |
| `scripts/rosetta_search_ted365m.py` | Search single-domain proteins via full ted_365M scan |
| `scripts/benchmark_rosetta_stone.py` | Rosetta Stone cross-join benchmark |
| `scripts/rosetta_explain_pair.py` | Step-by-step scoring trace for one protein pair |
| `scripts/predict_ppi.py` | End-to-end single-pair prediction tool |
| `tests/test_rosetta_template_index.py` | Unit tests: index builder (3 tests) |
| `tests/test_benchmark_rosetta_stone.py` | Unit tests: scoring functions (6 tests) |

All 13 unit tests pass (9 original + 4 self-interaction filter tests).

---

## Pipeline Architecture

```
[One-time setup]
TED pair list (129M pairs) + Zhang sub-DB domain names
        → rosetta_build_template_index.py
        → zhang_template_index.json
           {"domainX": ["domainY", "domainZ", ...], ...}
           = "which domain pairs co-occur in the same fusion protein"

[Per-protein search — cached to disk]
For each benchmark protein P:
  Path A (multi-domain): SQLite lookup → extract domain PDB from TED pairlist DB
  Path B (single-domain): linear scan of ted_365M → extract domain PDB
        → run: merizo search domain.pdb zhang_db/ output/ tmp/
        → saves: benchmark_cache/rosetta_searches/{P}_domNN_search.tsv
  Content: one row per Zhang template domain hit with TM score

[Scoring — fast, reads cached TSVs only]
For pair (P1, P2):
  H[P1] = {zhang_domain: best_TM}  ← read from P1's cached TSV files
  H[P2] = {zhang_domain: best_TM}  ← read from P2's cached TSV files
  For every template pair (A, B) in template_index:
    if A in H[P1] and B in H[P2]:
      # Self-interaction filter (NEW):
      if B already in H[P1] above self_filter_tm → DISCARD (trivial intrachain in P1)
      if A already in H[P2] above self_filter_tm → DISCARD (trivial intrachain in P2)
      bridge_score = min(H[P1][A], H[P2][B])
  Final score = max over all surviving bridges
```

merizo_search only runs at the per-protein search step. `predict_ppi.py --uid1/--uid2`
reads from cache with no merizo runs. `predict_ppi.py --pdb1/--pdb2` runs merizo_search
for new proteins not in the cache.

---

## Database Roles

| Database | Location | Size | Role |
|----------|----------|------|------|
| Zhang sub-DB | `zhang_pairlist_db/zhang_pairlist_db/ted_pairlist` | 2,753 domains | Search target — every protein is searched against this |
| Full TED pairlist DB | `merizo_pairlist_db/ted_pairlist` | Millions of domains | Domain extraction (multi-domain path) + source of the 129M pair list |
| ted_365M | `/mnt/bigstore/foldclass-db-ted/ted_365M.json` | 365M domains | Domain discovery + extraction (single-domain path) |

### Why search against Zhang sub-DB rather than full pairlist_db?

The template index keys are Zhang domain names. The bridge-finding step requires A and B
to exist in the template index, which only contains Zhang domains. Searching against the
full pairlist_db would produce millions of hits per protein that would all be ignored at
scoring time — much slower with zero benefit. The right long-term fix is building the
template index from all 129M TED pairs and searching the full DB, but this is out of scope.

---

## Setup Log

### Template Index Build

**Step 1: Dump Zhang sub-DB domain names**

    python scripts/rosetta_dump_zhang_domains.py \
        --zhang-db zhang_pairlist_db/zhang_pairlist_db/ted_pairlist \
        --output benchmark_cache/zhang_domain_names.txt

Result: 2,753 domain names written to `benchmark_cache/zhang_domain_names.txt`

**Step 2: Build template pair index**

    python scripts/rosetta_build_template_index.py \
        --pair-list /mnt/bigstore/ted/pair_list_20250128 \
        --zhang-domain-names benchmark_cache/zhang_domain_names.txt \
        --output benchmark_cache/zhang_template_index.json

Result:
- Pair list: 129,440,391 lines, 1,756 pairs retained
- Index: 2,753 unique domain entries
- Saved to: `benchmark_cache/zhang_template_index.json`

Only 1,756 of 129M pairs are retained because the template library is restricted to the
2,753 Zhang sub-DB domains. A fuller implementation would use templates from all of AFDB.

---

## Benchmark Results

> **STATUS: Numbers below are PRE-self-interaction-filter (old results).**
> After adding the self-interaction filter, the benchmark must be re-run on the server
> to produce updated numbers. See the "Self-interaction filter change" section below
> for the command and what to expect.

### Dataset

    Universe:           14,201 proteins
    Covered positives:  2,369 / 3,000  (79.0%)
    Negatives sampled:  11,845  (5:1 ratio)

### Threshold sweep (min_bridge_tm) — PRE-FILTER RESULTS

| min_bridge_tm | pos hit              | neg hit               | selectivity | AUCPR  | ×baseline | ROC-AUC    |
|---------------|---------------------|-----------------------|-------------|--------|-----------|------------|
| 0.0           | 1,832/2,369 (77.3%) | 8,069/11,845 (68.1%) | 1.13×       | 0.0271 | 2.73×     | 0.6388     |
| 0.3           | 1,739/2,369 (73.4%) | 7,285/11,845 (61.5%) | 1.19×       | 0.0271 | 2.73×     | **0.6400** |
| 0.5           | 857/2,369  (36.2%)  | 1,380/11,845 (11.7%) | 3.09×       | 0.0270 | 2.72×     | 0.6310     |
| 0.7           | 403/2,369  (17.0%)  | 228/11,845   (1.9%)  | 8.95×       | 0.0257 | 2.60×     | 0.5762     |

### Interpretation (pre-filter)

1. **AUCPR is stable across TM = 0.0, 0.3, 0.5 (all 2.73×).** The top-ranked pairs already
   have high TM scores — removing weak bridges does not affect the area under the curve.
   The signal is genuine and not an artefact of threshold choice.

2. **AUCPR drops only at TM = 0.7 (2.60×).** At this point 83% of positives score 0
   (no bridge strong enough on both sides), which hurts the tail of the curve.

3. **ROC-AUC peaks at TM = 0.3 (0.6400)** and is flat from TM=0.0 to 0.3. It drops at
   TM=0.5 and falls sharply at TM=0.7 (0.5762, near random).

4. **TM = 0.0 is the recommended operating point for coverage-sensitive use.** 77.3% of
   covered positives score non-zero; AUCPR 2.73× baseline.

5. **TM = 0.5 is the recommended operating point for precision-sensitive use.** Selectivity
   ratio 3.09× (pos 36.2% vs neg 11.7% score non-zero); AUCPR essentially unchanged (2.72×).

---

## Figures

Both figures generated from `benchmark_results_phase3_tm0.0/pair_scores.tsv`.

    python scripts/plot_benchmark_curves.py \
        --pair-scores benchmark_results_phase3_tm0.0/pair_scores.tsv \
        --labels "Rosetta Stone (paired-domain transfer)" \
        --output-dir figures/

### `figures/pr_curve.png` — Weighted Precision–Recall Curve (wp=0.01)

AUCPR = 0.0271, 2.73× random baseline (0.0099).

Shape of the curve:
- **Recall 0–0.02:** Precision near 1.0 — the very highest-scoring pairs are almost
  exclusively true positives. This is the strongest part of the signal.
- **Recall 0.02–0.20:** Sharp drop then gradual staircase descent from ~0.5 down toward
  baseline. The jagged steps reflect the discrete score distribution (many pairs share
  the same TM score value).
- **Recall 0.20–1.0:** Precision falls to and stays near random baseline. These are
  pairs that score 0 — either the positives have no bridge, or the method cannot
  distinguish them from negatives at this recall level.

Key strength: at low recall the method is highly precise. The top-ranked predictions are
almost certainly true positives — useful in practice for generating a short candidate list
to validate experimentally.

### `figures/roc_curve.png` — ROC Curve

AUC = 0.6388 (random baseline = 0.5000).

Shape of the curve:
- **FPR 0–0.15:** Steep initial rise — TPR reaches ~0.35 while only 15% of negatives are
  accepted. Confirms strong discrimination at the high-confidence end of the score range.
- **FPR 0.15–1.0:** Smooth, consistent curve above the diagonal throughout. No region
  where the method dips back to random — the rank ordering is meaningful all the way through.
- At FPR=0.5, TPR≈0.75 — the method recovers 75% of positives before half the negatives
  have been accepted.

ROC-AUC of 0.64 is a solid result for a zero-shot structural homology method with no
training, no co-expression or sequence coevolution signals. The curve confirms the method
is genuinely discriminative across the full 79% coverage evaluation set.

---

## Case Example: LYN × SRC (INVALIDATED by self-interaction filter)

> **This example is no longer valid after adding the self-interaction filter.**
> It is retained here to document WHY the filter is needed (and as the motivating
> example from the supervisor's feedback). See "New case example" below for the
> replacement.

**Old command (pre-filter):**

    python scripts/rosetta_explain_pair.py \
        --p1 P07948 --p2 P12931 --min-bridge-tm 0.5 \
        --search-cache-dir benchmark_cache/rosetta_searches \
        --template-index benchmark_cache/zhang_template_index.json

**Pair:** P07948 (LYN, Src-family kinase) × P12931 (SRC, Src-family kinase)
**Old score (pre-filter):** 0.9935  |  **Bridges found:** 141
**New score (post-filter):** 0.0  |  **All 141 bridges rejected**

**Why the score was high (and why it was wrong):**

    Template protein: P08631 (HCK — Hematopoietic cell kinase, Src-family)
    ├─ HCK_TED04  co-occurs with  HCK_TED03  in the same protein
    │
    ├─ LYN (P07948) → HCK_TED04   TM = 0.9989  ✓
    └─ SRC (P12931) → HCK_TED03   TM = 0.9935  ✓

    bridge_score = min(0.9989, 0.9935) = 0.9935  [BEFORE filter]

**Why the filter rejects all 141 bridges:**

LYN is itself a Src-family kinase with SH3 + SH2 + kinase domains — the same architecture
as HCK. Therefore LYN's hit map H[LYN] contains hits to BOTH HCK_TED03 and HCK_TED04
(and all equivalent domains from every Src-family kinase in the sub-DB). For every bridge
(A, B) that fires between LYN and SRC, domain B is already present in H[LYN] above the
self-filter threshold. The self-interaction filter correctly identifies that the A–B
co-occurrence in HCK is trivially explained by LYN's own intrachain contacts — there is
no genuine interchain evidence.

In the professor's terms: LYN and SRC "contain the exact same set of domains — so any
domain pairings you find will just be intrachain interactions and not new interchain
interactions."

**What this tells us about the method:**
The pre-filter result was not evidence of a PPI — it was evidence that LYN and SRC are
from the same structural family. The method was finding homologues, not interaction
partners. The filter makes this failure mode visible and removes it from the scored set.

---

## Self-interaction filter change (supervisor feedback — rosetta_stone_v2 branch)

### What changed and why

The supervisor identified a missing validity condition in the Rosetta Stone inference.
A bridge (A, B) is only genuine interchain evidence if the two domain types are **split**
across the two query proteins. If P1 already carries both an A-like AND a B-like domain,
the A–B co-occurrence in the template is trivially explained by P1's own intrachain
contacts — no interchain prediction is warranted. Same applies symmetrically for P2.

The filter was added to `score_pair_rosetta()`, `find_bridges()` (explain script), and
`find_bridges()` (predict_ppi.py). The new `--self-filter-tm` CLI argument controls the
threshold (defaults to `min_bridge_tm`).

### Scripts changed

| Script | Change |
|--------|--------|
| `benchmark_rosetta_stone.py` | `score_pair_rosetta()`: added self-interaction filter in both forward and reverse bridge loops; added `--self-filter-tm` CLI arg |
| `rosetta_explain_pair.py` | `find_bridges()`: same filter; updated console output to label active filter conditions |
| `predict_ppi.py` | `find_bridges()`: same filter; added `--self-filter-tm` CLI arg |
| `tests/test_benchmark_rosetta_stone.py` | +4 tests: P1-has-both rejected, P2-has-both rejected, valid-split accepted, threshold-boundary respected |

### Commands to re-run on the server

**Step 1 — Re-run the benchmark with the filter (all four threshold points):**

    # Default: self_filter_tm = min_bridge_tm (0.0)
    python scripts/benchmark_rosetta_stone.py \
        --search-cache-dir benchmark_cache/rosetta_searches \
        --template-index benchmark_cache/zhang_template_index.json \
        --output-dir benchmark_results_filtered_tm0.0 \
        --multi-domain --min-bridge-tm 0.0

    python scripts/benchmark_rosetta_stone.py \
        --search-cache-dir benchmark_cache/rosetta_searches \
        --template-index benchmark_cache/zhang_template_index.json \
        --output-dir benchmark_results_filtered_tm0.3 \
        --multi-domain --min-bridge-tm 0.3

    python scripts/benchmark_rosetta_stone.py \
        --search-cache-dir benchmark_cache/rosetta_searches \
        --template-index benchmark_cache/zhang_template_index.json \
        --output-dir benchmark_results_filtered_tm0.5 \
        --multi-domain --min-bridge-tm 0.5

    python scripts/benchmark_rosetta_stone.py \
        --search-cache-dir benchmark_cache/rosetta_searches \
        --template-index benchmark_cache/zhang_template_index.json \
        --output-dir benchmark_results_filtered_tm0.7 \
        --multi-domain --min-bridge-tm 0.7

**Step 2 — Confirm LYN × SRC is filtered (score should now be 0.0):**

    python scripts/rosetta_explain_pair.py \
        --p1 P07948 --p2 P12931 \
        --search-cache-dir benchmark_cache/rosetta_searches \
        --template-index benchmark_cache/zhang_template_index.json \
        --min-bridge-tm 0.0

**Step 3 — Find the new top-scoring surviving positive pair:**

    python scripts/rosetta_explain_pair.py \
        --find-examples \
        --controls benchmark_cache/benchmarks/positives_and_negatives.tsv \
        --search-cache-dir benchmark_cache/rosetta_searches \
        --template-index benchmark_cache/zhang_template_index.json \
        --min-bridge-tm 0.0

**Step 4 — Trace the new top pair in detail:**

    python scripts/rosetta_explain_pair.py \
        --p1 <NEW_P1> --p2 <NEW_P2> \
        --search-cache-dir benchmark_cache/rosetta_searches \
        --template-index benchmark_cache/zhang_template_index.json \
        --min-bridge-tm 0.0 --top-bridges 5

**Step 5 — Precision@k on new results:**

    python scripts/rosetta_precision_at_k.py \
        --pair-scores benchmark_results_filtered_tm0.0/pair_scores.tsv

**Step 6 — Regenerate figures:**

    python scripts/plot_benchmark_curves.py \
        --pair-scores benchmark_results_filtered_tm0.0/pair_scores.tsv \
        --labels "Rosetta Stone (paired-domain transfer, self-filter)" \
        --output-dir figures/

### What to expect after re-running

Same-family pairs (kinase × kinase, SH2 × SH2, etc.) will lose all bridges and score 0.
Cross-family pairs (e.g. kinase × adaptor, receptor × regulator) where one protein has
A-type domains and the other has structurally unrelated B-type domains will be unaffected.

Expected direction of change:
- **Positive hit rate will drop** — same-family true positives that scored for the wrong
  reason will drop to 0. These are "unreachable" by the corrected method.
- **Negative hit rate will also drop** — many high-scoring negatives were probably also
  same-family pairs (e.g. two unrelated kinases sharing a template kinase). Removing these
  may improve selectivity.
- **Net effect on AUCPR is uncertain** — the benchmark needs to be re-run to know.
- **The method is now conceptually correct** — any surviving non-zero score represents
  genuine interchain structural evidence, not intrachain self-similarity.

### What a valid new case example looks like

A good replacement for LYN × SRC should satisfy:
- P1 carries domain type A but has NO structural match to domain type B (H[P1] does not
  contain domain B above self_filter_tm)
- P2 carries domain type B but has NO structural match to domain type A
- The bridge (A, B) comes from a template protein T where A and B are physically adjacent
  but structurally unrelated folds
- Biologically: a known interaction between proteins from different structural families,
  e.g. a kinase interacting with an adaptor/scaffold, a receptor with a cytoplasmic
  effector, or an enzyme with a regulatory subunit

Run Step 3 above to find the actual top pair from the filtered results.

---

## New Case Example (TO BE FILLED after server re-run)

    Pair:         <P1_ACCESSION> × <P2_ACCESSION>
    Score:        <SCORE>
    Bridges:      <N>
    Template:     <TEMPLATE_PROTEIN>
    Bridge:       <TEMPLATE_DOM_A> + <TEMPLATE_DOM_B>
    P1 → dom_A:  TM = <TM_A>
    P2 → dom_B:  TM = <TM_B>
    Filter check: dom_B NOT in H[P1], dom_A NOT in H[P2]  ✓

---

## Precision at Top-k Results

Generated by `scripts/rosetta_precision_at_k.py` on `benchmark_results_phase3_tm0.0/pair_scores.tsv`.

Random baseline (prior): 16.67% (2,369 positives / 14,214 evaluated pairs)

| k | TP in top-k | Precision | vs random (lift) |
|---|---|---|---|
| 50 | 47 | 94.00% | 5.64× |
| 100 | 88 | 88.00% | 5.28× |
| 200 | 165 | 82.50% | 4.95× |
| 500 | 359 | 71.80% | 4.31× |
| 1,000 | 528 | 52.80% | 3.17× |

### Non-zero scoring pair analysis

| Category | Count | % of group |
|---|---|---|
| Total pairs evaluated | 14,214 | — |
| Non-zero scoring pairs | 9,901 | 69.7% of all |
| — Positives (score > 0) | 1,832 | 77.3% of all positives |
| — Negatives (score > 0) | 8,069 | 68.1% of all negatives |
| Precision (score > 0) | 18.50% | 1.11× random |
| Zero-scoring pairs | 4,313 | 30.3% of all |
| — Positives (score = 0) | 537 | structurally unreachable |
| — Negatives (score = 0) | 3,776 | — |

### Interpretation

Precision@k is very strong at the top of the ranked list (94% at k=50, 88% at k=100)
but degrades as k grows. This confirms the method is a high-confidence precision
instrument: when a strong bridge exists, the prediction is almost always correct.

Precision among non-zero-scoring pairs (18.5%, 1.11× random) is barely above random.
Having *any* bridge is nearly uninformative — weak bridges appear in negative pairs by
chance due to shared common folds. The signal is concentrated in the score magnitude.

The AUCPR of 2.73× understates the method's discrimination power at high confidence
because it averages over the entire ranked list, including the long tail of weak bridges.

---

## Key Numbers for Report

> Numbers marked [PRE-FILTER] are from before the self-interaction filter was added.
> Numbers marked [POST-FILTER] will be filled after re-running on the server.

| Metric | Value | Status | Source |
|--------|-------|--------|--------|
| Zhang sub-DB domain names | 2,753 | stable | rosetta_dump_zhang_domains.py |
| TED pair list total lines | 129,440,391 | stable | rosetta_build_template_index.py |
| Template pairs retained | 1,756 | stable | rosetta_build_template_index.py |
| Random AUCPR baseline | 0.0099 | stable | Derived from wp=0.01 |
| Random P@k baseline | 16.67% | stable | 2,369/14,214 evaluated pairs |
| Universe size | 14,201 proteins | stable | unchanged by filter |
| Covered positives | 2,369 / 3,000 (79.0%) | stable | unchanged by filter |
| AUCPR (TM=0.0) | 0.0271 (2.73× baseline) | **[PRE-FILTER]** | benchmark_results_phase3_tm0.0 |
| AUCPR (TM=0.3) | 0.0271 (2.73× baseline) | **[PRE-FILTER]** | benchmark_results_phase3_tm0.3 |
| AUCPR (TM=0.5) | 0.0270 (2.72× baseline) | **[PRE-FILTER]** | benchmark_results_phase3_tm0.5 |
| AUCPR (TM=0.7) | 0.0257 (2.60× baseline) | **[PRE-FILTER]** | benchmark_results_phase3_tm0.7 |
| ROC-AUC (TM=0.0) | 0.6388 | **[PRE-FILTER]** | benchmark_results_phase3_tm0.0 |
| ROC-AUC (TM=0.3) | 0.6400 (peak) | **[PRE-FILTER]** | benchmark_results_phase3_tm0.3 |
| ROC-AUC (TM=0.5) | 0.6310 | **[PRE-FILTER]** | benchmark_results_phase3_tm0.5 |
| ROC-AUC (TM=0.7) | 0.5762 | **[PRE-FILTER]** | benchmark_results_phase3_tm0.7 |
| Precision@50 | 94.0% (5.64× random) | **[PRE-FILTER]** | rosetta_precision_at_k.py |
| Precision@100 | 88.0% (5.28× random) | **[PRE-FILTER]** | rosetta_precision_at_k.py |
| Precision@200 | 82.5% (4.95× random) | **[PRE-FILTER]** | rosetta_precision_at_k.py |
| Precision@500 | 71.8% (4.31× random) | **[PRE-FILTER]** | rosetta_precision_at_k.py |
| Precision@1000 | 52.8% (3.17× random) | **[PRE-FILTER]** | rosetta_precision_at_k.py |
| Precision (score > 0) | 18.5% (1.11× random) | **[PRE-FILTER]** | rosetta_precision_at_k.py |
| Non-zero pairs | 9,901 / 14,214 (69.7%) | **[PRE-FILTER]** | rosetta_precision_at_k.py |
| Structurally unreachable positives | 537 (score = 0) | **[PRE-FILTER]** | rosetta_precision_at_k.py |
| AUCPR post-filter (TM=0.0) | **[TBD]** | [POST-FILTER] | benchmark_results_filtered_tm0.0 |
| ROC-AUC post-filter (TM=0.0) | **[TBD]** | [POST-FILTER] | benchmark_results_filtered_tm0.0 |
| Hits@1 post-filter (reachable) | **[TBD]** | [POST-FILTER] | rosetta_precision_at_k.py |

---

## Limitations to Acknowledge in Report

### Limitation 1: 20.1% of benchmark pairs are structurally uncoverable

The pipeline covers 2,369/3,000 (79.0%) of Zhang benchmark positive pairs. The remaining
631 pairs (20.1%) involve proteins with no AlphaFold2 predicted structure — absent from TED
entirely. This is a hard ceiling shared by all structure-based PPI prediction approaches.

**What to write in the report:**
> The pipeline achieves 79.0% coverage of Zhang benchmark positive pairs (2,369/3,000).
> The remaining 20.1% are proteins with no AlphaFold2 predicted structure, which cannot
> be covered by any domain-structure-based method. Within the covered set, the method
> achieves AUCPR of 2.73× random baseline and ROC-AUC of 0.6388.

### Limitation 2: Template library restricted to Zhang proteins (circularity)

The template index only contains co-occurring domain pairs where at least one domain belongs
to a Zhang benchmark protein. This means templates and evaluation proteins are drawn from
the same set. A true cross-organism Rosetta Stone would build the template index from all
129M TED pairs across all of AFDB — eliminating circularity entirely but requiring
significantly more compute.

### Limitation 3: Single structural signal only

No co-expression, co-localisation, or evolutionary co-variation signals are used. Real PPI
prediction systems combine structural, sequence coevolution, and interaction database
evidence. This method is purely structural-homology-based.

### Limitation 4: Proteins with no AlphaFold model

Proteins with no AlphaFold2 predicted structure cannot be covered by any domain-based
method. This is a hard ceiling shared by all structure-based PPI prediction approaches.

---

## Status

| Task | Status |
|------|--------|
| Algorithm design (Rosetta Stone) | Done |
| Template index built | Done — 1,756 pairs, 2,753 entries |
| Search cache — multi-domain proteins | Done — rosetta_search_both_sides.py |
| Search cache — single-domain proteins | Done — rosetta_search_ted365m.py (5,703 proteins) |
| Benchmark + threshold sweep (pre-filter) | Done — 2.73× baseline, 79.0% coverage |
| Self-interaction filter — code | **Done** — added to benchmark/explain/predict scripts |
| Self-interaction filter — unit tests | **Done** — 13/13 passing |
| Benchmark re-run with filter | **TODO** — run on server, fill in post-filter numbers |
| New case example | **TODO** — run --find-examples on server after filter re-run |
| Figures (updated) | **TODO** — regenerate after benchmark re-run |
| Report | In progress |
