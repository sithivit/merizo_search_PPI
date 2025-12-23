# Final Year Project: Merizo-search PPI Query Enhancement

## Project Goal

**Upgrade Merizo-search with filtering capabilities for protein-protein interaction (PPI) queries.**

### Core Functionality

Merizo-search returns high-probability protein pairs that are likely to interact. This project adds **fast filtering on those results** to enable:

1. **Multiple filter types** including:
   - Taxonomy ID (e.g., "show only human proteins")
   - Confidence level (e.g., "high-confidence domain segmentations only")
   - Domain properties (e.g., "CATH fold X", "globularity score > 0.8")
   - Domain instance/type
   - And more based on available metadata

2. **Fast querying via indexing:**
   - Use index systems or hash tables
   - Query only subsets of the database (not full scans)
   - Sub-second performance for filtering operations

3. **Integration with existing Merizo-search workflow:**
   - Merizo-search identifies potential PPIs
   - Filtering system narrows results by biological constraints
   - Returns relevant subset for downstream analysis

### Future Extension (Not Current Focus)

Later stages will use **inference logic**: if domain A interacts with domain B (from domain pairing data), and proteins contain these domains, then those proteins could interact. This is planned but not the immediate priority.

---

## Understanding the Data

### Key Conceptual Distinction: Domain Instance vs Domain Type

**Domain Instance** (what the TED files contain):
- A specific region of a specific protein structure
- Example ID: `AF-A0A000-F1-model_v4_TED01`
- Belongs to exactly **one** protein
- Each row in the domain summary file = one domain instance

**Domain Type/Fold/Family** (what repeats across proteins):
- An abstract category (e.g., CATH fold `3.40.50.300`)
- Many domain instances from many proteins can have the same type
- A protein can contain multiple domain instances
- A domain type can occur in many proteins via many instances

### Available Data Files

#### File 1: Domain Summary
**Path:** `/mnt/bigstore/ted/ted_365.domain_summary.cath.globularity.taxid.tsv`

**What it contains:**
- Each row = one domain instance
- Domain instances from Merizo segmentation
- Can be continuous (`54-288`) or discontinuous (`11-41_290-389`)

**Key columns:**
- Domain instance ID: `AF-..._TED0X`
- Residue ranges: e.g., `11-41_290-389` (underscore separates segments)
- Domain length, number of segments
- Segmentation confidence: high/medium
- Globularity score
- **Taxonomy source with TaxID** (e.g., `proteome-tax_id-67581-0_v4`)
- CATH/fold label (optional): e.g., `3.90.1150.10` or `-`
- Architecture class, assignment method
- Species name and full taxonomic lineage

**Why it matters:**
- Defines which domains exist and what proteins they belong to
- Enables taxonomy filtering
- Enables domain-level search/similarity

#### File 2: Domain Pair List
**Path:** `/mnt/bigstore/ted/pair_list_20250128`

**What it contains:**
- Each row = paired relationship between two domain instances
- **Intra-protein domain pairings** (domains within the same protein)
- Format: `domainA:domainB  valueA:valueB  pair_metric`

**What it tells us:**
- Which domain combinations are "paired" when co-present
- A domain-level relationship graph inside proteins
- Can indicate domains with high chance of interacting

**What it is NOT:**
- Not protein-protein interactions (yet)
- Not cross-protein domain interactions
- Not evolutionary co-occurrence

### How These Files Work Together

Think of them as a graph:
- **Nodes:** Domain instances (from `domain_summary...tsv`) with metadata (protein, taxonomy, fold, quality)
- **Edges:** Domain-domain pairings (from `pair_list...`) indicating which domains pair together

**For your query framework:**
1. **Taxonomy relevance:** Use TaxID/lineage to restrict proteins
2. **Domain relevance:** Restrict proteins by containing specific domain type/fold
3. **Domain similarity relevance:** Use Merizo-search to find similar domain instances, map back to proteins
4. **Domain pairing relevance:** Use domain-domain pairings to identify domains with high interaction potential

---

## Formal Problem Statement

Given:
- Universe of proteins **P**
- Domain mapping **D: P → {domain instances}**
- Taxonomy mapping **T: P → TaxID**
- Domain similarity function **S(d_query, d)** (Merizo-search)
- Domain pairing data (from pair list)

Build a query framework that:
1. Accepts constraints:
   - Taxonomy constraint (TaxID)
   - Domain constraint (exact domain ID / fold / label)
   - Domain similarity constraint (query structure → similar domains)
2. Produces relevant protein subset **P' ⊆ P** satisfying constraints
3. Returns domains with high chance of interacting based on pairing data

---

## Project Specifications (Answered)

### 1. Filter Types Available

**Multiple filters will be supported:**
- **Taxonomy-based:** Filter by TaxID or lineage
- **Quality-based:** Filter by confidence level (high/medium), globularity score
- **Domain-based:** Filter by CATH fold, architecture class, domain type
- **Sample-based:** Optional for testing infrastructure
- **Combination filters:** Any combination of the above

All filters should be combinable (e.g., "human proteins with high-confidence CATH fold 3.40.50.300 domains").

### 2. Query System Architecture

**Simple filtering integrated with Merizo-search:**

1. **Merizo-search** returns high-probability protein pairs that will interact
2. **Filtering system** (this project) enables fast filtering on those results by:
   - Taxonomy (e.g., keep only human-human interactions)
   - Domain properties (e.g., keep only interactions involving specific folds)
   - Confidence levels (e.g., keep only high-confidence domains)

3. **Performance optimization:**
   - Use **index systems** or **hash tables** for fast lookups
   - Query **only subsets** of the database (not full scans)
   - Target: sub-second query performance

**Workflow:**
```
Merizo-search → PPI candidates → Filter by taxonomy/domain → Relevant subset
```

### 3. Project Scope

**This is an upgrade/enhancement to the existing Merizo-search program.**

Deliverables:
- Enhanced filtering functionality integrated into Merizo-search workflow
- Indexing/hash table system for fast queries
- Query interface/API for applying multiple filters
- Documentation and example usage

### 4. Database/Testing Approach

**Test on existing Merizo-search database for now.**

- Use `examples/database/ted100_9606_small/` or similar existing databases
- Understand existing Merizo database format
- Build filtering layer on top of existing structure
- Don't rebuild from scratch unless necessary

### 5. PPI Inference Logic

**Planned for later stage (not current focus):**

The eventual inference logic:
- If domain A interacts with domain B (from domain pairing data)
- And protein X contains domain A
- And protein Y contains domain B
- Then proteins X and Y could interact

This domain-to-protein inference will be implemented in a future phase.

### 6. Current Phase Focus

**Focus areas for this phase:**
1. Understanding existing Merizo-search database structure
2. Designing and implementing indexing system (hash tables/indexes)
3. Building filtering functionality for:
   - Taxonomy
   - Domain properties
   - Confidence levels
4. Integrating filters with Merizo-search PPI results
5. Performance testing and optimization

---

## Database Schema Reference

Understanding the data structure (based on existing Merizo databases):

```sql
-- Proteins table
CREATE TABLE proteins (
    protein_id TEXT PRIMARY KEY,  -- e.g., AF-A0A000-F1-model_v4
    taxonomy_id INTEGER,
    species_name TEXT,
    lineage TEXT
);

-- Domains table
CREATE TABLE domains (
    domain_id TEXT PRIMARY KEY,  -- e.g., AF-A0A000-F1-model_v4_TED01
    protein_id TEXT,
    residue_ranges TEXT,
    domain_length INTEGER,
    num_segments INTEGER,
    confidence TEXT,
    globularity_score REAL,
    cath_fold TEXT,
    architecture_class TEXT,
    FOREIGN KEY (protein_id) REFERENCES proteins(protein_id)
);

-- Domain pairings table (interaction potential)
CREATE TABLE domain_pairings (
    domain_a TEXT,
    domain_b TEXT,
    value_a REAL,
    value_b REAL,
    pair_metric REAL,
    FOREIGN KEY (domain_a) REFERENCES domains(domain_id),
    FOREIGN KEY (domain_b) REFERENCES domains(domain_id)
);

-- Indexes for fast queries
CREATE INDEX idx_domains_protein ON domains(protein_id);
CREATE INDEX idx_domains_cath ON domains(cath_fold);
CREATE INDEX idx_proteins_taxonomy ON proteins(taxonomy_id);
CREATE INDEX idx_pairings_domain_a ON domain_pairings(domain_a);
CREATE INDEX idx_pairings_domain_b ON domain_pairings(domain_b);
```

---

## Example Filter Use Cases

These filters should be efficiently implementable using indexes/hash tables:

```python
# Use Case 1: Filter PPI results by taxonomy
# Input: Merizo-search PPI candidates
# Filter: Keep only human-human interactions (TaxID=9606)
# Output: Subset of PPIs involving only human proteins

# Use Case 2: Filter by domain type
# Input: Merizo-search PPI candidates
# Filter: Keep only interactions involving CATH fold 3.40.50.300
# Output: Subset of PPIs where at least one protein contains this fold

# Use Case 3: Filter by confidence
# Input: Merizo-search PPI candidates
# Filter: Keep only high-confidence domain segmentations
# Output: Subset of PPIs with reliable domain boundaries

# Use Case 4: Combined filters
# Input: Merizo-search PPI candidates
# Filter: Human proteins + high confidence + specific CATH fold
# Output: Highly-filtered, biologically-relevant PPI subset

# Use Case 5: Fast lookup by protein ID
# Input: Protein ID from PPI candidate
# Action: Retrieve all metadata (taxonomy, domains, confidence) instantly
# Implementation: Hash table or index for O(1) or O(log n) lookup
```

### Performance Requirements

**Key indexing strategies:**
- **Hash tables** for protein ID → metadata lookups
- **Inverted indexes** for taxonomy → protein IDs
- **Inverted indexes** for CATH fold → domain IDs → protein IDs
- **Bitmap indexes** for confidence levels (high/medium)

**Target performance:**
- Single filter query: < 100ms
- Combined filter query: < 500ms
- No full database scans for common queries

---

## Success Criteria

The filtering enhancement should demonstrate:

1. **Functional filters:**
   - ✅ Filter Merizo-search PPI results by taxonomy
   - ✅ Filter by domain type/fold (CATH)
   - ✅ Filter by confidence level
   - ✅ Combine multiple filters efficiently

2. **Performance:**
   - Single filter queries: < 100ms
   - Combined filter queries: < 500ms
   - No full database scans

3. **Correctness:**
   - Filters return accurate results
   - No false positives/negatives in filtering
   - Metadata lookups are correct

4. **Integration:**
   - Works seamlessly with existing Merizo-search output
   - Minimal changes to existing Merizo-search workflow
   - Clear API/interface for applying filters

5. **Documentation:**
   - Clear API documentation
   - Example filter usage
   - Performance benchmarks
   - Index/hash table design documentation

---

## Implementation Roadmap

### Phase 1: Understanding Merizo-search (Current)

1. **Examine existing database structure:**
   - Explore `examples/database/ted100_9606_small/`
   - Understand file formats and organization
   - Document current data schema

2. **Understand Merizo-search output:**
   - What format are PPI candidates returned in?
   - What information is available (protein IDs, scores, etc.)?
   - How are results currently accessed?

3. **Identify filterable attributes:**
   - Map available metadata fields
   - Determine what can be indexed
   - Identify common filter use cases

### Phase 2: Design Indexing System

1. **Choose indexing strategy:**
   - Hash tables for protein ID lookups
   - Inverted indexes for taxonomy/domain lookups
   - Bitmap indexes for categorical data (confidence levels)

2. **Design filter API:**
   - Input: Merizo-search PPI results + filter criteria
   - Output: Filtered PPI subset
   - Interface: Python API or command-line tool

3. **Plan integration:**
   - How to plug into existing Merizo-search workflow
   - Minimal disruption to current functionality
   - Backward compatibility considerations

### Phase 3: Implementation

1. **Build index/hash table system:**
   - Implement protein ID → metadata hash table
   - Implement taxonomy → protein ID inverted index
   - Implement CATH fold → domain → protein index
   - Implement confidence level bitmap index

2. **Implement filter functions:**
   - `filter_by_taxonomy(ppi_results, taxid)`
   - `filter_by_domain(ppi_results, cath_fold)`
   - `filter_by_confidence(ppi_results, level)`
   - `apply_combined_filters(ppi_results, filter_dict)`

3. **Optimize performance:**
   - Benchmark query times
   - Optimize slow operations
   - Ensure sub-second performance

### Phase 4: Testing & Documentation

1. **Test on existing database:**
   - Run filters on `ted100_9606_small` results
   - Verify correctness
   - Measure performance

2. **Create documentation:**
   - API reference
   - Usage examples
   - Performance benchmarks
   - Design documentation

3. **Integration testing:**
   - Test with real Merizo-search output
   - Ensure seamless workflow
   - Validate results

---

## Summary

**What you're building:**
A filtering enhancement for Merizo-search that:
- Takes PPI candidates from Merizo-search (high-probability protein pairs)
- Applies fast filters based on taxonomy, domain properties, and confidence
- Uses hash tables and indexes to query subsets efficiently (no full scans)
- Returns biologically-relevant PPI subsets in sub-second time
- Integrates seamlessly into existing Merizo-search workflow

**Key technologies:**
- **Hash tables** for O(1) protein metadata lookups
- **Inverted indexes** for taxonomy and domain filtering
- **Bitmap indexes** for categorical filtering (confidence levels)
- Python API for easy integration

**Why it matters:**
Merizo-search identifies potential PPIs at scale. This filtering system enables researchers to quickly narrow results to their specific biological context (e.g., "human proteins with high-confidence CATH fold X") without scanning the entire database. This makes large-scale PPI analysis practical and targeted.

**Future direction:**
Once filtering is established, the project can extend to **inference logic** where domain-domain interactions inform protein-protein interaction predictions.
