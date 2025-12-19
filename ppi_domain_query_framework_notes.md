# Protein–Protein Interaction Query Framework (Exploration Notes)

These notes consolidate the key concepts and dataset interpretations needed to design a query framework that **identifies and evaluates protein–protein interaction (PPI) records only within a biologically relevant subset of proteins**, where relevance is defined by **taxonomy** and/or **structural domain identity/similarity**, without scanning the full interaction database.

---

## 1) Official restatement of the problem

### Context
PPI datasets can be extremely large. Users often care only about a subset of proteins—for example:
- proteins from a specific **taxonomic group** (TaxID), and/or
- proteins containing a **particular domain**, and/or
- proteins whose domains are **structurally similar** to a query domain/structure.

Modern structure-based approaches (e.g., Merizo-search) operate at the **domain** level, while interaction records are typically **protein**-level, creating a domain–protein abstraction gap.

### Formal problem statement
Given:
- a universe of proteins **P**
- interaction records **I ⊆ P × P**
- taxonomy mapping **T: P → TaxID**
- domain mapping **D: P → {domain instances}**
- a domain similarity function **S(d_query, d)** (e.g., Merizo-search domain similarity)

Build a query framework that:
1. Accepts optional constraints:
   - taxonomy constraint (TaxID)
   - domain constraint (exact domain ID / fold / label)
   - domain similarity constraint (query structure → similar domains)
2. Produces a relevant protein subset **P' ⊆ P** satisfying provided constraints
3. Evaluates only the induced interaction subset:
   **I' = {(p_i, p_j) ∈ I | p_i ∈ P' and p_j ∈ P'}**

### Intended scope (what the framework is responsible for)
- Determining which interaction records are worth considering at all under given biological constraints
- Bridging domain-level search outputs to protein-level interaction evaluation

Not necessarily responsible for:
- predicting new PPIs from first principles
- performing full-database scans per query
- building new structural similarity methods (Merizo-search is treated as an external component)

---

## 2) Key conceptual distinction: domain instance vs domain type

This distinction resolves the “isn’t it the other way around?” confusion.

### Domain instance (what the TED files contain)
A **domain instance** is:
- a specific region of a specific protein structure

Example domain instance ID:
- `AF-A0A000-F1-model_v4_TED01`

It refers to **one** protein model (`AF-A0A000-F1-model_v4`) and **one** segmented domain (`TED01`).

**Rule:** A *domain instance* belongs to exactly **one** protein.

### Domain type / fold / family (what repeats across proteins)
A **domain type** (or fold/family/cluster) is an abstract category.
Many **domain instances** (from many proteins) can map to the same:
- CATH fold (e.g., `3.40.50.300`)
- embedding cluster
- “similar domains” returned by Merizo-search

**Rule:** A protein can contain multiple domain instances; a domain type can occur in many proteins via many instances.

---

## 3) File 1 — Domain definitions and metadata
`/mnt/bigstore/ted/ted_365.domain_summary.cath.globularity.taxid.tsv`

### What this file represents
- **Each row = one domain instance**
- Domain instances come from **Merizo segmentation**
- Domains can be **continuous or discontinuous** (multi-segment)
- Each row attaches **quality metrics**, **taxonomy**, and **structural classification**

### How to read the residue range field
Examples:
- `54-288` → one continuous segment
- `11-41_290-389` → discontinuous domain made of **two** segments
- `15-188_223-252_288-302` → discontinuous domain made of **three** segments

The underscore `_` separates segments.

### What the key columns mean (conceptual)
Although the file may not come with a header here, your sample rows indicate it includes:

- **Domain instance ID**: `AF-..._TED0X`  
  Unique identifier for a domain instance inside a specific protein model.
- **Segmentation confidence**: `high / medium`  
  Confidence in the domain boundary definition.
- **Residue ranges**: e.g., `11-41_290-389`  
  Defines which residues belong to the domain.
- **Domain length**: total residues across segments
- **Number of segments**: 1, 2, 3, ...
- **Globularity score**: how “domain-like” / compact the segment is
- **Taxonomy source**: includes TaxID (e.g., `proteome-tax_id-67581-0_v4`)
- **CATH / fold label (optional)**: e.g., `3.90.1150.10` or `-` if missing
- **Architecture class (optional)**: e.g., `H`, `T`, or `-`
- **Assignment method (optional)**: e.g., `foldseek`, `foldclass`, or `-`
- **Species name** and **full taxonomic lineage**

### Why this file matters
This file defines the domain universe used to connect:
- taxonomy filtering (TaxID / lineage)  
and
- domain-level search or similarity (Merizo-search outputs)  
to
- protein-level interaction evaluation.

---

## 4) File 2 — Domain pairs
`/mnt/bigstore/ted/pair_list_20250128`

### What this file represents
- **Each row = a paired relationship between two domain instances**
- In your examples, every pair is **within the same protein model**:
  - `AF-XXXX-model_v4_TED0A : AF-XXXX-model_v4_TED0B`
  - same `AF-XXXX-model_v4` prefix on both sides

So this is best interpreted as an **intra-protein domain–domain pairing/adjacency/interaction** list.

### Format (as observed)
Each line looks like:
- `domainA:domainB  valueA:valueB  pair_metric`

Where:
- `domainA:domainB` are domain instance IDs
- `valueA:valueB` are per-domain numeric values (often quality-like)
- `pair_metric` is a numeric score/measure for the pairing

(Exact semantics of the numeric fields depend on how this file was generated, but the structure is consistent.)

### What this file is NOT
- Not a protein–protein interaction database directly
- Not cross-protein domain interactions
- Not evolutionary co-occurrence

### What it IS
- A domain-level relationship graph **inside proteins**
- A description of which domain combinations are “paired” when co-present

---

## 5) What to do with these two files together (conceptually)

These two files are complementary:

### `domain_summary...tsv` defines the **nodes**
- Each domain instance is a node, with metadata:
  - protein association
  - residue boundaries
  - confidence / quality
  - taxonomy
  - fold labels (optional)

### `pair_list...` defines the **edges**
- Each row defines an edge between two domain instances:
  - “these two domains are paired”

Together they form a **domain interaction graph (intra-protein)**.

### How this supports your query framework
Your framework’s relevance definition can use:

1. **Taxonomy relevance**
   - Use TaxID / lineage to restrict proteins

2. **Domain relevance**
   - Restrict proteins by containing a specific domain instance/type/fold

3. **Domain similarity relevance**
   - Use Merizo-search to turn a query domain/structure into a set of similar domain instances
   - Map those domain instances back to their proteins

4. **Domain pairing relevance**
   - Use domain–domain pairing as a biological prior:
     - domains are not just present; they participate in meaningful pairings
   - This can help justify why a protein subset is biologically meaningful before evaluating PPIs

In one line:
> The domain summary file tells you **what domains exist and what proteins they belong to**, and the pair list tells you **which domains are paired**, enabling domain-aware constraints to define the relevant protein subset for PPI evaluation.

---

## 6) One-sentence takeaway

**Proteins contain domain instances; domain instances are uniquely defined by protein + residue ranges; domain types/folds repeat across proteins.**  
The `domain_summary` file defines domain instances (nodes + metadata), while `pair_list` defines domain–domain pairings (edges), and together they provide the domain-level structure needed to constrain PPI queries by taxonomy and/or domain identity/similarity without scanning the full interaction database.
