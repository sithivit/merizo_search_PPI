# Chapter 2: Background and Related Work — Content Plan

## Overall narrative arc

The chapter should tell a **story** that leads the reader logically from "why PPIs matter" to "why structural homology transfer via Merizo-search is a sensible approach that hasn't been tried yet." Each section should end by motivating the next one. The final section should land on the evaluation framework, so the reader understands how you'll measure success.

---

## 2.1 Protein–Protein Interactions (~1.5 pages)

**Purpose:** Establish the biological importance of PPIs and why mapping them matters.

**Content plan:**

- **What are PPIs?** Proteins don't work alone — they physically bind to form complexes that carry out nearly all cellular functions. PPIs include stable complexes (e.g. ribosomes, proteasomes) and transient signalling interactions (e.g. kinase–substrate binding).

- **Why do they matter?** A complete PPI map (the "interactome") for an organism would:
  - Reveal drug targets (many diseases involve disrupted interactions)
  - Explain disease mechanisms (e.g. oncogenic signalling cascades)
  - Guide functional annotation of uncharacterised proteins ("guilt by association")

- **The scale of the problem:** Human proteome has ~20,000 proteins → ~2×10⁸ possible pairs. Current experimental coverage is <10% of this space.

- **Experimental methods and their limitations:**
  - **Yeast two-hybrid (Y2H):** Tests binary interactions in vivo. High false-positive rate, biased toward nuclear/soluble proteins, misses membrane interactions.
  - **Affinity purification–mass spectrometry (AP-MS):** Captures complexes but can't distinguish direct from indirect interactions. Expensive per bait protein.
  - **Cross-linking mass spectrometry (XL-MS):** Provides distance constraints but low throughput.
  - **Key point:** Low overlap between independent experimental datasets (e.g. Y2H vs AP-MS) — each captures a different slice of the interactome.

- **Bridge to next section:** "Given the limitations of experimental approaches, computational methods have become essential for narrowing the search space."

**Key references to cite:** Zhang et al. 2025 (for interactome scale/benchmark context)

---

## 2.2 Computational PPI Prediction (~2 pages)

**Purpose:** Survey existing computational approaches and show their trade-offs, establishing the gap your project fills.

**Content plan:**

- **Overview:** Computational PPI prediction has been approached from multiple angles. This section reviews the main families of methods and their respective strengths and weaknesses.

- **Sequence-based co-evolution methods:**
  - Principle: Interacting proteins co-evolve — mutations in one are compensated by mutations in the other.
  - Methods: Correlated mutations, direct coupling analysis (DCA), EVcouplings.
  - Limitation: Require deep multiple sequence alignments (MSAs). Many protein families lack sufficient homologs, especially in eukaryotes.

- **Machine-learning classifiers:**
  - Principle: Train on known PPI pairs using features like sequence similarity, domain composition, gene expression correlation, subcellular localisation.
  - Methods: Random forests, SVMs, deep learning approaches.
  - Limitation: Performance depends on training data coverage and quality. Risk of learning dataset-specific biases. Don't generalise well across organisms.

- **Physics-based docking:**
  - Principle: Simulate physical binding between two protein structures.
  - Methods: ZDOCK, ClusPro, HADDOCK.
  - Limitation: Computationally expensive (minutes to hours per pair). Requires accurate input structures. Infeasible at proteome scale (2×10⁸ pairs).

- **Structure-based approaches (most relevant to your project):**
  - Principle: Use 3D structural information to predict whether two proteins can physically interact.
  - **AlphaFold-Multimer:** Predicts complex structures directly but very expensive per pair. Zhang et al. (2025) used AF-Multimer as one signal in their multi-feature ensemble.
  - **Template-based docking:** If a homolog of protein A is seen in a complex with a homolog of protein B in the PDB, infer A–B interaction.
  - **Domain-based approaches:** Because interactions are often mediated by specific domains, not whole proteins.

- **The gap your project fills:**
  - Most structure-based PPI methods either require expensive pairwise computation (docking, AF-Multimer) or rely on experimentally solved complexes (template-based).
  - The availability of AlphaFold2 predicted structures for 200M+ proteins, combined with fast structural search tools, creates the opportunity for a *scalable, domain-level structural homology transfer* approach.
  - This approach has been used for function annotation but **not systematically evaluated as a standalone PPI prediction signal**.

**Key references:** Zhang et al. 2025 (for the multi-feature ensemble context — their method combines AF-Multimer, co-evolution, and other signals; your project tests whether structural homology alone is useful)

---

## 2.3 AlphaFold2 and the Structural Biology Revolution (~1.5 pages)

**Purpose:** Explain how AlphaFold2 changed the landscape and why it enables your approach.

**Content plan:**

- **The protein structure prediction problem:** For decades, determining protein structure required experimental methods (X-ray crystallography, cryo-EM, NMR) — slow, expensive, and not all proteins are amenable.

- **AlphaFold2 breakthrough:**
  - Won CASP14 in 2020, published 2021.
  - Predicts tertiary structure from sequence alone with near-experimental accuracy (median GDT-TS ~92 on CASP14).
  - Architecture: MSA + pair representation → Evoformer → Structure Module with invariant point attention.
  - The key point for your project: it provides *reliable predicted structures* at massive scale.

- **The AlphaFold Protein Structure Database (AFDB):**
  - Initial release: 350,000 human proteome structures (2021).
  - Expanded to 200+ million structures across hundreds of organisms (2022).
  - Covers nearly all known protein sequences in UniProt.
  - Quality metric: pLDDT (per-residue confidence). Regions with pLDDT > 70 are generally reliable.

- **Why this matters for PPI prediction:**
  - Before AlphaFold2, structure-based methods were limited to the ~180,000 experimentally solved structures in the PDB.
  - Now, structures are available for virtually every known protein — enabling structural comparison at *proteome scale* for the first time.
  - This is the enabling shift that makes your project feasible.

- **Bridge to next section:** "To make use of these 200M+ structures for domain-level analysis, a systematic decomposition into structural domains is needed. The TED dataset provides exactly this."

**Key references:** alphafold2, ted_dataset

---

## 2.4 The TED and UniProt Datasets (~1.5 pages)

**Purpose:** Explain where your domain pair data comes from and why it's central to the pipeline.

**Content plan:**

- **UniProt as the protein catalogue:**
  - UniProt is the comprehensive resource for protein sequence and annotation data.
  - UniProt IDs are used as the standard protein identifiers throughout your pipeline and the AlphaFold database.

- **The TED dataset (The Encyclopaedia of Domains):**
  - Published: Lau et al. (Science, 2024).
  - What it is: A comprehensive decomposition of all 200M+ AlphaFold-predicted structures into their constituent structural domains.
  - Total: ~365 million domains identified.
  - Method: Uses Merizo (see Section 2.5) to segment each AlphaFold structure into domains.

- **Domain–domain contacts (the pair list):**
  - TED identifies *intra-protein* domain–domain contacts — pairs of domains within the same protein that are in physical contact.
  - The pair list contains >200 million domain pair entries.
  - **Why this matters for your project:** Each contact pair represents a potential interaction interface. If Domain B contacts Domain A within protein Y, and another protein contains a structural homolog B' of Domain B, then B' may interact with A-like domains too.
  - The pair list is the source of your "interaction templates."

- **Domain metadata:**
  - Each domain has associated metadata: taxonomy ID (organism), CATH fold classification, model confidence (pLDDT), globularity score, domain density.
  - This metadata enables the filtering infrastructure you built.

- **Bridge to next section:** "The TED dataset provides the domain decomposition and pair data. To search for structural homologs of these domains at scale, the project uses Merizo-search."

**Key references:** ted_dataset

---

## 2.5 Merizo-search (~2 pages)

**Purpose:** Explain the technical details of the tool your pipeline is built on.

### 2.5.1 Merizo: Domain Segmentation (~1 page)

- **What it does:** Segments a multi-domain protein structure into its constituent domains — identifies domain boundaries.

- **Architecture:**
  - Input: Protein structure (PDB/CIF file) — specifically, Cα coordinates.
  - Uses invariant point attention (IPA) — the same attention mechanism used in AlphaFold2's structure module.
  - Output: Per-residue domain assignment. Each residue is labelled as belonging to domain 1, 2, 3, ..., or as a linker region.

- **Why IPA?** Structure-aware attention that respects 3D geometry — rotations and translations don't change the output. This is essential for a domain segmentation tool that must work regardless of the protein's orientation.

- **Quality filtering:** Regions with low pLDDT (AlphaFold confidence) are treated as disordered and excluded. Only well-structured regions are assigned to domains.

- **Performance:** Processes a typical protein in ~50ms. Can segment the entire AFDB.

**Key references:** merizo

### 2.5.2 Foldclass: Structural Search (~1 page)

- **What it does:** Given a query domain structure, rapidly finds the most structurally similar domains in a precomputed database.

- **Two-stage search pipeline:**
  1. **Embedding stage:** Encode each domain as a 128-dimensional vector using an equivariant graph neural network (EGNN). The EGNN operates on the Cα backbone graph and produces embeddings that are invariant to rotation/translation.
  2. **Retrieval stage:** Cosine similarity search over the precomputed embedding database to retrieve the top-k most similar candidates.
  3. **Re-ranking stage:** TM-align structural alignment on the top candidates for precise scoring. TM-score ranges from 0 to 1; scores above 0.5 generally indicate the same fold.

- **Database format:**
  - Precomputed embeddings stored as memory-mapped binary files (128 × float32 per domain).
  - Index files for sequences, Cα coordinates, and metadata.
  - Configuration via JSON file.

- **Scalability:** The embedding search is the fast part (~100ms for 365M domains). TM-align verification is slower (~1–5s per hit) but only applied to the top-k candidates. This two-stage design makes it practical to search hundreds of millions of domains within minutes.

**Key references:** merizo_search, tmalign

---

## 2.6 Structural Homology Transfer (~1.5 pages)

**Purpose:** Explain the core principle your project is based on, and show it's established for function annotation but novel for PPI prediction.

**Content plan:**

- **The principle:** If two proteins share a structurally similar domain, they may share related functions. This is because protein structure is more conserved than sequence — two proteins with <20% sequence identity can have near-identical folds and similar functions.

- **Established use: function annotation transfer:**
  - CATH and SCOP classify proteins by structural similarity. Proteins within the same superfamily often share function.
  - Tools like Dali, FATCAT, and now Foldclass are routinely used to annotate "proteins of unknown function" by finding structural neighbours with known function.
  - This is well-established and widely accepted.

- **CATH classification (relevant to your filtering):**
  - Hierarchical: Class → Architecture → Topology → Homologous superfamily.
  - Domains within the same CATH superfamily share a common ancestor.
  - Your SQLite metadata index stores CATH fold classifications to enable fold-specific queries.

- **The gap: applying structural homology transfer to PPI prediction:**
  - The idea is simple: if domain B in protein Y mediates an interaction with domain A, and domain B' in protein X is structurally similar to B, then X may interact with A (or A-like proteins) through B'.
  - This is the "structural homology transfer" hypothesis applied to domain–domain interactions.
  - While this idea is intuitive, it has **not been systematically evaluated** as a standalone PPI prediction signal at proteome scale.
  - **Why not?** Until AlphaFold2 + TED + Merizo-search, the infrastructure didn't exist to do this at scale. Previously, you'd need experimentally solved structures (limited to PDB), and domain decomposition tools weren't fast enough for 200M+ proteins.

- **Limitations to acknowledge upfront:**
  - Structural similarity ≠ functional similarity in all cases (convergent evolution, paralogs with different partners).
  - Interactions mediated by intrinsically disordered regions won't be captured (no stable domain structure).
  - This is a single signal — real PPI prediction likely needs multiple signals (co-evolution, co-expression, etc.).

**Key references:** ted_dataset (for CATH context), merizo_search

---

## 2.7 Evaluation Framework (~1.5 pages)

**Purpose:** Introduce the Zhang et al. benchmark and the weighted precision–recall framework so the reader understands how you'll measure success.

**Content plan:**

- **The Zhang et al. (Science, 2025) benchmark:**
  - Most recent and comprehensive benchmark for human PPI prediction.
  - **Positive set:** 3,000 high-confidence PPI pairs, drawn from the intersection of three independent databases (STRING, BioGRID, UniProt). Requiring agreement across all three reduces false positives.
  - **Negative set:** 30,000 random protein pairs. These are assumed to be non-interacting (at a 1:10 ratio to positives in the benchmark set).
  - **Why this benchmark?** It's recent, uses stringent positive selection, and was designed specifically for evaluating proteome-scale PPI predictors.

- **Why standard precision–recall is insufficient:**
  - The benchmark has a 1:10 positive-to-negative ratio, but in reality, proteome-wide screening has approximately a 1:1,000 ratio (most protein pairs don't interact).
  - Standard precision–recall on the benchmark would overestimate real-world precision because the benchmark is artificially enriched for positives.
  - **Example:** A method with 50% precision on the benchmark might only have ~1% precision in a real proteome screen.

- **Weighted precision–recall framework:**
  - Solution: Re-weight the negatives to simulate the real 1:1,000 signal-to-noise ratio.
  - Positive weight wp = 0.01. This means each positive is weighted as if there are 100× more negatives than actually present in the benchmark.
  - The weighted precision at a given threshold reflects what you'd see in real-world proteome screening.
  - Metric: AUCPR (area under the weighted precision–recall curve).

- **Metrics used in this project:**
  - AUCPR (primary metric): Overall ranking quality.
  - Precision at recall = 0.1: How precise is the method when retrieving the top 10% of true interactions?
  - Precision at recall = 0.2 and 0.5: Broader coverage trade-offs.
  - TM-score threshold sweep (0.3 to 0.9): How does the structural similarity cutoff affect precision–recall?

- **Bridge to Chapter 3:** "With the background established, Chapter 3 describes the design and implementation of the PPI prediction pipeline."

**Key references:** zhang2025ppi

---

## Summary: Page estimates and key points per section

| Section | Est. pages | Key point for markers |
|---------|-----------|----------------------|
| 2.1 PPIs | 1.5 | Establishes biological motivation, quantifies the problem |
| 2.2 Computational PPI | 2 | Shows you understand the landscape and where your method fits |
| 2.3 AlphaFold2 | 1.5 | Demonstrates the enabling shift that makes your project timely |
| 2.4 TED/UniProt | 1.5 | Explains your primary data source in detail |
| 2.5 Merizo-search | 2 | Technical depth on the tools underpinning your pipeline |
| 2.6 Structural homology transfer | 1.5 | The core hypothesis — established for function, novel for PPI |
| 2.7 Evaluation framework | 1.5 | Shows critical thinking about metrics (weighted PR is non-trivial) |
| **Total** | **~11.5** | **Target: 10–12 pages** |

## What markers are looking for (from the marking form)

- **"Good evidence of extra-curricular academic reading"** → Sections 2.2 and 2.7 show you've read beyond course materials (co-evolution methods, weighted PR, Zhang et al. 2025 in Science).
- **"Critical thought and original interpretation"** → Section 2.6 identifies the gap (structural homology transfer used for function annotation but not PPI) and Section 2.7 explains *why* standard metrics are misleading.
- **"Clearly at Master's level"** → The technical depth in 2.5 (EGNN embeddings, IPA, cosine similarity + TM-align pipeline) demonstrates graduate-level understanding.
