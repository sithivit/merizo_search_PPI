# Merizo-Search PPI System Guide

**Complete System Architecture for Protein-Protein Interaction Prediction**

---

## Table of Contents

1. [System Overview](#system-overview)
2. [DDI vs PPI: Important Distinction](#ddi-vs-ppi-important-distinction)
3. [CLI Modes Reference](#cli-modes-reference)
4. [Component 1: Merizo (Domain Segmentation)](#component-1-merizo-domain-segmentation)
5. [Component 2: Foldclass (Structural Embeddings)](#component-2-foldclass-structural-embeddings)
6. [Component 3: Rosetta Stone (Fusion Database)](#component-3-rosetta-stone-fusion-database)
7. [Component 4: Promiscuity Filter](#component-4-promiscuity-filter)
8. [Complete Pipeline Flow](#complete-pipeline-flow)
9. [Data Flow & Storage](#data-flow--storage)
10. [GPU Memory Management](#gpu-memory-management)
11. [Usage Examples](#usage-examples)

---

## System Overview

The Merizo-Search PPI system predicts protein-protein interactions using the **Rosetta Stone method** with structural information.

### Core Concept: Rosetta Stone Method

```
If domains A and B are fused together in some organism (Rosetta Stone),
they likely interact in organisms where they appear as separate proteins.

Example:
  Organism 1:  [Domain A]----linker----[Domain B]  (fusion protein)
               ↓ Evidence suggests interaction ↓
  Organism 2:  [Domain A] ←→ [Domain B]  (separate proteins interact)
```

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MERIZO-SEARCH PPI SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│  │   MERIZO     │─────▶│  FOLDCLASS   │─────▶│  ROSETTA     │          │
│  │  Segments    │      │  Embeds      │      │  STONE       │          │
│  │  Domains     │      │  Structures  │      │  Predicts    │          │
│  └──────────────┘      └──────────────┘      │  PPIs        │          │
│        │                      │               └──────────────┘          │
│        │                      │                      │                  │
│        ▼                      ▼                      ▼                  │
│  Domain Boundaries    128-dim Vectors      Interaction List            │
│                                                      │                  │
│                                                      ▼                  │
│                                            ┌──────────────┐            │
│                                            │ PROMISCUITY  │            │
│                                            │   FILTER     │            │
│                                            │  (Optional)  │            │
│                                            └──────────────┘            │
│                                                      │                  │
│                                                      ▼                  │
│                                          High-Confidence PPIs           │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## DDI vs PPI: Important Distinction

### What This System Actually Predicts

**This system predicts Domain-Domain Interactions (DDI), which then infer Protein-Protein Interactions (PPI).**

```
┌─────────────────────────────────────────────────────────────────────┐
│  DOMAIN-DOMAIN INTERACTIONS → PROTEIN-PROTEIN INTERACTIONS          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  MECHANISM: Domain-Domain Interaction (DDI)                          │
│  ┌──────────────────────────────────────────────────────┐           │
│  │  Query Protein X:    [Domain A]                      │           │
│  │  Target Protein Y:   [Domain B]                      │           │
│  │                                                       │           │
│  │  Rosetta Stone Evidence:                             │           │
│  │  Fusion Protein Z:   [Domain A']--[Domain B']        │           │
│  │                           ↓           ↓               │           │
│  │                    similar to    similar to          │           │
│  │                           ↓           ↓               │           │
│  │                      Domain A     Domain B           │           │
│  │                                                       │           │
│  │  PREDICTION: Domain A ←→ Domain B                    │           │
│  │             (specific binding interface!)            │           │
│  └──────────────────────────────────────────────────────┘           │
│                          │                                           │
│                          ▼                                           │
│  BIOLOGICAL OUTCOME: Protein-Protein Interaction (PPI)               │
│  ┌──────────────────────────────────────────────────────┐           │
│  │  Protein X ←→ Protein Y                              │           │
│  │                                                       │           │
│  │  They interact via their domains:                    │           │
│  │  - Protein X provides Domain A                       │           │
│  │  - Protein Y provides Domain B                       │           │
│  │  - Binding occurs at the A-B interface               │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Why DDI Prediction is Better Than Traditional PPI Prediction

```
Traditional PPI Prediction:
  Output: "Protein X interacts with Protein Y"
  Problem: Doesn't tell you WHERE or HOW they interact

Domain-Domain Interaction (DDI) Prediction:
  Output: "Protein X (domain A, residues 50-150) interacts with
           Protein Y (domain B, residues 200-350)"
  Benefits:
    ✓ Identifies specific binding interfaces
    ✓ Explains mechanism of interaction
    ✓ Enables structure-based drug design
    ✓ Distinguishes different interaction modes
    ✓ More specific and actionable
```

### Example

```
Query: Human Kinase Protein (500 residues)
  ├─ Domain 1: Kinase domain (residues 50-300)
  └─ Domain 2: SH2 domain (residues 350-450)

DDI Predictions:
  1. Kinase domain (50-300) ←→ Substrate Protein Domain X
     → Prediction: Kinase phosphorylates Substrate
     → Confidence: 0.92
     → Evidence: 8 Rosetta Stones

  2. SH2 domain (350-450) ←→ Adaptor Protein Domain Y
     → Prediction: Kinase recruited to signaling complex
     → Confidence: 0.85
     → Evidence: 12 Rosetta Stones

PPI Inference:
  - Kinase protein interacts with Substrate protein (via kinase domain)
  - Kinase protein interacts with Adaptor protein (via SH2 domain)
  - Two different interaction modes for the same protein!
```

### Terminology Used in This System

- **DDI (Domain-Domain Interaction)**: The fundamental prediction unit
  - What the system directly predicts
  - Specific structural interfaces between domains

- **PPI (Protein-Protein Interaction)**: The biological outcome
  - Inferred from DDI predictions
  - Multiple DDIs can contribute to one PPI
  - One protein can have multiple PPIs via different domains

- **Rosetta Stone**: The evidence for DDI/PPI
  - Fusion proteins where domains appear together
  - Evolutionary evidence of functional relationship

---

## CLI Modes Reference

The Merizo-Search system provides 5 main modes:

```
┌─────────────────────────────────────────────────────────────────────┐
│  MERIZO-SEARCH CLI MODES                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. segment       - Domain segmentation only                         │
│  2. createdb      - Build Foldclass domain database                  │
│  3. search        - Search domains against Foldclass database        │
│  4. easy-search   - Segment + search in one command                  │
│  5. rosetta       - Rosetta Stone PPI prediction (DDI-based)         │
│                                                                       │
│  Usage: python merizo_search/merizo.py <mode> [args]                │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Mode 1: segment

**Purpose**: Segment multi-domain proteins into individual domains

```bash
# Basic segmentation
python merizo_search/merizo.py segment \
    input.pdb \
    output_prefix \
    -d cuda

# Multiple inputs with options
python merizo_search/merizo.py segment \
    input1.pdb input2.pdb \
    output_prefix \
    -d cuda \
    --save_domains \
    --save_pdf \
    --iterate \
    --output_headers

# Advanced: filter and save high-confidence domains
python merizo_search/merizo.py segment \
    input.pdb \
    output \
    -d cuda \
    --conf_filter 0.7 \
    --plddt_filter 70 \
    --save_domains
```

**Key Options**:
- `--save_domains`: Save each domain as separate PDB file
- `--save_pdf`: Generate domain boundary visualization
- `--iterate`: Iterative refinement for complex proteins
- `--conf_filter`: Only save domains above confidence threshold
- `--plddt_filter`: Only save domains above pLDDT threshold
- `--min_domain_size`: Minimum residues per domain (default: 50)

**Output**:
- `{output}_segment.tsv`: Segmentation results table
- `{output}_domain_0.pdb`, `{output}_domain_1.pdb`, ...: Individual domains (if `--save_domains`)
- `{output}_domains.pdf`: Visualization (if `--save_pdf`)

**Use Case**:
- Analyze protein domain architecture
- Extract domains for further analysis
- Visualize domain organization

---

### Mode 2: createdb

**Purpose**: Build Foldclass embedding database from domain structures

```bash
# Build from directory of PDB files
python merizo_search/merizo.py createdb \
    /path/to/pdb_directory/ \
    output_database_prefix \
    -d cuda

# Build from tar/zip archive
python merizo_search/merizo.py createdb \
    structures.tar.gz \
    output_database \
    -d cuda \
    --max-length 2000

# Specify temp directory
python merizo_search/merizo.py createdb \
    structures.zip \
    output_db \
    -d cuda \
    -t /tmp/merizo_temp
```

**Key Options**:
- `--max-length`: Maximum residues per protein (default: 2000)
- `-t, --tmpdir`: Temporary directory for extraction

**Output**:
- `{output_database}.pt`: PyTorch tensor of embeddings
- `{output_database}.index`: Metadata index file

**Database Format**:
```python
{
  'embeddings': torch.Tensor([N, 128]),  # N domain embeddings
  'names': List[str],                     # Domain IDs
  'metadata': List[dict]                  # Additional info
}
```

**Use Case**:
- Create searchable database of known domains
- Build custom domain libraries
- Prepare reference database for similarity search

---

### Mode 3: search

**Purpose**: Search query domains against Foldclass database (structural similarity)

```bash
# Basic search
python merizo_search/merizo.py search \
    query.pdb \
    database_prefix \
    output_prefix \
    tmp_dir \
    -d cuda

# Advanced search with validation
python merizo_search/merizo.py search \
    query.pdb \
    database_prefix \
    output \
    tmp \
    -d cuda \
    -k 10 \
    --mincos 0.6 \
    --mintm 0.5 \
    --mincov 0.7 \
    --output_headers

# Fast mode (skip TM-align)
python merizo_search/merizo.py search \
    query.pdb \
    database \
    output \
    tmp \
    -d cuda \
    --fastmode
```

**Key Options**:
- `-k, --topk`: Number of matches to return per domain (default: 1)
- `-s, --mincos`: Minimum cosine similarity threshold (default: 0.5)
- `-m, --mintm`: Minimum TM-align score threshold (default: 0.5)
- `-c, --mincov`: Minimum coverage threshold (default: 0.7)
- `-f, --fastmode`: Use fast TM-align (less accurate, faster)
- `--format`: Customize output columns

**Output**:
- `{output}_search.tsv`: Search results
- `{output}_search_insignificant.tsv`: Below-threshold hits (if `--report_insignificant_hits`)

**Result Columns**:
```
query, target, emb_rank, emb_score, q_len, t_len,
ali_len, seq_id, q_tm, t_tm, max_tm, rmsd, metadata
```

**Use Case**:
- Find structurally similar domains
- Identify domain families
- Structural annotation of unknown domains

---

### Mode 4: easy-search

**Purpose**: Combined segmentation + search pipeline

```bash
# One-step domain search
python merizo_search/merizo.py easy-search \
    query.pdb \
    database_prefix \
    output_prefix \
    tmp_dir \
    -d cuda

# Multi-domain protein with full output
python merizo_search/merizo.py easy-search \
    multi_domain_protein.pdb \
    database \
    output \
    tmp \
    -d cuda \
    -k 5 \
    --output_headers \
    --save_domains \
    --save_pdf

# Multi-domain interaction search
python merizo_search/merizo.py easy-search \
    query.pdb \
    database \
    output \
    tmp \
    -d cuda \
    --multi_domain_search \
    --multi_domain_mode exhaustive_tmalign
```

**Key Options**:
- All `segment` mode options (domain extraction)
- All `search` mode options (database search)
- `--multi_domain_search`: Find proteins matching ALL query domains
- `--multi_domain_mode`: Algorithm for multi-domain matching

**Output**:
- `{output}_segment.tsv`: Segmentation results
- `{output}_search.tsv`: Search results for each domain
- `{output}_search_multi_dom.tsv`: Full-length matches (if `--multi_domain_search`)

**Use Case**:
- Quick domain analysis + homology search
- Identify proteins with similar domain architecture
- Find fusion partners

---

### Mode 5: rosetta (Rosetta Stone PPI Prediction)

**Purpose**: Predict protein-protein interactions using domain fusion analysis

This mode has two subcommands: `build` and `search`

#### 5a. rosetta build (Build Fusion Database)

```bash
# Basic fusion database build
python merizo_search/merizo.py rosetta build \
    /path/to/structures/ \
    fusion_db_output/ \
    -d cuda

# Optimized for 6GB GPU
python merizo_search/merizo.py rosetta build \
    structures/ \
    fusion_db/ \
    -d cuda \
    --batch-size 4 \
    --max-protein-size 1800

# Skip promiscuity filtering (faster)
python merizo_search/merizo.py rosetta build \
    structures/ \
    fusion_db/ \
    -d cuda \
    --skip-promiscuity

# Custom parameters
python merizo_search/merizo.py rosetta build \
    structures/ \
    fusion_db/ \
    -d cuda \
    --min-domains 2 \
    --max-protein-size 2000 \
    --batch-size 2 \
    --promiscuity-threshold 30
```

**Key Options**:
- `--min-domains`: Minimum domains per protein (default: 2)
- `--max-protein-size`: Max residues to prevent OOM (default: 1800)
- `--batch-size`: Embedding batch size (default: 4, reduce to 2 if OOM)
- `--skip-promiscuity`: Skip promiscuity index building
- `--promiscuity-threshold`: Partner count threshold (default: 25)
- `-d, --device`: Device (cpu, cuda, mps)

**Output Files**:
```
fusion_db/
├── domain_embeddings.pt       # All domain embeddings [N, 128]
├── domain_metadata.index      # Domain information
├── fusion_metadata.index      # Fusion link information
└── promiscuity_index.pkl      # Promiscuity filter (if not skipped)
```

**Processing Flow**:
1. Segment each protein (skip single-domain proteins)
2. Embed each domain (128-dim vector)
3. Find fusion links (pairs of domains in same protein)
4. Store embeddings and metadata
5. Build promiscuity index (cluster domains, count partners)

**Performance** (RTX 3060 6GB):
- Throughput: 2-5 proteins/min
- Memory: < 3 GB (stable)
- 23,586 proteins: ~10 hours

#### 5b. rosetta search (Search for PPIs)

```bash
# Basic PPI search
python merizo_search/merizo.py rosetta search \
    query.pdb \
    fusion_db/ \
    output_prefix \
    -d cuda

# With structural validation
python merizo_search/merizo.py rosetta search \
    query.pdb \
    fusion_db/ \
    output \
    -d cuda \
    --validate-tm \
    --min-tm-score 0.5 \
    --fastmode

# Adjust sensitivity
python merizo_search/merizo.py rosetta search \
    query.pdb \
    fusion_db/ \
    output \
    -d cuda \
    --cosine-threshold 0.6 \
    --top-k 50 \
    --output-headers

# Skip promiscuity filter
python merizo_search/merizo.py rosetta search \
    query.pdb \
    fusion_db/ \
    output \
    -d cuda \
    --skip-filter
```

**Key Options**:
- `--cosine-threshold`: Similarity threshold (default: 0.7)
- `--top-k`: Number of top matches to consider (default: 20)
- `--validate-tm`: Run TM-align validation
- `--min-tm-score`: Minimum TM-score for validation (default: 0.5)
- `--fastmode`: Fast TM-align mode
- `--skip-filter`: Skip promiscuity filtering
- `--output-headers`: Include headers in output

**Output**:
- `{output}_rosetta.json`: Interaction predictions (JSON format)
- Console: Top 10 predictions summary

**Output Format**:
```json
{
  "query": "query.pdb",
  "num_predictions": 15,
  "predictions": [
    {
      "query_domain_id": "query_domain_0",
      "query_protein": "query",
      "query_range": [1, 120],
      "target_domain_id": "AF-P12345_domain_1",
      "target_protein": "AF-P12345",
      "target_range": [135, 280],
      "interaction_type": "inter",
      "confidence_score": 0.89,
      "cosine_similarity": 0.92,
      "tm_score": 0.72,
      "num_rosetta_stones": 5,
      "rosetta_stone_evidence": [
        {"rosetta_stone_id": "AF-Q98765", "linker_length": 15},
        {"rosetta_stone_id": "AF-P54321", "linker_length": 8}
      ],
      "promiscuity_filtered": false
    }
  ]
}
```

**Confidence Score Formula**:
```
confidence = (
  0.4 × cosine_similarity +      # Structural similarity
  0.3 × num_rosetta_stones/10 +  # Evidence count
  0.2 × (1 - promiscuity) +     # Domain specificity
  0.1 × min_conf                 # Merizo confidence
)
```

**Use Case**:
- Predict protein interaction partners
- Identify domain-domain interfaces
- Discover protein complexes
- Infer signaling pathways

---

### Mode Comparison Table

| Mode | Input | Output | Speed | Use Case |
|------|-------|--------|-------|----------|
| `segment` | PDB file | Domain boundaries | Fast (5-10s) | Domain architecture analysis |
| `createdb` | PDB directory | Embedding database | Medium (varies) | Build search database |
| `search` | PDB + database | Similar domains | Medium (30s) | Find structural homologs |
| `easy-search` | PDB + database | Segment + search | Medium (30s) | One-step homology search |
| `rosetta build` | PDB directory | Fusion database | Slow (hours) | Build PPI database (once) |
| `rosetta search` | PDB + fusion DB | PPI predictions | Fast (30s) | Predict interactions |

### Recommended Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  TYPICAL WORKFLOW                                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  SETUP (One-time):                                                   │
│  1. rosetta build  → Create fusion database from proteome           │
│                                                                       │
│  ANALYSIS (Per query):                                               │
│  2. segment        → Examine domain architecture (optional)          │
│  3. rosetta search → Predict interaction partners                    │
│                                                                       │
│  ALTERNATIVE:                                                        │
│  - Use easy-search for quick domain homology search                  │
│  - Use createdb + search for custom domain databases                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Merizo (Domain Segmentation)

### Purpose
**Segments multi-domain proteins into individual structural domains**

### How It Works

```
INPUT: Protein Structure (PDB/CIF file)
  │
  ▼
┌────────────────────────────────────────────────────┐
│  MERIZO NEURAL NETWORK                             │
│  - EGNN (E(3)-Equivariant Graph Network)          │
│  - Iterative segmentation (max 3 iterations)       │
│  - Per-residue domain assignment                   │
└────────────────────────────────────────────────────┘
  │
  ▼
OUTPUT: Domain Boundaries
  - domain_ids tensor: [N_residues] (0=no domain, 1=domain1, 2=domain2...)
  - conf_res tensor: [N_residues] (confidence per residue)
  - Residue indices and coordinates
```

### Detailed Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  MERIZO SEGMENTATION PIPELINE                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. LOAD PROTEIN STRUCTURE                                       │
│     ┌─────────────────┐                                          │
│     │ PDB/CIF File    │                                          │
│     │ AF-Q14686.pdb   │                                          │
│     └────────┬────────┘                                          │
│              │                                                    │
│              ▼                                                    │
│  2. EXTRACT FEATURES                                             │
│     ┌─────────────────┐                                          │
│     │ CA Coords       │  [N, 3] array                            │
│     │ Sequence        │  "MKKL..."                               │
│     │ Residue Indices │  [1,2,3,...,N]                           │
│     └────────┬────────┘                                          │
│              │                                                    │
│              ▼                                                    │
│  3. MERIZO FORWARD PASS (GPU)                                    │
│     ┌─────────────────────────────────────┐                      │
│     │  Graph Construction:                │                      │
│     │  - Nodes = residues                 │                      │
│     │  - Edges = spatial proximity        │                      │
│     │                                      │                      │
│     │  EGNN Layers (x2):                  │                      │
│     │  - Edge features: distances         │                      │
│     │  - Node features: positions         │                      │
│     │  - Attention mechanism              │                      │
│     │                                      │                      │
│     │  Iterative Refinement:              │                      │
│     │  Iteration 1 → boundaries           │                      │
│     │  Iteration 2 → refine boundaries    │                      │
│     │  Iteration 3 → final boundaries     │                      │
│     └────────┬────────────────────────────┘                      │
│              │                                                    │
│              ▼                                                    │
│  4. OUTPUT DOMAIN ASSIGNMENTS                                    │
│     ┌─────────────────────────────────────┐                      │
│     │ domain_ids: [0,0,1,1,1,...,2,2,2]  │  Per-residue labels │
│     │ conf_res:   [0.9,0.8,0.95,...]     │  Confidence scores  │
│     └────────┬────────────────────────────┘                      │
│              │                                                    │
│              ▼                                                    │
│  5. EXTRACT DOMAINS                                              │
│     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│     │  Domain 0    │  │  Domain 1    │  │  Domain 2    │       │
│     │  Res 1-120   │  │  Res 135-280 │  │  Res 295-450 │       │
│     │  Coords: ... │  │  Coords: ... │  │  Coords: ... │       │
│     │  Seq: "MKK"  │  │  Seq: "AST"  │  │  Seq: "VLD"  │       │
│     └──────────────┘  └──────────────┘  └──────────────┘       │
│              │                 │                 │               │
│              └─────────────────┴─────────────────┘               │
│                                │                                 │
│                                ▼                                 │
│                          List[Domain]                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

MEMORY CLEANUP POINTS (CRITICAL):
→ Before segmentation: torch.cuda.empty_cache() + gc.collect()
→ After extraction: del features + torch.cuda.synchronize()
→ Convert to numpy immediately: domain_ids_np = domain_ids_tensor.numpy()
```

### Key Parameters

```python
segment(
    pdb_path=str(pdb_path),
    network=self.merizo,              # Pre-loaded Merizo network
    device='cuda',                     # GPU acceleration
    iterate=True,                      # Iterative refinement
    max_iterations=3,                  # Number of refinement steps
    min_domain_size=50,                # Minimum 50 residues per domain
    min_fragment_size=10,              # Minimum fragment size
    domain_ave_size=200,               # Expected domain size
    conf_threshold=0.5,                # Confidence threshold
    pdb_chain='A'                      # Chain to process
)
```

---

## Component 2: Foldclass (Structural Embeddings)

### Purpose
**Converts 3D protein structure into fixed-size vector representation**

### How It Works

```
INPUT: Domain CA Coordinates [N_residues, 3]
  │
  ▼
┌────────────────────────────────────────────────────┐
│  FOLDCLASS NEURAL NETWORK                          │
│  - Positional Encoding (sinusoidal)                │
│  - EGNN Layers (x2) for structure processing       │
│  - Global Average Pooling                          │
└────────────────────────────────────────────────────┘
  │
  ▼
OUTPUT: Embedding Vector [128]
  - Fixed-size representation
  - Captures structural features
  - Used for similarity search
```

### Detailed Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  FOLDCLASS EMBEDDING PIPELINE                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. INPUT PREPARATION                                            │
│     ┌─────────────────────────────────────┐                      │
│     │  Domain CA Coordinates              │                      │
│     │  Shape: [n_residues, 3]            │                      │
│     │  Example: [[1.2, 3.4, 5.6],        │                      │
│     │            [2.3, 4.5, 6.7], ...]   │                      │
│     └────────┬────────────────────────────┘                      │
│              │                                                    │
│              ▼                                                    │
│  2. ADD BATCH DIMENSION                                          │
│     ┌─────────────────────────────────────┐                      │
│     │  coords_tensor.unsqueeze(0)         │                      │
│     │  Shape: [1, n_residues, 3]         │                      │
│     └────────┬────────────────────────────┘                      │
│              │                                                    │
│              ▼                                                    │
│  3. POSITIONAL ENCODING (GPU)                                    │
│     ┌─────────────────────────────────────┐                      │
│     │  Sinusoidal Position Encoding:      │                      │
│     │                                      │                      │
│     │  For each residue position i:        │                      │
│     │    PE(i, 2j)   = sin(i/10000^(2j/d))│                      │
│     │    PE(i, 2j+1) = cos(i/10000^(2j/d))│                      │
│     │                                      │                      │
│     │  Output: [1, n_residues, 128]       │                      │
│     └────────┬────────────────────────────┘                      │
│              │                                                    │
│              ▼                                                    │
│  4. EGNN LAYER 1                                                 │
│     ┌─────────────────────────────────────┐                      │
│     │  Edge Construction:                 │                      │
│     │  - Compute pairwise distances       │                      │
│     │  - Edge features: [feat_i, feat_j,  │                      │
│     │                     dist_ij^2]      │                      │
│     │                                      │                      │
│     │  Edge MLP:                          │                      │
│     │  - Linear(edge_dim → edge_dim*2)    │                      │
│     │  - SiLU activation                  │                      │
│     │  - Linear(edge_dim*2 → m_dim)       │                      │
│     │  - Gated attention                  │                      │
│     │                                      │                      │
│     │  Node Update:                       │                      │
│     │  - Aggregate edge messages          │                      │
│     │  - Node MLP + residual connection   │                      │
│     └────────┬────────────────────────────┘                      │
│              │                                                    │
│              ▼                                                    │
│  5. EGNN LAYER 2                                                 │
│     ┌─────────────────────────────────────┐                      │
│     │  (Same architecture as Layer 1)     │                      │
│     │  Further refinement of features     │                      │
│     └────────┬────────────────────────────┘                      │
│              │                                                    │
│              ▼                                                    │
│  6. GLOBAL AVERAGE POOLING                                       │
│     ┌─────────────────────────────────────┐                      │
│     │  out_feats.mean(dim=1)              │                      │
│     │                                      │                      │
│     │  Input:  [1, n_residues, 128]       │                      │
│     │  Output: [1, 128]                   │                      │
│     └────────┬────────────────────────────┘                      │
│              │                                                    │
│              ▼                                                    │
│  7. OUTPUT EMBEDDING                                             │
│     ┌─────────────────────────────────────┐                      │
│     │  Embedding vector: [128]            │                      │
│     │  - Fixed size regardless of length  │                      │
│     │  - Captures structural signature    │                      │
│     │  - Normalized (unit length)         │                      │
│     └─────────────────────────────────────┘                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

MEMORY CLEANUP POINTS:
→ After embedding: embedding_np = embedding.squeeze(0).cpu().numpy()
→ Delete tensors: del embedding, coords_tensor
→ Clear cache: torch.cuda.empty_cache()
```

### Key Properties

```python
# Embedding Characteristics:
- Dimension: 128 (fixed)
- Type: float32
- Range: Normalized (typically -1 to 1)
- Invariances: Translation, rotation (E(3) equivariant)
- Similarity metric: Cosine similarity

# Example similarity:
domain_A_emb = [0.12, -0.45, 0.78, ...]  # [128]
domain_B_emb = [0.10, -0.42, 0.81, ...]  # [128]
cosine_sim = np.dot(domain_A_emb, domain_B_emb)  # 0.95 (very similar)
```

---

## Component 3: Rosetta Stone (Fusion Database)

### Purpose
**Build searchable database of domain fusions and predict interactions**

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ROSETTA STONE SYSTEM                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  PHASE 1: DATABASE BUILDING (Preprocessing)             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Structure Database (AlphaFold, PDB)                             │
│         │                                                         │
│         ├─▶ Protein 1: [Domain A]----[Domain B]                 │
│         ├─▶ Protein 2: [Domain C]----[Domain D]----[Domain E]   │
│         └─▶ Protein N: [Domain X]----[Domain Y]                 │
│                       │                                           │
│                       ▼                                           │
│         ┌──────────────────────────────┐                         │
│         │  FUSION DATABASE BUILDER     │                         │
│         │  1. Segment each protein     │                         │
│         │  2. Embed each domain        │                         │
│         │  3. Find fusion pairs        │                         │
│         │  4. Store in database        │                         │
│         └──────────┬───────────────────┘                         │
│                    │                                              │
│                    ▼                                              │
│         ┌──────────────────────────────┐                         │
│         │  FUSION DATABASE FILES       │                         │
│         │  ├─ domain_embeddings.pt     │  All domain vectors    │
│         │  ├─ domain_metadata.index    │  Domain info           │
│         │  ├─ fusion_embeddings.pt     │  Fusion pair vectors   │
│         │  └─ fusion_metadata.index    │  Fusion link info      │
│         └──────────────────────────────┘                         │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  PHASE 2: QUERY SEARCH (Fast)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Query Protein                                                   │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────┐                                            │
│  │ Segment domains  │                                            │
│  └─────┬────────────┘                                            │
│        │                                                          │
│        ▼                                                          │
│  ┌──────────────────┐                                            │
│  │ Embed domains    │                                            │
│  └─────┬────────────┘                                            │
│        │                                                          │
│        ▼                                                          │
│  ┌──────────────────────────────────┐                            │
│  │ FAISS Similarity Search          │                            │
│  │ - Find similar domains in DB     │                            │
│  │ - Check for Rosetta Stone links  │                            │
│  │ - Rank by confidence             │                            │
│  └─────┬────────────────────────────┘                            │
│        │                                                          │
│        ▼                                                          │
│  ┌──────────────────────────────────┐                            │
│  │ INTERACTION PREDICTIONS          │                            │
│  │ - Query domain A ←→ Target X     │                            │
│  │ - Query domain B ←→ Target Y     │                            │
│  └──────────────────────────────────┘                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Database Building Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  FUSION DATABASE BUILDER - DETAILED FLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  FOR EACH PROTEIN IN DATABASE:                                   │
│                                                                   │
│  1. SEGMENT PROTEIN                                              │
│     ┌────────────────────────────────┐                           │
│     │ Merizo segmentation            │                           │
│     │ Input:  AF-Q14686.pdb          │                           │
│     │ Output: [Domain_0, Domain_1]   │                           │
│     └────────┬───────────────────────┘                           │
│              │                                                    │
│              ▼                                                    │
│  2. CHECK DOMAIN COUNT                                           │
│     ┌────────────────────────────────┐                           │
│     │ if len(domains) < 2:           │                           │
│     │    skip (not multi-domain)     │                           │
│     │ else:                          │                           │
│     │    continue processing         │                           │
│     └────────┬───────────────────────┘                           │
│              │                                                    │
│              ▼                                                    │
│  3. EMBED EACH DOMAIN                                            │
│     ┌────────────────────────────────┐                           │
│     │ For domain in domains:         │                           │
│     │   - Foldclass embedding        │                           │
│     │   - Store embedding [128]      │                           │
│     │   - Add to batch               │                           │
│     └────────┬───────────────────────┘                           │
│              │                                                    │
│              ▼                                                    │
│  4. FIND FUSION LINKS                                            │
│     ┌────────────────────────────────────────────┐               │
│     │ For each pair (domain_i, domain_j):        │               │
│     │                                             │               │
│     │   Check:                                    │               │
│     │   ✓ No overlap                             │               │
│     │   ✓ Linker length in range [0, 100]        │               │
│     │                                             │               │
│     │   Create FusionLink:                       │               │
│     │   - domain_A = domain_i                    │               │
│     │   - domain_B = domain_j                    │               │
│     │   - linker_length = residues between       │               │
│     │   - fusion_embedding = [emb_A | emb_B]     │  [256] concat│
│     └────────┬──────────────────────────────────┘               │
│              │                                                    │
│              ▼                                                    │
│  5. STORE TO DATABASE                                            │
│     ┌────────────────────────────────────────────┐               │
│     │ Accumulate in RAM (not GPU!):              │               │
│     │ - all_embeddings.append(emb_np)            │  numpy array │
│     │ - domain_metadata_list.append(...)         │  metadata    │
│     │ - fusion_metadata_list.append(...)         │  fusion info │
│     └────────┬──────────────────────────────────┘               │
│              │                                                    │
│              ▼                                                    │
│  6. PERIODIC CHECKPOINT (every 50 proteins)                      │
│     ┌────────────────────────────────────────────┐               │
│     │ Convert numpy → tensor                     │               │
│     │ Save to disk:                              │               │
│     │   - domain_embeddings.pt                   │               │
│     │   - domain_metadata.index                  │               │
│     │   - fusion_metadata.index                  │               │
│     │ Clear GPU cache                            │               │
│     └────────────────────────────────────────────┘               │
│                                                                   │
│  AFTER ALL PROTEINS PROCESSED:                                   │
│                                                                   │
│  7. FINAL SAVE                                                   │
│     ┌────────────────────────────────────────────┐               │
│     │ Convert all_embeddings → tensor            │               │
│     │ torch.save(embeddings, ...)                │               │
│     │ pickle.dump(metadata, ...)                 │               │
│     └────────────────────────────────────────────┘               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

MEMORY MANAGEMENT (CRITICAL):
┌───────────────────────────────────────────────────────────────┐
│ • Store embeddings as NUMPY in RAM (not torch tensors on GPU) │
│ • Convert to tensor only at checkpoint/save time               │
│ • Clear GPU after each protein                                 │
│ • Checkpoint every 50 proteins to prevent data loss            │
│ • Process domains individually (no batching) to avoid OOM      │
└───────────────────────────────────────────────────────────────┘
```

### Search Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│  ROSETTA STONE SEARCH - DETAILED FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. LOAD FUSION DATABASE                                         │
│     ┌────────────────────────────────────────┐                   │
│     │ Load from disk:                        │                   │
│     │ - domain_embeddings.pt   [N, 128]     │                   │
│     │ - fusion_embeddings.pt   [M, 256]     │                   │
│     │ - metadata files                       │                   │
│     └────────┬───────────────────────────────┘                   │
│              │                                                    │
│              ▼                                                    │
│  2. BUILD FAISS INDEX                                            │
│     ┌────────────────────────────────────────┐                   │
│     │ Create similarity search index:        │                   │
│     │ - Normalize embeddings (L2)            │                   │
│     │ - IndexFlatIP (inner product)          │                   │
│     │ - Optional: GPU acceleration           │                   │
│     └────────┬───────────────────────────────┘                   │
│              │                                                    │
│              ▼                                                    │
│  3. PROCESS QUERY PROTEIN                                        │
│     ┌────────────────────────────────────────┐                   │
│     │ Query: protein_X.pdb                   │                   │
│     │   ↓ Merizo                             │                   │
│     │ [Query_Domain_A, Query_Domain_B]       │                   │
│     │   ↓ Foldclass                          │                   │
│     │ [Embedding_A[128], Embedding_B[128]]   │                   │
│     └────────┬───────────────────────────────┘                   │
│              │                                                    │
│              ▼                                                    │
│  4. SEARCH FOR EACH QUERY DOMAIN                                 │
│     ┌────────────────────────────────────────────────────────┐   │
│     │ For Query_Domain_A:                                     │   │
│     │                                                          │   │
│     │   Search FAISS index:                                   │   │
│     │   ┌─────────────────────────────────────┐              │   │
│     │   │ Query: Embedding_A                  │              │   │
│     │   │ Return: Top-K similar domains       │              │   │
│     │   │         (cosine_similarity > 0.7)   │              │   │
│     │   └───────────┬─────────────────────────┘              │   │
│     │               │                                          │   │
│     │               ▼                                          │   │
│     │   ┌─────────────────────────────────────┐              │   │
│     │   │ Match 1: DB_Domain_X (sim: 0.92)   │              │   │
│     │   │ Match 2: DB_Domain_Y (sim: 0.85)   │              │   │
│     │   │ Match 3: DB_Domain_Z (sim: 0.78)   │              │   │
│     │   └───────────┬─────────────────────────┘              │   │
│     │               │                                          │   │
│     │               ▼                                          │   │
│     │   For each match, CHECK FUSION DATABASE:                │   │
│     │   ┌─────────────────────────────────────────┐          │   │
│     │   │ Is DB_Domain_X part of fusion link?     │          │   │
│     │   │ Example: [DB_Domain_X]--[DB_Domain_W]   │          │   │
│     │   │                                           │          │   │
│     │   │ If YES → Rosetta Stone evidence!        │          │   │
│     │   │ Predicted: Query_Domain_A ←→ DB_Domain_W│          │   │
│     │   └─────────────────────────────────────────┘          │   │
│     └────────────────────────────────────────────────────────┘   │
│              │                                                    │
│              ▼                                                    │
│  5. RANK PREDICTIONS                                             │
│     ┌────────────────────────────────────────────────────────┐   │
│     │ Confidence Score Calculation:                           │   │
│     │                                                          │   │
│     │ confidence = (                                          │   │
│     │   0.4 * cosine_similarity +        # Embedding match    │   │
│     │   0.3 * num_rosetta_stones / 10 +  # Evidence count    │   │
│     │   0.2 * (1 - promiscuity) +       # Domain specificity │   │
│     │   0.1 * min_conf                   # Merizo confidence │   │
│     │ )                                                       │   │
│     │                                                          │   │
│     │ Sort by confidence (descending)                         │   │
│     └────────┬───────────────────────────────────────────────┘   │
│              │                                                    │
│              ▼                                                    │
│  6. OPTIONAL: TM-ALIGN VALIDATION                                │
│     ┌────────────────────────────────────────┐                   │
│     │ For top predictions:                   │                   │
│     │ - Run TM-align between domains         │                   │
│     │ - TM-score > 0.5 → structural match    │                   │
│     │ - Update confidence score              │                   │
│     └────────┬───────────────────────────────┘                   │
│              │                                                    │
│              ▼                                                    │
│  7. OUTPUT PREDICTIONS                                           │
│     ┌────────────────────────────────────────────────────────┐   │
│     │ Prediction 1:                                           │   │
│     │   Query: Domain_A (res 1-120)                          │   │
│     │   Target: Protein_Y Domain_3 (res 200-350)             │   │
│     │   Confidence: 0.89                                      │   │
│     │   Evidence: 5 Rosetta Stones                            │   │
│     │   TM-score: 0.72                                        │   │
│     │                                                          │   │
│     │ Prediction 2: ...                                       │   │
│     └────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 4: Promiscuity Filter

### Purpose
**Filter out promiscuous domains that interact with many partners (low specificity)**

### Concept

```
Promiscuous Domain Example:
  - ATP-binding domain
  - Appears in 500+ different proteins
  - Forms fusions with 100+ different domain types
  - Low predictive value (interacts with everything!)

High-Specificity Domain:
  - Kinase-specific regulatory domain
  - Only fuses with 2-3 kinase domain types
  - High predictive value (specific interaction!)
```

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  PROMISCUITY FILTER - DETAILED FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PHASE 1: BUILD PROMISCUITY INDEX (One-time)                     │
│                                                                   │
│  1. LOAD DOMAIN EMBEDDINGS                                       │
│     ┌────────────────────────────────────┐                       │
│     │ domain_embeddings.pt [N, 128]      │                       │
│     └────────┬───────────────────────────┘                       │
│              │                                                    │
│              ▼                                                    │
│  2. CLUSTER DOMAINS (HDBSCAN)                                    │
│     ┌──────────────────────────────────────────────┐             │
│     │ Group structurally similar domains:          │             │
│     │                                               │             │
│     │ Cluster 0: ATP-binding domains (500 domains) │             │
│     │ Cluster 1: SH3 domains (120 domains)         │             │
│     │ Cluster 2: Kinase domains (300 domains)      │             │
│     │ ...                                           │             │
│     │ Cluster N: Novel domains (15 domains)        │             │
│     └────────┬────────────────────────────────────┘             │
│              │                                                    │
│              ▼                                                    │
│  3. ANALYZE FUSION LINKS                                         │
│     ┌──────────────────────────────────────────────────────┐     │
│     │ For each cluster, count linked clusters:             │     │
│     │                                                        │     │
│     │ Cluster 0 (ATP-binding):                             │     │
│     │   ├─ Links to Cluster 1 (5 times)                    │     │
│     │   ├─ Links to Cluster 2 (8 times)                    │     │
│     │   ├─ Links to Cluster 3 (12 times)                   │     │
│     │   ├─ ... (links to 85+ other clusters!)              │     │
│     │   └─ Total unique partners: 87                       │     │
│     │   → PROMISCUOUS (threshold: 25)                      │     │
│     │                                                        │     │
│     │ Cluster 50 (Specific kinase regulator):              │     │
│     │   ├─ Links to Cluster 2 (15 times)                   │     │
│     │   ├─ Links to Cluster 45 (8 times)                   │     │
│     │   └─ Total unique partners: 2                        │     │
│     │   → SPECIFIC (good predictor!)                       │     │
│     └────────┬───────────────────────────────────────────┘     │
│              │                                                    │
│              ▼                                                    │
│  4. CREATE PROMISCUITY INDEX                                     │
│     ┌────────────────────────────────────────┐                   │
│     │ Save to promiscuity_index.pkl:         │                   │
│     │                                         │                   │
│     │ {                                       │                   │
│     │   cluster_0: {                         │                   │
│     │     num_links: 87,                     │                   │
│     │     is_promiscuous: True               │                   │
│     │   },                                    │                   │
│     │   cluster_50: {                        │                   │
│     │     num_links: 2,                      │                   │
│     │     is_promiscuous: False              │                   │
│     │   },                                    │                   │
│     │   ...                                   │                   │
│     │ }                                       │                   │
│     └─────────────────────────────────────────┘                   │
│                                                                   │
│  PHASE 2: FILTER PREDICTIONS (Query time)                        │
│                                                                   │
│  5. LOAD PROMISCUITY INDEX                                       │
│     ┌────────────────────────────────────┐                       │
│     │ Read promiscuity_index.pkl         │                       │
│     └────────┬───────────────────────────┘                       │
│              │                                                    │
│              ▼                                                    │
│  6. CHECK EACH PREDICTION                                        │
│     ┌──────────────────────────────────────────────────────┐     │
│     │ Prediction: Domain_A ←→ Domain_X                     │     │
│     │                                                        │     │
│     │ Check:                                                 │     │
│     │ - Is Domain_A's cluster promiscuous? → NO             │     │
│     │ - Is Domain_X's cluster promiscuous? → YES (ATP)      │     │
│     │                                                        │     │
│     │ Decision: FLAG as LOW CONFIDENCE                      │     │
│     │           (one partner is promiscuous)                │     │
│     └────────┬───────────────────────────────────────────┘     │
│              │                                                    │
│              ▼                                                    │
│  7. OUTPUT FILTERED PREDICTIONS                                  │
│     ┌────────────────────────────────────────────────┐           │
│     │ HIGH CONFIDENCE PREDICTIONS:                   │           │
│     │ - Both domains in specific clusters            │           │
│     │ - Narrow interaction specificity               │           │
│     │                                                 │           │
│     │ REMOVED PREDICTIONS:                           │           │
│     │ - Involves promiscuous domain                  │           │
│     │ - Low predictive value                         │           │
│     └─────────────────────────────────────────────────┘           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

PARAMETERS:
┌────────────────────────────────────────────────────────────────┐
│ • Promiscuity Threshold: 25 unique partner clusters            │
│ • Clustering: HDBSCAN (min_cluster_size=5, min_samples=3)     │
│ • Distance Metric: Cosine similarity on embeddings             │
└────────────────────────────────────────────────────────────────┘
```

---

## Complete Pipeline Flow

### End-to-End System

```
┌═════════════════════════════════════════════════════════════════════════════┐
║                    MERIZO-SEARCH PPI COMPLETE PIPELINE                       ║
╚═════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║  STEP 1: PREPROCESSING (Run once)                                         ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Input: 23,586 AlphaFold structures                                       ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  FOR EACH STRUCTURE:                                      │            ║
║  │                                                            │            ║
║  │  1. Read PDB file                                         │            ║
║  │     ↓                                                      │            ║
║  │  2. MERIZO: Segment into domains                          │            ║
║  │     • GPU forward pass (2-6 GB memory)                    │            ║
║  │     • Extract domain boundaries                           │            ║
║  │     • GPU cleanup (critical!)                             │            ║
║  │     ↓                                                      │            ║
║  │  3. Filter: Keep only multi-domain proteins (≥2 domains)  │            ║
║  │     ↓                                                      │            ║
║  │  4. FOLDCLASS: Embed each domain                          │            ║
║  │     • Process one domain at a time                        │            ║
║  │     • Get 128-dim embedding                               │            ║
║  │     • Store as numpy in RAM (not GPU!)                    │            ║
║  │     • GPU cleanup after each domain                       │            ║
║  │     ↓                                                      │            ║
║  │  5. Find fusion links (pairwise domain combinations)      │            ║
║  │     • Check no overlap                                    │            ║
║  │     • Check linker length                                 │            ║
║  │     • Create fusion embedding [256] = [emb_A | emb_B]     │            ║
║  │     ↓                                                      │            ║
║  │  6. Accumulate in RAM                                     │            ║
║  │     • all_embeddings list (numpy arrays)                  │            ║
║  │     • metadata lists                                      │            ║
║  │     ↓                                                      │            ║
║  │  7. Checkpoint every 50 proteins                          │            ║
║  │     • Convert numpy → tensor                              │            ║
║  │     • Save to disk                                        │            ║
║  │     • Clear GPU thoroughly                                │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  FINAL DATABASE FILES:                                    │            ║
║  │  ├─ domain_embeddings.pt     [~50,000 domains × 128]     │            ║
║  │  ├─ domain_metadata.index    [pickled list]              │            ║
║  │  ├─ fusion_embeddings.pt     [~100,000 fusions × 256]    │            ║
║  │  └─ fusion_metadata.index    [pickled list]              │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  BUILD PROMISCUITY INDEX:                                 │            ║
║  │  1. Cluster domains (HDBSCAN)                            │            ║
║  │  2. Count cluster linkages                               │            ║
║  │  3. Flag promiscuous clusters (>25 partners)             │            ║
║  │  4. Save promiscuity_index.pkl                           │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║                                                                            ║
║  Time: ~10 hours for 23,586 proteins on RTX 3060 (6GB GPU)               ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║  STEP 2: QUERY SEARCH (Fast, ~30 seconds per protein)                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Input: Query protein (e.g., AF-Q14686.pdb)                               ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  1. Load Fusion Database                                  │            ║
║  │     • Load embeddings and metadata from disk              │            ║
║  │     • Build FAISS index for fast search                   │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  2. Segment Query Protein                                 │            ║
║  │     MERIZO: [Query_Domain_A, Query_Domain_B]             │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  3. Embed Query Domains                                   │            ║
║  │     FOLDCLASS: [Emb_A[128], Emb_B[128]]                  │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  4. Search for Each Query Domain                          │            ║
║  │     For Query_Domain_A:                                   │            ║
║  │       • FAISS search → top-K similar domains              │            ║
║  │       • Check if matched domains are in fusion links      │            ║
║  │       • If YES → Rosetta Stone evidence!                  │            ║
║  │       • Predict interaction with fusion partner           │            ║
║  │                                                            │            ║
║  │     Repeat for Query_Domain_B                             │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  5. Apply Promiscuity Filter                              │            ║
║  │     • Check if domains are in promiscuous clusters        │            ║
║  │     • Flag or remove low-specificity predictions          │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  6. Rank by Confidence                                    │            ║
║  │     • Cosine similarity (40%)                             │            ║
║  │     • Number of Rosetta Stones (30%)                      │            ║
║  │     • Domain specificity (20%)                            │            ║
║  │     • Merizo confidence (10%)                             │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  7. Optional: TM-align Validation                         │            ║
║  │     • Structural alignment of matched domains             │            ║
║  │     • TM-score > 0.5 confirms structural similarity       │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║         ↓                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐            ║
║  │  OUTPUT: Ranked PPI Predictions                           │            ║
║  │                                                            │            ║
║  │  Prediction 1: [Confidence: 0.89]                        │            ║
║  │    Query Domain A ←→ Protein_Y Domain_3                  │            ║
║  │    Evidence: 5 Rosetta Stones                            │            ║
║  │    Cosine similarity: 0.92                               │            ║
║  │    TM-score: 0.72                                        │            ║
║  │                                                            │            ║
║  │  Prediction 2: [Confidence: 0.81]                        │            ║
║  │    Query Domain B ←→ Protein_Z Domain_1                  │            ║
║  │    ...                                                    │            ║
║  └──────────────────────────────────────────────────────────┘            ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow & Storage

### File Structure

```
fusion_db/
├── domain_embeddings.pt          # [N_domains, 128] torch.Tensor (float32)
├── domain_metadata.index         # List of (domain_id, coords, seq, embedding)
├── fusion_embeddings.pt          # [N_fusions, 256] torch.Tensor (float32)
├── fusion_metadata.index         # List of fusion link dictionaries
├── promiscuity_index.pkl         # Cluster → promiscuity score mapping
└── domain_registry.pkl           # domain_id → Domain object (optional)
```

### Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  DURING DATABASE BUILDING                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  RAM (Host Memory):                                              │
│  ┌──────────────────────────────────────────┐                   │
│  │ all_embeddings = []  ← numpy arrays      │  Main storage    │
│  │ domain_metadata_list = []                │                   │
│  │ fusion_metadata_list = []                │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  GPU Memory (VRAM):                                              │
│  ┌──────────────────────────────────────────┐                   │
│  │ Merizo network weights      ~2 GB        │  Persistent      │
│  │ Foldclass network weights   ~500 MB      │  Persistent      │
│  │ ───────────────────────────────────────  │                   │
│  │ Active tensors (per protein)             │  Temporary       │
│  │ - Segmentation tensors      1-3 GB       │  (freed after)   │
│  │ - Embedding tensor          <100 MB      │  (freed after)   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  Target GPU memory: < 3 GB total                                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DURING QUERY SEARCH                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  RAM:                                                            │
│  ┌──────────────────────────────────────────┐                   │
│  │ Fusion database loaded (~1-2 GB)         │                   │
│  │ FAISS index (in-memory)                  │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  GPU:                                                            │
│  ┌──────────────────────────────────────────┐                   │
│  │ Merizo + Foldclass networks ~2.5 GB      │                   │
│  │ Query processing          <500 MB         │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## GPU Memory Management

### Critical Memory Leak Fixes

```
┌─────────────────────────────────────────────────────────────────┐
│  9 CRITICAL GPU MEMORY ISSUES FIXED                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Issue #1: MASSIVE Memory Leak in Incremental Saves             │
│  ─────────────────────────────────────────────────────────────   │
│  Problem: Loading entire embeddings file to GPU every batch      │
│  Fix: Accumulate as numpy in RAM, convert to tensor only once    │
│                                                                   │
│  Issue #2: Merizo Features Tensor Leak                           │
│  ─────────────────────────────────────────────────────────────   │
│  Problem: GPU tensors from segment() never freed                 │
│  Fix: Move to CPU immediately, delete dict, clear cache          │
│                                                                   │
│  Issue #3: No Per-Protein GPU Cleanup                            │
│  ─────────────────────────────────────────────────────────────   │
│  Problem: Cache only cleared every 100 proteins                  │
│  Fix: torch.cuda.empty_cache() + gc.collect() after each         │
│                                                                   │
│  Issue #4: Error Handler Leaks Memory                            │
│  ─────────────────────────────────────────────────────────────   │
│  Problem: Failed proteins don't cleanup GPU                      │
│  Fix: Aggressive cleanup in exception handler                    │
│                                                                   │
│  Issue #5: Embedding Tensors Not Freed                           │
│  ─────────────────────────────────────────────────────────────   │
│  Problem: Foldclass tensors stay on GPU                          │
│  Fix: del embeddings + torch.cuda.empty_cache()                  │
│                                                                   │
│  Issue #6: Batch Size Too Large (6GB GPU)                        │
│  ─────────────────────────────────────────────────────────────   │
│  Problem: Default batch_size=32 → OOM                            │
│  Fix: Reduced to batch_size=4, configurable to 1                 │
│                                                                   │
│  Issue #7: No Checkpointing                                      │
│  ─────────────────────────────────────────────────────────────   │
│  Problem: OOM crash loses all data                               │
│  Fix: Checkpoint every 50 proteins                               │
│                                                                   │
│  Issue #8: Tensor Dimension Mismatch                             │
│  ─────────────────────────────────────────────────────────────   │
│  Problem: Batching with padding causes errors                    │
│  Fix: Process domains individually (no batching)                 │
│                                                                   │
│  Issue #9: Memory Leak During Segmentation (CRITICAL)            │
│  ─────────────────────────────────────────────────────────────   │
│  Problem: Merizo segment() allocates 2-6 GB, never freed         │
│  Fix: Clear GPU BEFORE segmentation, wrap in no_grad,            │
│       convert to numpy immediately, double gc + cache clear      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Management Pattern

```python
# BEFORE each protein segmentation:
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()

# DURING segmentation:
with torch.no_grad():
    features = segment(...)

# AFTER segmentation (immediately):
domain_ids_np = features['domain_ids'].cpu().numpy()
conf_res_np = features['conf_res'].cpu().numpy()
del features

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

# AFTER embedding computation:
embedding_np = embedding.squeeze(0).cpu().numpy()
del embedding, coords_tensor
torch.cuda.empty_cache()

# Store in RAM (not GPU):
embeddings_list.append(embedding_np)  # numpy array

# AFTER each protein:
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()  # Second clear after gc

# CHECKPOINT every 50 proteins:
checkpoint_embeddings = torch.tensor(np.array(all_embeddings))
torch.save(checkpoint_embeddings, path)
```

---

## Usage Examples

### 1. Build Fusion Database

```bash
# Basic build
python merizo_search/merizo.py rosetta build \
    examples/database/ \
    fusion_db/ \
    -d cuda

# With custom batch size (if still OOM)
python merizo_search/merizo.py rosetta build \
    examples/database/ \
    fusion_db/ \
    -d cuda \
    --batch-size 2

# Build without promiscuity filtering
python merizo_search/merizo.py rosetta build \
    examples/database/ \
    fusion_db/ \
    -d cuda \
    --skip-promiscuity
```

### 2. Search for Interactions

```bash
# Basic search
python merizo_search/merizo.py rosetta search \
    query_protein.pdb \
    fusion_db/ \
    results \
    -d cuda

# With TM-align validation
python merizo_search/merizo.py rosetta search \
    query_protein.pdb \
    fusion_db/ \
    results \
    -d cuda \
    --validate-tm \
    --min-tm-score 0.5

# Adjust sensitivity
python merizo_search/merizo.py rosetta search \
    query_protein.pdb \
    fusion_db/ \
    results \
    -d cuda \
    --cosine-threshold 0.6 \
    --top-k 50
```

### 3. Monitor GPU Usage

```bash
# In separate terminal
watch -n 0.5 nvidia-smi

# Expected behavior:
# - GPU Utilization: 70-95%
# - Memory Usage: 1.5-3 GB (stable)
# - Memory freed after each protein
```

### 4. Output Format

```json
{
  "query": "AF-Q14686.pdb",
  "num_predictions": 15,
  "predictions": [
    {
      "query_domain_id": "AF-Q14686_domain_0",
      "query_protein": "AF-Q14686",
      "query_range": [1, 120],
      "target_domain_id": "AF-P12345_domain_1",
      "target_protein": "AF-P12345",
      "target_range": [135, 280],
      "num_rosetta_stones": 5,
      "rosetta_stone_ids": ["AF-Q98765", "AF-P54321", ...],
      "cosine_similarity": 0.92,
      "tm_score": 0.72,
      "confidence": 0.89,
      "promiscuity_filtered": false,
      "interaction_type": "inter"
    }
  ]
}
```

---

## Performance Benchmarks

```
┌─────────────────────────────────────────────────────────────────┐
│  SYSTEM PERFORMANCE (RTX 3060 6GB GPU)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Database Building:                                              │
│  ├─ Throughput: 2-5 proteins/min                                │
│  ├─ GPU Memory: < 3 GB (stable)                                 │
│  ├─ 23,586 proteins: ~10 hours                                  │
│  └─ Checkpoint frequency: every 50 proteins (~10 min)           │
│                                                                   │
│  Query Search:                                                   │
│  ├─ Segmentation: ~5-10 seconds                                 │
│  ├─ Embedding: ~2-5 seconds                                     │
│  ├─ FAISS search: <1 second                                     │
│  ├─ Total: ~30 seconds per query                                │
│  └─ With TM-align: ~2-3 minutes per query                       │
│                                                                   │
│  Database Size:                                                  │
│  ├─ 50,000 domains → ~25 MB embeddings                          │
│  ├─ 100,000 fusion links → ~100 MB embeddings                   │
│  └─ Total database: ~500 MB - 2 GB                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Common Issues

```
Issue: CUDA out of memory
Solution:
  1. Reduce batch size to 2 or 1
  2. Restart Python to clear leaked memory
  3. Check other GPU processes with nvidia-smi
  4. Ensure PyTorch CUDA allocator is set:
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Issue: Memory accumulating between proteins
Solution:
  - Verify GPU cleanup logs show memory freed
  - Check torch.cuda.memory_allocated() is decreasing
  - Ensure all tensors converted to numpy immediately

Issue: Tensor dimension mismatch
Solution:
  - Already fixed: processes domains individually
  - No batching with padding

Issue: Very slow processing
Solution:
  - Verify GPU is being used: torch.cuda.is_available()
  - Check GPU utilization with nvidia-smi (should be 70-95%)
  - Install CUDA-enabled PyTorch if needed
```

---

## System Requirements

```
Minimum:
├─ GPU: 6 GB VRAM (RTX 3060, Tesla T4)
├─ RAM: 16 GB
├─ Storage: 50 GB free
└─ Python: 3.8+

Recommended:
├─ GPU: 8+ GB VRAM (RTX 3070, A4000)
├─ RAM: 32 GB
├─ Storage: 100 GB SSD
└─ Python: 3.9+

Dependencies:
├─ torch (CUDA-enabled)
├─ numpy
├─ h5py
├─ faiss-cpu (or faiss-gpu)
├─ hdbscan
├─ scikit-learn
└─ tqdm
```

---

**End of Guide**

For implementation details, see `IMPLEMENTATION.md`
For GPU memory fixes, see `GPU_MEMORY_FIXES.md`
For usage instructions, see `README.md`
