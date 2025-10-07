# Merizo-Search Architecture Guide

## Table of Contents
1. [Overview](#overview)
2. [Mode 1: Segment](#mode-1-segment)
3. [Mode 2: Search](#mode-2-search)
4. [Mode 3: Easy-Search](#mode-3-easy-search)
5. [Mode 4: CreateDB](#mode-4-createdb)
6. [Neural Network Architectures](#neural-network-architectures)
   - [Merizo Network](#merizo-network-architecture)
   - [Foldclass Network](#foldclass-network-architecture)

---

## Overview

Merizo-search provides 4 operational modes, each calling a specific chain of functions. Two neural networks power the system:
- **Merizo**: Domain segmentation using Invariant Point Attention (IPA)
- **Foldclass**: Structure embedding using Equivariant Graph Neural Networks (EGNN)

---

## Mode 1: Segment

**Purpose**: Segment multi-domain proteins into individual domains

### Function Call Chain

```
merizo.py::segment()
    ↓
merizo.py::segment_pdb() [alias for programs.Merizo.predict.run_merizo]
    ↓
programs/Merizo/predict.py::run_merizo()
    ↓
    ├─→ Load Merizo neural network weights
    │   └─→ programs/Merizo/predict.py::read_split_weight_files()
    │       └─→ Loads from programs/Merizo/weights/*.pt
    │
    ├─→ For each input PDB file:
    │   ├─→ programs/Merizo/predict.py::segment()
    │   │   ↓
    │   │   ├─→ programs/Merizo/model/utils/features.py::generate_features_domain()
    │   │   │   └─→ Extracts CA coords, sequences, generates rotation/translation frames
    │   │   │
    │   │   ├─→ Merizo Network Forward Pass
    │   │   │   └─→ programs/Merizo/model/network.py::Merizo.forward()
    │   │   │       ↓
    │   │   │       ├─→ Linear projections (s_in, z_in)
    │   │   │       ├─→ IPA Encoder (6 blocks)
    │   │   │       └─→ Mask Decoder (10 transformer layers)
    │   │   │           └─→ Returns domain_ids, conf_res
    │   │   │
    │   │   ├─→ [Optional] Iterative Segmentation
    │   │   │   └─→ programs/Merizo/predict.py::iterative_segmentation()
    │   │   │       └─→ Re-segments large domains (>200 residues)
    │   │   │           └─→ Calls Merizo network on subdomain masks
    │   │   │
    │   │   ├─→ Post-processing
    │   │   │   ├─→ programs/Merizo/model/utils/utils.py::separate_components()
    │   │   │   ├─→ programs/Merizo/model/utils/utils.py::clean_domains()
    │   │   │   └─→ programs/Merizo/model/utils/utils.py::clean_singletons()
    │   │   │
    │   │   └─→ Return segmented features
    │   │
    │   └─→ programs/Merizo/predict.py::generate_outputs()
    │       ├─→ [Optional] write_pdb_predictions()
    │       ├─→ [Optional] write_pdf_predictions()
    │       ├─→ [Optional] write_fasta()
    │       └─→ [Optional] write_domain_idx()
    │
    └─→ programs/utils.py::write_segment_results()
        └─→ Writes *_segment.tsv output file
```

### Neural Network Used: Merizo

**Input**: Protein structure (CA coordinates + sequence)
**Output**: Domain boundaries and confidence scores

---

## Mode 2: Search

**Purpose**: Search single-domain queries against a pre-built database

### Function Call Chain

```
merizo.py::search()
    ↓
merizo.py::dbsearch() [alias for programs.Foldclass.dbsearch.run_dbsearch]
    ↓
programs/Foldclass/dbsearch.py::run_dbsearch()
    ↓
    ├─→ Setup Foldclass Network
    │   └─→ programs/Foldclass/dbsearch.py::network_setup()
    │       └─→ Loads programs/Foldclass/FINAL_foldclass_model.pt
    │
    ├─→ Read Database
    │   └─→ programs/Foldclass/dbsearch.py::read_database()
    │       ├─→ Standard DB: Loads {db_name}.pt + {db_name}.index
    │       └─→ Faiss DB: Loads {db_name}.json config
    │
    ├─→ Search Branch A: Standard PyTorch Database
    │   └─→ For each query PDB:
    │       └─→ programs/Foldclass/dbsearch.py::dbsearch()
    │           ↓
    │           ├─→ programs/Foldclass/utils.py::read_pdb()
    │           │   └─→ Reads CA coords and sequence
    │           │
    │           ├─→ Foldclass Network Forward Pass
    │           │   └─→ programs/Foldclass/nndef_fold_egnn_embed.py::FoldClassNet.forward()
    │           │       ↓
    │           │       ├─→ PositionalEncoder (sinusoidal encoding)
    │           │       ├─→ EGNN layers (2 blocks)
    │           │       │   └─→ programs/Foldclass/my_egnn_nocoords.py::EGNN.forward()
    │           │       └─→ Mean pooling → 128-dim embedding
    │           │
    │           ├─→ Embedding Search
    │           │   └─→ programs/Foldclass/dbsearch.py::search_query_against_db()
    │           │       └─→ F.cosine_similarity() with coverage mask
    │           │           └─→ torch.topk() to get top-k hits
    │           │
    │           └─→ TM-align Verification (unless skip_tmalign=True)
    │               └─→ For each top-k hit:
    │                   ├─→ programs/Foldclass/utils.py::write_pdb() (temp files)
    │                   └─→ programs/Foldclass/utils.py::run_tmalign()
    │                       └─→ Filters by mintm threshold
    │
    └─→ Search Branch B: Faiss Database (for large-scale search)
        └─→ programs/Foldclass/dbsearch.py::dbsearch_faiss()
            ↓
            ├─→ Memory-map database
            │   └─→ programs/Foldclass/dbutil.py::db_memmap()
            │
            ├─→ Embed all queries in batch
            │   └─→ For each query: FoldClassNet.forward()
            │       └─→ Normalize embeddings (F.normalize)
            │
            ├─→ Batch search using Faiss
            │   └─→ knn_exact_faiss() (custom implementation)
            │       ├─→ Iterates over DB in chunks (search_batchsize)
            │       ├─→ faiss.IndexFlat with METRIC_INNER_PRODUCT
            │       └─→ faiss.ResultHeap to accumulate top-k
            │
            ├─→ Retrieve hit information
            │   └─→ programs/Foldclass/dbutil.py
            │       ├─→ retrieve_names_by_idx() (domain names)
            │       ├─→ retrieve_start_end_by_idx() (byte offsets)
            │       └─→ retrieve_bytes() (sequences, coords, metadata)
            │
            └─→ TM-align all hits (parallel possible)
                └─→ For each hit: run_tmalign()
                    └─→ Filter by mintm threshold
    │
    ├─→ [Optional] Multi-Domain Search
    │   └─→ programs/Foldclass/dbsearch_fulllength.py::multi_domain_search()
    │       └─→ See dedicated section below
    │
    └─→ programs/utils.py::write_search_results()
        ├─→ Writes *_search.tsv (significant hits)
        └─→ [Optional] *_search_insignificant.tsv
```

### Neural Network Used: Foldclass

**Input**: Single domain structure (CA coordinates)
**Output**: 128-dimensional structure embedding

### Multi-Domain Search Workflow

When `--multi_domain_search` is enabled:

```
programs/Foldclass/dbsearch_fulllength.py::multi_domain_search()
    ↓
    ├─→ Group query domains by chain
    │   └─→ Extract chain IDs from domain names
    │
    ├─→ Build candidate target list
    │   └─→ For each query domain's hits:
    │       └─→ Find all domains from same target chain
    │           └─→ domid2chainid_fn() extracts chain from domain name
    │
    ├─→ Mode: exhaustive_tmalign
    │   ↓
    │   ├─→ Write query and target PDBs to temp directory
    │   │   └─→ programs/Foldclass/utils.py::write_pdb()
    │   │
    │   ├─→ Pairwise TM-align (all query domains vs all target domains)
    │   │   └─→ programs/Foldclass/utils.py::pairwise_parallel_fill_tmalign_array()
    │   │       └─→ Parallelized TM-align execution
    │   │           └─→ Returns NxM matrix of TM-scores
    │   │
    │   └─→ Analyze TM-score matrix for valid matches
    │       └─→ tmalign_submatrix_to_hits()
    │           ├─→ Filters out rows/columns with no valid hits
    │           ├─→ Enumerates all valid domain mappings
    │           │   └─→ itertools.product() over hit paths
    │           │
    │           └─→ Classifies match category:
    │               ├─→ Category 0: Unordered (bag-of-domains)
    │               ├─→ Category 1: Ordered with gaps (discontiguous)
    │               ├─→ Category 2: Ordered, end gaps only (contiguous)
    │               └─→ Category 3: Exact MDA match (same number, same order)
    │
    └─→ programs/utils.py::write_all_dom_search_results()
        └─→ Writes *_search_multi_dom.tsv
```

---

## Mode 3: Easy-Search

**Purpose**: Combined workflow - segment then search

### Function Call Chain

```
merizo.py::easy_search()
    ↓
    ├─→ Step 1: Segmentation
    │   └─→ programs/Merizo/predict.py::run_merizo()
    │       └─→ [Same as Mode 1: Segment]
    │       └─→ Returns: segment_domains (list of domain dicts)
    │                    segment_results (metadata for TSV)
    │
    ├─→ programs/utils.py::write_segment_results()
    │   └─→ Writes *_segment.tsv
    │
    ├─→ Check for valid domains
    │   └─→ If len(segment_domains) == 0: gracefully exit
    │
    ├─→ Step 2: Search segmented domains
    │   └─→ programs/Foldclass/dbsearch.py::run_dbsearch()
    │       ├─→ inputs_are_ca=True (domains are dict with coords/seq)
    │       └─→ [Same as Mode 2: Search]
    │           └─→ Returns: search_results, all_search_results
    │
    ├─→ programs/utils.py::write_search_results()
    │   ├─→ Writes *_search.tsv
    │   └─→ [Optional] *_search_insignificant.tsv
    │
    └─→ [Optional] Multi-Domain Search
        └─→ programs/Foldclass/dbsearch_fulllength.py::multi_domain_search()
            ├─→ inputs_from_easy_search=True
            │   └─→ Infers chain IDs from domain names (pattern: {name}_merizo_{nn})
            └─→ Writes *_search_multi_dom.tsv
```

**Key Difference from Search Mode**:
- Easy-search passes domain coordinate dictionaries (not PDB files) to dbsearch
- `inputs_are_ca=True` flag tells dbsearch to skip PDB reading
- Chain IDs are inferred from segmentation results

---

## Mode 4: CreateDB

**Purpose**: Build custom Foldclass database from PDB files

### Function Call Chain

```
merizo.py::createdb()
    ↓
merizo.py::createdb_from_pdb() [alias for programs.Foldclass.makedb.run_createdb]
    ↓
programs/Foldclass/makedb.py::run_createdb()
    ↓
    ├─→ Parse input (directory, tar, tar.gz, or zip)
    │   └─→ Collect all .pdb files
    │       └─→ Sort alphabetically for consistency
    │
    ├─→ Setup Foldclass Network
    │   └─→ programs/Foldclass/makedb.py::network_setup()
    │       └─→ Loads FINAL_foldclass_model.pt
    │
    ├─→ For each PDB file:
    │   ├─→ Parse CA atoms and sequences (inline parsing)
    │   │   └─→ Reads ATOM records with ' CA ' identifier
    │   │       └─→ Uses programs/Foldclass/constants.py::three_to_single_aa
    │   │
    │   ├─→ Truncate to 2000 residues (max length)
    │   │
    │   ├─→ Foldclass Network Forward Pass
    │   │   └─→ programs/Foldclass/nndef_fold_egnn_embed.py::FoldClassNet.forward()
    │   │       ↓
    │   │       ├─→ PositionalEncoder
    │   │       ├─→ EGNN (2 layers)
    │   │       └─→ Mean pooling → 128-dim embedding
    │   │
    │   └─→ Accumulate:
    │       ├─→ tvecs: list of embeddings
    │       └─→ targets: list of (pdb_path, coords, seq) tuples
    │
    ├─→ Concatenate all embeddings
    │   └─→ torch.cat(tvecs, dim=0) → shape: [N, 128]
    │
    └─→ Save database files
        ├─→ torch.save() → {out_db}.pt (embedding tensor)
        └─→ pickle.dump() → {out_db}.index (metadata list)
```

### Neural Network Used: Foldclass

**Input**: PDB structure (CA coordinates)
**Output**: 128-dimensional embedding per structure

**Database Format**:
- `.pt` file: PyTorch tensor of shape [N, 128]
- `.index` file: Pickled list of tuples: `[(name, coords, sequence), ...]`

---

## Neural Network Architectures

### Merizo Network Architecture

**Purpose**: Segment multi-domain proteins into constituent domains using Invariant Point Attention

**File**: `programs/Merizo/model/network.py`

#### High-Level Architecture

```
Input Features (PDB Structure)
    ↓
[Linear Projections]
    ├─→ s (sequence/node features): 20 → 512
    └─→ z (pair features): 1 → 32
    ↓
[IPA Encoder - 6 Blocks]
    ├─→ Invariant Point Attention (rotation-equivariant)
    ├─→ Residual connections
    └─→ Structure Module Transition (FFN)
    ↓
[Mask Decoder - 10 Transformer Layers]
    ├─→ Class embeddings (20 domain classes)
    ├─→ Multi-head self-attention with ALiBi positional bias
    ├─→ Domain classification head
    ├─→ Background residue prediction (GRU)
    └─→ Confidence scoring (GRU per domain)
    ↓
Output: (domain_ids, confidence_scores)
```

#### Detailed Component Breakdown

##### 1. Input Feature Generation
**File**: `programs/Merizo/model/utils/features.py::generate_features_domain()`

```python
Input: PDB file path
    ↓
Parse PDB → Extract:
    ├─→ s: Node features [1, N, 20] (one-hot amino acid encoding)
    ├─→ z: Edge features [1, N, N, 1] (Cβ-Cβ distances)
    ├─→ r: Rotation matrices [1, N, 3, 3] (local coordinate frames)
    ├─→ t: Translation vectors [1, N, 3] (CA positions in local frames)
    └─→ ri: Residue indices [1, N, 1] (positional information)
```

##### 2. IPA Encoder Block
**File**: `programs/Merizo/model/ipa/ipa_encoder.py::ipa_block`

**Configuration**:
```
c_s = 512          # Single representation dimension
c_z = 32           # Pair representation dimension
c_ipa = 512        # IPA hidden dimension
no_blocks = 6      # Number of IPA blocks
no_heads = 16      # Attention heads
no_qk_points = 4   # Query/Key point attention heads
no_v_points = 8    # Value point attention heads
dropout_rate = 0.0
```

**Per-Block Operation**:
```
For each of 6 blocks:
    ├─→ Layer Normalization (s, z)
    ├─→ Invariant Point Attention
    │   └─→ programs/Merizo/model/ipa/nndef_ipa.py::InvariantPointAttention
    │       ├─→ Scalar attention: Q·K (traditional self-attention)
    │       ├─→ Point attention: Computes attention based on 3D distances
    │       │   └─→ Uses Rigid transformations (rotation + translation)
    │       ├─→ Combine scalar + point attention
    │       └─→ Output: [1, N, 512]
    ├─→ Residual connection: s = s + ipa_out
    ├─→ Layer Normalization
    └─→ Structure Module Transition (2-layer FFN)
        └─→ Residual connection
```

**Key Innovation - Invariant Point Attention**:
- Operates in 3D coordinate space using SE(3)-equivariant operations
- Attention weights depend on both:
  1. Feature similarity (scalar attention)
  2. Geometric proximity in 3D space (point attention)
- Uses local coordinate frames (Rigid objects) to maintain rotation/translation invariance

##### 3. Mask Decoder (Transformer)
**File**: `programs/Merizo/model/decoders/mask_decoder.py::MaskTransformer`

**Configuration**:
```
n_cls = 20          # Number of domain classes
n_layers = 10       # Transformer decoder layers
n_heads = 16        # Multi-head attention
d_model = 512       # Model dimension
d_ff = 512          # Feedforward dimension
```

**Architecture**:

```
Input: IPA encoder output [1, N, 512]
    ↓
├─→ Learnable class embeddings [1, 20, 512]
│   └─→ Concatenate with input: [1, N+20, 512]
    ↓
├─→ 10 Transformer Decoder Layers
│   └─→ For each layer:
│       ├─→ Layer Norm
│       ├─→ Multi-Head Self-Attention
│       │   ├─→ Q, K, V projections
│       │   ├─→ ALiBi positional bias (no learned pos. embeddings)
│       │   └─→ Attention scores: softmax(Q·K^T + bias)
│       ├─→ Residual connection
│       ├─→ Layer Norm
│       ├─→ FFN (Linear → GELU → Linear)
│       └─→ Residual connection
    ↓
├─→ Separate patch and class tokens
│   ├─→ Patch features: [1, N, 512] @ proj_patch
│   └─→ Class features: [1, 20, 512] @ proj_classes
    ↓
├─→ Domain Classification
│   └─→ Dot product: patches @ classes.T = [1, N, 20]
│       └─→ argmax(dim=-1) → predicted_domain_ids
    ↓
├─→ Background Prediction (2-layer BiGRU)
│   └─→ Input: patch features [1, N, 512]
│       ├─→ GRU(hidden=256, bidirectional=True) → [1, N, 512]
│       ├─→ Linear(512, 2) → background scores
│       └─→ argmax → background mask (0 or 1)
    ↓
├─→ Apply background mask
│   └─→ domain_ids = predicted_ids * background_mask
    ↓
└─→ Confidence Scoring (per-domain 2-layer BiGRU)
    └─→ For each unique domain:
        ├─→ Extract domain mask predictions [1, n_res_in_domain, 20]
        ├─→ GRU(hidden=512, bidirectional=True)
        ├─→ Linear(512, 1) → confidence score
        └─→ Clamp to [0, 1]
```

**ALiBi Positional Bias**:
- File: `programs/Merizo/model/posenc/alibi.py::AlibiPositionalBias`
- Linear bias based on token distance: `bias[i,j] = -slope * |i-j|`
- Different slopes for each attention head
- Allows extrapolation to longer sequences than training data

##### 4. Post-Processing
**File**: `programs/Merizo/model/utils/utils.py`

```
Raw domain predictions
    ↓
├─→ separate_components()
│   └─→ Connected component analysis on domain adjacency graph
│       └─→ Splits non-contiguous segments into separate domains
    ↓
├─→ clean_domains(min_domain_size=50)
│   └─→ Removes domains with < 50 residues
    ↓
└─→ clean_singletons(min_fragment_size=10)
    └─→ Removes isolated fragments < 10 residues
```

##### 5. Iterative Segmentation (Optional)
**File**: `programs/Merizo/predict.py::iterative_segmentation()`

**Purpose**: Re-segment large domains that may contain multiple subdomains

```
Conditions for iteration:
    ├─→ domain_size > 200 residues (default)
    └─→ max_iterations not reached (default: 3)

For each large domain:
    ├─→ Create domain mask
    ├─→ Run Merizo network on masked features
    │   └─→ network.forward(features, mask=domain_mask)
    ├─→ Check if segmented into >1 domains
    │   ├─→ Yes: Accept subdivision
    │   │   └─→ Offset new domain IDs to avoid conflicts
    │   └─→ No: Mark domain as unsplittable
    └─→ Update domain_ids and confidence scores
```

#### Summary - Merizo Network

**Input Dimensions**:
- s (node features): [1, N, 20]
- z (edge features): [1, N, N, 1]
- r (rotations): [1, N, 3, 3]
- t (translations): [1, N, 3]

**Architecture**:
- IPA Encoder: 6 blocks, 16 heads, SE(3)-equivariant
- Mask Decoder: 10 transformer layers, 16 heads, 20 domain classes

**Output**:
- domain_ids: [N] - Integer domain assignments (0=background)
- conf_res: [N] - Per-residue confidence scores [0, 1]

**Key Features**:
- Rotation/translation equivariant via IPA
- ALiBi for length generalization
- Iterative refinement for complex proteins
- Confidence estimation per domain

---

### Foldclass Network Architecture

**Purpose**: Generate structure embeddings for protein domains using Equivariant Graph Neural Networks

**File**: `programs/Foldclass/nndef_fold_egnn_embed.py`

#### High-Level Architecture

```
Input: CA Coordinates [batch, N, 3]
    ↓
[Positional Encoding]
    ↓
[EGNN Layer 1]
    ├─→ Edge updates (based on distances)
    └─→ Node updates (message passing)
    ↓
[EGNN Layer 2]
    ├─→ Edge updates (based on distances)
    └─→ Node updates (message passing)
    ↓
[Mean Pooling]
    ↓
Output: Structure Embedding [batch, 128]
```

#### Detailed Component Breakdown

##### 1. FoldClassNet Main Module
**File**: `programs/Foldclass/nndef_fold_egnn_embed.py::FoldClassNet`

**Configuration**:
```python
width = 128             # Embedding dimension
n_egnn_layers = 2       # Number of EGNN layers
m_dim = 256             # Edge message dimension (width * 2)
```

**Forward Pass**:
```python
def forward(x):  # x: [batch, nres, 3] CA coordinates
    ↓
    ├─→ Positional Encoding
    │   └─→ seq_feats = posenc_as(x)  # [batch, nres, 128]
    │       └─→ Sinusoidal position encoding (learned=False)
    ↓
    ├─→ EGNN Processing
    │   └─→ out_feats = encode_ca_egnn((seq_feats, x, None))
    │       │   # Input tuple: (node_features, coordinates, edges)
    │       │   # edges=None (fully connected graph)
    │       └─→ Returns: (updated_feats, coords, edges)
    │           # Coordinates not updated (nocoords version)
    ↓
    └─→ Global Pooling
        └─→ embed = out_feats.mean(dim=1)  # [batch, 128]
            └─→ Mean over residue dimension
```

##### 2. Positional Encoder
**File**: `programs/Foldclass/nndef_fold_egnn_embed.py::PositionalEncoder`

**Purpose**: Inject sequential information into node features

```python
Parameters:
    d_model = 128       # Feature dimension
    max_len = 3000      # Maximum sequence length
    learned = False     # Use fixed sinusoidal encoding

Encoding formula (for position pos, dimension i):
    PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

Output: [1, nres, 128]
```

##### 3. EGNN Layer (E(n) Equivariant Graph Neural Network)
**File**: `programs/Foldclass/my_egnn_nocoords.py::EGNN`

**Configuration**:
```python
dim = 128              # Node feature dimension
m_dim = 256           # Edge message dimension
edge_dim = 0          # Additional edge features (unused)
radius = 10.0         # Distance radius (not enforced in this version)
init_eps = 1e-3       # Weight initialization scale
```

**Architecture**:

```
Input: (feats, coords, edges)
    # feats: [batch, n, 128]
    # coords: [batch, n, 3]
    # edges: None (fully connected)

Step 1: Compute Pairwise Distances
    ├─→ rel_coors = coords[:, i, :] - coords[:, j, :]  # [batch, n, n, 3]
    └─→ dist = ||rel_coors||  # [batch, n, n, 1]

Step 2: Edge Message Computation
    ├─→ Broadcast node features
    │   ├─→ feats_i = repeat(feats, 'b i d -> b i 1 d')  # Source nodes
    │   └─→ feats_j = repeat(feats, 'b j d -> b 1 j d')  # Target nodes
    │
    ├─→ Concatenate edge inputs
    │   └─→ edge_input = [feats_i, feats_j, dist²]  # [batch, n, n, 128+128+1]
    │
    ├─→ Edge MLP (3-layer)
    │   └─→ edge_mlp(edge_input) → [batch, n, n, m_dim]
    │       ├─→ Linear(257, 514)
    │       ├─→ SiLU activation
    │       ├─→ Linear(514, 256)
    │       └─→ SiLU activation
    │
    └─→ Edge Gating
        └─→ m_ij = edge_features * sigmoid(Linear(edge_features))
            # Soft attention mechanism

Step 3: Message Aggregation
    └─→ m_i = sum_j(m_ij)  # [batch, n, m_dim]
        # Sum messages from all neighbors

Step 4: Node Update
    ├─→ node_input = [feats, m_i]  # [batch, n, 128+256]
    │
    └─→ Node MLP (3-layer)
        └─→ node_mlp(node_input) + feats  # Residual connection
            ├─→ Linear(384, 256)
            ├─→ SiLU activation
            ├─→ Linear(256, 128)
            └─→ Residual: output + input_feats

Output: (updated_feats, coords, edges)
    # coords unchanged in this version
```

**Key Properties of EGNN**:

1. **E(n) Equivariance**:
   - Equivariant to rotations, translations, reflections
   - Uses pairwise distances (invariant quantities)
   - Node updates based on relative positions

2. **Message Passing**:
   - Each node aggregates information from all other nodes
   - Messages weighted by edge features and gating
   - Fully connected graph (all-to-all communication)

3. **Distance-Based Interactions**:
   - Edge features computed from squared distances (dist²)
   - No explicit radius cutoff in this implementation
   - Soft attention via edge gating

4. **Residual Connections**:
   - Node features: `out = MLP([feats, messages]) + feats`
   - Stabilizes training and gradient flow

##### 4. Two-Layer EGNN Stack

```
Layer 1: EGNN(dim=128, m_dim=256)
    ├─→ Input: (pos_encoding, coords, None)
    └─→ Output: (feats_1, coords, None)
        ↓
Layer 2: EGNN(dim=128, m_dim=256)
    ├─→ Input: (feats_1, coords, None)
    └─→ Output: (feats_2, coords, None)
```

**Information Flow**:
- Layer 1: Captures local structural patterns
- Layer 2: Integrates broader geometric context
- Two layers allow 2-hop message passing (each residue sees neighbors-of-neighbors)

##### 5. Global Pooling

```python
embed = out_feats.mean(dim=1)  # [batch, nres, 128] → [batch, 128]
```

**Purpose**: Aggregate per-residue features into single domain representation

**Properties**:
- Permutation invariant (order of residues doesn't matter)
- Fixed-size output regardless of domain length
- Simple but effective for structure comparison

#### Summary - Foldclass Network

**Input**:
- CA coordinates: [batch, N, 3]
- N can vary (up to 2000 residues)

**Architecture**:
- Positional Encoding: Sinusoidal, 128-dim
- EGNN: 2 layers, 128 node features, 256 message features
- Pooling: Mean over residues

**Output**:
- Structure embedding: [batch, 128]
- L2-normalized for cosine similarity search

**Key Features**:
- E(n) equivariant (rotation/translation/reflection invariant)
- Fully connected graph (all-to-all message passing)
- Captures geometric relationships via distance-based edges
- Efficient for structure comparison via embedding space

**Use Cases**:
1. **CreateDB**: Embed database structures for fast lookup
2. **Search**: Embed query, find similar embeddings via cosine similarity
3. **Verification**: TM-align validates geometric similarity

---

## Database Search Process (Detailed)

### Standard PyTorch Database Search

```
Query Structure → FoldClassNet → Query Embedding [128]
    ↓
Database Embeddings [N, 128]
    ↓
Cosine Similarity Computation
    ├─→ scores = F.cosine_similarity(db_embs, query_emb)
    ├─→ Apply coverage mask: scores *= mask
    │   └─→ mask = (query_len >= target_len * mincov)
    └─→ top_scores, top_indices = torch.topk(scores, k)
    ↓
Filter by mincos threshold
    ↓
For each candidate hit:
    ├─→ Retrieve coords and sequence from index
    ├─→ Write temporary PDB files
    ├─→ Run TM-align
    │   └─→ External binary: programs/Foldclass/bin/TMalign
    └─→ Parse TM-align output
        ├─→ TM-score (query), TM-score (target)
        ├─→ RMSD, alignment length, sequence identity
        └─→ Filter by mintm threshold
```

### Faiss Database Search (Large-Scale)

**Purpose**: Handle databases too large for GPU/RAM

```
Initialization:
    ├─→ Memory-map database files
    │   ├─→ Embeddings: db.bin (binary float32)
    │   ├─→ Names: db_names.dat
    │   ├─→ Sequences: db_seqs.dat + .index
    │   ├─→ Coordinates: db_coords.dat + .index
    │   └─→ Metadata: db_metadata.dat + .index
    │
    └─→ Create DB iterator with batchsize (e.g., 262144)

Query Processing:
    ├─→ Embed all queries: FoldClassNet → [num_queries, 128]
    └─→ Normalize embeddings: F.normalize()

Batch Search:
    └─→ For each DB batch:
        ├─→ Load batch_embeddings from memory-map
        ├─→ faiss.IndexFlat(128, METRIC_INNER_PRODUCT)
        ├─→ index.add(batch_embeddings)
        ├─→ D, I = index.search(query_embeddings, k)
        ├─→ Accumulate in faiss.ResultHeap
        ├─→ index.reset()
        └─→ Increment batch offset

Result Retrieval:
    ├─→ Extract hit indices from ResultHeap
    ├─→ Memory-map file reading:
    │   ├─→ retrieve_names_by_idx()
    │   ├─→ retrieve_start_end_by_idx()
    │   └─→ retrieve_bytes()
    │
    └─→ TM-align validation (same as standard search)
```

---

## Key Design Decisions

### 1. Two-Stage Search Strategy

**Rationale**: Balance speed and accuracy

```
Stage 1: Fast Embedding Search (Foldclass)
    ├─→ O(N) cosine similarity computation
    ├─→ Reduces N database entries to k candidates
    └─→ ~1000x faster than structural alignment

Stage 2: Precise Structural Alignment (TM-align)
    ├─→ Validates top-k candidates
    ├─→ Provides accurate TM-score and RMSD
    └─→ Only run on promising hits
```

### 2. Iterative Segmentation

**Purpose**: Handle complex multi-domain proteins

```
Problem: Long proteins (>500 residues) challenging for single pass
    ↓
Solution: Hierarchical segmentation
    ├─→ Initial segmentation
    ├─→ Re-segment domains >200 residues
    ├─→ Repeat up to max_iterations (default: 3)
    └─→ Stop when domains stabilize or max reached
```

### 3. Multi-Domain Matching Categories

**Category 0** (Unordered):
- All query domains match target domains
- No order constraints
- Example: {D1, D2, D3} matches {D3, D1, D2}

**Category 1** (Discontiguous):
- Order preserved, gaps allowed
- Example: Query {D1, D2, D3} matches Target {D1, Dx, D2, Dy, D3}

**Category 2** (Contiguous):
- Order preserved, end gaps only
- Example: Query {D1, D2} matches Target {Dx, D1, D2, Dy}

**Category 3** (Exact MDA):
- Same domain count, same order, no gaps
- Example: Query {D1, D2} matches Target {D1, D2}

---

## Performance Considerations

### Memory Usage

**Merizo**:
- IPA encoder: ~100MB model weights
- Per-protein: O(N²) for pair features
- Max practical length: ~2000 residues

**Foldclass**:
- Model: ~20MB weights
- Per-domain: O(N²) for distance matrix
- Max length: 2000 residues (enforced in createdb)

### Computational Complexity

**Segmentation** (Merizo):
- IPA: O(N² × blocks × heads) ≈ O(N²)
- Decoder: O(N² × layers) ≈ O(N²)
- Iterative: multiply by iterations (1-3)

**Embedding** (Foldclass):
- EGNN: O(N² × layers) = O(2N²)
- Fully connected graph (all pairwise distances)

**Search**:
- Standard DB: O(N × embed_dim) + O(k × TM-align)
- Faiss DB: O(N/batch × embed_dim) + O(k × TM-align)
- TM-align: O(query_len × target_len)

### Parallelization

**Multi-domain Search TM-align**:
- File: `programs/Foldclass/utils.py::pairwise_parallel_fill_tmalign_array()`
- Uses multiprocessing to parallelize TM-align calls
- Speedup: ~linear with CPU cores (I/O bound)

---

## File Format Specifications

### Standard Database Files

**{db_name}.pt**:
- PyTorch tensor, shape: [N, 128]
- dtype: float32
- Contains: Normalized structure embeddings

**{db_name}.index**:
- Pickled Python list
- Each entry: `(name: str, coords: np.ndarray[M,3], seq: str)`
- Indexed by database row number

**{db_name}.metadata** (optional):
- Plain text, JSON per line
- Variable fields per database

**{db_name}.metadata.index** (optional):
- Pickled list of (start_byte, end_byte) tuples

### Faiss Database Files

**{db_name}.json**:
```json
{
  "DB_SIZE": 12345678,
  "DB_DIM": 128,
  "dbfname_IP": "db_embeddings.bin",
  "db_names_f": "db_names.dat",
  "sif": "db_seqs.index",
  "sdf": "db_seqs.dat",
  "cif": "db_coords.index",
  "cdf": "db_coords.dat",
  "mif": "db_metadata.index",
  "mdf": "db_metadata.dat"
}
```

**db_embeddings.bin**:
- Binary file: float32 × DB_SIZE × DB_DIM
- Memory-mapped for efficient access

**db_names.dat**:
- Null-terminated C strings, concatenated

**db_seqs.dat + .index**:
- .dat: Concatenated amino acid sequences
- .index: Binary (start_byte, end_byte) pairs

**db_coords.dat + .index**:
- .dat: Binary float32 triplets (x, y, z)
- .index: Binary (start_byte, end_byte) pairs

---

## References

### Papers

**Merizo**:
- Lau, et al., 2023. "Merizo: a rapid and accurate domain segmentation method using invariant point attention." bioRxiv. doi: 10.1101/2023.02.19.529114

**Invariant Point Attention**:
- Jumper, et al., 2021. "Highly accurate protein structure prediction with AlphaFold." Nature. (IPA originally from AlphaFold2)

**EGNN**:
- Satorras, et al., 2021. "E(n) Equivariant Graph Neural Networks." ICML. arXiv: 2102.09844

**TM-align**:
- Zhang & Skolnick, 2005. "TM-align: a protein structure alignment algorithm based on the TM-score." Nucleic Acids Research.

**ALiBi Positional Encoding**:
- Press, et al., 2021. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." arXiv: 2108.12409

### Code Attribution

- IPA implementation derived from AlphaFold2 (Apache 2.0 License)
- EGNN implementation derived from lucidrains/egnn-pytorch
- Mask decoder inspired by Segmenter (Strudel, et al.)

---

## Appendix: Key File Locations

### Model Weights
```
programs/Merizo/weights/*.pt        # Merizo model (split files)
programs/Foldclass/FINAL_foldclass_model.pt
```

### External Binaries
```
programs/Foldclass/bin/TMalign     # TM-align executable
```

### Core Modules
```
merizo_search/merizo.py                           # Main CLI entry
programs/Merizo/predict.py                        # Segmentation logic
programs/Merizo/model/network.py                  # Merizo network
programs/Merizo/model/ipa/ipa_encoder.py         # IPA blocks
programs/Merizo/model/decoders/mask_decoder.py   # Domain decoder
programs/Foldclass/makedb.py                      # Database creation
programs/Foldclass/dbsearch.py                    # Single-domain search
programs/Foldclass/dbsearch_fulllength.py        # Multi-domain search
programs/Foldclass/nndef_fold_egnn_embed.py      # Foldclass network
programs/Foldclass/my_egnn_nocoords.py           # EGNN layers
programs/utils.py                                 # Output formatting
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-06
**Generated By**: Claude Code
