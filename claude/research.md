# Research: Merizo-search PPI Filtering Enhancement

This document tracks research findings for implementing fast filtering on Merizo-search PPI results.

---

## Research Objectives

### 1. Understand How Merizo-search Works
- How does it access the database?
- What is the output format?
- What are the database structures?
- What files are accessed?

### 2. Optimization Approaches
- What indexing strategies are most suitable?
- Hash tables vs other indexing methods
- Performance trade-offs
- Best practices for filtering operations

---

## 1. Merizo-search System Analysis

### 1.1 Codebase Structure

**Status:** ✅ Complete

**Key files examined:**
- [x] Main entry points / command-line interface
- [x] Database access modules
- [x] Search/query logic
- [x] Output formatting
- [x] Configuration files

**Findings:**

**Primary Entry Point:** `merizo_search/merizo.py`

**Four Main Modes:**
1. **`segment`** - Domain segmentation using Merizo
   - Function: `segment(args)` (line 33-99)
   - Calls: `segment_pdb()` from `programs.Merizo.predict`

2. **`createdb`** - Create database from PDB directory
   - Function: `createdb(args)` (line 102-124)
   - Calls: `createdb_from_pdb()` from `programs.Foldclass.makedb`

3. **`search`** - Search query structures against database
   - Function: `search(args)` (line 127-227)
   - Calls: `dbsearch()` from `programs.Foldclass.dbsearch`

4. **`easy-search`** - Combined segment + search workflow
   - Function: `easy_search(args)` (line 230-413)
   - Combines segmentation and searching

**Core Modules:**
- `programs/Foldclass/dbsearch.py` (597 lines) - Search engine
- `programs/Foldclass/dbutil.py` (205 lines) - Database utilities
- `programs/Foldclass/makedb.py` (113 lines) - Database creation
- `programs/Foldclass/nndef_fold_egnn_embed.py` (62 lines) - Neural network
- `programs/utils.py` (185 lines) - Output formatting

---

### 1.2 Database Access Patterns

**Status:** ✅ Complete

**Findings:**

**Two Database Formats Supported:**

1. **PyTorch Format (Legacy, for smaller databases):**
   - Files: `{name}.pt`, `{name}.index`
   - `.pt` file: Torch tensor of shape (N_domains, 128) - embeddings
   - `.index` file: Pickle file with list of (pdb_name, coords, sequence) tuples
   - Optional: `{name}.metadata`, `{name}.metadata.index`
   - Entire database loaded into RAM

2. **Faiss Format (Modern, for large databases):**
   - Configuration: `{name}.json`
   - Binary data files: embeddings, coordinates, sequences, metadata
   - **Memory-mapped I/O** - files accessed on-demand, not loaded entirely
   - Allows searching databases larger than available RAM

**Database Detection Logic** (`dbsearch.py:52-76`):
```python
# Checks for .pt/.index first (PyTorch format)
# Falls back to .json (Faiss format)
```

**Key Database Access Functions** (`dbutil.py`):
- `read_dbinfo()` - Read JSON config
- `db_memmap()` - Memory-map binary files
- `retrieve_names_by_idx()` - Get domain names by indices
- `retrieve_bytes()` - Read bytes from memory-mapped files
- `db_iterator()` - Batch iterator for large datasets

**Current Indexing:**
- Binary index files store byte offsets (start/end positions)
- Domain names stored in fixed-width format (33 bytes each)
- No taxonomy or domain-type indexes currently exist
- All filtering happens post-search (not pre-filtered)

---

### 1.3 Database Structure

**Status:** ✅ Complete

**Examination of:** `examples/database/ted100_9606_small/`

**Database Files (Faiss Format):**

| Filename | Size | Type | Purpose |
|----------|------|------|---------|
| `ted100_9606_small_raw_128d_norm.db` | 34.3 MB | Binary (float32) | 128-dimensional normalized embeddings |
| `ted100_9606_small_ca.db` | 96.2 MB | Binary | Alpha-carbon coordinates (float32 triplets) |
| `ted100_9606_small_ca.index` | 1.0 MB | Binary | Byte offsets for coordinate data |
| `ted100_9606_small_seq.db` | 8.0 MB | Binary | Protein sequences (ASCII encoded) |
| `ted100_9606_small_seq.index` | 1.0 MB | Binary | Byte offsets for sequence data |
| `ted100_9606_small_metadata.db` | 14.5 MB | Text | JSON metadata for each domain |
| `ted100_9606_small_metadata.index` | 1.0 MB | Binary | Byte offsets for metadata |
| `ted100_9606_small_raw_128d.index_names` | 2.2 MB | Text | Domain names/identifiers |
| `ted100_9606_small.json` | 368 bytes | JSON | Database configuration |

**Total database size:** ~153 MB for 66,943 domains

**Configuration JSON Structure** (`ted100_9606_small.json`):
```json
{
  "dbfname_IP": "ted100_9606_small_raw_128d_norm.db",
  "DB_SIZE": 66943,
  "DB_DIM": 128,
  "db_names_f": "ted100_9606_small_raw_128d.index_names",
  "sif": "ted100_9606_small_seq.index",
  "sdf": "ted100_9606_small_seq.db",
  "cif": "ted100_9606_small_ca.index",
  "cdf": "ted100_9606_small_ca.db",
  "mif": "ted100_9606_small_metadata.index",
  "mdf": "ted100_9606_small_metadata.db"
}
```

**Storage Format Details:**
- **Index files:** 2 × int64 values per entry (start/end byte positions)
- **Names:** Fixed-width 33 bytes per domain identifier
- **Coordinates:** 3 × float32 per residue (x, y, z)
- **Sequences:** ASCII characters (1 byte per residue)
- **Embeddings:** 128 × float32 per domain (normalized)
- **Metadata:** Variable-length JSON strings per domain

**Metadata JSON Example:**
```json
{
  "taxonomy_id": "9606",
  "species": "Homo sapiens",
  "cath_fold": "3.40.50.300",
  "confidence": "high",
  "globularity_score": 0.85
}
```

**Key Insights:**
- Database uses **memory-mapped files** for efficiency
- **No pre-built indexes** for taxonomy or domain properties
- Metadata is stored as JSON but accessed sequentially
- Current design optimized for embedding similarity search, not filtering

---

### 1.4 Merizo-search Output Format

**Status:** ✅ Complete

**Findings:**

**Primary Output Format:** Tab-separated values (TSV)

**Search Results File** (`_search.tsv`):

**Default columns (search mode):**
```
query, emb_rank, target, emb_score, q_len, t_len, ali_len, seq_id,
q_tm, t_tm, max_tm, rmsd, metadata
```

**Default columns (easy-search mode):**
```
query, chopping, conf, plddt, emb_rank, target, emb_score, q_len, t_len,
ali_len, seq_id, q_tm, t_tm, max_tm, rmsd, metadata
```

**Column Descriptions:**
- `query` - Query domain/protein identifier
- `target` - Database hit domain identifier
- `emb_rank` - Rank based on embedding similarity
- `emb_score` - Cosine similarity score (0-1)
- `q_len`, `t_len` - Query and target lengths (residues)
- `ali_len` - Alignment length
- `seq_id` - Sequence identity (%)
- `q_tm`, `t_tm` - Query and target TM-align scores
- `max_tm` - Maximum of q_tm and t_tm
- `rmsd` - Root mean square deviation
- `metadata` - JSON metadata from database (includes taxonomy, CATH fold, etc.)
- `chopping` - Domain boundaries (easy-search only)
- `conf`, `plddt` - Confidence scores (easy-search only)

**Example Search Result Row:**
```tsv
AF-Q9Y6K1-F1_TED01	1	AF-P12345-F1_TED02	0.87	150	145	140	35.2	0.92	0.89	0.92	1.2	{"taxonomy_id":"9606","cath_fold":"3.40.50.300","confidence":"high"}
```

**Optional Outputs:**
- `_search_insignificant.tsv` - Hits below `--mintm` threshold
- `_search.tsv.hit_metadata.json` - Structured metadata (if `--metadata_json` flag)
- `_segment.tsv` - Segmentation results (easy-search mode)
- `_search_multi_dom.tsv` - Multi-domain alignment results (if enabled)

**Multi-Domain Search Results:**
- Columns: `query_chain, nqd, hit_chain, nhd, match_category, match_info, hit_metadata`
- Match categories:
  - 0: Bag-of-domains (unordered)
  - 1: Gapped alignment with end gaps
  - 2: Gapped alignment without interstitial gaps
  - 3: Exact multi-domain alignment match

**Key Insight for Filtering:**
The metadata column contains JSON with taxonomy_id, cath_fold, confidence, etc. This is the data we need to index for fast filtering!

---

### 1.5 File Access Patterns

**Status:** ✅ Complete

**Findings:**

**Search Workflow File Access:**

1. **Database loading** (happens once per search session):
   - Read `.json` config file
   - Memory-map all binary files:
     - Embeddings DB (`.db`)
     - Coordinates DB + index (`.ca.db`, `.ca.index`)
     - Sequences DB + index (`.seq.db`, `.seq.index`)
     - Metadata DB + index (`.metadata.db`, `.metadata.index`)
     - Domain names file (`.index_names`)

2. **Query processing** (per query):
   - Read query PDB file
   - Generate embedding (forward pass through neural network)
   - Search embedding database (sequential read of embedding file in blocks)
   - Retrieve metadata for top hits (random access via memory-mapped files)
   - Retrieve coordinates for TM-align (random access via memory-mapped files)

3. **TM-align verification** (per hit):
   - Write temporary PDB files
   - Execute TM-align subprocess
   - Read TM-align output
   - Parse scores

**Memory-Mapped File Access:**
- Implemented in `dbutil.py:28-30` using `numpy.memmap()`
- Allows random access without loading entire file
- OS handles caching automatically
- Critical for large databases (>RAM size)

**Batch Processing:**
- Embeddings searched in blocks (default: 262,144 domains at a time)
- Reduces peak memory usage
- Trade-off: more I/O vs less memory

**I/O Bottlenecks:**
1. **Metadata retrieval** - Currently sequential JSON parsing
   - Each metadata lookup requires:
     - Index read (find byte offsets)
     - DB read (retrieve JSON string)
     - JSON parsing
   - No caching of parsed metadata

2. **No pre-filtering** - Must search entire database
   - Cannot skip domains that don't match taxonomy/domain criteria
   - All filtering happens after similarity search

**Critical Finding - No Subset Loading:**

**Question:** Does Merizo-search load only a subset of the database when searching?

**Answer:** **NO** - It ALWAYS searches the entire database, regardless of filter criteria.

**Evidence from code** (`dbsearch.py`):

1. **Line 285-286:** Entire database is memory-mapped and iterated
   ```python
   dbmm = db_memmap(filename=dbfname, shape=(dbinfo['DB_SIZE'], dbinfo['DB_DIM']))
   dbi = db_iterator(dbmm, search_batchsize)  # Iterator over ENTIRE database
   ```

2. **Line 332:** Search processes entire database iterator
   ```python
   D, I = knn_exact_faiss(query_embeddings.cpu(), dbi, topk, ...)
   # dbi iterates through ALL domains in blocks
   ```

3. **Line 244-253:** Inside `knn_exact_faiss()`, loops through ALL blocks:
   ```python
   for xbi in db_iterator:  # Processes every single block
       index.add(xbi)
       D, I = index.search(xq, k)
       rh.add_result(D, I)
   ```

4. **Developer TODO Comment (Line 323-326):**
   ```python
   # TODO: SMK this strategy does not allow us to easily
   # implement length/coverage filtering as we have for pytorch version.
   # Another way is to apply mincov filter in post for both versions
   ```

   This confirms filtering is designed to happen POST-search, not pre-search.

**What filtering exists:**
- ✅ Post-search cosine similarity filter (`mincos >= 0.5`)
- ✅ Post-search TM-align filter (`mintm >= 0.5`)
- ❌ NO pre-filtering by taxonomy
- ❌ NO pre-filtering by domain type/fold
- ❌ NO pre-filtering by confidence level
- ❌ NO way to skip irrelevant domains during search

**Why this matters:**
- If searching for human proteins only (TaxID=9606), Merizo-search still:
  - Loads embeddings for ALL 66,943 domains (or all 365M in full database)
  - Computes similarity against ALL domains
  - Only filters by taxonomy AFTER similarity computation

**Computational waste example:**
- Database: 66,943 domains
- Human domains (TaxID=9606): ~30,000 domains (~45%)
- Wasted computation: ~37,000 unnecessary similarity calculations (~55%)
- For 365M domain database: potentially billions of wasted calculations!

**Opportunity for Optimization:**
Build auxiliary index files that map:
- taxonomy_id → list of domain indices
- cath_fold → list of domain indices
- confidence → list of domain indices

Then modify search to:
1. Apply filters BEFORE similarity search
2. Create filtered database iterator with only relevant indices
3. Search only the subset
4. Massive performance improvement for filtered queries

---

## 2. Indexing & Optimization Research

### 2.1 Hash Table Approaches

**Use case:** Domain/Protein ID → Metadata lookups

**Status:** ✅ Complete

**Python Hash Table Options:**

| Implementation | Lookup Time | Memory | Persistence | Best For |
|----------------|-------------|---------|-------------|----------|
| `dict` | O(1) avg | High | No | In-memory, frequently accessed |
| `collections.defaultdict` | O(1) avg | High | No | Same as dict, with default values |
| `shelve` | O(1) avg | Low | Yes | Persistent, infrequent access |
| `dbm` | O(1) avg | Low | Yes | Simple key-value persistence |
| `sqlite3` | O(1)* | Low | Yes | Complex queries, transactions |

*With proper indexing

**Findings and Recommendations:**

**1. In-Memory Hash Table (Python `dict`):**
```python
# Build once at startup
metadata_cache = {}
for idx in range(db_size):
    domain_id = get_domain_name(idx)
    metadata = parse_metadata(idx)
    metadata_cache[domain_id] = {
        'taxonomy_id': metadata['taxonomy_id'],
        'cath_fold': metadata.get('cath_fold'),
        'confidence': metadata['confidence'],
        'index': idx  # Store index for later retrieval
    }

# O(1) lookups
domain_meta = metadata_cache[domain_id]
```

**Pros:**
- Fastest possible lookup (O(1))
- No disk I/O after initial load
- Simple implementation

**Cons:**
- Memory usage: ~1KB per domain × 66,943 domains = ~65 MB (for example DB)
- For 365M domains: ~365 GB (too large!)
- Must rebuild on every run

**2. SQLite with Indexes:**
```python
import sqlite3

# Build once
conn = sqlite3.connect('metadata_index.db')
conn.execute('''
    CREATE TABLE domains (
        domain_id TEXT PRIMARY KEY,
        index_num INTEGER,
        taxonomy_id INTEGER,
        cath_fold TEXT,
        confidence TEXT
    )
''')
conn.execute('CREATE INDEX idx_taxonomy ON domains(taxonomy_id)')
conn.execute('CREATE INDEX idx_cath ON domains(cath_fold)')

# Fast lookups
cursor.execute('SELECT * FROM domains WHERE domain_id = ?', (domain_id,))
```

**Pros:**
- Persistent (build once, use many times)
- Supports complex queries
- Indexes for fast lookups
- Reasonable memory usage

**Cons:**
- Slower than in-memory dict (disk I/O)
- Still need to build the index initially

**3. Hybrid Approach (RECOMMENDED):**
```python
# Pre-build SQLite index with all metadata
# At runtime, load only the subset needed into memory

# Load taxonomy subset into memory
taxonomy_filter = 9606  # Human
cursor.execute('SELECT domain_id, index_num FROM domains WHERE taxonomy_id = ?',
               (taxonomy_filter,))
subset_dict = {row[0]: row[1] for row in cursor.fetchall()}

# Now subset_dict is small enough for memory
# Contains only relevant domains for fast filtering
```

**Recommendation:**
- Build persistent SQLite index once
- Load filtered subsets into memory dicts for ultra-fast lookups
- Best of both worlds: persistence + speed

---

### 2.2 Inverted Index Approaches

**Use case:** Taxonomy/Domain → Protein IDs

**Considerations:**
- Index building time vs query time trade-off
- Storage overhead
- Update frequency
- Compression

**Approaches to evaluate:**
- Simple dict of lists: `{taxid: [protein_ids]}`
- Sorted arrays with binary search
- B-tree indexes
- Specialized libraries (e.g., `whoosh`, `pyterrier`)

**Findings:**

*(To be filled as research progresses)*

---

### 2.3 Bitmap Index Approaches

**Use case:** Categorical filters (confidence: high/medium)

**Considerations:**
- Memory efficiency for categorical data
- Fast boolean operations (AND/OR for combined filters)
- Sparse vs dense bitmaps

**Approaches to evaluate:**
- Simple boolean arrays (NumPy)
- Compressed bitmaps (e.g., `roaringbitmap`)
- Database bitmap indexes (SQLite, PostgreSQL)

**Findings:**

*(To be filled as research progresses)*

---

### 2.4 Hybrid Approaches

**Combining multiple indexing strategies:**
- Hash table for primary lookups
- Inverted indexes for filtering
- Bitmaps for categorical combinations

**Architecture considerations:**
- Pre-build indexes vs build-on-demand
- Index update frequency
- Memory vs disk trade-offs
- Index serialization format

**Findings:**

*(To be filled as research progresses)*

---

### 2.5 Performance Benchmarking Plan

**Metrics to measure:**
- Index building time
- Index size (memory/disk)
- Single filter query time
- Combined filter query time
- Scalability with data size

**Test cases:**
- Filter by single taxonomy
- Filter by single CATH fold
- Filter by confidence
- Combined: taxonomy + domain + confidence

**Findings:**

*(To be filled as research progresses)*

---

## 3. Existing Solutions & Literature

### 3.1 Similar Systems

**Systems to examine:**
- How do other PPI databases handle filtering? (STRING, BioGRID, IntAct)
- How do protein databases handle queries? (UniProt, PDB)
- How do domain databases handle filtering? (Pfam, InterPro, CATH)

**Findings:**

*(To be filled as research progresses)*

---

### 3.2 Relevant Papers/Documentation

**Topics to research:**
- Indexing strategies for biological databases
- Fast filtering for large-scale protein data
- Domain-based PPI prediction methods

**References:**

*(To be filled as research progresses)*

---

## 4. Decision Matrix

### Indexing Strategy Comparison

| Strategy | Use Case | Lookup Time | Build Time | Memory | Disk | Complexity |
|----------|----------|-------------|------------|--------|------|------------|
| Hash Table | Protein→Metadata | O(1) | O(n) | High | Low | Low |
| Inverted Index | Tax/Domain→Proteins | O(1)-O(log n) | O(n) | Medium | Medium | Medium |
| Bitmap Index | Categorical filters | O(n/w)* | O(n) | Low** | Low | Medium |
| B-tree | Range queries | O(log n) | O(n log n) | Medium | Medium | High |

*w = word size for bitmap operations
**If compressed

---

### Detailed Strategy Explanations with Examples

#### Strategy 1: Hash Table (Direct Lookups)

**Use Case:** You have a domain/protein ID and want to quickly get its metadata.

**Example Scenario:**
```
Input: "AF-Q9Y6K1-F1_TED01" (domain ID from search result)
Question: "What's the taxonomy and CATH fold for this domain?"
```

**How it works:**
```python
# Pre-built hash table (Python dict)
metadata_hash = {
    "AF-Q9Y6K1-F1_TED01": {
        "index": 1234,
        "taxonomy_id": 9606,
        "cath_fold": "3.40.50.300",
        "confidence": "high"
    },
    "AF-P12345-F1_TED02": {
        "index": 5678,
        "taxonomy_id": 10090,
        "cath_fold": "2.60.40.10",
        "confidence": "medium"
    },
    # ... 66,943 entries total
}

# O(1) lookup
result = metadata_hash["AF-Q9Y6K1-F1_TED01"]
# Returns: {"index": 1234, "taxonomy_id": 9606, ...}
```

**When to use:** Post-search filtering - you already have domain IDs and need their metadata fast.

**NOT suitable for:** Pre-filtering - can't answer "give me ALL domains with taxonomy 9606" efficiently (would need to scan entire hash table).

---

#### Strategy 2: Inverted Index (Filter → Domain List)

**Use Case:** You want to filter domains by a property BEFORE searching.

**Example Scenario:**
```
Question: "Which domains have CATH fold 3.40.50.300?"
Answer: "Give me the list of domain indices so I can search only those."
```

**How it works:**
```python
# Pre-built inverted indexes
taxonomy_index = {
    9606: [0, 1, 5, 12, 15, 23, ...],      # Human: indices of all human domains
    10090: [2, 3, 4, 8, 10, ...],          # Mouse
    7227: [6, 7, 9, 11, ...],              # Fruit fly
    # ... all taxonomies
}

cath_index = {
    "3.40.50.300": [1, 15, 42, 103, ...],  # Indices of domains with this fold
    "2.60.40.10": [2, 8, 25, 67, ...],     # Different fold
    "1.10.8.10": [5, 12, 33, ...],
    # ... all CATH folds
}

confidence_index = {
    "high": [0, 1, 2, 5, 8, 12, ...],      # High confidence domains
    "medium": [3, 4, 6, 7, 9, ...],        # Medium confidence
}
```

**Example: Filter for specific CATH fold BEFORE searching:**
```python
# User wants: "Search only domains with CATH fold 3.40.50.300"

# Step 1: Look up which domains have this fold
target_cath = "3.40.50.300"
domain_indices = cath_index[target_cath]  # [1, 15, 42, 103, ...]
# Returns in O(1) time!

# Step 2: Create filtered database iterator
# Only load embeddings for these specific indices
filtered_embeddings = embeddings[domain_indices]  # Only these domains!

# Step 3: Search ONLY the filtered subset
D, I = faiss_search(query, filtered_embeddings)

# Result: Searched 1,000 domains instead of 66,943!
# 98.5% reduction in computation!
```

**Example: Combined filters (taxonomy + CATH fold):**
```python
# User wants: "Search only human domains with CATH fold 3.40.50.300"

# Step 1: Get indices for each filter
human_indices = taxonomy_index[9606]           # [0, 1, 5, 12, 15, 23, ...]
cath_indices = cath_index["3.40.50.300"]      # [1, 15, 42, 103, ...]

# Step 2: Find intersection (domains that match BOTH filters)
filtered_indices = set(human_indices) & set(cath_indices)
# Result: [1, 15]  (only 2 domains match both!)

# Step 3: Search only these 2 domains
filtered_embeddings = embeddings[list(filtered_indices)]
D, I = faiss_search(query, filtered_embeddings)

# Result: Searched 2 domains instead of 66,943!
# 99.997% reduction!
```

**When to use:** Pre-filtering - you want to search only a subset of the database.

**Performance:** O(1) to lookup each index, O(k) to combine filters (k = number of filters).

---

#### Strategy 3: Bitmap Index (Fast Boolean Operations)

**Use Case:** Combining multiple categorical filters very quickly.

**Example Scenario:**
```
Question: "Give me domains that are:
  - High confidence AND
  - Human (9606) AND
  - CATH fold 3.40.50.300"
```

**How it works:**
```python
import numpy as np

# Pre-built bitmap indexes (one bit per domain)
# Database has 66,943 domains = 66,943 bits per bitmap

# Taxonomy bitmap (one bitmap per taxonomy)
is_human = np.array([1,1,0,0,0,1,0,0,1,1,0,1,1,0,0,1,...])  # 66,943 bits
is_mouse = np.array([0,0,1,1,1,0,1,1,0,0,1,0,0,1,1,0,...])

# CATH fold bitmap (one bitmap per fold)
is_fold_3_40_50_300 = np.array([0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,...])

# Confidence bitmap
is_high_confidence = np.array([1,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1,...])
is_medium_confidence = np.array([0,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,...])
```

**Example: Fast boolean filter combination:**
```python
# User wants: High confidence AND Human AND CATH fold 3.40.50.300

# Combine using bitwise AND (VERY fast - single CPU instruction per word)
result = is_high_confidence & is_human & is_fold_3_40_50_300
# result = [0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,...]
#          Only domains 1, 9, and 12 match all criteria!

# Convert bitmap to indices
filtered_indices = np.where(result == 1)[0]  # [1, 9, 12]

# Search only these domains
filtered_embeddings = embeddings[filtered_indices]
```

**Performance comparison:**

| Method | Time for 3 filters on 66,943 domains |
|--------|--------------------------------------|
| Inverted Index (set intersection) | ~5-10 ms |
| Bitmap Index (bitwise AND) | ~0.1 ms |

**When to use:** Many categorical filters that change frequently, need ultra-fast combination.

**Trade-off:** Requires more disk space (one bitmap per category value), but queries are incredibly fast.

---

#### Strategy 4: B-tree (Range Queries)

**Use Case:** Filtering by numeric ranges (less common for this project).

**Example Scenario:**
```
Question: "Give me domains with globularity score between 0.8 and 0.95"
```

**How it works:**
```python
# B-tree index on globularity_score
# (Conceptual - usually implemented by database like SQLite)

# Tree structure allows efficient range queries
results = btree_range_query(
    field="globularity_score",
    min_value=0.8,
    max_value=0.95
)
# Returns domain indices with scores in range
```

**When to use:** Numeric range filters (globularity score, domain length, etc.).

**For this project:** Less critical since most filters are categorical (taxonomy, fold, confidence).

---

### Recommended Hybrid Architecture for Your Project

**Goal:** Filter domains by taxonomy and/or CATH fold BEFORE searching.

**Recommended approach:**

```python
# 1. Build once (preprocessing step)
# Create SQLite database with indexes

import sqlite3
conn = sqlite3.connect('domain_filters.db')

# Create table
conn.execute('''
    CREATE TABLE domains (
        domain_idx INTEGER PRIMARY KEY,
        domain_id TEXT,
        taxonomy_id INTEGER,
        cath_fold TEXT,
        confidence TEXT,
        globularity REAL
    )
''')

# Create indexes (these ARE inverted indexes under the hood)
conn.execute('CREATE INDEX idx_taxonomy ON domains(taxonomy_id)')
conn.execute('CREATE INDEX idx_cath ON domains(cath_fold)')
conn.execute('CREATE INDEX idx_confidence ON domains(confidence)')

# Populate with metadata from Merizo database
for idx in range(66943):
    metadata = parse_metadata(idx)
    conn.execute('INSERT INTO domains VALUES (?, ?, ?, ?, ?, ?)',
                 (idx, metadata['domain_id'], metadata['taxonomy_id'], ...))

# 2. At query time - get filtered indices BEFORE searching

# Example: User wants human domains with specific CATH fold
cursor = conn.execute('''
    SELECT domain_idx FROM domains
    WHERE taxonomy_id = 9606 AND cath_fold = '3.40.50.300'
''')
filtered_indices = [row[0] for row in cursor.fetchall()]
# Returns: [15, 42, 103] in ~1ms using the indexes!

# 3. Modify Merizo-search database iterator
# Instead of iterating ALL domains:
# dbi = db_iterator(dbmm, search_batchsize)  # OLD

# Create filtered iterator:
dbi_filtered = db_iterator_filtered(dbmm, filtered_indices, search_batchsize)  # NEW

# 4. Search only the filtered subset
D, I = knn_exact_faiss(query, dbi_filtered, topk)
```

**Why this works best:**
- ✅ **Persistent** - build once, use many times
- ✅ **Fast queries** - SQLite indexes are optimized inverted indexes
- ✅ **Flexible** - easy to add new filter types
- ✅ **Familiar** - SQL queries are intuitive
- ✅ **Low memory** - indexes stay on disk, only load filtered subset

**Performance gain example:**
```
Database size: 66,943 domains
Filter: taxonomy_id = 9606 (human)
Result: ~30,000 domains

Without pre-filtering: Search 66,943 domains (~100% work)
With pre-filtering: Search 30,000 domains (~45% work)
Speedup: ~2.2x

Filter: taxonomy_id = 9606 AND cath_fold = '3.40.50.300'
Result: ~50 domains

Without pre-filtering: Search 66,943 domains (100% work)
With pre-filtering: Search 50 domains (~0.07% work)
Speedup: ~1,340x !!!
```

---

### Summary: Which Strategy When?

| Your Need | Best Strategy | Implementation |
|-----------|---------------|----------------|
| "Get metadata for domain ID X" | Hash Table | In-memory dict for fast lookups |
| "Find all domains with taxonomy Y" | Inverted Index | SQLite with index on taxonomy_id |
| "Find domains with CATH fold Z" | Inverted Index | SQLite with index on cath_fold |
| "Combine multiple filters" | Inverted Index (SQL) | Single SQL query with multiple WHERE clauses |
| "Ultra-fast filter combinations (>10 filters)" | Bitmap Index | NumPy bitwise operations |
| "Filter by score range" | B-tree (SQL) | SQLite with index + BETWEEN query |

**Preliminary recommendation:**

**Use SQLite with B-tree indexes** (which internally implement inverted indexes) for the filtering system. This provides:
- Fast pre-filtering queries (< 10ms)
- Persistent storage (no rebuild needed)
- Easy integration with existing Python code
- Scalable to 365M domains with proper indexing

---

## 5. Integration Points

### 5.1 Where to Hook into Merizo-search

**Potential integration points:**
- Post-processing filter on results
- Pre-filter on database before search
- Integrated filter during search
- Separate filtering module/API

**Findings:**

*(To be filled as research progresses)*

---

### 5.2 API Design Considerations

**Interface options:**
- Python function calls
- Command-line wrapper
- Configuration file
- REST API

**Findings:**

*(To be filled as research progresses)*

---

## 6. Next Steps

**Immediate actions:**
1. [ ] Explore Merizo-search codebase structure
2. [ ] Examine `examples/database/ted100_9606_small/` directory
3. [ ] Run Merizo-search with example queries (if possible)
4. [ ] Document findings above
5. [ ] Prototype simple hash table lookup
6. [ ] Benchmark different indexing approaches
7. [ ] Propose final architecture

---

## Research Log

*(Chronological notes as research progresses)*

### [DATE] - Initial Setup
- Created research.md framework
- Defined research objectives
- Ready to begin codebase exploration

---
