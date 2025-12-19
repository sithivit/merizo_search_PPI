# System Relationship: Merizo-search vs Your PPI Framework

## Quick Answer

**Your PPI query framework is a SEPARATE system that CALLS Merizo-search when needed.**

Think of it like this:
- **Merizo-search** = Google Search (finds things)
- **Your framework** = A research assistant (filters results and answers specific questions)

---

## Concrete Example: User Query Flow

### Scenario: User wants PPIs for human proteins with domains similar to a query structure

```
USER QUERY:
"Show me interactions between human proteins that have domains
 similar to my protein structure (query.pdb)"

                    ↓

┌─────────────────────────────────────────────────────────────┐
│  YOUR PPI FRAMEWORK processes this                          │
└─────────────────────────────────────────────────────────────┘

Step 1: Your framework says "I need to find similar domains"
        ↓ (calls Merizo-search)

┌─────────────────────────────────────────────────────────────┐
│  MERIZO-SEARCH runs:                                        │
│  $ merizo-search search query.pdb --db ted100_9606_small   │
│                                                             │
│  Returns:                                                   │
│  - AF-Q96HM7-F1-model_v4_TED01 (similarity: 0.85)          │
│  - AF-P12345-F1-model_v4_TED02 (similarity: 0.78)          │
│  - AF-Q99999-F1-model_v4_TED01 (similarity: 0.72)          │
│  ... (100 more results)                                     │
└─────────────────────────────────────────────────────────────┘

        ↓ (results returned to your framework)

Step 2: Your framework processes Merizo results
        - Extracts protein IDs: {AF-Q96HM7-F1, AF-P12345-F1, AF-Q99999-F1, ...}
        - Checks taxonomy: filters to only tax_id=9606 (human)
        - Result: P' = {AF-Q96HM7-F1, AF-P12345-F1} (2 human proteins)

Step 3: Your framework queries YOUR PPI database
        - Looks up: which PPIs involve proteins in P'?
        - Finds: AF-Q96HM7-F1 <-> AF-P12345-F1 (confidence: 0.9)

                    ↓

FINAL ANSWER TO USER:
"Found 1 interaction between 2 human proteins with similar domains:
 - AF-Q96HM7-F1 interacts with AF-P12345-F1"
```

---

## File Organization

### Option 1: Separate Repository (Recommended for clean separation)

```
Your_Computer/
├── merizo_search/                    ← Existing (don't modify)
│   ├── merizo_search/
│   │   └── programs/
│   │       └── Foldclass/
│   ├── examples/
│   │   └── database/                 ← Merizo's databases
│   │       ├── ted100_9606_small/
│   │       └── cath-dataset.../
│   └── ...
│
└── ppi_query_framework/              ← Your new project
    ├── src/
    │   ├── models.py
    │   ├── indexing.py
    │   ├── query_engine.py
    │   └── merizo_integration.py     ← Wrapper that calls merizo-search
    ├── data/                         ← Your data sources
    │   ├── domain_summary.tsv        ← From /mnt/bigstore/ted/...
    │   └── string_ppis.tsv           ← PPI database you download
    ├── indices/                      ← Your built indices
    │   ├── taxonomy.idx
    │   ├── domain.idx
    │   └── ppi.idx
    ├── ppi_query.py                  ← Your CLI
    └── implementation.md
```

**How they connect:**
```python
# In your ppi_query_framework/src/merizo_integration.py
from merizo_search.programs.Foldclass.dbsearch import run_dbsearch  # Import!

class MerizoSearchIntegration:
    def search_similar_domains(self, query_pdb: str):
        # Call Merizo-search function
        results = run_dbsearch(...)
        return results
```

### Option 2: Subdirectory (If you want everything together)

```
merizo_search/                        ← Existing repo
├── merizo_search/                    ← Merizo code (don't modify)
├── examples/
├── ppi_query_framework/              ← Your new subdirectory
│   ├── src/
│   ├── data/
│   └── indices/
└── implementation.md
```

**I recommend Option 1** for cleaner separation and easier version control.

---

## Data Sources

### Data That Already Exists (You'll use as-is)

**Merizo-search domain database** (already built):
- Location: `examples/database/ted100_9606_small/`
- Contains: Domain embeddings, sequences, coordinates
- Purpose: Domain similarity search
- **You use this**: Pass path to Merizo-search when doing similarity queries

### Data You Need to Obtain

**1. Domain Summary File** (metadata about all domains):
- Original location: `/mnt/bigstore/ted/ted_365.domain_summary.cath.globularity.taxid.tsv`
- Contains: Domain IDs, taxonomy, CATH labels, quality scores
- **You need this**: To build your taxonomy and domain indices
- Size: ~10-50 GB (depending on version)
- **Action**: Download or get access from your supervisor

**2. PPI Database** (protein interactions):
- Sources: STRING, IntAct, BioGRID, etc.
- Contains: Protein pairs, confidence scores, evidence
- **You need this**: To build your PPI index
- Size: 100 MB - 10 GB (depending on scope)
- **Action**: Download from public databases
  - STRING: https://string-db.org/cgi/download
  - Example: `9606.protein.links.v12.0.txt` (human PPIs)

---

## What You Build

### 1. Parsers (Read external data)
```python
# src/parsers.py
def parse_domain_summary(tsv_file):
    """Parse /mnt/bigstore/ted/ted_365.domain_summary... file"""
    # Returns: List[DomainInstance]

def parse_ppi_records(string_file):
    """Parse STRING database file"""
    # Returns: List[PPIRecord]
```

### 2. Indices (Your databases)
```python
# Built once from parsed data, saved to disk
indices/
├── taxonomy.idx        # TaxID → Protein IDs
├── domain.idx          # CATH fold → Protein IDs
└── ppi.idx             # Protein pairs with metadata
```

### 3. Query Engine (Your main logic)
```python
# src/query_engine.py
class QueryEngine:
    def query(self, constraints):
        # 1. Look up relevant proteins from YOUR indices
        # 2. Optionally call Merizo-search for similarity
        # 3. Filter PPIs from YOUR PPI index
        # 4. Return results
```

### 4. Merizo Integration (Wrapper)
```python
# src/merizo_integration.py
from merizo_search.programs.Foldclass.dbsearch import run_dbsearch

class MerizoSearchIntegration:
    def __init__(self, merizo_db_path):
        # Points to: examples/database/ted100_9606_small/...
        self.db_path = merizo_db_path

    def search_similar_domains(self, query_pdb):
        # Calls Merizo-search's run_dbsearch()
        # Returns protein IDs with similar domains
```

---

## Key Differences

| Aspect | Merizo-search | Your PPI Framework |
|--------|---------------|-------------------|
| **Purpose** | Find structurally similar domains | Filter PPIs by biological constraints |
| **Input** | PDB file (protein structure) | Query constraints (taxonomy, domains) |
| **Output** | List of similar domains with scores | List of filtered PPI records |
| **Database** | Domain embeddings (128D vectors) | Taxonomy + Domain + PPI indices |
| **Technology** | FAISS, PyTorch, neural networks | Python dicts/SQLite, set operations |
| **Your role** | **Use as-is** (external tool) | **Build from scratch** (your project) |

---

## Workflow Summary

### Without Structure Similarity (Simple)
```
User: "Show human-human PPIs where both proteins have CATH fold 3.40.50.300"

Your Framework:
1. Query taxonomy index → human proteins
2. Query domain index → proteins with that CATH fold
3. Intersect → relevant protein set P'
4. Query PPI index → filter to P' × P'
5. Return results

(No need to call Merizo-search!)
```

### With Structure Similarity (Advanced)
```
User: "Show PPIs for human proteins similar to my query structure"

Your Framework:
1. Call Merizo-search: search query.pdb → similar domain IDs
2. Map domain IDs → protein IDs
3. Query taxonomy index → human proteins
4. Intersect → relevant protein set P'
5. Query PPI index → filter to P' × P'
6. Return results

(Merizo-search used only in step 1!)
```

---

## Installation Setup

### Step 1: Install Merizo-search (if not already done)
```bash
cd merizo_search/
pip install -e .  # Install in editable mode
```

### Step 2: Create your project
```bash
cd ..  # Go up one level
mkdir ppi_query_framework
cd ppi_query_framework
```

### Step 3: Install dependencies
```bash
# Create requirements.txt
cat > requirements.txt << EOF
numpy>=1.20
pandas>=1.3
# merizo-search already installed above
EOF

pip install -r requirements.txt
```

### Step 4: Import Merizo-search in your code
```python
# In your code:
from merizo_search.programs.Foldclass.dbsearch import run_dbsearch
# This works because merizo-search is installed!
```

---

## Summary

✅ **Separate systems**: Merizo-search and your PPI framework are different projects

✅ **Your framework uses Merizo-search**: Like using a library/tool

✅ **Two different databases**:
- Merizo has domain embeddings (already exists)
- You build taxonomy/domain/PPI indices (your work)

✅ **Data flow**:
- Merizo answers: "What domains are similar?"
- Your framework answers: "What PPIs exist among relevant proteins?"

✅ **Your project scope**:
- Build parsers for domain summary + PPI files
- Build three indices (taxonomy, domain, PPI)
- Build query engine with filtering logic
- Optionally integrate Merizo-search for similarity queries

Does this clarify the relationship? The key insight is: **Merizo-search is a tool you call; your framework is the system that orchestrates everything.**
