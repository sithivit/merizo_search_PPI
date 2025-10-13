# RosettaStone: Protein-Protein Interaction Prediction

Predicts protein-protein interactions via structural domain fusion analysis using the Rosetta Stone method.

## Overview

The Rosetta Stone method identifies protein-protein interactions by finding domain fusion patterns in multi-domain proteins. If domains A and B appear fused together in one organism, they likely interact when they appear as separate proteins in another organism.

## Installation

Install additional dependencies:

```bash
pip install -r requirements_rosetta.txt
```

## Quick Start

### 1. Build Fusion Database

Build a database from multi-domain proteins:

```bash
python merizo.py rosetta build \
    /path/to/pdb_files/ \
    fusion_db/ \
    --device cuda
```

### 2. Search for Interactions

Search for interactions in a query protein:

```bash
python merizo.py rosetta search \
    query_protein.pdb \
    fusion_db/ \
    results \
    --validate-tm \
    --device cuda
```

## Detailed Usage

### Building Fusion Database

```bash
python merizo.py rosetta build INPUT OUTPUT [OPTIONS]
```

**Arguments:**
- `INPUT`: Directory of PDB files or file list
- `OUTPUT`: Output directory for fusion database

**Options:**
- `--min-domains N`: Minimum domains per protein (default: 2)
- `--device DEVICE`: cpu, cuda, or mps (default: cuda)
- `--skip-promiscuity`: Skip promiscuity index building
- `--promiscuity-threshold N`: Promiscuity threshold (default: 25)

**Example:**
```bash
# Build from directory
python merizo.py rosetta build alphafold_structures/ fusion_db/ --device cuda

# Build from file list
python merizo.py rosetta build pdb_list.txt fusion_db/ --min-domains 3
```

### Searching for Interactions

```bash
python merizo.py rosetta search QUERY DATABASE OUTPUT [OPTIONS]
```

**Arguments:**
- `QUERY`: Query PDB file
- `DATABASE`: Fusion database directory
- `OUTPUT`: Output file prefix

**Options:**
- `--cosine-threshold F`: Cosine similarity threshold (default: 0.7)
- `--top-k N`: Number of top matches (default: 20)
- `--validate-tm`: Validate with TM-align
- `--min-tm-score F`: Minimum TM-score (default: 0.5)
- `--fastmode`: Use fast TM-align
- `--skip-filter`: Skip promiscuity filtering
- `--device DEVICE`: cpu, cuda, or mps (default: cuda)

**Example:**
```bash
# Basic search
python merizo.py rosetta search query.pdb fusion_db/ results --device cuda

# Search with TM-align validation
python merizo.py rosetta search query.pdb fusion_db/ results \
    --validate-tm \
    --min-tm-score 0.5 \
    --fastmode \
    --device cuda
```

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "query": "query_protein.pdb",
  "num_predictions": 10,
  "predictions": [
    {
      "query_domain_id": "protein1_query_domain_0",
      "query_protein": "protein1",
      "query_range": [1, 150],
      "target_domain_id": "protein2_domain_1",
      "target_protein": "protein2",
      "target_range": [200, 350],
      "num_rosetta_stones": 3,
      "rosetta_stone_ids": ["fusion_protein1", "fusion_protein2"],
      "cosine_similarity": 0.85,
      "tm_score": 0.72,
      "confidence": 0.68,
      "promiscuity_filtered": false,
      "interaction_type": "inter"
    }
  ]
}
```

## Database Structure

The fusion database consists of:

- `domain_embeddings.pt`: Domain embeddings (PyTorch tensor)
- `domain_metadata.index`: Domain metadata (pickled list)
- `fusion_embeddings.pt`: Fusion link embeddings
- `fusion_metadata.index`: Fusion link metadata
- `domain_registry.pkl`: Domain registry
- `promiscuity_index.pkl`: Promiscuity scores

## Algorithm

1. **Segmentation**: Multi-domain proteins are segmented using Merizo
2. **Embedding**: Each domain is embedded using Foldclass (128-dim vectors)
3. **Fusion Links**: Domain pairs within proteins are recorded
4. **Clustering**: Domains are clustered by structural similarity (HDBSCAN)
5. **Promiscuity Scoring**: Clusters linking to many others are flagged
6. **Search**: Query domains are matched against fusion patterns using FAISS
7. **Filtering**: Promiscuous domains are filtered out
8. **Validation**: Optional TM-align structural validation

## Performance

- **Database building**: ~1-2 proteins/second (depends on domain count)
- **Search**: < 1 second for most queries (with FAISS indexing)
- **Memory**: ~1-2 GB per 10,000 proteins

## Python API

```python
from programs.RosettaStone import (
    FusionDatabaseBuilder,
    StructuralRosettaStoneSearch,
    DomainPromiscuityFilter
)
from pathlib import Path

# Build database
builder = FusionDatabaseBuilder(
    output_dir=Path('fusion_db'),
    device='cuda'
)
builder.build_from_structure_list(structure_paths)

# Search
search_engine = StructuralRosettaStoneSearch(
    fusion_db_dir=Path('fusion_db'),
    device='cuda'
)
predictions = search_engine.search_interactions(
    query_pdb_path=Path('query.pdb'),
    validate_tm=True
)

# Filter
filter_engine = DomainPromiscuityFilter(
    fusion_db_dir=Path('fusion_db')
)
filter_engine.load_promiscuity_index(Path('fusion_db/promiscuity_index.pkl'))
filtered, removed = filter_engine.filter_predictions(predictions)
```

## Citation

If you use this module, please cite:

- Merizo: Domain segmentation
- Foldclass: Structure embeddings
- Rosetta Stone method: Original concept from comparative genomics

## Troubleshooting

**"No domains found in query"**
- Check that PDB file is valid and has CA atoms
- Try adjusting `--min-domain-size` in Merizo settings

**"FAISS index error"**
- Ensure faiss-cpu or faiss-gpu is installed
- Check database was built completely

**"Promiscuity index not found"**
- Run database build without `--skip-promiscuity`
- Or use `--skip-filter` when searching

**Memory errors**
- Reduce `--top-k` parameter
- Use CPU device instead of CUDA
- Process smaller database batches
