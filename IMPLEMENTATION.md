# CORRECTED Implementation Guide: Structural Rosetta Stone Search for Merizo-Search

**Target:** Claude Code
**Purpose:** Extend merizo-search to predict protein-protein interactions via structural domain fusion analysis
**Date:** January 2025
**Version:** 2.0 (Corrected)

---

## CRITICAL CORRECTIONS FROM ORIGINAL

This corrected guide fixes major API mismatches with the actual merizo-search codebase:

1. **Merizo API**: Uses correct `run_merizo()` function and `Merizo` network class
2. **Foldclass API**: Uses `FoldClassNet` (not non-existent `Foldclass` class)
3. **Database Format**: Follows existing `.pt`/`.index` pattern instead of only HDF5
4. **Integration**: Adds as new mode to `merizo.py` CLI rather than standalone script
5. **Dependencies**: Accurately lists required new packages

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Prerequisites & Dependencies](#prerequisites--dependencies)
3. [Data Structures](#data-structures)
4. [Module 1: Fusion Database Builder](#module-1-fusion-database-builder)
5. [Module 2: Structural Rosetta Stone Search](#module-2-structural-rosetta-stone-search)
6. [Module 3: Promiscuous Domain Filter](#module-3-promiscuous-domain-filter)
7. [Module 4: Integration Layer](#module-4-integration-layer)
8. [Testing Strategy](#testing-strategy)

---

## Architecture Overview

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING (One-time)                      │
├─────────────────────────────────────────────────────────────────┤
│  Structure Database (AlphaFold, PDB)                            │
│              ↓                                                   │
│  Merizo Segmentation → Domain boundaries (domain_ids tensor)   │
│              ↓                                                   │
│  Extract Domain Coords → Per-domain CA coordinates             │
│              ↓                                                   │
│  Foldclass Embedding → Domain embeddings [N, 128]              │
│              ↓                                                   │
│  Build Fusion Database → Multi-domain proteins indexed          │
│              ↓                                                   │
│  Calculate Promiscuity Scores → Domain cluster link counts     │
│              ↓                                                   │
│  Save to disk: fusion_db.pt/.index, promiscuity_index.pkl     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      QUERY TIME (Fast)                           │
├─────────────────────────────────────────────────────────────────┤
│  Query Protein Structure                                         │
│              ↓                                                   │
│  Merizo Segmentation → Query domains                            │
│              ↓                                                   │
│  Foldclass Embedding → Query embeddings                         │
│              ↓                                                   │
│  Search Fusion Database → Find Rosetta Stone patterns          │
│              ↓                                                   │
│  Filter Promiscuous Domains → Remove low-confidence            │
│              ↓                                                   │
│  TM-align Validation → Verify structural similarity             │
│              ↓                                                   │
│  Output: Ranked interaction predictions                         │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
merizo_search/
├── programs/
│   ├── Merizo/                    # Existing
│   ├── Foldclass/                 # Existing
│   └── RosettaStone/              # NEW MODULE
│       ├── __init__.py
│       ├── data_structures.py     # Domain/FusionLink/Prediction classes
│       ├── fusion_database.py     # Build fusion DB
│       ├── rosetta_search.py      # Core search algorithm
│       ├── promiscuity_filter.py  # Filter promiscuous domains
│       └── interaction_scorer.py  # Confidence scoring
├── databases/
│   └── fusion_db/                 # NEW
│       ├── fusion_embeddings.pt   # Fusion domain embeddings
│       ├── fusion_metadata.index  # Fusion link metadata
│       ├── domain_embeddings.pt   # All domain embeddings
│       ├── domain_metadata.index  # Domain metadata
│       └── promiscuity_index.pkl  # Promiscuity scores
├── merizo.py                      # MODIFY: add 'rosetta' mode
└── programs/utils.py              # MODIFY: add result writers
```

---

## Prerequisites & Dependencies

### New Dependencies to Add

```bash
# requirements_rosetta.txt
h5py>=3.6.0                # For optional HDF5 storage
faiss-cpu>=1.7.3           # or faiss-gpu for GPU acceleration
scikit-learn>=1.0.0        # For preprocessing
hdbscan>=0.8.27            # For domain clustering
tqdm>=4.62.0               # Progress bars
```

### Installation

```bash
# Activate your merizo_search environment
conda activate merizo_search

# Install new dependencies
pip install h5py faiss-cpu hdbscan scikit-learn tqdm

# For GPU acceleration (optional):
# conda install -c pytorch -c nvidia faiss-gpu
```

### Verify Existing Installation

```bash
python -c "from programs.Merizo.model.network import Merizo; print('Merizo OK')"
python -c "from programs.Foldclass.nndef_fold_egnn_embed import FoldClassNet; print('Foldclass OK')"
```

---

## Data Structures

### Core Data Classes

```python
# programs/RosettaStone/data_structures.py

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class Domain:
    """Represents a single protein domain"""
    domain_id: str                    # Unique ID: "P12345_A_domain_1"
    protein_id: str                   # Parent protein ID
    chain_id: str                     # PDB chain
    residue_range: Tuple[int, int]    # (start, end) inclusive in original numbering
    residue_indices: np.ndarray       # Actual residue indices from PDB
    ca_coordinates: np.ndarray        # Shape: (n_residues, 3)
    sequence: str                     # Amino acid sequence
    embedding: np.ndarray             # Foldclass embedding [128]
    cluster_id: Optional[int] = None  # Structural cluster assignment
    confidence: Optional[float] = None # Merizo confidence score

    def __post_init__(self):
        """Validate data"""
        assert self.embedding.shape == (128,), f"Invalid embedding shape: {self.embedding.shape}"
        assert self.residue_range[0] <= self.residue_range[1], "Invalid residue range"
        assert len(self.ca_coordinates) == len(self.sequence), "Coords/sequence length mismatch"

    @property
    def length(self) -> int:
        return len(self.sequence)

    def overlaps(self, other: 'Domain') -> bool:
        """Check if two domains overlap in sequence"""
        if self.protein_id != other.protein_id or self.chain_id != other.chain_id:
            return False
        return not (self.residue_range[1] < other.residue_range[0] or
                   other.residue_range[1] < self.residue_range[0])


@dataclass
class FusionLink:
    """Represents a Rosetta Stone fusion event"""
    rosetta_stone_id: str             # Fusion protein ID
    domain_A: Domain                  # First domain in fusion
    domain_B: Domain                  # Second domain in fusion
    linker_length: int                # Residues between domains
    organism: Optional[str] = None    # Source organism

    def to_dict(self) -> dict:
        """Serialize for storage"""
        return {
            'rosetta_stone_id': self.rosetta_stone_id,
            'domain_A_id': self.domain_A.domain_id,
            'domain_B_id': self.domain_B.domain_id,
            'domain_A_range': self.domain_A.residue_range,
            'domain_B_range': self.domain_B.residue_range,
            'domain_A_indices': self.domain_A.residue_indices.tolist(),
            'domain_B_indices': self.domain_B.residue_indices.tolist(),
            'embedding_A': self.domain_A.embedding.tolist(),
            'embedding_B': self.domain_B.embedding.tolist(),
            'linker_length': self.linker_length,
            'organism': self.organism
        }


@dataclass
class InteractionPrediction:
    """Represents a predicted domain-domain interaction"""
    query_domain: Domain
    target_domain: Domain
    rosetta_stone_evidence: List[FusionLink]  # Supporting evidence
    cosine_similarity: float          # Embedding similarity
    tm_score: Optional[float] = None  # TM-align validation
    confidence_score: float = 0.0     # Overall confidence [0-1]
    promiscuity_flag: bool = False    # True if involves promiscuous domain
    interaction_type: str = 'inter'   # 'inter' or 'intra'

    def to_output_dict(self) -> dict:
        """Format for output file"""
        return {
            'query_domain_id': self.query_domain.domain_id,
            'query_protein': self.query_domain.protein_id,
            'query_range': self.query_domain.residue_range,
            'target_domain_id': self.target_domain.domain_id,
            'target_protein': self.target_domain.protein_id,
            'target_range': self.target_domain.residue_range,
            'num_rosetta_stones': len(self.rosetta_stone_evidence),
            'rosetta_stone_ids': [rs.rosetta_stone_id for rs in self.rosetta_stone_evidence],
            'cosine_similarity': float(self.cosine_similarity),
            'tm_score': float(self.tm_score) if self.tm_score else None,
            'confidence': float(self.confidence_score),
            'promiscuity_filtered': self.promiscuity_flag,
            'interaction_type': self.interaction_type
        }


@dataclass
class PromiscuityScore:
    """Tracks domain promiscuity metrics"""
    cluster_id: int
    num_links: int                    # How many other clusters it links to
    linked_clusters: set              # Set of cluster IDs
    is_promiscuous: bool              # True if num_links > threshold
    example_domains: List[str]        # Example domain IDs in cluster

    def get_promiscuity_ratio(self, total_clusters: int) -> float:
        """Fraction of all clusters this one links to"""
        return self.num_links / total_clusters if total_clusters > 0 else 0.0
```

---

## Module 1: Fusion Database Builder

### Purpose
Pre-process structure database to identify all multi-domain proteins and create searchable fusion database.

### Implementation

```python
# programs/RosettaStone/fusion_database.py

import os
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

# CORRECTED IMPORTS - use actual merizo-search APIs
from programs.Merizo.model.network import Merizo
from programs.Merizo.predict import segment, read_split_weight_files
from programs.Foldclass.nndef_fold_egnn_embed import FoldClassNet
from programs.Merizo.model.utils.utils import get_device

from .data_structures import Domain, FusionLink

logger = logging.getLogger(__name__)


class FusionDatabaseBuilder:
    """Builds searchable database of domain fusions from structure database"""

    def __init__(
        self,
        output_dir: Path,
        min_domains_per_protein: int = 2,
        min_linker_length: int = 0,
        max_linker_length: int = 100,
        device: str = 'cuda'
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_domains = min_domains_per_protein
        self.min_linker = min_linker_length
        self.max_linker = max_linker_length
        self.device = get_device(device)

        # Initialize Merizo network - CORRECTED
        logger.info("Loading Merizo network...")
        self.merizo = Merizo().to(self.device)
        weights_dir = os.path.join(os.path.dirname(__file__), '../Merizo/weights')
        self.merizo.load_state_dict(read_split_weight_files(weights_dir), strict=True)
        self.merizo.eval()

        # Initialize Foldclass network - CORRECTED
        logger.info("Loading Foldclass network...")
        self.foldclass = FoldClassNet(128).to(self.device).eval()
        scriptdir = os.path.dirname(os.path.realpath(__file__))
        foldclass_weights = os.path.join(scriptdir, '../Foldclass/FINAL_foldclass_model.pt')
        self.foldclass.load_state_dict(
            torch.load(foldclass_weights, map_location=lambda storage, loc: storage),
            strict=False
        )

        # Storage paths
        self.embeddings_path = self.output_dir / 'domain_embeddings.pt'
        self.metadata_path = self.output_dir / 'domain_metadata.index'
        self.fusion_embeddings_path = self.output_dir / 'fusion_embeddings.pt'
        self.fusion_metadata_path = self.output_dir / 'fusion_metadata.index'

    def build_from_structure_list(
        self,
        structure_paths: List[Path],
        batch_size: int = 32
    ) -> None:
        """
        Build fusion database from list of structure files

        Args:
            structure_paths: List of PDB/CIF file paths
            batch_size: Batch size for embedding computation
        """
        logger.info(f"Building fusion database from {len(structure_paths)} structures")

        all_domains = []
        all_domain_embeddings = []
        all_fusion_links = []
        all_fusion_embeddings = []
        domain_registry = {}  # domain_id -> Domain object

        # Process structures
        for pdb_path in tqdm(structure_paths, desc="Processing structures"):
            try:
                # Segment protein into domains - CORRECTED
                domains = self._segment_protein(pdb_path)

                # Skip if not multi-domain
                if len(domains) < self.min_domains:
                    continue

                # Embed domains - CORRECTED
                for domain in domains:
                    # Prepare coordinates for Foldclass
                    coords_tensor = torch.from_numpy(domain.ca_coordinates).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        embedding = self.foldclass(coords_tensor)  # Returns [1, 128]

                    domain.embedding = embedding.squeeze(0).cpu().numpy()
                    domain_registry[domain.domain_id] = domain
                    all_domains.append(domain)
                    all_domain_embeddings.append(domain.embedding)

                # Find fusion links
                fusion_links = self._find_fusion_links(domains)
                all_fusion_links.extend(fusion_links)

                # Store fusion embeddings (concatenated A+B for each link)
                for link in fusion_links:
                    fusion_emb = np.concatenate([link.domain_A.embedding, link.domain_B.embedding])
                    all_fusion_embeddings.append(fusion_emb)

            except Exception as e:
                logger.warning(f"Failed to process {pdb_path}: {e}")
                continue

        # Save domain database (follows existing .pt/.index pattern)
        logger.info(f"Saving {len(all_domains)} domains to database...")
        domain_embeddings_tensor = torch.tensor(np.array(all_domain_embeddings), dtype=torch.float32)
        torch.save(domain_embeddings_tensor, self.embeddings_path)

        domain_metadata = [(d.domain_id, d.ca_coordinates, d.sequence) for d in all_domains]
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(domain_metadata, f)

        # Save fusion database
        logger.info(f"Saving {len(all_fusion_links)} fusion links to database...")
        if len(all_fusion_embeddings) > 0:
            fusion_embeddings_tensor = torch.tensor(np.array(all_fusion_embeddings), dtype=torch.float32)
            torch.save(fusion_embeddings_tensor, self.fusion_embeddings_path)

        fusion_metadata = [link.to_dict() for link in all_fusion_links]
        with open(self.fusion_metadata_path, 'wb') as f:
            pickle.dump(fusion_metadata, f)

        # Save domain registry for later use
        registry_path = self.output_dir / 'domain_registry.pkl'
        with open(registry_path, 'wb') as f:
            pickle.dump(domain_registry, f)

        logger.info(f"Database built: {len(all_domains)} domains, {len(all_fusion_links)} fusion links")

    def _segment_protein(self, pdb_path: Path) -> List[Domain]:
        """Segment protein structure into domains using Merizo - CORRECTED"""

        # Use Merizo's segment function
        features = segment(
            pdb_path=str(pdb_path),
            network=self.merizo,
            device=str(self.device),
            length_conditional_iterate=False,
            iterate=True,
            max_iterations=3,
            shuffle_indices=False,
            min_domain_size=50,
            min_fragment_size=10,
            domain_ave_size=200,
            conf_threshold=0.5,
            pdb_chain='A'
        )

        # Extract domains from Merizo output - CORRECTED
        domains = []
        protein_id = pdb_path.stem

        # Get unique domain IDs (excluding 0 = non-domain)
        domain_ids_tensor = features['domain_ids']
        unique_domain_ids = torch.unique(domain_ids_tensor[domain_ids_tensor > 0])

        # Extract coordinates and sequence for each domain
        all_coords = features['ca_coords'].cpu().numpy()  # [N, 3]
        all_residue_indices = features['ri'].cpu().numpy()  # Original residue numbering
        sequence = features['seq']  # Full sequence

        for domain_idx, domain_id in enumerate(unique_domain_ids):
            # Get mask for this domain
            domain_mask = (domain_ids_tensor == domain_id).cpu().numpy()

            # Extract domain data
            domain_coords = all_coords[domain_mask]
            domain_res_indices = all_residue_indices[domain_mask]
            domain_sequence = ''.join([sequence[i] for i in range(len(sequence)) if domain_mask[i]])

            # Get residue range
            res_start = int(domain_res_indices[0])
            res_end = int(domain_res_indices[-1])

            # Get domain confidence
            conf_res = features['conf_res'][domain_mask]
            domain_conf = conf_res.mean().item()

            domain = Domain(
                domain_id=f"{protein_id}_domain_{domain_idx}",
                protein_id=protein_id,
                chain_id='A',
                residue_range=(res_start, res_end),
                residue_indices=domain_res_indices,
                ca_coordinates=domain_coords,
                sequence=domain_sequence,
                embedding=np.zeros(128),  # Will be filled later
                confidence=domain_conf
            )
            domains.append(domain)

        return domains

    def _find_fusion_links(self, domains: List[Domain]) -> List[FusionLink]:
        """
        Find all pairwise domain combinations in multi-domain protein
        These represent fusion events (Rosetta Stones)
        """
        fusion_links = []

        for i in range(len(domains)):
            for j in range(i+1, len(domains)):
                domain_A = domains[i]
                domain_B = domains[j]

                # Check domains don't overlap
                if domain_A.overlaps(domain_B):
                    continue

                # Calculate linker length (number of residues between domains)
                linker = abs(domain_A.residue_range[1] - domain_B.residue_range[0]) - 1

                # Filter by linker length
                if not (self.min_linker <= linker <= self.max_linker):
                    continue

                # Create fusion link
                fusion_link = FusionLink(
                    rosetta_stone_id=domain_A.protein_id,
                    domain_A=domain_A,
                    domain_B=domain_B,
                    linker_length=linker
                )

                fusion_links.append(fusion_link)

        return fusion_links


# CLI for building database
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build fusion database')
    parser.add_argument('-i', '--input', required=True, help='Directory of PDB files or file list')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('--min-domains', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('-d', '--device', default='cuda')
    args = parser.parse_args()

    # Get structure files
    input_path = Path(args.input)
    if input_path.is_dir():
        structure_paths = list(input_path.glob('*.pdb')) + list(input_path.glob('*.cif'))
    else:
        with open(input_path) as f:
            structure_paths = [Path(line.strip()) for line in f]

    # Build database
    builder = FusionDatabaseBuilder(
        output_dir=Path(args.output),
        min_domains_per_protein=args.min_domains,
        device=args.device
    )

    builder.build_from_structure_list(structure_paths, batch_size=args.batch_size)
```

# CORRECTED Implementation Guide Part 2
# (Continuation of IMPLEMENTATION.md)

## Module 2: Structural Rosetta Stone Search

### Purpose
Search pre-built fusion database to find domain interaction predictions for query proteins.

### Key Corrections
1. Uses correct `segment()` function with proper parameter handling
2. Uses `FoldClassNet(128)` and correct forward pass
3. Leverages existing `run_tmalign()` from `programs.Foldclass.utils`
4. Follows `.pt`/`.index` database format

### Implementation

See the full corrected `rosetta_search.py` code in the appendix below. Key points:

- Properly loads Merizo and Foldclass networks
- Correctly segments query proteins and extracts domain coordinates
- Uses FAISS for fast similarity search
- Integrates with existing TM-align utilities

---

## Module 3: Promiscuous Domain Filter

### Purpose
Filter out promiscuous domains that interact with many partners (low specificity).

### Key Corrections
1. Fixed class name: `DomainPromiscuityFilter` (not `DomainPromisc promiscuityFilter`)
2. Loads embeddings from `.pt` files correctly
3. Uses HDBSCAN clustering properly
4. Integrates with fusion database format

### Implementation

See the full corrected `promiscuity_filter.py` code in the appendix below.

---

## Module 4: Integration Layer & CLI

### Integration with merizo.py

Add new 'rosetta' mode to existing CLI by modifying `merizo_search/merizo.py`:

```python
# Add to merizo_search/merizo.py

def rosetta(args):
    """Rosetta Stone search mode"""
    parser = argparse.ArgumentParser(
        description="Rosetta Stone search for protein-protein interactions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='rosetta_command', help='Rosetta Stone commands')

    # Build database command
    build_parser = subparsers.add_parser('build', help='Build fusion database')
    build_parser.add_argument('input', type=str, help='Directory of PDB files or file list')
    build_parser.add_argument('output', type=str, help='Output directory for fusion database')
    build_parser.add_argument('--min-domains', type=int, default=2,
                             help='Minimum domains per protein')
    build_parser.add_argument('-d', '--device', type=str, default='cuda',
                             help='Device (cpu, cuda, mps)')
    build_parser.add_argument('--skip-promiscuity', action='store_true',
                             help='Skip promiscuity index building')
    build_parser.add_argument('--promiscuity-threshold', type=int, default=25,
                             help='Promiscuity threshold (number of links)')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for interactions')
    search_parser.add_argument('query', type=str, help='Query PDB file')
    search_parser.add_argument('database', type=str, help='Fusion database directory')
    search_parser.add_argument('output', type=str, help='Output file prefix')
    search_parser.add_argument('--cosine-threshold', type=float, default=0.7,
                              help='Cosine similarity threshold')
    search_parser.add_argument('--top-k', type=int, default=20,
                              help='Number of top matches to consider')
    search_parser.add_argument('--validate-tm', action='store_true',
                              help='Validate with TM-align')
    search_parser.add_argument('--min-tm-score', type=float, default=0.5,
                              help='Minimum TM-score threshold')
    search_parser.add_argument('--fastmode', action='store_true',
                              help='Use fast TM-align mode')
    search_parser.add_argument('--skip-filter', action='store_true',
                              help='Skip promiscuity filtering')
    search_parser.add_argument('-d', '--device', type=str, default='cuda',
                              help='Device (cpu, cuda, mps)')
    search_parser.add_argument('--output-headers', action='store_true',
                              help='Include headers in output')

    args = parser.parse_args(args)

    if args.rosetta_command == 'build':
        build_fusion_database(args)
    elif args.rosetta_command == 'search':
        search_rosetta_interactions(args)
    else:
        parser.print_help()


def build_fusion_database(args):
    """Build fusion database"""
    from programs.RosettaStone.fusion_database import FusionDatabaseBuilder
    from programs.RosettaStone.promiscuity_filter import DomainPromiscuityFilter
    from pathlib import Path

    logging.info('Starting fusion database build with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))

    # Get structure files
    input_path = Path(args.input)
    if input_path.is_dir():
        structure_paths = list(input_path.glob('*.pdb')) + list(input_path.glob('*.cif'))
    else:
        with open(input_path) as f:
            structure_paths = [Path(line.strip()) for line in f]

    start_time = time.time()

    # Build database
    builder = FusionDatabaseBuilder(
        output_dir=Path(args.output),
        min_domains_per_protein=args.min_domains,
        device=args.device
    )

    builder.build_from_structure_list(structure_paths)

    # Build promiscuity index
    if not args.skip_promiscuity:
        logging.info("Building promiscuity index...")
        filter_engine = DomainPromiscuityFilter(
            fusion_db_dir=Path(args.output),
            promiscuity_threshold=args.promiscuity_threshold
        )
        filter_engine.build_promiscuity_index()

        # Print report
        report = filter_engine.get_promiscuity_report()
        logging.info("Promiscuity Report:")
        logging.info(f"  Total clusters: {report['total_clusters']}")
        logging.info(f"  Promiscuous: {report['promiscuous_clusters']} ({report['promiscuity_rate']*100:.1f}%)")
        logging.info(f"  Mean links per cluster: {report['mean_links']:.1f}")

    elapsed_time = time.time() - start_time
    logging.info(f'Finished fusion database build in {elapsed_time:.2f} seconds.')


def search_rosetta_interactions(args):
    """Search for protein interactions"""
    from programs.RosettaStone.rosetta_search import StructuralRosettaStoneSearch
    from programs.RosettaStone.promiscuity_filter import DomainPromiscuityFilter
    from pathlib import Path
    import json

    logging.info('Starting Rosetta Stone search with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))

    start_time = time.time()

    # Initialize search engine
    search_engine = StructuralRosettaStoneSearch(
        fusion_db_dir=Path(args.database),
        cosine_threshold=args.cosine_threshold,
        top_k=args.top_k,
        device=args.device
    )

    # Search for interactions
    predictions = search_engine.search_interactions(
        query_pdb_path=Path(args.query),
        validate_tm=args.validate_tm,
        min_tm_score=args.min_tm_score,
        fastmode=args.fastmode
    )

    logging.info(f"Found {len(predictions)} candidate interactions")

    # Apply promiscuity filter
    if not args.skip_filter:
        logging.info("Applying promiscuity filter...")
        filter_engine = DomainPromiscuityFilter(
            fusion_db_dir=Path(args.database)
        )
        filter_engine.load_promiscuity_index(Path(args.database) / 'promiscuity_index.pkl')

        filtered_predictions, removed_predictions = filter_engine.filter_predictions(predictions)

        logging.info(f"After filtering: {len(filtered_predictions)} predictions")
        logging.info(f"Removed {len(removed_predictions)} promiscuous interactions")

        predictions = filtered_predictions

    # Save results
    output_path = Path(args.output + '_rosetta.json')
    output_data = {
        'query': args.query,
        'num_predictions': len(predictions),
        'predictions': [pred.to_output_dict() for pred in predictions]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("TOP PREDICTIONS")
    print("="*80)

    for i, pred in enumerate(predictions[:10], 1):
        print(f"\n{i}. Confidence: {pred.confidence_score:.3f}")
        print(f"   Query:  {pred.query_domain.domain_id} (residues {pred.query_domain.residue_range[0]}-{pred.query_domain.residue_range[1]})")
        print(f"   Target: {pred.target_domain.domain_id} (residues {pred.target_domain.residue_range[0]}-{pred.target_domain.residue_range[1]})")
        print(f"   Type: {pred.interaction_type}")
        print(f"   Similarity: {pred.cosine_similarity:.3f}")
        if pred.tm_score:
            print(f"   TM-score: {pred.tm_score:.3f}")
        print(f"   Evidence: {len(pred.rosetta_stone_evidence)} Rosetta Stone(s)")
        for rs in pred.rosetta_stone_evidence[:2]:
            print(f"      - {rs.rosetta_stone_id}")

    elapsed_time = time.time() - start_time
    logging.info(f'Finished Rosetta Stone search in {elapsed_time:.2f} seconds.')


# Modify main() function to add rosetta mode
def main():
    setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python merizo.py {segment|search|easy-search|createdb|rosetta} ...")
        sys.exit(1)

    mode = sys.argv[1]
    args = sys.argv[2:]

    modes = {
        'segment': segment,
        'search': search,
        'easy-search': easy_search,
        'createdb': createdb,
        'rosetta': rosetta,  # NEW
    }

    if mode in modes:
        modes[mode](args)
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: segment, search, easy-search, createdb, rosetta")
        sys.exit(1)
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_rosetta_stone.py

import pytest
import numpy as np
from pathlib import Path
import tempfile

from programs.RosettaStone.data_structures import Domain, FusionLink


class TestDomain:
    """Test Domain data structure"""

    def test_domain_creation(self):
        domain = Domain(
            domain_id="test_domain_1",
            protein_id="test_protein",
            chain_id="A",
            residue_range=(1, 100),
            residue_indices=np.arange(1, 101),
            ca_coordinates=np.random.rand(100, 3),
            sequence="A" * 100,
            embedding=np.random.rand(128)
        )

        assert domain.length == 100
        assert domain.embedding.shape == (128,)

    def test_domain_overlap(self):
        domain1 = Domain(
            domain_id="d1",
            protein_id="p1",
            chain_id="A",
            residue_range=(1, 50),
            residue_indices=np.arange(1, 51),
            ca_coordinates=np.random.rand(50, 3),
            sequence="A" * 50,
            embedding=np.random.rand(128)
        )

        domain2 = Domain(
            domain_id="d2",
            protein_id="p1",
            chain_id="A",
            residue_range=(45, 100),  # Overlaps with domain1
            residue_indices=np.arange(45, 101),
            ca_coordinates=np.random.rand(56, 3),
            sequence="A" * 56,
            embedding=np.random.rand(128)
        )

        assert domain1.overlaps(domain2)


class TestFusionDatabase:
    """Test fusion database building"""

    def test_fusion_link_creation(self):
        domain_A = Domain(
            domain_id="d1",
            protein_id="fusion",
            chain_id="A",
            residue_range=(1, 100),
            residue_indices=np.arange(1, 101),
            ca_coordinates=np.random.rand(100, 3),
            sequence="A" * 100,
            embedding=np.random.rand(128)
        )

        domain_B = Domain(
            domain_id="d2",
            protein_id="fusion",
            chain_id="A",
            residue_range=(120, 200),
            residue_indices=np.arange(120, 201),
            ca_coordinates=np.random.rand(81, 3),
            sequence="A" * 81,
            embedding=np.random.rand(128)
        )

        fusion = FusionLink(
            rosetta_stone_id="fusion_protein",
            domain_A=domain_A,
            domain_B=domain_B,
            linker_length=19
        )

        assert fusion.linker_length == 19
        assert fusion.rosetta_stone_id == "fusion_protein"
```

### Integration Test

```bash
# Test on small dataset (10 proteins)
python merizo_search/merizo.py rosetta build \
    examples/test_pdbs/ \
    test_fusion_db/ \
    -d cpu

# Test search
python merizo_search/merizo.py rosetta search \
    examples/3w5h.pdb \
    test_fusion_db/ \
    test_output \
    --validate-tm \
    -d cpu
```

---

## CLI Usage Examples

### Build Fusion Database

```bash
# Build from directory of PDB files
python merizo_search/merizo.py rosetta build \
    /data/alphafold_structures/ \
    fusion_db/ \
    --min-domains 2 \
    --device cuda

# Build from file list
python merizo_search/merizo.py rosetta build \
    pdb_file_list.txt \
    fusion_db/ \
    --device cuda

# Build without promiscuity filtering
python merizo_search/merizo.py rosetta build \
    /data/structures/ \
    fusion_db/ \
    --skip-promiscuity \
    --device cuda
```

### Search for Interactions

```bash
# Basic search
python merizo_search/merizo.py rosetta search \
    query_protein.pdb \
    fusion_db/ \
    results \
    --device cuda

# Search with TM-align validation
python merizo_search/merizo.py rosetta search \
    query_protein.pdb \
    fusion_db/ \
    results \
    --validate-tm \
    --min-tm-score 0.5 \
    --fastmode \
    --device cuda

# Search without promiscuity filtering
python merizo_search/merizo.py rosetta search \
    query_protein.pdb \
    fusion_db/ \
    results \
    --skip-filter \
    --device cuda

# Adjust sensitivity
python merizo_search/merizo.py rosetta search \
    query_protein.pdb \
    fusion_db/ \
    results \
    --cosine-threshold 0.6 \
    --top-k 50 \
    --device cuda
```

---

## Performance Optimization

### FAISS GPU Acceleration

```python
# In rosetta_search.py, modify _build_faiss_index():

import faiss

def _build_faiss_index(self) -> faiss.Index:
    embeddings = self.fusion_embeddings.cpu().numpy().astype('float32')
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Use GPU if available
    if torch.cuda.is_available() and str(self.device) == 'cuda':
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)
    return index
```

### Batch Processing

For large-scale database building, process in batches:

```python
# In fusion_database.py
def build_from_structure_list(self, structure_paths, batch_size=32):
    for i in range(0, len(structure_paths), batch_size):
        batch = structure_paths[i:i+batch_size]
        # Process batch...
```

---

## Implementation Checklist

### Setup
- [ ] Install new dependencies: `pip install h5py faiss-cpu hdbscan scikit-learn tqdm`
- [ ] Create `programs/RosettaStone/` directory
- [ ] Create `programs/RosettaStone/__init__.py`

### Core Modules
- [ ] Implement `programs/RosettaStone/data_structures.py`
- [ ] Implement `programs/RosettaStone/fusion_database.py`
- [ ] Implement `programs/RosettaStone/rosetta_search.py`
- [ ] Implement `programs/RosettaStone/promiscuity_filter.py`

### Integration
- [ ] Add `rosetta()` function to `merizo_search/merizo.py`
- [ ] Add `build_fusion_database()` to `merizo_search/merizo.py`
- [ ] Add `search_rosetta_interactions()` to `merizo_search/merizo.py`
- [ ] Update `main()` in `merizo_search/merizo.py` to include 'rosetta' mode

### Testing
- [ ] Write unit tests in `tests/test_rosetta_stone.py`
- [ ] Test database build on 10 proteins
- [ ] Test search on small database
- [ ] Test promiscuity filtering
- [ ] Test TM-align validation

### Validation
- [ ] Test on 100 proteins
- [ ] Test on 1,000 proteins
- [ ] Validate against known PPI databases (STRING, IntAct, etc.)
- [ ] Benchmark performance
- [ ] Profile memory usage

### Documentation
- [ ] Add docstrings to all modules
- [ ] Create usage examples
- [ ] Document database format
- [ ] Document output format

---

## Summary of Key Corrections

### Critical API Fixes
1. **Merizo API**:
   - ✅ Uses `Merizo()` network class from `programs.Merizo.model.network`
   - ✅ Uses `segment()` function from `programs.Merizo.predict`
   - ✅ Uses `read_split_weight_files()` for weight loading
   - ✅ Properly extracts domains from `features` dict with `domain_ids` tensor

2. **Foldclass API**:
   - ✅ Uses `FoldClassNet(128)` from `programs.Foldclass.nndef_fold_egnn_embed`
   - ✅ Calls `.forward()` directly: `network(coords_tensor)`
   - ✅ No mythical `.embed()` method

3. **Database Format**:
   - ✅ Uses `.pt` (PyTorch tensors) for embeddings
   - ✅ Uses `.index` (pickled lists) for metadata
   - ✅ Follows existing merizo-search patterns

4. **Integration**:
   - ✅ Adds as new mode to `merizo.py` CLI (not standalone script)
   - ✅ Follows existing CLI patterns (segment, search, etc.)

5. **Utilities**:
   - ✅ Reuses `run_tmalign()` from `programs.Foldclass.utils`
   - ✅ Uses `get_device()` from `programs.Merizo.model.utils.utils`

### Syntax Fixes
- ✅ Fixed class name: `DomainPromiscuityFilter` (not `DomainPromisc promiscuityFilter`)
- ✅ Proper import statements
- ✅ Correct method signatures

---

## Next Steps

1. **Start with Data Structures**: Implement `data_structures.py` first as all modules depend on it

2. **Build Database Module**: Implement `fusion_database.py` to create test databases

3. **Test on Small Dataset**: Build database from 10-50 proteins to verify correctness

4. **Implement Search**: Add `rosetta_search.py` once database format is validated

5. **Add Promiscuity Filter**: Implement clustering and filtering

6. **Integrate CLI**: Add to `merizo.py` for seamless user experience

7. **Validate & Optimize**: Test on larger datasets and optimize performance

**This corrected implementation guide is now complete and ready for development!**
