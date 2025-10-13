"""
Data structures for Rosetta Stone protein interaction prediction.

This module defines the core data structures used throughout the Rosetta Stone
implementation, including Domain, FusionLink, InteractionPrediction, and PromiscuityScore.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Domain:
    """Represents a single protein domain

    Attributes:
        domain_id: Unique identifier (e.g., "P12345_A_domain_1")
        protein_id: Parent protein identifier
        chain_id: PDB chain identifier
        residue_range: Tuple of (start, end) residue numbers (inclusive)
        residue_indices: Actual residue indices from PDB
        ca_coordinates: C-alpha coordinates, shape (n_residues, 3)
        sequence: Amino acid sequence
        embedding: Foldclass embedding vector [128]
        cluster_id: Structural cluster assignment (optional)
        confidence: Merizo confidence score (optional)
    """
    domain_id: str
    protein_id: str
    chain_id: str
    residue_range: Tuple[int, int]
    residue_indices: np.ndarray
    ca_coordinates: np.ndarray
    sequence: str
    embedding: np.ndarray
    cluster_id: Optional[int] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        """Validate data after initialization"""
        assert self.embedding.shape == (128,), f"Invalid embedding shape: {self.embedding.shape}"
        assert self.residue_range[0] <= self.residue_range[1], "Invalid residue range"
        if len(self.ca_coordinates) > 0:
            assert len(self.ca_coordinates) == len(self.sequence), \
                f"Coords/sequence length mismatch: {len(self.ca_coordinates)} vs {len(self.sequence)}"

    @property
    def length(self) -> int:
        """Return domain length"""
        return len(self.sequence)

    def overlaps(self, other: 'Domain') -> bool:
        """Check if two domains overlap in sequence

        Args:
            other: Another Domain instance

        Returns:
            True if domains overlap, False otherwise
        """
        if self.protein_id != other.protein_id or self.chain_id != other.chain_id:
            return False
        return not (self.residue_range[1] < other.residue_range[0] or
                   other.residue_range[1] < self.residue_range[0])


@dataclass
class FusionLink:
    """Represents a Rosetta Stone fusion event

    A fusion link connects two domains that appear together in the same
    multi-domain protein, suggesting they may interact in other contexts.

    Attributes:
        rosetta_stone_id: Identifier of the fusion protein
        domain_A: First domain in the fusion
        domain_B: Second domain in the fusion
        linker_length: Number of residues between domains
        organism: Source organism (optional)
    """
    rosetta_stone_id: str
    domain_A: Domain
    domain_B: Domain
    linker_length: int
    organism: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize fusion link to dictionary for storage

        Returns:
            Dictionary representation of the fusion link
        """
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

    @classmethod
    def from_dict(cls, data: dict, domain_registry: dict) -> 'FusionLink':
        """Deserialize fusion link from dictionary

        Args:
            data: Dictionary containing fusion link data
            domain_registry: Registry mapping domain IDs to Domain objects

        Returns:
            FusionLink instance
        """
        domain_A_id = data['domain_A_id']
        domain_B_id = data['domain_B_id']

        # Try to get from registry, otherwise reconstruct
        domain_A = domain_registry.get(domain_A_id)
        domain_B = domain_registry.get(domain_B_id)

        if domain_A is None:
            domain_A = Domain(
                domain_id=domain_A_id,
                protein_id=data['rosetta_stone_id'],
                chain_id='A',
                residue_range=tuple(data['domain_A_range']),
                residue_indices=np.array(data['domain_A_indices']),
                ca_coordinates=np.array([]),
                sequence='',
                embedding=np.array(data['embedding_A'])
            )

        if domain_B is None:
            domain_B = Domain(
                domain_id=domain_B_id,
                protein_id=data['rosetta_stone_id'],
                chain_id='A',
                residue_range=tuple(data['domain_B_range']),
                residue_indices=np.array(data['domain_B_indices']),
                ca_coordinates=np.array([]),
                sequence='',
                embedding=np.array(data['embedding_B'])
            )

        return cls(
            rosetta_stone_id=data['rosetta_stone_id'],
            domain_A=domain_A,
            domain_B=domain_B,
            linker_length=data['linker_length'],
            organism=data.get('organism')
        )


@dataclass
class InteractionPrediction:
    """Represents a predicted domain-domain interaction

    Attributes:
        query_domain: Query domain
        target_domain: Target domain predicted to interact
        rosetta_stone_evidence: List of fusion links supporting prediction
        cosine_similarity: Embedding similarity score
        tm_score: TM-align structural similarity (optional)
        confidence_score: Overall confidence [0-1]
        promiscuity_flag: True if involves promiscuous domain
        interaction_type: 'inter' (between proteins) or 'intra' (within protein)
    """
    query_domain: Domain
    target_domain: Domain
    rosetta_stone_evidence: List[FusionLink]
    cosine_similarity: float
    tm_score: Optional[float] = None
    confidence_score: float = 0.0
    promiscuity_flag: bool = False
    interaction_type: str = 'inter'

    def to_output_dict(self) -> dict:
        """Format prediction for output file

        Returns:
            Dictionary with prediction details for JSON output
        """
        return {
            'query_domain_id': self.query_domain.domain_id,
            'query_protein': self.query_domain.protein_id,
            'query_range': self.query_domain.residue_range,
            'query_sequence': self.query_domain.sequence,
            'target_domain_id': self.target_domain.domain_id,
            'target_protein': self.target_domain.protein_id,
            'target_range': self.target_domain.residue_range,
            'target_sequence': self.target_domain.sequence,
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
    """Tracks domain promiscuity metrics

    Promiscuous domains interact with many partners (low specificity).
    This class tracks which structural clusters are promiscuous.

    Attributes:
        cluster_id: Structural cluster identifier
        num_links: Number of other clusters this cluster links to
        linked_clusters: Set of linked cluster IDs
        is_promiscuous: True if num_links exceeds threshold
        example_domains: Example domain IDs in this cluster
    """
    cluster_id: int
    num_links: int
    linked_clusters: set
    is_promiscuous: bool
    example_domains: List[str]

    def get_promiscuity_ratio(self, total_clusters: int) -> float:
        """Calculate fraction of all clusters this one links to

        Args:
            total_clusters: Total number of clusters in database

        Returns:
            Promiscuity ratio [0-1]
        """
        return self.num_links / total_clusters if total_clusters > 0 else 0.0
