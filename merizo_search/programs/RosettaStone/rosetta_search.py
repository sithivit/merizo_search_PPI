"""
Structural Rosetta Stone Search for Protein Interaction Prediction.

This module searches a pre-built fusion database to find domain interaction
predictions for query proteins using FAISS-accelerated similarity search.
"""

import os
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import tempfile

from .data_structures import Domain, FusionLink, InteractionPrediction
from programs.Merizo.model.network import Merizo
from programs.Merizo.predict import segment, read_split_weight_files
from programs.Foldclass.nndef_fold_egnn_embed import FoldClassNet
from programs.Foldclass.utils import run_tmalign, write_pdb
from programs.Merizo.model.utils.utils import get_device

logger = logging.getLogger(__name__)


class StructuralRosettaStoneSearch:
    """Search for domain interactions using structural fusion analysis"""

    def __init__(
        self,
        fusion_db_dir: Path,
        cosine_threshold: float = 0.7,
        top_k: int = 20,
        device: str = 'cuda'
    ):
        """Initialize the Rosetta Stone search engine

        Args:
            fusion_db_dir: Directory containing fusion database files
            cosine_threshold: Minimum cosine similarity for matches
            top_k: Number of top matches to retrieve
            device: Device for neural network inference
        """
        self.fusion_db_dir = Path(fusion_db_dir)
        self.cosine_threshold = cosine_threshold
        self.top_k = top_k
        self.device = get_device(device)

        # Load domain registry (or build from metadata if missing)
        logger.info("Loading domain registry...")
        registry_path = self.fusion_db_dir / 'domain_registry.pkl'
        if registry_path.exists():
            with open(registry_path, 'rb') as f:
                self.domain_registry = pickle.load(f)
        else:
            logger.warning("domain_registry.pkl not found, will reconstruct from metadata")
            self.domain_registry = {}

        # Load domain embeddings database
        logger.info("Loading domain embeddings...")
        self.domain_embeddings = torch.load(
            self.fusion_db_dir / 'domain_embeddings.pt'
        ).to(self.device)

        with open(self.fusion_db_dir / 'domain_metadata.index', 'rb') as f:
            self.domain_metadata = pickle.load(f)

        # Load fusion database
        logger.info("Loading fusion database...")
        self.fusion_embeddings = torch.load(
            self.fusion_db_dir / 'fusion_embeddings.pt'
        ).to(self.device)

        with open(self.fusion_db_dir / 'fusion_metadata.index', 'rb') as f:
            self.fusion_metadata = pickle.load(f)

        # Build FAISS index for fast similarity search
        logger.info("Building FAISS index...")
        self.faiss_index = self._build_faiss_index()

        # Initialize networks for query processing
        logger.info("Loading Merizo network...")
        self.merizo = Merizo().to(self.device)
        weights_dir = os.path.join(os.path.dirname(__file__), '../Merizo/weights')
        self.merizo.load_state_dict(read_split_weight_files(weights_dir), strict=True)
        self.merizo.eval()

        logger.info("Loading Foldclass network...")
        self.foldclass = FoldClassNet(128).to(self.device).eval()
        scriptdir = os.path.dirname(os.path.realpath(__file__))
        foldclass_weights = os.path.join(scriptdir, '../Foldclass/FINAL_foldclass_model.pt')
        self.foldclass.load_state_dict(
            torch.load(foldclass_weights, map_location=lambda storage, loc: storage),
            strict=False
        )

        logger.info("Rosetta Stone search engine ready")

    def _build_faiss_index(self) -> faiss.Index:
        """Build FAISS index for fast cosine similarity search"""
        # Use fusion embeddings (concatenated domain pairs)
        embeddings = self.fusion_embeddings.cpu().numpy().astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Use IndexFlatIP for inner product (cosine similarity after normalization)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        logger.info(f"FAISS index built with {index.ntotal} fusion embeddings")
        return index

    def search_interactions(
        self,
        query_pdb_path: Path,
        validate_tm: bool = True,
        min_tm_score: float = 0.5,
        fastmode: bool = True
    ) -> List[InteractionPrediction]:
        """Find domain interactions for query protein

        Args:
            query_pdb_path: Path to query PDB file
            validate_tm: Whether to run TM-align validation
            min_tm_score: Minimum TM-score threshold
            fastmode: Use fast TM-align mode

        Returns:
            List of interaction predictions sorted by confidence
        """
        logger.info(f"Searching for interactions in {query_pdb_path}")

        # Step 1: Segment query protein
        query_domains = self._segment_and_embed_query(query_pdb_path)

        if len(query_domains) < 1:
            logger.warning("No domains found in query")
            return []

        logger.info(f"Found {len(query_domains)} domains in query")

        # Step 2: Search for Rosetta Stone patterns
        predictions = []

        # Check for intra-protein interactions
        if len(query_domains) >= 2:
            intra_predictions = self._find_intra_protein_interactions(query_domains)
            predictions.extend(intra_predictions)

        # Check for inter-protein interactions
        inter_predictions = self._find_inter_protein_interactions(query_domains)
        predictions.extend(inter_predictions)

        logger.info(f"Found {len(predictions)} candidate interactions")

        # Step 3: Validate with TM-align (optional)
        if validate_tm and len(predictions) > 0:
            predictions = self._validate_with_tmalign(predictions, min_tm_score, fastmode)
            logger.info(f"{len(predictions)} predictions passed TM-align validation")

        # Step 4: Calculate confidence scores
        for pred in predictions:
            pred.confidence_score = self._calculate_confidence(pred)

        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence_score, reverse=True)

        return predictions

    def _segment_and_embed_query(self, pdb_path: Path) -> List[Domain]:
        """Segment query protein and compute embeddings"""
        # Segment with Merizo
        features = segment(
            pdb_path=str(pdb_path),
            network=self.merizo,
            device=str(self.device),
            length_conditional_iterate=False,
            iterate=True,
            max_iterations=3,
            shuffle_indices=False,
            min_domain_size=50,
            pdb_chain='A'
        )

        domains = []
        protein_id = pdb_path.stem

        # Extract domains from Merizo output
        # Remove batch dimension [1, N] -> [N]
        domain_ids_tensor = features['domain_ids'].squeeze(0)
        conf_res_tensor = features['conf_res'].squeeze(0)
        unique_domain_ids = torch.unique(domain_ids_tensor[domain_ids_tensor > 0])

        # Get CA coordinates from pdb structured array
        pdb_array = features['pdb']
        ca_atoms = pdb_array[pdb_array['n'] == 'CA']
        all_coords = np.column_stack([ca_atoms['x'], ca_atoms['y'], ca_atoms['z']])
        all_residue_indices = features['ri'].squeeze(0).cpu().numpy()

        # Get sequence from pdb
        from programs.Merizo.model.utils.features import pdb_to_fasta
        sequence = pdb_to_fasta(pdb_array)

        for domain_idx, domain_id in enumerate(unique_domain_ids):
            domain_mask = (domain_ids_tensor == domain_id).cpu().numpy()

            domain_coords = all_coords[domain_mask]
            domain_res_indices = all_residue_indices[domain_mask]
            domain_sequence = ''.join([sequence[i] for i in range(len(sequence)) if domain_mask[i]])

            res_start = int(domain_res_indices[0])
            res_end = int(domain_res_indices[-1])

            domain_conf = conf_res_tensor[domain_mask].mean().item()

            # Create domain object
            domain = Domain(
                domain_id=f"{protein_id}_query_domain_{domain_idx}",
                protein_id=protein_id,
                chain_id='A',
                residue_range=(res_start, res_end),
                residue_indices=domain_res_indices,
                ca_coordinates=domain_coords,
                sequence=domain_sequence,
                embedding=np.zeros(128),
                confidence=domain_conf
            )

            # Compute embedding with Foldclass (must be float32)
            coords_tensor = torch.from_numpy(domain.ca_coordinates).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.foldclass(coords_tensor)
            domain.embedding = embedding.squeeze(0).cpu().numpy()

            domains.append(domain)

        return domains

    def _find_intra_protein_interactions(
        self,
        query_domains: List[Domain]
    ) -> List[InteractionPrediction]:
        """Find interactions between domains within the same query protein"""
        predictions = []

        # Check all domain pairs
        for i in range(len(query_domains)):
            for j in range(i+1, len(query_domains)):
                domain_A = query_domains[i]
                domain_B = query_domains[j]

                # Skip if domains overlap
                if domain_A.overlaps(domain_B):
                    continue

                # Search for Rosetta Stone evidence
                rosetta_stones = self._search_fusion_pattern(
                    domain_A.embedding,
                    domain_B.embedding
                )

                if rosetta_stones:
                    # Calculate similarity score
                    similarities = [
                        self._compute_cosine_similarity(domain_A.embedding, rs.domain_A.embedding)
                        for rs in rosetta_stones
                    ]
                    max_similarity = max(similarities)

                    pred = InteractionPrediction(
                        query_domain=domain_A,
                        target_domain=domain_B,
                        rosetta_stone_evidence=rosetta_stones,
                        cosine_similarity=max_similarity,
                        interaction_type='intra'
                    )
                    predictions.append(pred)

        return predictions

    def _find_inter_protein_interactions(
        self,
        query_domains: List[Domain]
    ) -> List[InteractionPrediction]:
        """Find interactions between query domains and external proteins"""
        predictions = []

        for query_domain in query_domains:
            # Search FAISS index for similar fusions
            similar_fusions = self._search_similar_fusions(
                query_domain.embedding,
                k=self.top_k
            )

            # For each similar fusion, extract partner domain
            for fusion_idx, similarity in similar_fusions:
                fusion_metadata = self.fusion_metadata[fusion_idx]

                # Determine which domain in fusion matches query
                emb_A = np.array(fusion_metadata['embedding_A'])
                emb_B = np.array(fusion_metadata['embedding_B'])

                sim_A = self._compute_cosine_similarity(query_domain.embedding, emb_A)
                sim_B = self._compute_cosine_similarity(query_domain.embedding, emb_B)

                # Partner is the one with lower similarity (the other domain)
                if sim_A > sim_B:
                    partner_domain_id = fusion_metadata['domain_B_id']
                else:
                    partner_domain_id = fusion_metadata['domain_A_id']

                # Get partner domain from registry
                if partner_domain_id in self.domain_registry:
                    partner_domain = self.domain_registry[partner_domain_id]

                    # Reconstruct FusionLink
                    fusion_link = self._reconstruct_fusion_link(fusion_metadata)

                    pred = InteractionPrediction(
                        query_domain=query_domain,
                        target_domain=partner_domain,
                        rosetta_stone_evidence=[fusion_link],
                        cosine_similarity=similarity,
                        interaction_type='inter'
                    )
                    predictions.append(pred)

        return predictions

    def _search_fusion_pattern(
        self,
        embedding_A: np.ndarray,
        embedding_B: np.ndarray
    ) -> List[FusionLink]:
        """Search for fusion proteins containing both domain patterns"""
        # Create concatenated query embedding
        query_fusion_emb = np.concatenate([embedding_A, embedding_B])

        # Normalize
        query_fusion_emb = query_fusion_emb / np.linalg.norm(query_fusion_emb)

        # Search FAISS
        distances, indices = self.faiss_index.search(
            query_fusion_emb.reshape(1, -1).astype('float32'),
            k=self.top_k
        )

        # Filter by threshold and reconstruct FusionLinks
        rosetta_stones = []
        for idx, dist in zip(indices[0], distances[0]):
            if dist >= self.cosine_threshold:
                fusion_metadata = self.fusion_metadata[idx]
                fusion_link = self._reconstruct_fusion_link(fusion_metadata)
                rosetta_stones.append(fusion_link)

        return rosetta_stones

    def _search_similar_fusions(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> List[Tuple[int, float]]:
        """Search for fusions containing domains similar to query"""
        # Expand query to match fusion embedding size
        query_expanded = np.concatenate([query_embedding, np.zeros_like(query_embedding)])

        # Normalize
        query_norm = query_expanded / np.linalg.norm(query_expanded)

        # Search FAISS
        distances, indices = self.faiss_index.search(
            query_norm.reshape(1, -1).astype('float32'),
            k=k
        )

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if dist >= self.cosine_threshold:
                results.append((int(idx), float(dist)))

        return results

    def _reconstruct_fusion_link(self, fusion_metadata: dict) -> FusionLink:
        """Reconstruct FusionLink from stored metadata"""
        domain_A_id = fusion_metadata['domain_A_id']
        domain_B_id = fusion_metadata['domain_B_id']

        # Try to get from registry
        domain_A = self.domain_registry.get(domain_A_id)
        domain_B = self.domain_registry.get(domain_B_id)

        if domain_A is None:
            domain_A = Domain(
                domain_id=domain_A_id,
                protein_id=fusion_metadata['rosetta_stone_id'],
                chain_id='A',
                residue_range=tuple(fusion_metadata['domain_A_range']),
                residue_indices=np.array(fusion_metadata['domain_A_indices']),
                ca_coordinates=np.array([]),
                sequence='',
                embedding=np.array(fusion_metadata['embedding_A'])
            )

        if domain_B is None:
            domain_B = Domain(
                domain_id=domain_B_id,
                protein_id=fusion_metadata['rosetta_stone_id'],
                chain_id='A',
                residue_range=tuple(fusion_metadata['domain_B_range']),
                residue_indices=np.array(fusion_metadata['domain_B_indices']),
                ca_coordinates=np.array([]),
                sequence='',
                embedding=np.array(fusion_metadata['embedding_B'])
            )

        return FusionLink(
            rosetta_stone_id=fusion_metadata['rosetta_stone_id'],
            domain_A=domain_A,
            domain_B=domain_B,
            linker_length=fusion_metadata['linker_length']
        )

    def _validate_with_tmalign(
        self,
        predictions: List[InteractionPrediction],
        min_score: float,
        fastmode: bool
    ) -> List[InteractionPrediction]:
        """Validate predictions using TM-align"""
        validated = []

        for pred in predictions:
            # Skip if coordinates not available
            if len(pred.query_domain.ca_coordinates) == 0:
                validated.append(pred)
                continue

            # Get Rosetta Stone structure
            rosetta_stone = pred.rosetta_stone_evidence[0]
            if len(rosetta_stone.domain_A.ca_coordinates) == 0:
                validated.append(pred)
                continue

            # Write temporary PDB files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f1:
                write_pdb(
                    f1,
                    pred.query_domain.ca_coordinates,
                    pred.query_domain.sequence
                )
                query_pdb = f1.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f2:
                write_pdb(
                    f2,
                    rosetta_stone.domain_A.ca_coordinates,
                    rosetta_stone.domain_A.sequence
                )
                target_pdb = f2.name

            try:
                # Run TM-align using existing utility
                tm_score = run_tmalign(query_pdb, target_pdb, fastmode=fastmode)
                pred.tm_score = tm_score

                # Keep if passes threshold
                if tm_score >= min_score:
                    validated.append(pred)
            except Exception as e:
                logger.warning(f"TM-align failed: {e}")
                validated.append(pred)  # Keep anyway
            finally:
                # Cleanup
                Path(query_pdb).unlink(missing_ok=True)
                Path(target_pdb).unlink(missing_ok=True)

        return validated

    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def _calculate_confidence(self, pred: InteractionPrediction) -> float:
        """Calculate overall confidence score

        Combines:
        - Cosine similarity (0-0.5)
        - Number of Rosetta Stone evidence (0-0.25)
        - TM-score if available (0-0.25)
        """
        score = 0.0

        # Cosine similarity component
        score += pred.cosine_similarity * 0.5

        # Evidence count component
        evidence_score = min(len(pred.rosetta_stone_evidence) / 5.0, 1.0) * 0.25
        score += evidence_score

        # TM-score component
        if pred.tm_score is not None:
            score += pred.tm_score * 0.25

        return score
