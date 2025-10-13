"""
Promiscuous Domain Filter for Rosetta Stone Predictions.

This module filters out promiscuous domains (domains that interact with many
partners) to improve prediction specificity. Uses HDBSCAN clustering to group
structurally similar domains and identifies clusters that link to many others.
"""

import numpy as np
import pickle
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging
from collections import defaultdict
from sklearn.cluster import HDBSCAN

from .data_structures import Domain, PromiscuityScore, InteractionPrediction

logger = logging.getLogger(__name__)


class DomainPromiscuityFilter:
    """Filter out promiscuous domains from interaction predictions"""

    def __init__(
        self,
        fusion_db_dir: Path,
        promiscuity_threshold: int = 25,
        clustering_min_samples: int = 5
    ):
        """Initialize the promiscuity filter

        Args:
            fusion_db_dir: Directory containing fusion database
            promiscuity_threshold: Max links before cluster is promiscuous
            clustering_min_samples: Minimum cluster size for HDBSCAN
        """
        self.fusion_db_dir = Path(fusion_db_dir)
        self.promiscuity_threshold = promiscuity_threshold
        self.clustering_min_samples = clustering_min_samples

        self.cluster_assignments = {}  # domain_id -> cluster_id
        self.promiscuity_scores = {}   # cluster_id -> PromiscuityScore

        logger.info("Initializing promiscuity filter")

    def build_promiscuity_index(
        self,
        output_path: Optional[Path] = None
    ) -> None:
        """Build promiscuity index by clustering and analyzing fusion patterns

        Steps:
        1. Load all domain embeddings
        2. Cluster embeddings into structural families
        3. Count fusion links for each cluster
        4. Identify promiscuous clusters

        Args:
            output_path: Path to save index (default: fusion_db_dir/promiscuity_index.pkl)
        """
        logger.info("Building promiscuity index...")

        # Step 1: Load all domain embeddings
        embeddings, domain_ids = self._load_all_embeddings()
        logger.info(f"Loaded {len(embeddings)} domain embeddings")

        # Step 2: Cluster embeddings into structural families
        logger.info("Clustering embeddings into structural families...")
        cluster_labels = self._cluster_embeddings(embeddings)

        # Create cluster assignments
        for domain_id, cluster_id in zip(domain_ids, cluster_labels):
            self.cluster_assignments[domain_id] = int(cluster_id)

        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(f"Found {num_clusters} structural clusters")

        # Step 3: Count fusion links per cluster
        logger.info("Counting fusion links per cluster...")
        link_counts = self._count_cluster_links()

        # Step 4: Calculate promiscuity scores
        for cluster_id, linked_clusters in link_counts.items():
            num_links = len(linked_clusters)
            is_promiscuous = num_links > self.promiscuity_threshold

            # Get example domains in this cluster
            example_domains = [
                did for did, cid in self.cluster_assignments.items()
                if cid == cluster_id
            ][:5]

            self.promiscuity_scores[cluster_id] = PromiscuityScore(
                cluster_id=cluster_id,
                num_links=num_links,
                linked_clusters=linked_clusters,
                is_promiscuous=is_promiscuous,
                example_domains=example_domains
            )

        # Statistics
        total_clusters = len(self.promiscuity_scores)
        promiscuous_clusters = sum(1 for ps in self.promiscuity_scores.values() if ps.is_promiscuous)

        logger.info(f"Promiscuity analysis complete:")
        logger.info(f"  Total clusters: {total_clusters}")
        if total_clusters > 0:
            logger.info(f"  Promiscuous clusters: {promiscuous_clusters} ({promiscuous_clusters/total_clusters*100:.1f}%)")
            logger.info(f"  Specific clusters: {total_clusters - promiscuous_clusters} ({(total_clusters-promiscuous_clusters)/total_clusters*100:.1f}%)")
        else:
            logger.info(f"  Promiscuous clusters: {promiscuous_clusters}")
            logger.info(f"  Specific clusters: {total_clusters - promiscuous_clusters}")

        # Save index
        if output_path is None:
            output_path = self.fusion_db_dir / 'promiscuity_index.pkl'

        self._save_index(output_path)
        logger.info(f"Promiscuity index saved to {output_path}")

    def load_promiscuity_index(self, index_path: Path) -> None:
        """Load pre-built promiscuity index

        Args:
            index_path: Path to promiscuity index file
        """
        logger.info(f"Loading promiscuity index from {index_path}")

        with open(index_path, 'rb') as f:
            data = pickle.load(f)

        self.cluster_assignments = data['cluster_assignments']
        self.promiscuity_scores = data['promiscuity_scores']

        logger.info(f"Loaded index with {len(self.promiscuity_scores)} clusters")

    def filter_predictions(
        self,
        predictions: List[InteractionPrediction]
    ) -> Tuple[List[InteractionPrediction], List[InteractionPrediction]]:
        """Filter predictions to remove those involving promiscuous domains

        Args:
            predictions: List of interaction predictions

        Returns:
            Tuple of (filtered_predictions, removed_predictions)
        """
        filtered = []
        removed = []

        for pred in predictions:
            # Get cluster IDs for both domains
            query_cluster = self._get_cluster_for_domain(pred.query_domain)
            target_cluster = self._get_cluster_for_domain(pred.target_domain)

            # Check if either cluster is promiscuous
            query_promiscuous = self._is_cluster_promiscuous(query_cluster)
            target_promiscuous = self._is_cluster_promiscuous(target_cluster)

            if query_promiscuous or target_promiscuous:
                pred.promiscuity_flag = True
                removed.append(pred)
            else:
                filtered.append(pred)

        logger.info(f"Filtered {len(predictions)} predictions:")
        logger.info(f"  Kept: {len(filtered)}")
        logger.info(f"  Removed (promiscuous): {len(removed)}")

        return filtered, removed

    def get_promiscuity_report(self) -> Dict:
        """Generate statistical report on domain promiscuity

        Returns:
            Dictionary with promiscuity statistics
        """
        if not self.promiscuity_scores:
            return {}

        link_counts = [ps.num_links for ps in self.promiscuity_scores.values()]

        report = {
            'total_clusters': len(self.promiscuity_scores),
            'promiscuous_clusters': sum(1 for ps in self.promiscuity_scores.values() if ps.is_promiscuous),
            'promiscuity_rate': sum(1 for ps in self.promiscuity_scores.values() if ps.is_promiscuous) / len(self.promiscuity_scores),
            'mean_links': float(np.mean(link_counts)),
            'median_links': float(np.median(link_counts)),
            'max_links': int(np.max(link_counts)),
            'min_links': int(np.min(link_counts)),
            'percentiles': {
                '95th': float(np.percentile(link_counts, 95)),
                '99th': float(np.percentile(link_counts, 99))
            }
        }

        # Distribution bins
        bins = [0, 5, 10, 25, 50, 100, 1000]
        hist, _ = np.histogram(link_counts, bins=bins)
        report['distribution'] = {
            f'{bins[i]}-{bins[i+1]}': int(hist[i])
            for i in range(len(hist))
        }

        return report

    def _load_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Load all domain embeddings from database

        Returns:
            Tuple of (embeddings array, domain IDs list)
        """
        embeddings_tensor = torch.load(self.fusion_db_dir / 'domain_embeddings.pt')
        embeddings = embeddings_tensor.cpu().numpy()

        with open(self.fusion_db_dir / 'domain_metadata.index', 'rb') as f:
            metadata = pickle.load(f)

        domain_ids = [entry[0] for entry in metadata]  # (domain_id, coords, seq)

        return embeddings, domain_ids

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings using HDBSCAN

        HDBSCAN advantages:
        - No need to specify number of clusters
        - Handles varying cluster densities
        - Identifies noise points
        - Works well with high-dimensional data

        Args:
            embeddings: Domain embeddings array

        Returns:
            Cluster labels array
        """
        clusterer = HDBSCAN(
            min_cluster_size=self.clustering_min_samples,
            min_samples=3,
            metric='euclidean',
            cluster_selection_method='eom',
            n_jobs=-1
        )

        cluster_labels = clusterer.fit_predict(embeddings)
        return cluster_labels

    def _count_cluster_links(self) -> Dict[int, Set[int]]:
        """Count how many other clusters each cluster links to via Rosetta Stones

        Returns:
            Dict mapping cluster_id -> set of linked cluster_ids
        """
        link_counts = defaultdict(set)

        # Load fusion metadata
        with open(self.fusion_db_dir / 'fusion_metadata.index', 'rb') as f:
            fusion_metadata = pickle.load(f)

        for fusion_dict in fusion_metadata:
            domain_A_id = fusion_dict['domain_A_id']
            domain_B_id = fusion_dict['domain_B_id']

            # Get cluster assignments
            cluster_A = self.cluster_assignments.get(domain_A_id, -1)
            cluster_B = self.cluster_assignments.get(domain_B_id, -1)

            # Skip noise clusters
            if cluster_A == -1 or cluster_B == -1:
                continue

            # Record bidirectional link
            link_counts[cluster_A].add(cluster_B)
            link_counts[cluster_B].add(cluster_A)

        return dict(link_counts)

    def _get_cluster_for_domain(self, domain: Domain) -> int:
        """Get cluster ID for a domain

        Args:
            domain: Domain instance

        Returns:
            Cluster ID (or -1 if not in database)
        """
        # Check if domain is in our registry
        if domain.domain_id in self.cluster_assignments:
            return self.cluster_assignments[domain.domain_id]

        # Otherwise, find nearest cluster centroid (for query domains)
        return self._assign_to_nearest_cluster(domain.embedding)

    def _assign_to_nearest_cluster(self, embedding: np.ndarray) -> int:
        """Assign embedding to nearest cluster centroid

        Args:
            embedding: Domain embedding vector

        Returns:
            Cluster ID
        """
        # Calculate centroids for each cluster
        if not hasattr(self, 'cluster_centroids'):
            self._compute_cluster_centroids()

        # Find nearest centroid
        similarities = {}
        for cluster_id, centroid in self.cluster_centroids.items():
            sim = np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid))
            similarities[cluster_id] = sim

        # Return cluster with highest similarity
        return max(similarities, key=similarities.get)

    def _compute_cluster_centroids(self) -> None:
        """Compute centroid embedding for each cluster"""
        self.cluster_centroids = {}

        # Load embeddings
        embeddings, domain_ids = self._load_all_embeddings()

        # Group by cluster
        cluster_embeddings = defaultdict(list)
        for emb, domain_id in zip(embeddings, domain_ids):
            cluster_id = self.cluster_assignments.get(domain_id, -1)
            if cluster_id != -1:
                cluster_embeddings[cluster_id].append(emb)

        # Calculate centroids
        for cluster_id, embs in cluster_embeddings.items():
            self.cluster_centroids[cluster_id] = np.mean(embs, axis=0)

    def _is_cluster_promiscuous(self, cluster_id: int) -> bool:
        """Check if cluster is promiscuous

        Args:
            cluster_id: Cluster identifier

        Returns:
            True if cluster is promiscuous
        """
        if cluster_id == -1:  # Noise cluster
            return False

        ps = self.promiscuity_scores.get(cluster_id)
        if ps is None:
            return False

        return ps.is_promiscuous

    def _save_index(self, output_path: Path) -> None:
        """Save promiscuity index to disk

        Args:
            output_path: Path to save index
        """
        data = {
            'cluster_assignments': self.cluster_assignments,
            'promiscuity_scores': self.promiscuity_scores,
            'threshold': self.promiscuity_threshold,
            'clustering_params': {
                'min_samples': self.clustering_min_samples
            }
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
