#!/usr/bin/env python
"""
Filter query interface for retrieving domain indices from SQLite filter database.

This module provides a simple interface for querying the filter database
created by metadata_extractor.py to get domain indices matching various criteria.
"""

import sqlite3
from typing import List, Optional, Dict


class FilterQuery:
    """
    Interface for querying filter database.

    Usage:
        fq = FilterQuery('examples/database/ted100_9606_small/filters.db')

        # Single filter
        indices = fq.filter_by_taxonomy(9606)

        # Combined filters
        indices = fq.filter_combined(
            taxonomy_id=9606,
            cath_fold='3.40.50.300',
            confidence='high'
        )
    """

    def __init__(self, filter_db_path: str):
        """
        Initialize filter query interface.

        Args:
            filter_db_path: Path to SQLite filter database
        """
        self.conn = sqlite3.connect(filter_db_path)
        self.cursor = self.conn.cursor()

    def filter_by_taxonomy(self, taxonomy_id: int) -> List[int]:
        """
        Get domain indices for specific taxonomy.

        Args:
            taxonomy_id: Taxonomy ID (e.g., 9606 for human)

        Returns:
            List of domain indices
        """
        self.cursor.execute(
            'SELECT domain_idx FROM domains WHERE taxonomy_id = ?',
            (taxonomy_id,)
        )
        return [row[0] for row in self.cursor.fetchall()]

    def filter_by_cath_fold(self, cath_fold: str) -> List[int]:
        """
        Get domain indices for specific CATH fold.

        Args:
            cath_fold: CATH fold ID (e.g., '3.40.50.300')

        Returns:
            List of domain indices
        """
        self.cursor.execute(
            'SELECT domain_idx FROM domains WHERE cath_fold = ?',
            (cath_fold,)
        )
        return [row[0] for row in self.cursor.fetchall()]

    def filter_by_confidence(self, confidence: str) -> List[int]:
        """
        Get domain indices for specific confidence level.

        Args:
            confidence: Confidence level ('high' or 'medium')

        Returns:
            List of domain indices
        """
        self.cursor.execute(
            'SELECT domain_idx FROM domains WHERE confidence = ?',
            (confidence,)
        )
        return [row[0] for row in self.cursor.fetchall()]

    def filter_by_globularity(self, min_score: float, max_score: float = 100.0) -> List[int]:
        """
        Get domain indices with globularity score in range.

        Args:
            min_score: Minimum globularity score
            max_score: Maximum globularity score

        Returns:
            List of domain indices
        """
        self.cursor.execute(
            'SELECT domain_idx FROM domains WHERE globularity_score BETWEEN ? AND ?',
            (min_score, max_score)
        )
        return [row[0] for row in self.cursor.fetchall()]

    def filter_combined(self,
                       taxonomy_id: Optional[int] = None,
                       cath_fold: Optional[str] = None,
                       confidence: Optional[str] = None,
                       min_globularity: Optional[float] = None,
                       max_globularity: float = 100.0) -> List[int]:
        """
        Get domain indices matching multiple criteria.

        Args:
            taxonomy_id: Optional taxonomy filter
            cath_fold: Optional CATH fold filter
            confidence: Optional confidence filter
            min_globularity: Optional minimum globularity score
            max_globularity: Optional maximum globularity score

        Returns:
            List of domain indices matching ALL criteria

        Example:
            # Human domains with CATH fold 3.40.50.300 and high confidence
            indices = fq.filter_combined(
                taxonomy_id=9606,
                cath_fold='3.40.50.300',
                confidence='high'
            )
        """
        conditions = []
        params = []

        if taxonomy_id is not None:
            conditions.append('taxonomy_id = ?')
            params.append(taxonomy_id)

        if cath_fold is not None:
            conditions.append('cath_fold = ?')
            params.append(cath_fold)

        if confidence is not None:
            conditions.append('confidence = ?')
            params.append(confidence)

        if min_globularity is not None:
            conditions.append('globularity_score BETWEEN ? AND ?')
            params.extend([min_globularity, max_globularity])

        if not conditions:
            # No filters - return all indices
            self.cursor.execute('SELECT domain_idx FROM domains ORDER BY domain_idx')
        else:
            query = f'SELECT domain_idx FROM domains WHERE {" AND ".join(conditions)}'
            self.cursor.execute(query, params)

        return [row[0] for row in self.cursor.fetchall()]

    def get_metadata(self, domain_idx: int) -> Dict:
        """
        Get full metadata for a domain.

        Args:
            domain_idx: Domain index

        Returns:
            Dictionary of metadata
        """
        self.cursor.execute(
            'SELECT * FROM domains WHERE domain_idx = ?',
            (domain_idx,)
        )
        row = self.cursor.fetchone()

        if row is None:
            return {}

        return {
            'domain_idx': row[0],
            'domain_id': row[1],
            'taxonomy_id': row[2],
            'species': row[3],
            'cath_fold': row[4],
            'confidence': row[5],
            'globularity_score': row[6],
            'architecture_class': row[7],
            'domain_length': row[8]
        }

    def get_statistics(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with counts by filter type
        """
        stats = {}

        # Total domains
        self.cursor.execute('SELECT COUNT(*) FROM domains')
        stats['total_domains'] = self.cursor.fetchone()[0]

        # Domains by taxonomy
        self.cursor.execute('''
            SELECT taxonomy_id, COUNT(*)
            FROM domains
            WHERE taxonomy_id IS NOT NULL
            GROUP BY taxonomy_id
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ''')
        stats['top_taxonomies'] = dict(self.cursor.fetchall())

        # Domains by CATH fold
        self.cursor.execute('''
            SELECT cath_fold, COUNT(*)
            FROM domains
            WHERE cath_fold != ''
            GROUP BY cath_fold
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ''')
        stats['top_cath_folds'] = dict(self.cursor.fetchall())

        # Domains by confidence
        self.cursor.execute('''
            SELECT confidence, COUNT(*)
            FROM domains
            GROUP BY confidence
        ''')
        stats['confidence_distribution'] = dict(self.cursor.fetchall())

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()
