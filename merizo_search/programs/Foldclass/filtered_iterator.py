#!/usr/bin/env python
"""
Filtered database iterator for subset searching.

This module provides iterators that only yield specified domain embeddings,
allowing Merizo-search to search only a filtered subset of the database
instead of all domains.
"""

import numpy as np


def db_iterator_filtered(embeddings_mm, filtered_indices, batch_size=262144):
    """
    Iterator that yields only specified domain embeddings.

    This is a filtered version of dbutil.db_iterator() that only yields
    embeddings for domains matching filter criteria.

    Args:
        embeddings_mm: Memory-mapped embeddings array (shape: [DB_SIZE, 128])
        filtered_indices: List or array of domain indices to include
        batch_size: Number of domains per batch

    Yields:
        np.ndarray: Batch of embeddings (shape: [batch_size, 128])

    Example:
        # Original iterator (searches all 66,943 domains)
        dbi = db_iterator(embeddings_mm, batch_size=262144)

        # Filtered iterator (searches only 1,000 human domains)
        filtered_indices = [0, 5, 12, 15, ...]  # From filter query
        dbi_filtered = db_iterator_filtered(embeddings_mm, filtered_indices, batch_size=262144)
    """
    filtered_indices = np.array(filtered_indices, dtype=np.int64)
    n_filtered = len(filtered_indices)

    for start_idx in range(0, n_filtered, batch_size):
        end_idx = min(start_idx + batch_size, n_filtered)
        batch_indices = filtered_indices[start_idx:end_idx]

        # Extract embeddings for this batch
        batch_embeddings = embeddings_mm[batch_indices]

        yield batch_embeddings


class FilteredIndexMapper:
    """
    Maps filtered search results back to original database indices.

    When using filtered iterator, Faiss returns indices 0, 1, 2, ...
    relative to the filtered subset. This class maps them back to
    original database indices.

    Example:
        filtered_indices = [15, 42, 103, 205]  # Domains matching filter
        mapper = FilteredIndexMapper(filtered_indices)

        # Faiss returns index 2 (3rd domain in filtered subset)
        faiss_index = 2
        original_index = mapper.to_original(faiss_index)  # Returns 103
    """

    def __init__(self, filtered_indices):
        """
        Initialize mapper with filtered indices.

        Args:
            filtered_indices: List or array of domain indices in the filtered subset
        """
        self.filtered_indices = np.array(filtered_indices, dtype=np.int64)

    def to_original(self, filtered_idx):
        """
        Map filtered index to original database index.

        Args:
            filtered_idx: Index in filtered subset (int or array)

        Returns:
            Original database index (int or array)
        """
        if isinstance(filtered_idx, (list, np.ndarray)):
            return self.filtered_indices[filtered_idx]
        return self.filtered_indices[filtered_idx]

    def __len__(self):
        """Return number of domains in filtered subset."""
        return len(self.filtered_indices)
