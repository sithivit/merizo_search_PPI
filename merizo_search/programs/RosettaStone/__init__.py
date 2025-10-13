"""
RosettaStone: Protein-Protein Interaction Prediction via Structural Domain Fusion Analysis

This module implements the Rosetta Stone method for predicting protein-protein interactions
by identifying domain fusion patterns in multi-domain proteins.

Main components:
- FusionDatabaseBuilder: Build searchable database of domain fusions
- StructuralRosettaStoneSearch: Search for interaction predictions
- DomainPromiscuityFilter: Filter out promiscuous domains
"""

__version__ = "1.0.0"

# Import only if needed to avoid circular dependencies
def __getattr__(name):
    """Lazy imports to avoid loading heavy dependencies unless needed"""
    if name == 'Domain':
        from .data_structures import Domain
        return Domain
    elif name == 'FusionLink':
        from .data_structures import FusionLink
        return FusionLink
    elif name == 'InteractionPrediction':
        from .data_structures import InteractionPrediction
        return InteractionPrediction
    elif name == 'PromiscuityScore':
        from .data_structures import PromiscuityScore
        return PromiscuityScore
    elif name == 'FusionDatabaseBuilder':
        from .fusion_database import FusionDatabaseBuilder
        return FusionDatabaseBuilder
    elif name == 'StructuralRosettaStoneSearch':
        from .rosetta_search import StructuralRosettaStoneSearch
        return StructuralRosettaStoneSearch
    elif name == 'DomainPromiscuityFilter':
        from .promiscuity_filter import DomainPromiscuityFilter
        return DomainPromiscuityFilter
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
