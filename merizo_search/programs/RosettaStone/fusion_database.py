"""
Fusion Database Builder for Rosetta Stone Interaction Prediction.

This module builds a searchable database of domain fusions by:
1. Segmenting multi-domain proteins into individual domains
2. Computing structural embeddings for each domain
3. Identifying fusion links (domain co-occurrence patterns)
4. Storing the results in an efficient searchable format
"""

import os
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

# Import actual merizo-search APIs
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
        device: str = 'cuda',
        max_protein_size: int = 1800
    ):
        """Initialize the fusion database builder

        Args:
            output_dir: Directory to store database files
            min_domains_per_protein: Minimum domains required for fusion
            min_linker_length: Minimum linker length between domains
            max_linker_length: Maximum linker length between domains
            device: Device for neural network inference ('cuda', 'cpu', 'mps')
            max_protein_size: Maximum protein size (residues) to prevent OOM on small GPUs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_domains = min_domains_per_protein
        self.min_linker = min_linker_length
        self.max_linker = max_linker_length
        self.max_protein_size = max_protein_size
        self.device = get_device(device)

        # Set PyTorch memory allocator to avoid fragmentation
        # This helps prevent OOM errors from memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Log device information
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("CUDA not available, using CPU (this will be slow!)")

        # Initialize Merizo network
        logger.info("Loading Merizo network...")
        self.merizo = Merizo().to(self.device)
        weights_dir = os.path.join(os.path.dirname(__file__), '../Merizo/weights')
        self.merizo.load_state_dict(read_split_weight_files(weights_dir), strict=True)
        self.merizo.eval()
        logger.info(f"Merizo network on device: {next(self.merizo.parameters()).device}")

        # Initialize Foldclass network
        logger.info("Loading Foldclass network...")
        self.foldclass = FoldClassNet(128).to(self.device).eval()
        scriptdir = os.path.dirname(os.path.realpath(__file__))
        foldclass_weights = os.path.join(scriptdir, '../Foldclass/FINAL_foldclass_model.pt')
        self.foldclass.load_state_dict(
            torch.load(foldclass_weights, map_location=lambda storage, loc: storage),
            strict=False
        )
        logger.info(f"Foldclass network on device: {next(self.foldclass.parameters()).device}")

        # Storage paths
        self.embeddings_path = self.output_dir / 'domain_embeddings.pt'
        self.metadata_path = self.output_dir / 'domain_metadata.index'
        self.fusion_embeddings_path = self.output_dir / 'fusion_embeddings.pt'
        self.fusion_metadata_path = self.output_dir / 'fusion_metadata.index'

    def build_from_structure_list(
        self,
        structure_paths: List[Path],
        batch_size: int = 4  # Reduced to 4 for 6GB GPU
    ) -> None:
        """Build fusion database from list of structure files

        Args:
            structure_paths: List of PDB/CIF file paths
            batch_size: Batch size for embedding computation (GPU batching)
        """
        logger.info(f"Building fusion database from {len(structure_paths)} structures")

        # Use temporary lists that are periodically flushed
        domain_batch = []
        coord_batch = []
        domain_metadata_list = []
        fusion_metadata_list = []
        all_embeddings = []  # Accumulate as numpy arrays, convert to tensor at end
        domain_count = 0
        fusion_count = 0

        # Process structures
        for pdb_idx, pdb_path in enumerate(tqdm(structure_paths, desc="Processing structures")):
            try:
                # CRITICAL: Check protein size BEFORE segmentation to avoid OOM on large proteins
                # The IPA attention mechanism has O(N²) memory requirements
                # For 6GB GPU, proteins > 1800 residues often cause OOM
                num_residues = self._count_protein_residues(pdb_path)
                if num_residues > self.max_protein_size:
                    logger.warning(
                        f"SKIPPING {pdb_path.name}: {num_residues} residues (> {self.max_protein_size} limit). "
                        f"Protein too large for GPU. IPA attention would require ~{(num_residues/1000)**2 * 2:.1f} GB."
                    )
                    continue
                elif num_residues > int(self.max_protein_size * 0.67):  # Warn at 67% of limit
                    logger.info(f"{pdb_path.name} has {num_residues} residues (large - may be slow)")

                # Log GPU memory before segmentation
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    logger.info(f"GPU memory before {pdb_path.name}: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

                # Segment protein into domains
                logger.info(f"Segmenting {pdb_path.name}...")
                domains = self._segment_protein(pdb_path)

                # Skip if not multi-domain
                if len(domains) < self.min_domains:
                    logger.info(f"Skipping {pdb_path.name}: only {len(domains)} domain(s)")
                    continue

                logger.info(f"Found {len(domains)} domains in {pdb_path.name}, computing embeddings...")

                # Batch domains for GPU processing
                for domain in domains:
                    domain_batch.append(domain)
                    coord_batch.append(domain.ca_coordinates)

                    # Process batch when full
                    if len(coord_batch) >= batch_size:
                        self._process_domain_batch(domain_batch, coord_batch, domain_metadata_list, all_embeddings)
                        domain_count += len(domain_batch)
                        domain_batch = []
                        coord_batch = []

                # Find fusion links for this protein
                fusion_links = self._find_fusion_links(domains)
                logger.info(f"Found {len(fusion_links)} fusion links in {pdb_path.name}")

                # Store fusion metadata (embeddings computed after batching)
                for link in fusion_links:
                    fusion_metadata_list.append(link.to_dict())

                fusion_count += len(fusion_links)

                # Clear domain objects to free memory
                del domains

                # CRITICAL: Aggressive GPU cleanup after each protein to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for all GPU operations
                    torch.cuda.empty_cache()  # Free cached memory

                    # Force garbage collection
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()  # Free again after gc

                # Periodic checkpointing to prevent data loss
                if (pdb_idx + 1) % 50 == 0:
                    logger.info(f"Processed {pdb_idx + 1} structures, saving checkpoint...")
                    if len(domain_batch) > 0:
                        self._process_domain_batch(domain_batch, coord_batch, domain_metadata_list, all_embeddings)
                        domain_count += len(domain_batch)
                        domain_batch = []
                        coord_batch = []

                    # Save checkpoint of embeddings and metadata
                    if len(all_embeddings) > 0:
                        checkpoint_embeddings = torch.tensor(np.array(all_embeddings), dtype=torch.float32)
                        torch.save(checkpoint_embeddings, self.embeddings_path)
                        logger.info(f"Checkpoint: saved {len(all_embeddings)} domain embeddings")

                    if len(domain_metadata_list) > 0:
                        with open(self.metadata_path, 'wb') as f:
                            pickle.dump(domain_metadata_list, f)
                        logger.info(f"Checkpoint: saved {len(domain_metadata_list)} domain metadata")

                    if len(fusion_metadata_list) > 0:
                        with open(self.fusion_metadata_path, 'wb') as f:
                            pickle.dump(fusion_metadata_list, f)
                        logger.info(f"Checkpoint: saved {len(fusion_metadata_list)} fusion links")

                    # Force garbage collection
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Failed to process {pdb_path}: {e}")
                import traceback
                logger.debug(traceback.format_exc())

                # CRITICAL: Aggressive GPU memory cleanup on error
                # OOM errors leave tensors allocated, need to force cleanup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for all operations
                    torch.cuda.empty_cache()  # Free cached memory

                    # Force garbage collection to break references
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()  # Free again after gc

                continue

        # Process remaining domains
        if len(domain_batch) > 0:
            logger.info("Processing final batch of domains...")
            self._process_domain_batch(domain_batch, coord_batch, domain_metadata_list, all_embeddings)
            domain_count += len(domain_batch)

        # Save all data to final database
        logger.info(f"Saving final database with {domain_count} domains and {fusion_count} fusion links...")

        if len(all_embeddings) > 0:
            logger.info(f"Converting {len(all_embeddings)} embeddings to tensor...")
            final_embeddings = torch.tensor(np.array(all_embeddings), dtype=torch.float32)
            torch.save(final_embeddings, self.embeddings_path)
            logger.info(f"Saved embeddings: {final_embeddings.shape}")

        if len(domain_metadata_list) > 0:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(domain_metadata_list, f)
            logger.info(f"Saved {len(domain_metadata_list)} domain metadata entries")

        if len(fusion_metadata_list) > 0:
            with open(self.fusion_metadata_path, 'wb') as f:
                pickle.dump(fusion_metadata_list, f)
            logger.info(f"Saved {len(fusion_metadata_list)} fusion link entries")

        # Build domain registry (domain_id -> Domain object mapping)
        logger.info("Building domain registry...")
        domain_registry = {}
        for domain_id, ca_coords, sequence, embedding in domain_metadata_list:
            # Parse protein_id from domain_id (format: "protein_domain_N")
            parts = domain_id.rsplit('_domain_', 1)
            protein_id = parts[0] if len(parts) == 2 else domain_id

            # Create Domain object
            domain = Domain(
                domain_id=domain_id,
                protein_id=protein_id,
                chain_id='A',
                residue_range=(0, len(ca_coords)),  # Approximate, exact range in metadata
                residue_indices=np.arange(len(ca_coords)),
                ca_coordinates=ca_coords,
                sequence=sequence,
                embedding=embedding,
                confidence=1.0
            )
            domain_registry[domain_id] = domain

        registry_path = self.output_dir / 'domain_registry.pkl'
        with open(registry_path, 'wb') as f:
            pickle.dump(domain_registry, f)
        logger.info(f"Saved domain registry with {len(domain_registry)} domains")

        # Build fusion embeddings (concatenated domain pair embeddings)
        logger.info("Building fusion embeddings...")
        fusion_embeddings_list = []

        # Create embedding lookup for fast access
        embedding_lookup = {domain_id: emb for domain_id, _, _, emb in domain_metadata_list}

        for fusion_meta in fusion_metadata_list:
            domain_A_id = fusion_meta['domain_A_id']
            domain_B_id = fusion_meta['domain_B_id']

            # Get embeddings from lookup
            if domain_A_id in embedding_lookup and domain_B_id in embedding_lookup:
                emb_A = embedding_lookup[domain_A_id]
                emb_B = embedding_lookup[domain_B_id]

                # Concatenate embeddings [256]
                fusion_embedding = np.concatenate([emb_A, emb_B])
                fusion_embeddings_list.append(fusion_embedding)
            else:
                logger.warning(f"Missing embeddings for fusion: {domain_A_id} - {domain_B_id}")

        if len(fusion_embeddings_list) > 0:
            fusion_embeddings_tensor = torch.tensor(np.array(fusion_embeddings_list), dtype=torch.float32)
            torch.save(fusion_embeddings_tensor, self.fusion_embeddings_path)
            logger.info(f"Saved fusion embeddings: {fusion_embeddings_tensor.shape}")

        logger.info(f"Database built successfully: {domain_count} domains, {fusion_count} fusion links")

    def _process_domain_batch(
        self,
        domain_batch: List[Domain],
        coord_batch: List[np.ndarray],
        metadata_list: List[Tuple],
        embeddings_list: List[np.ndarray]
    ) -> None:
        """Process a batch of domains with GPU acceleration

        Args:
            domain_batch: List of Domain objects to process
            coord_batch: List of coordinate arrays
            metadata_list: List to append metadata tuples to
            embeddings_list: List to append embeddings to (as numpy arrays)
        """
        if len(coord_batch) == 0:
            return

        # Filter out any domains with too few residues (< 10 CA atoms)
        valid_indices = []
        filtered_coords = []
        filtered_domains = []

        for i, coords in enumerate(coord_batch):
            if coords.shape[0] >= 10:  # Minimum domain size
                valid_indices.append(i)
                filtered_coords.append(coords)
                filtered_domains.append(domain_batch[i])
            else:
                logger.warning(f"Skipping domain {domain_batch[i].domain_id}: only {coords.shape[0]} residues")

        if len(filtered_coords) == 0:
            logger.warning("No valid domains in batch after filtering")
            return

        # Process one domain at a time to avoid dimension mismatch issues
        # This is safer than batching and avoids padding-related problems
        for i, (domain, coords) in enumerate(zip(filtered_domains, filtered_coords)):
            try:
                # Add batch dimension: [1, n_residues, 3]
                coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Compute embedding
                with torch.no_grad():
                    embedding = self.foldclass(coords_tensor)  # [1, 128]

                # Move to CPU and remove batch dimension
                embedding_np = embedding.squeeze(0).cpu().numpy()  # [128]

                # Free GPU memory immediately
                del embedding, coords_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Append embedding to list (as numpy - stays in RAM, not GPU!)
                embeddings_list.append(embedding_np)

                # Store metadata
                metadata_list.append((
                    domain.domain_id,
                    domain.ca_coordinates,
                    domain.sequence,
                    embedding_np
                ))

            except Exception as e:
                logger.warning(f"Failed to compute embedding for {domain.domain_id}: {e}")
                # Skip this domain and continue with others
                continue

    def _count_protein_residues(self, pdb_path: Path) -> int:
        """Quickly count the number of residues in a PDB/CIF file

        This is used to filter out very large proteins that would cause OOM
        on small GPUs (e.g., 6GB).

        Args:
            pdb_path: Path to PDB/CIF file

        Returns:
            Number of CA atoms (= number of residues)
        """
        try:
            ca_count = 0
            with open(pdb_path, 'r') as f:
                for line in f:
                    # Check PDB format
                    if line.startswith('ATOM') and ' CA ' in line:
                        ca_count += 1
                    # Check mmCIF format
                    elif line.startswith('_atom_site.') or line.startswith('ATOM'):
                        # For CIF, we need to parse more carefully
                        if pdb_path.suffix == '.cif':
                            # Simple heuristic: count lines with 'CA' in atom name column
                            parts = line.split()
                            if len(parts) > 3 and parts[3] == 'CA':
                                ca_count += 1
            return ca_count
        except Exception as e:
            logger.warning(f"Failed to count residues in {pdb_path.name}: {e}")
            return 0

    def _segment_protein(self, pdb_path: Path) -> List[Domain]:
        """Segment protein structure into domains using Merizo

        Args:
            pdb_path: Path to PDB/CIF file

        Returns:
            List of Domain objects
        """
        # CRITICAL: Clear GPU memory BEFORE segmentation to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            import gc
            gc.collect()  # Force garbage collection

        # Ensure device is a string ('cuda' or 'cpu') for segment function
        device_str = str(self.device).replace('cuda:', 'cuda') if 'cuda' in str(self.device) else 'cpu'

        # Use Merizo's segment function - this runs on GPU
        # Wrap in no_grad to ensure no computation graphs are created
        with torch.no_grad():
            features = segment(
                pdb_path=str(pdb_path),
                network=self.merizo,
                device=device_str,
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

        # Extract domains from Merizo output
        domains = []
        protein_id = pdb_path.stem

        # IMPORTANT: Move all tensors to CPU immediately to free GPU memory
        # Get unique domain IDs (excluding 0 = non-domain)
        domain_ids_tensor = features['domain_ids'].squeeze(0).cpu()  # Move to CPU
        conf_res_tensor = features['conf_res'].squeeze(0).cpu()      # Move to CPU
        unique_domain_ids = torch.unique(domain_ids_tensor[domain_ids_tensor > 0])

        # Extract coordinates and sequence for each domain
        # Get CA coordinates from pdb structured array
        pdb_array = features['pdb']
        ca_atoms = pdb_array[pdb_array['n'] == 'CA']
        all_coords = np.column_stack([ca_atoms['x'], ca_atoms['y'], ca_atoms['z']])  # [N, 3]
        all_residue_indices = features['ri'].squeeze(0).cpu().numpy()  # Already moved to CPU

        # Get sequence from pdb
        from programs.Merizo.model.utils.features import pdb_to_fasta
        sequence = pdb_to_fasta(pdb_array)

        # Convert tensors to numpy to break torch references
        domain_ids_np = domain_ids_tensor.numpy()
        conf_res_np = conf_res_tensor.numpy()

        # CRITICAL: Free GPU memory immediately after extraction
        # Delete all references to features dict and its tensors
        del features, domain_ids_tensor, conf_res_tensor

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all GPU operations to complete
            torch.cuda.empty_cache()  # Free cached memory
            import gc
            gc.collect()  # Force Python garbage collection

        for domain_idx, domain_id in enumerate(unique_domain_ids):
            # Get mask for this domain
            domain_mask = (domain_ids_np == domain_id.item())

            # Extract domain data
            domain_coords = all_coords[domain_mask]
            domain_res_indices = all_residue_indices[domain_mask]
            domain_sequence = ''.join([sequence[i] for i in range(len(sequence)) if domain_mask[i]])

            # Get residue range
            res_start = int(domain_res_indices[0])
            res_end = int(domain_res_indices[-1])

            # Get domain confidence (use numpy array now)
            domain_conf = float(conf_res_np[domain_mask].mean())

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
        """Find all pairwise domain combinations in multi-domain protein

        These represent fusion events (Rosetta Stones).

        Args:
            domains: List of Domain objects from same protein

        Returns:
            List of FusionLink objects
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
    parser.add_argument('--max-protein-size', type=int, default=1800,
                        help='Maximum protein size (residues) to prevent OOM. '
                             'Default 1800 is safe for 6GB GPU. Increase for larger GPUs.')
    parser.add_argument('--batch-size', type=int, default=4, help='Embedding batch size (reduce to 2 if still OOM)')
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
        max_protein_size=args.max_protein_size,
        device=args.device
    )

    builder.build_from_structure_list(structure_paths, batch_size=args.batch_size)
