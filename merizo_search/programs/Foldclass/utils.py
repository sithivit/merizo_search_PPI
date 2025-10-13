import os
import re
import uuid
import logging
import subprocess
from multiprocessing import Pool, cpu_count

import numpy as np

from .constants import single_to_three_aa, three_to_single_aa

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)

def write_pdb(tmp, coords, sequence, name=None):
    """
    Write a set of coordinates and sequence to a randomly named PDB file in a tmp dir.

    Args:
        tmp         (str)           Temporary directory for writing things into
        coords      (numpy.ndarray) Numpy array of size (N,3) containing CA coordinates
        sequence    (str)           One-letter aa codes as a single string

    Returns:
        filename    (str)           Name of the pdb file that was generated.

    """
    assert len(coords) == len(sequence), "Number of coordinates should match number of amino acids"

    if name is None:
        name = str(uuid.uuid4())

    filename = os.path.join(tmp, name + ".pdb")

    with open(filename, 'w') as pdb_file:
        for i, (coord, amino_acid) in enumerate(zip(coords, sequence), start=1):
            pdb_file.write(f"ATOM  {i: >5}  CA  {single_to_three_aa.get(amino_acid): >3} A{i: >4}    {coord[0]: >8.3f}{coord[1]: >8.3f}{coord[2]: >8.3f}  1.00  0.00\n")
        pdb_file.write("END\n")

    return filename


def read_pdb(pdbfile: str, pdb_chain: str="A"):# -> dict[str, Any]
    """
    Read the coordinates and sequence of a pdb file into a dict.

    Args:
        pdbfile (str): Path to a pdb file.

    Returns:
        dict:
            coords  (numpy.ndarray) Numpy array of size (N,3) containing CA coordinates
            seq     (str)           One-letter aa codes as a single string
            name    (str)           Name of the pdb file
    """
    if len(pdb_chain) != 1:
        logger.error("Invalid chain ID: '%s'" % pdb_chain)
        exit(127)
    with open(pdbfile, 'r') as fn:
        coords, seq = [], []
        for line in fn:
            if line[:5] == 'MODEL':
                logger.warning(f'Input PDB file {pdbfile} has MODEL records; only reading the first one.')
            if line[:4] == 'ATOM' and line[12:16] == ' CA ':
                if line[21] == pdb_chain:
                    pdb_fields = [line[:6], line[6:11], line[12:16], line[17:20], line[21], line[22:26], line[30:38], line[38:46], line[46:54]]
                    coords.append(np.array([float(pdb_fields[6]), float(pdb_fields[7]), float(pdb_fields[8])]))
                    seq.append(three_to_single_aa.get(pdb_fields[3], 'X'))
            if line[:6] == 'ENDMDL':
                break

    coords = np.asarray(coords, dtype=np.float32) #[:2000]
    sequence = ''.join(seq)
    if len(seq) == 0:
        logger.error("Chain ID '%s' not read from PDB file %s." % (pdb_chain, pdbfile))
        exit(128)
    return {'coords': coords, 'seq': sequence, 'name': pdbfile}


def run_tmalign(structure1_path: str, structure2_path: str, options: str = None, keep_pdbs=False) -> str:
    """
    Run TM-align as a subprocess.

    Args:
        structure1_path (str): Path to the first structure file.
        structure2_path (str): Path to the second structure file.
        options (str, optional): Additional options for TM-align.

    Returns:
        str: TM-align output.
    """
    tmalign_path = os.path.join(SCRIPTDIR, 'tmalign')

    # Run tmalign as a subprocess
    if options is None:
        process = subprocess.Popen([tmalign_path, structure1_path, structure2_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        process = subprocess.Popen([tmalign_path, structure1_path, structure2_path, options], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output, error = process.communicate()

    if process.returncode != 0:
        logger.error(f"Error running tmalign: {error}")
        return ""

    if not keep_pdbs:
        # Delete structure files
        try:
            os.remove(structure1_path)
            os.remove(structure2_path)
        except OSError as e:
            logger.error(f"Error deleting structure files: {e}")

    return extract_tmalign_values(output)


def extract_tmalign_values(tmalign_output: str, return_alignment: bool = False):
    """
    Parse the output from TMalign.

    Args:
        tmalign_output      (str)   TM-align output.
        return_alignment    (bool)  Whether to return the alignment lines (Default: False)

    Retrns:
        dict:
            len_ali     (int)   Length of the aligned region
            rmsd        (float) RMSD in angstroms
            seq_id      (float) sequence identity of the alignment
            qtm         (float) TMalign score normalised by the query length
            ttm         (float) TMalign score normalised by the target length
            alignment   (str)   Optional: returns the alignment lines from TMalign although this isn't really parsed
    """
    # Define regular expressions to extract values
    aligned_length_pattern = re.compile(r'Aligned length=\s*(\d+),\s+RMSD=\s*([0-9.]+),\s+Seq_ID=n_identical/n_aligned=\s*([0-9.]+)')
    tm_score_pattern = re.compile(r'TM-score=\s*([0-9.]+)')

    # Extract values using regular expressions
    aligned_length_match = aligned_length_pattern.search(tmalign_output)
    tm_score_matches = tm_score_pattern.finditer(tmalign_output)

    # Extract values
    aligned_length = int(aligned_length_match.group(1)) if aligned_length_match else None
    rmsd = float(aligned_length_match.group(2)) if aligned_length_match else None
    seq_identity = float(aligned_length_match.group(3)) if aligned_length_match else None
    tm_scores = [float(match.group(1)) for match in tm_score_matches]

    result = {
        'len_ali': aligned_length,
        'rmsd': rmsd,
        'seq_id': seq_identity,
        'qtm': tm_scores[0],
        'ttm': tm_scores[1],
    }

    if return_alignment:
        # Capture three lines of alignment
        alignment_start_index = tmalign_output.find('(":" denotes residue pairs')
        alignment_lines = tmalign_output[alignment_start_index:].split('\n')[1:4]

        result['alignment'] = alignment_lines

    return result


def run_tmalign2(args):
    """Wrapper function for run_tmalign, for use with Pool.map().

    Args:
        args (iterable): Positional arguments to run_tmalign().

    Returns:
        (str): Output of run_tmalign().
    """
    x, y, options, keep_pdbs = args
    return run_tmalign(x, y, options, keep_pdbs)


def pairwise_parallel_fill_tmalign_array(qfnames: list[str],
                                tfnames: list[str],
                                ncpu: int = -1,
                                mintm: float = 0.5,
                                options: str = None,
                                keep_pdbs: bool = True
                                ):
    """Multi-thread TM-align runs to fill a pairwise query-target matrix of
        TM-align scores.

    Args:
        qfnames (list[str]): Query PDB filenames
        tfnames (list[str]): Target PDB filename
        ncpu (int, optional): Number of parallel processes. Defaults to -1.
        mintm (float, optional): TM-align scores <= mintm are set to zero.
            Defaults to 0.5.
        options (str, optional): options for TM-align. Defaults to None.
        keep_pdbs (bool, optional): Whether to keep the PDB files after
            alignment. Defaults to True.
        pairwise_q_t (bool, optional): Whether all possible pairs of `qfnames`
            and `tfnames` should be used for alignment. If False, `qfnames` and
            `tfnames` should have the same length

    Returns:
        np.array[float], shape:(len(qfnames), len(tfnames)): Pairwise TM-align
            scores. NB: only max(qTM, tTM) is returned.
    """
    nrow = len(qfnames)
    ncol = len(tfnames)

    if ncpu <= 0:  # wiseguy eh
        ncpu = min(nrow*ncol, cpu_count())

    # Create lists of all (i, j) combinations
    tm_args = [(qfname, tfname, options, keep_pdbs) for \
                qfname in qfnames for tfname in tfnames]

    with Pool(ncpu) as pool:
        results = pool.map(run_tmalign2, tm_args)

    tmalign_scores = [max(d['qtm'], d['ttm']) for d in results]
    tmalign_scores = np.asarray(tmalign_scores).reshape((nrow, ncol))

    tmalign_scores[tmalign_scores < mintm] = 0.0

    return tmalign_scores


def parallel_fill_tmalign_array(qfnames: list[str],
                                tfnames: list[str],
                                ncpu: int = -1,
                                options: str = None,
                                keep_pdbs: bool = True
                                ):
    """Multi-thread TM-align runs to create a pairwise query-target table of
        TM-align scores.

    Args:
        qfnames (list[str]): Query PDB filenames
        tfnames (list[str]): Target PDB filename
        ncpu (int, optional): Number of parallel processes. Defaults to -1.
        mintm (float, optional): TM-align scores <= mintm are set to zero.
            Defaults to 0.5.
        options (str, optional): options for TM-align. Defaults to None.
        keep_pdbs (bool, optional): Whether to keep the PDB files after
            alignment. Defaults to True.

    Returns:
        results (list[dict]): result dicts from run_tmalign
    """
    nrow = len(qfnames)
    ncol = len(tfnames)

    if ncpu <= 0:  # wiseguy eh
        ncpu = min(nrow*ncol, cpu_count())

    # check that qfnames and tfnames have same length
    assert nrow == ncol
    tm_args = [(qfname, tfname, options, keep_pdbs) for \
                (qfname, tfname) in zip(qfnames, tfnames)]

    with Pool(ncpu) as pool:
        results = pool.map(run_tmalign2, tm_args)

    return results
