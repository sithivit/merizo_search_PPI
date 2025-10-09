import argparse
import sys
import os
import shutil
import logging
import time
import uuid

from programs.Merizo.predict import run_merizo as segment_pdb
from programs.Foldclass.makedb import run_createdb as createdb_from_pdb
from programs.Foldclass.dbsearch import run_dbsearch as dbsearch
from programs.Foldclass.dbsearch_fulllength import multi_domain_search
from programs.utils import (
    parse_output_format,
    write_search_results,
    write_segment_results,
    write_all_dom_search_results,
    check_for_database
)

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPTDIR, 'programs'))

def munge_tmp_with_uuid(path: str) -> str:
    # make a uuid to be appended to tmp path name
    uuid_suffix = uuid.uuid4()
    return path.rstrip('/')+'-'+uuid_suffix.hex

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Function to handle segment mode
def segment(args):
    parser = argparse.ArgumentParser(description="Merizo segment is used to segment a multidomain protein.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, nargs="+", help="Specify path to pdb file input. Can also take multiple inputs (e.g. '/path/to/file.pdb' or '/path/to/*.pdb').")
    parser.add_argument("output", type=str, help="Output file prefix to write segment results to. Results will be called _segment.tsv.")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Hardware to run on. Options: 'cpu', 'cuda', 'mps'.")
    parser.add_argument("--merizo_output", type=str, default=os.getcwd(), help="Designate where to save the merizo outputs to.")
    parser.add_argument("--save_pdf", action="store_true", default=False, help="Include to save the domain map as a pdf.")
    parser.add_argument("--save_pdb", action="store_true", default=False,
                        help="Include to save the result as a pdb file. All domains will be included unless --conf_filter or --plddt_filter is used.")
    parser.add_argument("--save_domains", action="store_true", default=False, help="Include to save parsed domains as separate pdb files. Also saves the full pdb.")
    parser.add_argument("--save_fasta", action="store_true", default=False, help="Include to save a fasta file of the input pdb.")
    parser.add_argument("--output_headers", action="store_true", default=False, help="Print headers in output TSV files.")
    parser.add_argument("--conf_filter", type=float, default=None, help="(float, [0.0-1.0]) If specified, only domains with a pIoU above this threshold will be saved.")
    parser.add_argument("--plddt_filter", type=float, default=None,
                        help="(float, [0.0-1.0]) If specified, only domain with a plDDT above this threshold will be saved. Note: if used on a non-AF structure, this will correspond to crystallographic b-factors.")
    parser.add_argument("--iterate", action="store_true", help="If used, domains under a length threshold (see --min_domain_size) will be re-segmented.")
    parser.add_argument("--length_conditional_iterate", action="store_true", help=
                        "If used, --iterate is set to True when the input sequence length is greater than 512 residues or greater.")
    parser.add_argument("--max_iterations", type=int, default=3, help="(int [1, inf]) Specify the maximum number of re-segmentations that can occur.")
    parser.add_argument("--shuffle_indices", action="store_true", default=False, help="Shuffle domain indices - increases contrast between domain colours in PyMOL.")
    parser.add_argument("--return_indices", action="store_true", default=False, help="Return the domain indices for all residues.")
    parser.add_argument("--min_domain_size", type=int, default=50, help="The minimum domain size that is accepted.")
    parser.add_argument("--min_fragment_size", type=int, default=10, help="Minimum number of residues in a segment.")
    parser.add_argument("--domain_ave_size", type=int, default=200, help="[For iteration mode] Controls the size threshold to be used for further iterations.")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="[For iteration mode] Controls the minimum confidence to accept for iteration move.")
    parser.add_argument("--pdb_chain", type=str, dest="pdb_chain", default="A",
                        help="Select which PDB Chain you are analysing. Default is chain A for all input PDBs. You can provide a comma separated list if you can provide more than one input pdb.")
    parser.add_argument('-t', '--threads', type=int, default=-1, required=False, help="Number of CPU threads to use.")

    args = parser.parse_args(args)

    logging.info('Starting segment with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))

    segment_output = args.output + '_segment.tsv'
    if os.path.exists(segment_output):
        logging.warning(f"Segment output file '{segment_output}' already exists. Results will be overwritten!")

    start_time = time.time()

    _, segment_results = segment_pdb(
        input_paths=args.input,
        device=args.device,
        max_iterations=args.max_iterations,
        return_indices=args.return_indices,
        length_conditional_iterate=args.length_conditional_iterate,
        iterate=args.iterate,
        shuffle_indices=args.shuffle_indices,
        save_pdb=args.save_pdb,
        save_domains=args.save_domains,
        save_fasta=args.save_fasta,
        save_pdf=args.save_pdf,
        conf_filter=args.conf_filter,
        plddt_filter=args.plddt_filter,
        return_domains_as_list=True,
        conf_threshold=args.conf_threshold,
        merizo_output=args.merizo_output,
        pdb_chain=args.pdb_chain,
        threads=args.threads,
    )

    elapsed_time = time.time() - start_time
    logging.info(f'Finished segment in {elapsed_time} seconds.')

    write_segment_results(results=segment_results, output_file=segment_output, header=args.output_headers)

# Function to handle createdb mode
def createdb(args):
    parser = argparse.ArgumentParser(description="Call the createdb function of Foldclass to embed a collection of pdb files into a Foldclass database.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='Directory/.tar/.tar.gz/.zip containing PDB files. Will read all .pdb files inside.')
    parser.add_argument('out_db', type=str, help='Output prefix for the created Foldclass db.')
    parser.add_argument('-d', '--device', type=str, default='cpu', required=False, help="Hardware to run on. Options: 'cpu', 'cuda', 'mps'.")
    parser.add_argument('-t', '--tmpdir', type=str, required=False, help='Path to temporary directory; used only if input is not a directory.')
    args = parser.parse_args(args)

    logging.info('Starting createdb with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))

    start_time = time.time()

    createdb_from_pdb(
        pdb_files=args.input,
        out_db=args.out_db,
        device=args.device
    )

    elapsed_time = time.time() - start_time
    logging.info(f'Finished createdb in {elapsed_time} seconds.')

# Function to handle search mode
def search(args):
    parser = argparse.ArgumentParser(description="Calls the run_search function of Foldclass and searches query PDBs against a given database.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, nargs="+", help="Specify path to pdb file input.")
    parser.add_argument('db_name', type=str, help="Prefix of Foldclass database to search against.")
    parser.add_argument("output", type=str, help="Output file prefix to write search results to. Results will be called _search.tsv.")
    parser.add_argument('tmp', type=str, help="Temporary directory to write things to.")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Hardware to run vector search on. Options: 'cpu', 'cuda', 'mps'.")
    parser.add_argument('-k', '--topk', type=int, default=1, required=False, help="Max number of domain matches to return for each segmented domain.")
    parser.add_argument('-t', '--threads', type=int, default=-1, required=False, help="Number of CPU threads to use.")
    parser.add_argument('-s', '--mincos', type=float, default=0.5, required=False, help="(float, [0.0-1.0]) Filter hits by minumum cosine similarity to database matches.")
    parser.add_argument('-m', '--mintm', type=float, default=0.5, required=False, help="(float, [0.0-1.0]) Filter hits by minimum TM-align score to database matches.")
    parser.add_argument('-c', '--mincov', type=float, default=0.7, required=False, help="(float, [0.0-1.0]) Filter hits by minimum coverage of database matches.")
    parser.add_argument('-f', '--fastmode', action='store_true', required=False, help="Use the fast mode of TM-align to verify hits.")
    parser.add_argument("--format", type=str, default="query,emb_rank,target,emb_score,q_len,t_len,ali_len,seq_id,q_tm,t_tm,max_tm,rmsd,metadata",
                        help="Comma-separated list of variable names to output. Choose from: [query, target, emb_rank, emb_score, q_len, t_len, ali_len, seq_id, q_tm, t_tm, max_tm, rmsd].")
    parser.add_argument("--output_headers", action="store_true", default=False, help="Print headers in output TSV files.")
    parser.add_argument("--pdb_chain", type=str, dest="pdb_chain", default="A",
                        help="Select which PDB Chain you are analysing. Defaut is chain A. You can provide a comma separated list if more than one input pdb is provided.")
    parser.add_argument('--search_batchsize', type=int, default=262144, required=False,
                        help='For searches against Faiss databases, the search batchsize to use. Ignored otherwise.')
    parser.add_argument('--search_metric', type=str, default='IP', required=False,
                        help='For searches against Faiss databases, the search metric to use. Ignored otherwise. Currently only \'IP\' (cosine similarity) is supported.')
    parser.add_argument("--report_insignificant_hits", action="store_true", default=False,
                        help="Output a second results_search file that contains hits with TM-align scores less than --mintm threshold.")
    parser.add_argument("--metadata_json", action="store_true", default=False, help="Output metadata for hits in JSON format.")
    parser.add_argument("--multi_domain_search", action="store_true", default=False,
                        help="Search DB for entries that match all query domains (all query structures are treated as single domains coming from one chain).")
    parser.add_argument("--multi_domain_mode", type=str, default='exhaustive_tmalign', choices=['exhaustive_tmalign'],
                        help="If --multi_domain_search is used, specifies the multi-domain search mode. Currently only 'exhaustive_tmalign' is supported.")
                         #Run pairwise TM-align for each query domain and potential hit domain. If all query domains can be aligned (tm> --mintm) to domains in the hit, it is a full-length hit.")

    args = parser.parse_args(args)
    tmp = munge_tmp_with_uuid(args.tmp)
    logging.info('Starting search with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))

    # Check that the database is valid
    check_for_database(args.db_name)

    search_output = args.output + '_search.tsv'
    all_search_output = args.output + '_search_insignificant.tsv'
    if os.path.exists(search_output):
        logging.warning(f"Search output file '{search_output}' already exists. Results will be overwritten!")
    if os.path.exists(all_search_output):
        logging.warning(f"Search output file '{all_search_output}' already exists. Results will be overwritten!")

    if args.multi_domain_search:
        multi_domain_search_output = args.output + '_search_multi_dom.tsv'
        if os.path.exists(multi_domain_search_output):
            logging.warning(f"Multi-domain search output file '{multi_domain_search_output}' already exists. Results will be overwritten!")

    output_fields = parse_output_format(
        format_str=args.format,
        expected_str="query,emb_rank,target,emb_score,q_len,t_len,ali_len,seq_id,q_tm,t_tm,max_tm,rmsd,metadata"
    )

    start_time = time.time()

    search_results, all_search_results = dbsearch(
        inputs=args.input,
        db_name=args.db_name,
        tmp=tmp,
        device=args.device,
        topk=args.topk,
        fastmode=args.fastmode,
        threads=args.threads,
        mincos=args.mincos,
        mintm=args.mintm,
        mincov=args.mincov,
        inputs_are_ca=False,
        pdb_chain=args.pdb_chain,
        search_batchsize=args.search_batchsize,
        search_type=args.search_metric,
        skip_tmalign=False #args.multi_domain_search
    )
    write_search_results(results=search_results, output_file=search_output, format_list=output_fields, header=args.output_headers, metadata_json=args.metadata_json)
    if args.report_insignificant_hits:
        write_search_results(results=all_search_results, output_file=all_search_output, format_list=output_fields, header=args.output_headers,metadata_json=args.metadata_json)

    if args.multi_domain_search:
        fl_search_results = multi_domain_search(
            queries=args.input,
            search_results = search_results,
            db_name=args.db_name,
            tmp_root=tmp,
            device=args.device,
            fastmode=args.fastmode,
            threads=args.threads,
            mintm=args.mintm,
            inputs_from_easy_search=False,
            pdb_chain=args.pdb_chain,
            mode=args.multi_domain_mode
        )

        write_all_dom_search_results(fl_search_results, multi_domain_search_output, args.output_headers)

    elapsed_time = time.time() - start_time
    logging.info(f'Finished search in {elapsed_time} seconds.')
    shutil.rmtree(tmp)

# Function to handle easy-search mode
def easy_search(args):
    parser = argparse.ArgumentParser(description="Easy_search runs segment on a multidomain chain and then searches it against a database.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, nargs="+", help="Specify path to PDB file input. Can also take multiple inputs (e.g. '/path/to/file.pdb' or '/path/to/*.pdb').")
    parser.add_argument("db_name", type=str, help="Prefix of Foldclass database to search against.")
    parser.add_argument("output", type=str, help="Output file prefix to write segment and search results to. Results will be called _segment.tsv and _search.tsv.")
    parser.add_argument("tmp", type=str, help="Temporary directory to write things to.")
    parser.add_argument("--format", type=str, default="query,chopping,conf,plddt,emb_rank,target,emb_score,q_len,t_len,ali_len,seq_id,q_tm,t_tm,max_tm,rmsd,metadata",
                        help="Comma-separated list of variable names to output. Choose from: [query, target, conf, plddt, chopping, emb_rank, emb_score, q_len, t_len, ali_len, seq_id, q_tm, t_tm, max_tm, rmsd].")
    parser.add_argument("--output_headers", action="store_true", default=False, help="Print headers in output TSV files.")
    parser.add_argument("--multi_domain_search", action="store_true", default=False,
                        help="Search DB for entries that match all query domains for each query chain (domain ordering not currently considered).")
    # TODO this could be a subparser, has better-looking help output
    parser.add_argument("--multi_domain_mode", type=str, default='exhaustive_tmalign', choices=['exhaustive_tmalign'],
                        help="If --multi_domain_search is used, specifies the multi-domain search mode. Currently only 'exhaustive_tmalign' is supported.")
                        #: Run pairwise TM-align for each query domain and potential hit domain. If all query domains can be aligned (tm> --mintm) to domains in the hit, it is a full-length hit.")

    # TODO we could organise these into argument groups, will make help easier to understand
    # Foldclass (search) options
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Hardware to run vector search on. Options: 'cpu', 'cuda', 'mps'.")
    parser.add_argument('-k', '--topk', type=int, default=1, required=False, help="Max number of domain matches to return for each segmented domain.")
    parser.add_argument('-t', '--threads', type=int, default=-1, required=False, help="Number of CPU threads to use.")
    parser.add_argument('-s', '--mincos', type=float, default=0.5, required=False, help="(float, [0.0-1.0]) Filter hits by minumum cosine similarity to database matches.")
    parser.add_argument('-m', '--mintm', type=float, default=0.5, required=False, help="(float, [0.0-1.0]) Filter hits by minimum TM-align score to database matches.")
    parser.add_argument('-c', '--mincov', type=float, default=0.7, required=False, help="(float, [0.0-1.0]) Filter hits by minimum coverage of database matches.")
    parser.add_argument('-f', '--fastmode', action='store_true', required=False, help="Use the fast mode of TM-align to verify hits.")
    parser.add_argument('--search_batchsize', type=int, default=262144, required=False,
                        help='For searches against Faiss databases, the search batchsize to use. Ignored otherwise.')
    parser.add_argument('--search_metric', type=str, default='IP', required=False,
                        help='For searches against Faiss databases, the search metric to use. Ignored otherwise. Currently only \'IP\' (cosine similarity) is supported')
    parser.add_argument("--report_insignificant_hits", action="store_true", default=False,
                        help="Output a second results_search file that contains hits with TM-align scores less than the --mintm threshold.")
    parser.add_argument("--metadata_json", action="store_true", default=False, help="Output metadata for hits in JSON format.")

    # Merizo options
    parser.add_argument("--merizo_output", type=str, default=os.getcwd(), help="Designate where to save the merizo outputs to.")
    parser.add_argument("--save_pdf", action="store_true", default=False, help="Include to save the domain map as a pdf.")
    parser.add_argument("--save_pdb", action="store_true", default=False,
                        help="Include to save the result as a pdb file. All domains will be included unless --conf_filter and/or --plddt_filter are used.")
    parser.add_argument("--save_domains", action="store_true", default=False, help="Include to save parsed domains as separate pdb files. Also saves the full pdb.")
    parser.add_argument("--save_fasta", action="store_true", default=False, help="Include to save a fasta file of the input pdb.")
    parser.add_argument("--conf_filter", type=float, default=None,
                        help="(float, [0.0-1.0]) If specified, segmented domains will onyl be returned if they have a pIoU above this threshold. ")
    parser.add_argument("--plddt_filter", type=float, default=None,
                        help="(float, [0.0-1.0]) If specified, segmented domains will only be returned if they have a plDDT above this threshold. Note: if used on an X-ray structure, this will correspond to crystallographic B-factors.")
    parser.add_argument("--iterate", action="store_true",
                        help="If used, domains under a length threshold (see --min_domain_size) will be re-segmented.")
    parser.add_argument("--length_conditional_iterate", action="store_true",
                        help="If used, --iterate is set to True when the input sequence length is 512 residues or greater.")
    parser.add_argument("--max_iterations", type=int, default=3, help="(int [1, inf]) Specify the maximum number of re-segmentation passes that can occur.")
    parser.add_argument("--shuffle_indices", action="store_true", default=False, help="Shuffle domain indices - increases contrast between domain colours in PyMOL.")
    parser.add_argument("--return_indices", action="store_true", default=False, help="Return the domain indices for all residues.")
    parser.add_argument("--min_domain_size", type=int, default=50, help="The minimum domain size that is accepted.")
    parser.add_argument("--min_fragment_size", type=int, default=10, help="Minimum number of residues in a segment.")
    parser.add_argument("--domain_ave_size", type=int, default=200, help="[For iteration mode] Controls the size threshold to be used for further iterations.")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="[For iteration mode] Controls the minimum confidence to accept for iteration move.")
    parser.add_argument("--pdb_chain", type=str, dest="pdb_chain", default="A",
                        help="Select which PDB Chain you are analysing. Defaut is chain A. You can provide a comma separated list if you can provide more than one input pdb")

    args = parser.parse_args(args)
    tmp = munge_tmp_with_uuid(args.tmp)
    logging.info('Starting easy-search with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))

    # Check that the database is valid
    check_for_database(args.db_name)

    pdb_chain = args.pdb_chain.rstrip(",")
    pdb_chains = pdb_chain.split(",")

    if len(args.input) != len(pdb_chains):
        if len(pdb_chains) == 1:
            pdb_chains = pdb_chains * len(args.input)
        else:
            logging.error('Number of specified chain IDs not equal to number of input PDB files.')
            sys.exit(1)

    segment_output = args.output + '_segment.tsv'
    if os.path.exists(segment_output):
        logging.warning(f"Segment output file '{segment_output}' already exists. Results will be overwritten!")

    search_output = args.output + '_search.tsv'
    all_search_output = args.output + '_search_insignificant.tsv'
    if os.path.exists(search_output):
        logging.warning(f"Search output file '{search_output}' already exists. Results will be overwritten!")
    if os.path.exists(all_search_output):
        logging.warning(f"Search output file '{all_search_output}' already exists. Results will be overwritten!")

    if args.multi_domain_search:
        multi_domain_search_output = args.output + '_search_multi_dom.tsv'
        if os.path.exists(multi_domain_search_output):
            logging.warning(f"Multi-domain search output file '{multi_domain_search_output}' already exists. Results will be overwritten!")

    output_fields = parse_output_format(
        format_str=args.format,
        expected_str="query,chopping,conf,plddt,emb_rank,target,emb_score,q_len,t_len,ali_len,seq_id,q_tm,t_tm,max_tm,rmsd,metadata"
    )

    start_time = time.time()

    segment_domains, segment_results = segment_pdb(
        input_paths=args.input,
        device=args.device,
        max_iterations=args.max_iterations,
        return_indices=args.return_indices,
        length_conditional_iterate=args.length_conditional_iterate,
        iterate=args.iterate,
        shuffle_indices=args.shuffle_indices,
        save_pdb=args.save_pdb,
        save_domains=args.save_domains,
        save_fasta=args.save_fasta,
        save_pdf=args.save_pdf,
        conf_filter=args.conf_filter,
        plddt_filter=args.plddt_filter,
        return_domains_as_list=True,
        conf_threshold=args.conf_threshold,
        merizo_output=args.merizo_output,
        pdb_chain=args.pdb_chain,
        threads=args.threads
    )

    write_segment_results(results=segment_results, output_file=segment_output, header=args.output_headers)
    # if there are no valid inputs for the next step we should gracefully terminate here.
    if len(segment_domains) == 0:
        elapsed_time = time.time() - start_time
        logging.info('easy-search finished after segmentation. No domains could be identified.')
        logging.info(f'Finished easy-search in {elapsed_time} seconds.')
        exit(0)

    # pdb_chains_dict = {os.path.basename(k):v for k,v in zip(args.input, pdb_chains) }

    pdb_chains_for_search = []
    # for res in segment_results:
    #     for _ in range(res['num_domains']):
    #         pdb_chains_for_search.append(pdb_chains_dict[res['name']])
    for result in segment_results:
        chain = result['name'].split("_")[-1]
        nrep = result['num_domains']
        pdb_chains_for_search.extend([chain] * nrep)

    pdb_chains_for_search = ','.join(pdb_chains_for_search)

    search_results, all_search_results = dbsearch(
        inputs=segment_domains,
        db_name=args.db_name,
        tmp=tmp,
        device=args.device,
        topk=args.topk,
        fastmode=args.fastmode,
        threads=args.threads,
        mincos=args.mincos,
        mintm=args.mintm,
        mincov=args.mincov,
        inputs_are_ca=True,
        pdb_chain=pdb_chains_for_search,
        search_batchsize=args.search_batchsize,
        search_type=args.search_metric,
        skip_tmalign=False #args.multi_domain_search
    )

    write_search_results(results=search_results, output_file=search_output, format_list=output_fields, header=args.output_headers, metadata_json=args.metadata_json)
    if args.report_insignificant_hits:
        write_search_results(results=all_search_results, output_file=all_search_output, format_list=output_fields, header=args.output_headers, metadata_json=args.metadata_json)

    if args.multi_domain_search:
        fl_search_results = multi_domain_search(
            queries=segment_domains,
            search_results = search_results,
            db_name=args.db_name,
            tmp_root=tmp,
            device=args.device,
            fastmode=args.fastmode,
            threads=args.threads,
            mintm=args.mintm,
            inputs_from_easy_search=True,
            mode=args.multi_domain_mode,
            pdb_chain=None
        )
        if fl_search_results is not None:
            write_all_dom_search_results(fl_search_results, multi_domain_search_output, args.output_headers)
    elapsed_time = time.time() - start_time
    logging.info(f'Finished easy-search in {elapsed_time:.3f} seconds.')
    shutil.rmtree(tmp)


# Function to handle rosetta mode
def rosetta(args):
    """Rosetta Stone search for protein-protein interactions"""
    parser = argparse.ArgumentParser(
        description="Rosetta Stone search for protein-protein interactions via domain fusion analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='rosetta_command', help='Rosetta Stone commands')

    # Build database command
    build_parser = subparsers.add_parser('build', help='Build fusion database from multi-domain proteins')
    build_parser.add_argument('input', type=str, help='Directory of PDB files or file list')
    build_parser.add_argument('output', type=str, help='Output directory for fusion database')
    build_parser.add_argument('--min-domains', type=int, default=2,
                             help='Minimum domains per protein')
    build_parser.add_argument('--max-protein-size', type=int, default=1800,
                             help='Maximum protein size (residues) to prevent OOM. '
                                  'Default 1800 is safe for 6GB GPU. Increase for larger GPUs.')
    build_parser.add_argument('-d', '--device', type=str, default='cuda',
                             help='Device (cpu, cuda, mps)')
    build_parser.add_argument('--skip-promiscuity', action='store_true',
                             help='Skip promiscuity index building')
    build_parser.add_argument('--promiscuity-threshold', type=int, default=25,
                             help='Promiscuity threshold (number of links)')
    build_parser.add_argument('--batch-size', type=int, default=4,
                             help='Embedding batch size (reduce to 2 if GPU OOM)')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for protein-protein interactions')
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
    """Build fusion database for Rosetta Stone search"""
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
        max_protein_size=args.max_protein_size,
        device=args.device
    )

    builder.build_from_structure_list(structure_paths, batch_size=args.batch_size)

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
        if report:
            logging.info("Promiscuity Report:")
            logging.info(f"  Total clusters: {report['total_clusters']}")
            logging.info(f"  Promiscuous: {report['promiscuous_clusters']} ({report['promiscuity_rate']*100:.1f}%)")
            logging.info(f"  Mean links per cluster: {report['mean_links']:.1f}")
        else:
            logging.info("Promiscuity Report: No clusters found (dataset too small for clustering)")

    elapsed_time = time.time() - start_time
    logging.info(f'Finished fusion database build in {elapsed_time:.2f} seconds.')


def search_rosetta_interactions(args):
    """Search for protein-protein interactions using Rosetta Stone method"""
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


# Main function to parse arguments and call respective functions
def main():
    setup_logging()
    usage = """Usage: python merizo.py <mode> <args>
    <mode> is one of: 'segment', 'createdb', 'search', 'easy-search', or 'rosetta'.
    Detailed help is available for each mode:
        python merizo.py segment --help
        python merizo.py createdb --help
        python merizo.py search --help
        python merizo.py easy-search --help
        python merizo.py rosetta --help
    """

    if len(sys.argv) < 2:
        print(usage)
        return

    mode = sys.argv[1]
    args = sys.argv[2:]

    if mode == "segment":
        segment(args)
    elif mode == "createdb":
        createdb(args)
    elif mode == "search":
        search(args)
    elif mode == "easy-search":
        easy_search(args)
    elif mode == "rosetta":
        rosetta(args)
    elif mode == "-h" or mode == "--help":
        print(usage)
    else:
        print("Invalid mode. Please choose one of 'segment', 'createdb', 'search', 'easy-search', or 'rosetta'.")

if __name__ == "__main__":
    main()