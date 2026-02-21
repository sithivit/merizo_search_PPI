#!/bin/bash
#
# search-potential-interactions.sh
#
# Predict protein-protein interactions via domain structural homology transfer.
#
# Usage:
#   ./search-potential-interactions.sh <protein_id> [options]
#
# Example:
#   ./search-potential-interactions.sh Q9Y6K1 -o -t 0.3
#

set -e  # Exit on error

# ============================================================================
# Configuration and Defaults
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Database paths (adjust if needed)
PAIRLIST_DB="${PROJECT_ROOT}/merizo_pairlist_db/ted_pairlist"
FILTER_DB="${PROJECT_ROOT}/merizo_pairlist_db/ted_pairlist_filters.db"
PAIR_LIST="/mnt/bigstore/ted/pair_list_20250128"
DOMAIN_SUMMARY="/mnt/bigstore/ted/ted_365.domain_summary.cath.globularity.taxid.tsv"

# Default parameters
FILTER_ORGANISM=0  # 0=off, 1=on
TM_THRESHOLD=0.5
TOPK=50
OUTPUT_DIR="${PROJECT_ROOT}/interaction_predictions"
SEARCH_BATCHSIZE=2097152

# ============================================================================
# Functions
# ============================================================================

usage() {
    cat << EOF
Usage: $0 <protein_id> [options]

Predict protein-protein interactions via domain structural homology.

Arguments:
  protein_id          Protein ID (e.g., Q9Y6K1, P12345)

Options:
  -o, --same-organism Filter to same organism only (default: off)
  -t, --tm-threshold  TM-align threshold (default: 0.5)
                      Lower = more hits but weaker similarity
                      Recommended: 0.5 (strict), 0.3 (permissive)
  -k, --topk          Number of top hits to return (default: 50)
  -d, --output-dir    Output directory (default: interaction_predictions/)
  -h, --help          Show this help message

Examples:
  # Standard search (TM ≥ 0.5, all organisms)
  $0 Q9Y6K1

  # Same organism only, lower threshold
  $0 Q9Y6K1 -o -t 0.3

  # Custom output directory
  $0 Q9Y6K1 -o -t 0.5 -d my_results/

Output:
  <output_dir>/<protein_id>/
    ├── domain_pairs.txt              # Available domain pairs from pair_list
    ├── domain_b.pdb                  # Extracted Domain B structure
    ├── search_results.tsv            # Full search results
    ├── candidate_domains.txt         # Top similar domains
    ├── candidate_proteins.txt        # Predicted interaction partners
    └── analysis_summary.txt          # Summary with quality tiers

EOF
    exit 1
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

# ============================================================================
# Argument Parsing
# ============================================================================

if [ $# -eq 0 ]; then
    usage
fi

PROTEIN_ID="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--same-organism)
            FILTER_ORGANISM=1
            shift
            ;;
        -t|--tm-threshold)
            TM_THRESHOLD="$2"
            shift 2
            ;;
        -k|--topk)
            TOPK="$2"
            shift 2
            ;;
        -d|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

PROTEIN_OUTPUT="${OUTPUT_DIR}/${PROTEIN_ID}"
mkdir -p "${PROTEIN_OUTPUT}"

log "=============================================="
log "Protein Interaction Prediction Workflow"
log "=============================================="
log "Protein ID: ${PROTEIN_ID}"
log "TM threshold: ${TM_THRESHOLD}"
log "Same organism filter: $([ ${FILTER_ORGANISM} -eq 1 ] && echo 'YES' || echo 'NO')"
log "Output directory: ${PROTEIN_OUTPUT}"
log ""

# ============================================================================
# Step 1: Find Domain Pairs
# ============================================================================

log "Step 1: Finding domain pairs for ${PROTEIN_ID} in pair_list..."

DOMAIN_PAIRS="${PROTEIN_OUTPUT}/domain_pairs.txt"

grep "AF-${PROTEIN_ID}-F1-model_v4" "${PAIR_LIST}" > "${DOMAIN_PAIRS}" || {
    error "Protein ${PROTEIN_ID} not found in pair_list. Is it a multi-domain protein?"
}

PAIR_COUNT=$(wc -l < "${DOMAIN_PAIRS}")
log "  Found ${PAIR_COUNT} domain pair(s)"

if [ "${PAIR_COUNT}" -eq 0 ]; then
    error "No domain pairs found. Protein might not be in the pairlist database."
fi

# Display pairs
log ""
log "Available domain pairs:"
cat -n "${DOMAIN_PAIRS}"
log ""

# ============================================================================
# Step 2: Select Domain Pair
# ============================================================================

if [ "${PAIR_COUNT}" -eq 1 ]; then
    SELECTED_PAIR=1
    log "Using the only available pair (line 1)"
else
    log "Multiple pairs found. Using first pair (line 1)."
    log "To use a different pair, edit the script or re-run with desired pair."
    SELECTED_PAIR=1
fi

# Parse the selected pair
PAIR_LINE=$(sed -n "${SELECTED_PAIR}p" "${DOMAIN_PAIRS}")
DOMAIN_A=$(echo "${PAIR_LINE}" | awk -F':| ' '{print $1}')
DOMAIN_B=$(echo "${PAIR_LINE}" | awk -F':| ' '{print $2}')
PAE=$(echo "${PAIR_LINE}" | awk '{print $3}')

log ""
log "Selected pair:"
log "  Domain A: ${DOMAIN_A}"
log "  Domain B: ${DOMAIN_B} (query for homology search)"
log "  PAE: ${PAE}"
log ""

# ============================================================================
# Step 3: Get Metadata and Taxonomy
# ============================================================================

log "Step 3: Retrieving domain metadata..."

METADATA_A=$(grep "^${DOMAIN_A}" "${DOMAIN_SUMMARY}" | head -1)
METADATA_B=$(grep "^${DOMAIN_B}" "${DOMAIN_SUMMARY}" | head -1)

if [ -z "${METADATA_B}" ]; then
    error "Domain ${DOMAIN_B} not found in domain_summary file"
fi

# Extract taxonomy ID from column 13 (format: proteome-tax_id-9606-18_v4)
TAXONOMY_ID=$(echo "${METADATA_B}" | awk -F'\t' '{print $13}' | sed 's/.*tax_id-\([0-9]*\)-.*/\1/')

if [ -z "${TAXONOMY_ID}" ]; then
    log "  Warning: Could not determine taxonomy ID"
    TAXONOMY_ID=""
else
    log "  Taxonomy ID: ${TAXONOMY_ID}"
fi

# Extract CATH fold from column 14
CATH_FOLD=$(echo "${METADATA_B}" | awk -F'\t' '{print $14}')
log "  CATH fold (Domain B): ${CATH_FOLD}"
log ""

# ============================================================================
# Step 4: Extract Domain B
# ============================================================================

log "Step 4: Extracting Domain B structure..."

DOMAIN_B_PDB="${PROTEIN_OUTPUT}/domain_b.pdb"

python "${SCRIPT_DIR}/extract_domain_pdb.py" \
    "${PAIRLIST_DB}.json" \
    "${DOMAIN_B}" \
    "${DOMAIN_B_PDB}" || error "Failed to extract domain PDB"

RESIDUE_COUNT=$(grep -c "^ATOM" "${DOMAIN_B_PDB}" || echo "0")
log "  Extracted ${RESIDUE_COUNT} CA atoms"
log "  Saved to: ${DOMAIN_B_PDB}"
log ""

# ============================================================================
# Step 5: Run Search
# ============================================================================

log "Step 5: Searching for structurally similar domains..."
log "  This may take 2-5 minutes for 201M domains..."

SEARCH_OUTPUT="${PROTEIN_OUTPUT}/search"

# Build search command
SEARCH_CMD="python ${PROJECT_ROOT}/merizo_search/merizo.py search \
    ${DOMAIN_B_PDB} \
    ${PAIRLIST_DB} \
    ${SEARCH_OUTPUT} \
    tmp \
    --search_batchsize ${SEARCH_BATCHSIZE} \
    --mintm ${TM_THRESHOLD} \
    --topk ${TOPK}"

# Add organism filter if requested
if [ ${FILTER_ORGANISM} -eq 1 ] && [ -n "${TAXONOMY_ID}" ]; then
    SEARCH_CMD="${SEARCH_CMD} --filter-db ${FILTER_DB} --filter-taxonomy ${TAXONOMY_ID}"
    log "  Filtering to taxonomy ${TAXONOMY_ID} only"
fi

# Execute search
eval ${SEARCH_CMD} || error "Search failed"

SEARCH_RESULTS="${SEARCH_OUTPUT}_search.tsv"

if [ ! -f "${SEARCH_RESULTS}" ]; then
    error "Search results file not found: ${SEARCH_RESULTS}"
fi

HIT_COUNT=$(tail -n +2 "${SEARCH_RESULTS}" | wc -l)
log "  Found ${HIT_COUNT} hits (TM ≥ ${TM_THRESHOLD})"
log ""

# ============================================================================
# Step 6: Extract Candidate Domains and Proteins
# ============================================================================

log "Step 6: Extracting candidate proteins..."

CANDIDATE_DOMAINS="${PROTEIN_OUTPUT}/candidate_domains.txt"
CANDIDATE_PROTEINS="${PROTEIN_OUTPUT}/candidate_proteins.txt"

# Extract top domain IDs (skip self-hit)
tail -n +2 "${SEARCH_RESULTS}" | \
    tail -n +2 | \
    head -20 | \
    cut -f3 > "${CANDIDATE_DOMAINS}"

# Extract protein IDs
cat "${CANDIDATE_DOMAINS}" | \
    sed 's/AF-\([^-]*\)-F1-model_v4.*/\1/' | \
    sort -u > "${CANDIDATE_PROTEINS}"

CANDIDATE_COUNT=$(wc -l < "${CANDIDATE_PROTEINS}")
log "  Found ${CANDIDATE_COUNT} unique candidate proteins"
log ""

# ============================================================================
# Step 7: Analyze Results
# ============================================================================

log "Step 7: Analyzing predictions..."

SUMMARY="${PROTEIN_OUTPUT}/analysis_summary.txt"

{
    echo "=============================================="
    echo "Protein Interaction Prediction Summary"
    echo "=============================================="
    echo ""
    echo "Query Protein: ${PROTEIN_ID}"
    echo "Domain Pair: ${DOMAIN_A} ↔ ${DOMAIN_B}"
    echo "PAE: ${PAE}"
    echo "CATH Fold (Domain B): ${CATH_FOLD}"
    echo "Taxonomy: ${TAXONOMY_ID}"
    echo ""
    echo "Search Parameters:"
    echo "  TM threshold: ${TM_THRESHOLD}"
    echo "  Organism filter: $([ ${FILTER_ORGANISM} -eq 1 ] && echo 'YES' || echo 'NO')"
    echo "  Total hits: ${HIT_COUNT}"
    echo ""
    echo "=============================================="
    echo "Quality Tiers (by TM-score)"
    echo "=============================================="
    echo ""

    # High confidence (TM ≥ 0.9)
    echo "⭐⭐⭐ STRONG Predictions (TM ≥ 0.9):"
    tail -n +2 "${SEARCH_RESULTS}" | \
        awk -v cath="${CATH_FOLD}" '$11 >= 0.9 {
            match($13, /"cath": "([^"]*)"/, arr);
            cath_match = (arr[1] == cath) ? "✓ SAME CATH" : "  " arr[1];
            printf "  Rank %2d: %s (TM=%.2f) %s\n", $2, $3, $11, cath_match
        }' | head -10

    echo ""

    # Medium confidence (0.5 ≤ TM < 0.9)
    echo "⭐⭐ MEDIUM Predictions (0.5 ≤ TM < 0.9):"
    tail -n +2 "${SEARCH_RESULTS}" | \
        awk '$11 >= 0.5 && $11 < 0.9 {printf "  Rank %2d: %s (TM=%.2f)\n", $2, $3, $11}' | \
        head -10

    echo ""

    # Low confidence (TM < 0.5)
    echo "⭐ WEAK Predictions (TM < 0.5):"
    tail -n +2 "${SEARCH_RESULTS}" | \
        awk '$11 < 0.5 {printf "  Rank %2d: %s (TM=%.2f)\n", $2, $3, $11}' | \
        head -10

    echo ""
    echo "=============================================="
    echo "Candidate Interaction Partners (Top 20)"
    echo "=============================================="
    echo ""

    # Analyze each candidate protein
    for protein in $(head -20 "${CANDIDATE_PROTEINS}"); do
        # Count domains
        domain_count=$(grep "^AF-${protein}" "${DOMAIN_SUMMARY}" 2>/dev/null | wc -l || echo "?")

        # Get TM score for this protein's domain
        domain_id=$(grep "AF-${protein}" "${CANDIDATE_DOMAINS}" | head -1)
        tm_score=$(grep "${domain_id}" "${SEARCH_RESULTS}" | awk '{print $11}')
        rank=$(grep "${domain_id}" "${SEARCH_RESULTS}" | awk '{print $2}')

        if [ "${domain_count}" -eq 1 ]; then
            echo "  ${protein} - SINGLE-DOMAIN (Rank ${rank}, TM=${tm_score}) ✓ simpler"
        else
            echo "  ${protein} - MULTI-DOMAIN (${domain_count} domains, Rank ${rank}, TM=${tm_score})"
        fi
    done

    echo ""
    echo "=============================================="
    echo "Biological Interpretation"
    echo "=============================================="
    echo ""
    echo "HYPOTHESIS:"
    echo "  Protein ${PROTEIN_ID} contains Domain A (${DOMAIN_A##*_}) and Domain B (${DOMAIN_B##*_})."
    echo "  The candidate proteins contain domains structurally similar to Domain B."
    echo ""
    echo "PREDICTION:"
    echo "  ${PROTEIN_ID} might interact with these proteins via"
    echo "  Domain A binding to their Domain-B-like domains."
    echo ""
    echo "CONFIDENCE RANKING:"
    echo "  1. STRONG (TM ≥ 0.9, same CATH): High confidence"
    echo "  2. MEDIUM (0.5 ≤ TM < 0.9): Moderate confidence"
    echo "  3. WEAK (TM < 0.5): Low confidence, might be false positives"
    echo ""
    echo "CAVEATS:"
    echo "  ✓ Same organism: $([ ${FILTER_ORGANISM} -eq 1 ] && echo 'YES - filtered' || echo 'NO - check taxonomy')"
    echo "  ? Same cellular compartment: Unknown (requires additional data)"
    echo "  ? Same expression timing: Unknown (requires expression data)"
    echo "  ? Domain accessibility: Unknown for multi-domain proteins"
    echo ""
    echo "NEXT STEPS:"
    echo "  1. Prioritize STRONG predictions (TM ≥ 0.9, same CATH fold)"
    echo "  2. Focus on single-domain targets (simpler validation)"
    echo "  3. Check known PPI databases (STRING, BioGRID) for validation"
    echo "  4. Experimental validation: Co-IP, Y2H, etc."
    echo ""

} > "${SUMMARY}"

cat "${SUMMARY}"

# ============================================================================
# Final Output
# ============================================================================

log ""
log "=============================================="
log "Workflow Complete!"
log "=============================================="
log ""
log "Output files:"
log "  Domain pairs:        ${DOMAIN_PAIRS}"
log "  Extracted PDB:       ${DOMAIN_B_PDB}"
log "  Search results:      ${SEARCH_RESULTS}"
log "  Candidate domains:   ${CANDIDATE_DOMAINS}"
log "  Candidate proteins:  ${CANDIDATE_PROTEINS}"
log "  Analysis summary:    ${SUMMARY}"
log ""
log "Top 5 candidate proteins:"
head -5 "${CANDIDATE_PROTEINS}" | while read protein; do
    log "  - ${protein}"
done
log ""
log "Review the full analysis: cat ${SUMMARY}"
log ""
