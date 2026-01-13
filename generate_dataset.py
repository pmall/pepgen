"""
Dataset Generation Script for Protein-Protein Interaction Data

This script generates a dataset of interacting and non-interacting peptide sequences
from a PostgreSQL database containing protein-protein interaction data.

The dataset contains:
- INTERACTING: Human protein + viral peptide pairs from curated interactions
- NON_INTERACTING: Randomly sampled 20aa sequences from viral proteins

Database Schema Overview:
-------------------------
- dataset: Materialized view containing protein-protein interactions
  - type: 'hh' (human-human) or 'vh' (virus-human)
  - accession1: Human protein UniProt accession (always human in vh)
  - accession2: Viral protein UniProt accession (in vh interactions)
  - mapping2: JSON array of interaction mappings with peptide sequences
  - is_obsolete1/is_obsolete2: Whether proteins are obsolete in UniProt
  - deleted_at: Soft delete timestamp

- proteins: Protein records with sequences
  - type: 'h' (human) or 'v' (viral)
  - accession: UniProt accession
  - version: UniProt release version (e.g., '2019_01', '2021_02')
  - sequences: JSON object mapping accession to amino acid sequence

- proteins_versions: Tracks protein version history
  - accession: UniProt accession
  - version: UniProt release when this record was imported
  - current_version: Latest release where this protein still exists
  
  To get the latest proteins, we join proteins with proteins_versions on
  (accession, version) and filter where current_version = max(current_version).
  This gives us all proteins that still exist in the latest UniProt release.
"""

import argparse
import os
import random

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Peptide length constraints for interacting sequences
# Only peptides between 4 and 20 amino acids are considered valid
MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 20

# Length of randomly sampled non-interacting peptides
SAMPLE_LENGTH = 20

# Target number of non-interacting samples
# This is set to make interacting samples ~1% of the final dataset
NON_INTERACTING_COUNT = 200_000


def get_db_connection():
    """Create a PostgreSQL database connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def get_interacting_pairs(cursor):
    """
    Fetch all valid virus-human interacting peptide pairs from the database.
    
    Query explanation:
    - type = 'vh': Only virus-human interactions (not human-human)
    - is_obsolete1/2 = false: Both proteins must still exist in UniProt
    - deleted_at IS NULL: Interaction hasn't been soft-deleted
    - mapping2 contains JSON array of peptide mappings for protein2 (viral)
    
    Returns:
        tuple: (unique_combinations set, interacting_peptides set)
        - unique_combinations: Set of (human_accession, peptide_sequence) tuples
        - interacting_peptides: Set of all peptide sequences (for exclusion later)
    """
    cursor.execute("""
        SELECT accession1, mapping2
        FROM dataset
        WHERE type = 'vh'
          AND is_obsolete1 = false
          AND is_obsolete2 = false
          AND deleted_at IS NULL
          AND mapping2 IS NOT NULL
          AND mapping2::text != '[]'
    """)

    unique_combinations = set()
    interacting_peptides = set()

    for row in cursor:
        human_accession = row[0]
        mappings = row[1]  # JSON array of mapping objects

        # Each mapping contains a 'sequence' field with the peptide sequence
        for mapping in mappings:
            sequence = mapping.get("sequence", "")
            seq_len = len(sequence)

            # Filter peptides by length (4-20 amino acids)
            if seq_len < MIN_PEPTIDE_LENGTH or seq_len > MAX_PEPTIDE_LENGTH:
                continue

            unique_combinations.add((human_accession, sequence))
            interacting_peptides.add(sequence)

    return unique_combinations, interacting_peptides


def get_viral_sequences(cursor):
    """
    Fetch canonical sequences for all viral proteins at their latest version.
    
    Query explanation:
    - proteins table contains protein records at specific UniProt releases (accession, version)
    - proteins_versions tracks each protein's lifecycle:
      - version: the UniProt release when this record was imported
      - current_version: the latest release where this protein still exists
    - Join on (accession, version) links each protein to its version metadata
    - Filter by current_version = latest ensures we get proteins that still exist
    - type = 'v': Only viral proteins
    - sequences is a JSON object like {"P0DTC2": "MFVFL..."}
      where the key is the accession and value is the amino acid sequence
    
    Returns:
        list: List of (accession, sequence) tuples for viral proteins
              Only includes proteins with sequences >= SAMPLE_LENGTH
    """
    # First, get the latest current_version
    cursor.execute("SELECT MAX(current_version) FROM proteins_versions")
    latest_version = cursor.fetchone()[0]

    cursor.execute("""
        SELECT p.accession, p.sequences
        FROM proteins p
        JOIN proteins_versions pv ON p.accession = pv.accession AND p.version = pv.version
        WHERE p.type = 'v'
          AND pv.current_version = %s
          AND p.sequences IS NOT NULL
    """, (latest_version,))

    viral_sequences = []
    for row in cursor:
        accession = row[0]
        sequences = row[1]  # JSON object: {accession: sequence}
        
        # Extract the canonical sequence (keyed by accession)
        if sequences and accession in sequences:
            seq = sequences[accession]
            # Only include if sequence is long enough to sample from
            if len(seq) >= SAMPLE_LENGTH:
                viral_sequences.append((accession, seq))

    return viral_sequences


def sample_non_interacting(viral_sequences, interacting_peptides, count, verbose=False):
    """
    Randomly sample non-interacting peptide sequences from viral proteins.
    
    Sampling strategy:
    1. Distribute samples evenly across all viral proteins
       - Each protein gets (count / num_proteins) samples
       - Remainder is distributed to first N proteins
    2. For each protein, randomly select start positions and extract 20aa windows
    3. Reject any peptide that matches a known interacting sequence
    4. If a protein can't provide enough samples, fill from random proteins
    
    Args:
        viral_sequences: List of (accession, sequence) tuples
        interacting_peptides: Set of peptide sequences to exclude
        count: Target number of non-interacting samples
        verbose: Whether to print progress to stderr
    
    Returns:
        set: Set of unique non-interacting peptide sequences
    """
    # Calculate total possible sampling positions across all proteins
    total_positions = sum(len(seq) - SAMPLE_LENGTH + 1 for _, seq in viral_sequences)

    if verbose:
        import sys
        print(f"Total viral proteins: {len(viral_sequences)}", file=sys.stderr)
        print(f"Total sample positions: {total_positions}", file=sys.stderr)

    # Distribute samples evenly across proteins
    samples_per_protein = count // len(viral_sequences)
    extra_samples = count % len(viral_sequences)

    non_interacting = set()
    attempts = 0
    max_attempts = count * 10  # Safety limit to prevent infinite loops

    # Phase 1: Sample evenly from each protein
    for idx, (accession, seq) in enumerate(viral_sequences):
        # First 'extra_samples' proteins get one additional sample
        target = samples_per_protein + (1 if idx < extra_samples else 0)
        protein_samples = 0
        protein_attempts = 0
        max_protein_attempts = target * 20

        while protein_samples < target and protein_attempts < max_protein_attempts:
            protein_attempts += 1
            attempts += 1
            
            # Random position within the sequence
            pos = random.randint(0, len(seq) - SAMPLE_LENGTH)
            peptide = seq[pos:pos + SAMPLE_LENGTH]

            # Accept only if not an interacting peptide and not already sampled
            if peptide not in interacting_peptides and peptide not in non_interacting:
                non_interacting.add(peptide)
                protein_samples += 1

        if attempts > max_attempts:
            break

    # Phase 2: Fill remaining quota from random proteins if needed
    while len(non_interacting) < count and attempts < max_attempts:
        attempts += 1
        accession, seq = random.choice(viral_sequences)
        pos = random.randint(0, len(seq) - SAMPLE_LENGTH)
        peptide = seq[pos:pos + SAMPLE_LENGTH]

        if peptide not in interacting_peptides and peptide not in non_interacting:
            non_interacting.add(peptide)

    return non_interacting


def generate_dataset(verbose=False):
    """
    Main function to generate the complete dataset.
    
    Output format (TSV to stdout):
        INTERACTING     <human_accession>   <peptide_sequence>
        NON_INTERACTING                     <peptide_sequence>
    
    Statistics (if verbose) go to stderr.
    """
    import sys

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Step 1: Get all interacting human-viral peptide pairs
        if verbose:
            print("Fetching interacting pairs...", file=sys.stderr)

        interacting_pairs, interacting_peptides = get_interacting_pairs(cursor)

        if verbose:
            print(f"Found {len(interacting_pairs)} interacting pairs", file=sys.stderr)
            print(f"Found {len(interacting_peptides)} unique interacting peptides", file=sys.stderr)
            print("Fetching viral sequences...", file=sys.stderr)

        # Step 2: Get all viral protein sequences at their latest version
        viral_sequences = get_viral_sequences(cursor)

        if verbose:
            print(f"Found {len(viral_sequences)} viral proteins with sequences >= {SAMPLE_LENGTH} aa", file=sys.stderr)
            print(f"Sampling {NON_INTERACTING_COUNT} non-interacting peptides...", file=sys.stderr)

        # Step 3: Sample non-interacting peptides from viral proteins
        non_interacting = sample_non_interacting(
            viral_sequences, interacting_peptides, NON_INTERACTING_COUNT, verbose
        )

        # Step 4: Output the dataset
        # Interacting pairs: flag + human accession + peptide
        for human_accession, peptide in sorted(interacting_pairs):
            print(f"INTERACTING\t{human_accession}\t{peptide}")

        # Non-interacting: flag + empty accession + peptide
        for peptide in sorted(non_interacting):
            print(f"NON_INTERACTING\t\t{peptide}")

        # Step 5: Print statistics if verbose
        if verbose:
            total = len(interacting_pairs) + len(non_interacting)
            pct = len(interacting_pairs) / total * 100 if total > 0 else 0
            print(file=sys.stderr)
            print("--- Statistics ---", file=sys.stderr)
            print(f"Interacting pairs: {len(interacting_pairs)}", file=sys.stderr)
            print(f"Non-interacting samples: {len(non_interacting)}", file=sys.stderr)
            print(f"Total dataset size: {total}", file=sys.stderr)
            print(f"Interacting percentage: {pct:.2f}%", file=sys.stderr)

        cursor.close()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from PostgreSQL")
    parser.add_argument("-v", "--verbose", action="store_true", help="Output statistics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    generate_dataset(verbose=args.verbose)
