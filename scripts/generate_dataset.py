"""
Dataset Generation Script for Protein-Protein Interaction Data

This script generates a dataset of interacting and non-interacting peptide sequences
from a PostgreSQL database containing protein-protein interaction data.

The dataset contains:
- INTERACTING: Human protein + viral peptide pairs from curated interactions (~2k)
- NON_INTERACTING: Randomly sampled sequences from viral proteins (~500k)
  The length distribution (4-20 AA) matches the interacting sequences to prevent bias.

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

# Peptide length constraints
# Only peptides between 4 and 20 amino acids are considered valid
MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 20

# Target number of non-interacting samples
NON_INTERACTING_COUNT = 500_000


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
              Only includes proteins with sequences >= MIN_PEPTIDE_LENGTH
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
            if len(seq) >= MIN_PEPTIDE_LENGTH:
                viral_sequences.append((accession, seq))

    return viral_sequences


def compute_length_distribution(interacting_pairs):
    """
    Compute the length distribution of interacting peptide sequences.
    
    Args:
        interacting_pairs: Set of (human_accession, peptide_sequence) tuples
    
    Returns:
        tuple: (lengths list, probabilities list) for use with random.choices()
    """
    from collections import Counter
    
    length_counts = Counter(len(peptide) for _, peptide in interacting_pairs)
    
    # Sort by length for reproducibility
    lengths = sorted(length_counts.keys())
    counts = [length_counts[l] for l in lengths]
    total = sum(counts)
    probs = [c / total for c in counts]
    
    return lengths, probs


def sample_non_interacting(viral_sequences, interacting_peptides, count, length_distribution, verbose=False):
    """
    Randomly sample non-interacting peptide sequences from viral proteins.
    
    Sampling strategy:
    1. For each sample, pick a length according to the interacting length distribution
    2. Randomly select a viral protein and extract a substring of that length
    3. Reject any peptide that matches a known interacting sequence
    
    This ensures non-interacting samples have the same length distribution as
    interacting samples, preventing the model from using length as a discriminative feature.
    
    Args:
        viral_sequences: List of (accession, sequence) tuples
        interacting_peptides: Set of peptide sequences to exclude
        count: Target number of non-interacting samples
        length_distribution: Tuple of (lengths, probabilities) from compute_length_distribution()
        verbose: Whether to print progress to stderr
    
    Returns:
        set: Set of unique non-interacting peptide sequences
    """
    lengths, probs = length_distribution
    
    if verbose:
        import sys
        print(f"Total viral proteins: {len(viral_sequences)}", file=sys.stderr)
        print(f"Length distribution: {dict(zip(lengths, [f'{p:.1%}' for p in probs]))}", file=sys.stderr)

    # Pre-filter proteins by length for efficiency
    # For each possible length, keep only proteins that can provide that length
    proteins_by_min_length = {}
    for length in lengths:
        proteins_by_min_length[length] = [
            (acc, seq) for acc, seq in viral_sequences if len(seq) >= length
        ]

    non_interacting = set()
    attempts = 0
    max_attempts = count * 10  # Safety limit to prevent infinite loops

    while len(non_interacting) < count and attempts < max_attempts:
        attempts += 1
        
        # Pick a length according to the distribution
        sample_length = random.choices(lengths, weights=probs, k=1)[0]
        
        # Pick a random protein that can provide this length
        eligible_proteins = proteins_by_min_length[sample_length]
        if not eligible_proteins:
            continue
            
        accession, seq = random.choice(eligible_proteins)
        
        # Random position within the sequence
        max_pos = len(seq) - sample_length
        pos = random.randint(0, max_pos)
        peptide = seq[pos:pos + sample_length]

        # Accept only if not an interacting peptide and not already sampled
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

        # Step 2: Compute length distribution from interacting sequences
        length_distribution = compute_length_distribution(interacting_pairs)
        
        if verbose:
            lengths, probs = length_distribution
            print(f"Length distribution: min={min(lengths)}, max={max(lengths)}", file=sys.stderr)

        # Step 3: Get all viral protein sequences at their latest version
        viral_sequences = get_viral_sequences(cursor)

        if verbose:
            print(f"Found {len(viral_sequences)} viral proteins with sequences >= {MIN_PEPTIDE_LENGTH} aa", file=sys.stderr)
            print(f"Sampling {NON_INTERACTING_COUNT} non-interacting peptides...", file=sys.stderr)

        # Step 4: Sample non-interacting peptides from viral proteins
        # Uses the same length distribution as interacting sequences
        non_interacting = sample_non_interacting(
            viral_sequences, interacting_peptides, NON_INTERACTING_COUNT, length_distribution, verbose
        )

        # Step 5: Output the dataset as CSV
        import csv
        import os
        
        from pathlib import Path
        
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "dataset.csv"
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "target", "sequence"])
            
            # Interacting pairs: label=1, target=human_accession, sequence=peptide
            for human_accession, peptide in sorted(interacting_pairs):
                writer.writerow([1, human_accession, peptide])
            
            # Non-interacting: label=0, no target, sequence=peptide
            for peptide in sorted(non_interacting):
                writer.writerow([0, "", peptide])
        
        if verbose:
            print(f"Dataset written to {output_path}", file=sys.stderr)

        # Step 6: Print statistics if verbose
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
