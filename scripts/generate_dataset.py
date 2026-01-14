"""
Dataset Generation Script for Protein-Protein Interaction Data

This script generates a dataset of interacting and non-interacting peptide sequences
from a PostgreSQL database containing protein-protein interaction data.

The dataset contains:
- INTERACTING (label=1): Human protein + peptide pairs from curated interactions
  - Virus-Human (vh): mapping2 peptides target accession1 (human)
  - Human-Human (hh): mapping1 peptides target accession2, mapping2 peptides target accession1
- BACKGROUND (label=0): Random peptide sequences sampled from:
  - Human proteins: 500k samples
  - Viral proteins: 500k samples

Constraints:
- Peptide lengths: 4-20 amino acids
- Background peptide lengths follow the distribution of interacting peptides
- Background peptides are unique and not present in interacting set
- Background sampling is evenly distributed across proteins (round-robin)

Database Schema:
- dataset: Materialized view with protein-protein interactions
- proteins: Protein records with sequences (type 'h' or 'v')
- proteins_versions: Tracks protein version history for latest version filtering
"""

import argparse
import os
import random
import sys
from collections import Counter

import psycopg2
from dotenv import load_dotenv

load_dotenv()

MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 20
HUMAN_BACKGROUND_COUNT = 500_000
VIRAL_BACKGROUND_COUNT = 500_000


def get_db_connection():
    """Create a PostgreSQL database connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def extract_peptides_from_mapping(mapping):
    """
    Extract valid peptide sequences from a mapping JSON array.

    Yields peptides with length between MIN_PEPTIDE_LENGTH and MAX_PEPTIDE_LENGTH.
    """
    if not mapping:
        return

    for item in mapping:
        sequence = item.get("sequence", "")
        if MIN_PEPTIDE_LENGTH <= len(sequence) <= MAX_PEPTIDE_LENGTH:
            yield sequence


def get_hh_interacting_pairs(cursor):
    """
    Fetch human-human interacting peptide pairs.

    For hh interactions:
    - mapping1 peptides (from protein1) target accession2
    - mapping2 peptides (from protein2) target accession1
    """
    cursor.execute(
        """
        SELECT accession1, accession2, mapping1, mapping2
        FROM dataset
        WHERE type = 'hh'
          AND is_obsolete1 = false
          AND is_obsolete2 = false
          AND deleted_at IS NULL
    """
    )

    pairs = set()
    for accession1, accession2, mapping1, mapping2 in cursor:
        for peptide in extract_peptides_from_mapping(mapping1):
            pairs.add((accession2, peptide))
        for peptide in extract_peptides_from_mapping(mapping2):
            pairs.add((accession1, peptide))

    return pairs


def get_vh_interacting_pairs(cursor):
    """
    Fetch virus-human interacting peptide pairs.

    For vh interactions, mapping2 peptides (from viral protein) target accession1 (human).
    """
    cursor.execute(
        """
        SELECT accession1, mapping2
        FROM dataset
        WHERE type = 'vh'
          AND is_obsolete1 = false
          AND is_obsolete2 = false
          AND deleted_at IS NULL
          AND mapping2 IS NOT NULL
          AND mapping2::text != '[]'
    """
    )

    pairs = set()
    for human_accession, mapping2 in cursor:
        for peptide in extract_peptides_from_mapping(mapping2):
            pairs.add((human_accession, peptide))

    return pairs


def get_all_interacting_pairs(cursor, verbose=False):
    """
    Fetch all interacting peptide pairs from both vh and hh interactions.

    Returns:
        tuple: (unique_pairs, unique_peptides)
    """
    hh_pairs = get_hh_interacting_pairs(cursor)
    vh_pairs = get_vh_interacting_pairs(cursor)

    unique_pairs = hh_pairs | vh_pairs
    unique_peptides = {peptide for _, peptide in unique_pairs}

    if verbose:
        hh_peptides = {peptide for _, peptide in hh_pairs}
        vh_peptides = {peptide for _, peptide in vh_pairs}
        print(
            f"  Human peptides: {len(hh_peptides)} unique, {len(hh_pairs)} pairs",
            file=sys.stderr,
        )
        print(
            f"  Viral peptides: {len(vh_peptides)} unique, {len(vh_pairs)} pairs",
            file=sys.stderr,
        )
        print(
            f"  Total: {len(unique_peptides)} unique, {len(unique_pairs)} pairs",
            file=sys.stderr,
        )

    return unique_pairs, unique_peptides


def get_protein_sequences(cursor, protein_type):
    """
    Fetch canonical sequences for proteins of given type at their latest version.

    Args:
        protein_type: 'h' for human or 'v' for viral

    Returns:
        list: List of (accession, sequence) tuples
    """
    cursor.execute("SELECT MAX(current_version) FROM proteins_versions")
    latest_version = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT p.accession, p.sequences
        FROM proteins p
        JOIN proteins_versions pv ON p.accession = pv.accession AND p.version = pv.version
        WHERE p.type = %s
          AND pv.current_version = %s
          AND p.sequences IS NOT NULL
    """,
        (protein_type, latest_version),
    )

    sequences = []
    for accession, seq_data in cursor:
        if seq_data and accession in seq_data:
            seq = seq_data[accession]
            if len(seq) >= MIN_PEPTIDE_LENGTH:
                sequences.append((accession, seq))

    return sequences


def compute_length_distribution(peptides):
    """
    Compute length distribution of peptide sequences.

    Returns:
        tuple: (lengths, probabilities) for use with random.choices()
    """
    length_counts = Counter(len(p) for p in peptides)
    lengths = sorted(length_counts.keys())
    counts = [length_counts[l] for l in lengths]
    total = sum(counts)
    probs = [c / total for c in counts]
    return lengths, probs


def sample_background(
    protein_sequences, exclude_peptides, count, length_distribution, verbose=False
):
    """
    Sample background peptides using round-robin across shuffled proteins.

    Ensures even distribution across protein diversity, unique peptides,
    and length distribution matching interacting peptides.
    """
    lengths, probs = length_distribution

    if not protein_sequences:
        return set()

    shuffled = list(protein_sequences)
    random.shuffle(shuffled)

    min_length = min(lengths)
    eligible = [(acc, seq) for acc, seq in shuffled if len(seq) >= min_length]

    if verbose:
        print(
            f"    {len(eligible)} proteins eligible (len >= {min_length})",
            file=sys.stderr,
        )

    background = set()
    proteins_used = set()
    idx = 0
    attempts = 0
    max_attempts = count * 20

    while len(background) < count and attempts < max_attempts:
        attempts += 1
        accession, seq = eligible[idx % len(eligible)]
        idx += 1

        sample_length = random.choices(lengths, weights=probs, k=1)[0]
        if len(seq) < sample_length:
            continue

        pos = random.randint(0, len(seq) - sample_length)
        peptide = seq[pos : pos + sample_length]

        if peptide not in exclude_peptides and peptide not in background:
            background.add(peptide)
            proteins_used.add(accession)

    if verbose:
        print(
            f"    Sampled {len(background)} unique peptides from {len(proteins_used)} proteins",
            file=sys.stderr,
        )

    return background


def generate_dataset(verbose=False):
    """Generate the complete dataset and write to data/dataset.csv."""
    import csv
    from pathlib import Path

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        if verbose:
            print("Fetching interacting pairs...", file=sys.stderr)

        interacting_pairs, interacting_peptides = get_all_interacting_pairs(
            cursor, verbose
        )
        length_distribution = compute_length_distribution(interacting_peptides)

        if verbose:
            lengths, _ = length_distribution
            print(
                f"Peptide length distribution: min={min(lengths)}, max={max(lengths)}",
                file=sys.stderr,
            )
            print("Fetching protein sequences...", file=sys.stderr)

        human_sequences = get_protein_sequences(cursor, "h")
        viral_sequences = get_protein_sequences(cursor, "v")

        if verbose:
            print(f"  Human proteins: {len(human_sequences)}", file=sys.stderr)
            print(f"  Viral proteins: {len(viral_sequences)}", file=sys.stderr)
            print(
                f"Sampling {HUMAN_BACKGROUND_COUNT} background peptides from human proteins...",
                file=sys.stderr,
            )

        human_background = sample_background(
            human_sequences,
            interacting_peptides,
            HUMAN_BACKGROUND_COUNT,
            length_distribution,
            verbose,
        )

        if verbose:
            print(
                f"Sampling {VIRAL_BACKGROUND_COUNT} background peptides from viral proteins...",
                file=sys.stderr,
            )

        exclude_from_viral = interacting_peptides | human_background
        viral_background = sample_background(
            viral_sequences,
            exclude_from_viral,
            VIRAL_BACKGROUND_COUNT,
            length_distribution,
            verbose,
        )

        all_background = human_background | viral_background

        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "dataset.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "target", "sequence"])

            for target, peptide in sorted(interacting_pairs):
                writer.writerow([1, target, peptide])

            for peptide in sorted(all_background):
                writer.writerow([0, "", peptide])

        if verbose:
            total = len(interacting_pairs) + len(all_background)
            pct = len(interacting_pairs) / total * 100 if total > 0 else 0
            print(f"\nDataset written to {output_path}", file=sys.stderr)
            print("\n--- Statistics ---", file=sys.stderr)
            print(f"Interacting pairs: {len(interacting_pairs)}", file=sys.stderr)
            print(
                f"Background: {len(all_background)} (human: {len(human_background)}, viral: {len(viral_background)})",
                file=sys.stderr,
            )
            print(f"Total: {total}", file=sys.stderr)
            print(f"Interacting: {pct:.2f}%", file=sys.stderr)

        cursor.close()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PPI dataset from PostgreSQL")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Output statistics"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    generate_dataset(verbose=args.verbose)
