import argparse
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 20


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def generate_dataset(verbose=False):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

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
        total_interactions = 0
        total_peptides_checked = 0
        peptides_filtered_out = 0

        for row in cursor:
            total_interactions += 1
            human_accession = row[0]
            mappings = row[1]

            for mapping in mappings:
                sequence = mapping.get("sequence", "")
                total_peptides_checked += 1
                seq_len = len(sequence)

                if seq_len < MIN_PEPTIDE_LENGTH or seq_len > MAX_PEPTIDE_LENGTH:
                    peptides_filtered_out += 1
                    continue

                unique_combinations.add((human_accession, sequence))

        for human_accession, peptide in sorted(unique_combinations):
            print(f"{human_accession}\t{peptide}")

        if verbose:
            print(file=__import__("sys").stderr)
            print("--- Statistics ---", file=__import__("sys").stderr)
            print(f"Total vh interactions: {total_interactions}", file=__import__("sys").stderr)
            print(f"Total peptides checked: {total_peptides_checked}", file=__import__("sys").stderr)
            print(f"Peptides filtered out (length): {peptides_filtered_out}", file=__import__("sys").stderr)
            print(f"Unique (human_accession, peptide) pairs: {len(unique_combinations)}", file=__import__("sys").stderr)

        cursor.close()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from PostgreSQL")
    parser.add_argument("-v", "--verbose", action="store_true", help="Output statistics")
    args = parser.parse_args()

    generate_dataset(verbose=args.verbose)
