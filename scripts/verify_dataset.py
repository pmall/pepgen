"""
Dataset Verification Script

Verifies that the generated dataset follows all constraints:
1. Peptide lengths are between 4-20 amino acids
2. Labels are valid (0 or 1)
3. Interacting rows have a target, background rows have empty target
4. Interacting pairs (target, sequence) are unique
5. Background sequences are unique
6. Background sequences are not present in interacting set
7. All sequences are unique across entire dataset (accounting for valid duplicates)
"""

import csv
import sys
from collections import Counter
from pathlib import Path


MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 20


def verify_dataset(dataset_path, verbose=False):
    """
    Verify dataset constraints.
    
    Returns:
        bool: True if all constraints pass, False otherwise
    """
    interacting_sequences = set()
    interacting_pairs = set()
    background_sequences = set()
    all_sequences = []
    
    interacting_lengths = []
    background_lengths = []
    
    # Track constraint violations
    invalid_labels = []
    invalid_lengths = []
    missing_targets = []
    unexpected_targets = []
    duplicate_interacting_pairs = []
    duplicate_background_seqs = []
    
    print("\n📋 Parsing dataset...", file=sys.stderr)
    
    with open(dataset_path, newline="") as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, start=2):
            label = row.get("label", "")
            target = row.get("target", "")
            sequence = row.get("sequence", "")
            
            # Check label validity
            if label not in ("0", "1"):
                invalid_labels.append((row_num, label))
                continue
            
            label = int(label)
            seq_len = len(sequence)
            
            # Check sequence length
            if seq_len < MIN_PEPTIDE_LENGTH or seq_len > MAX_PEPTIDE_LENGTH:
                invalid_lengths.append((row_num, seq_len))
            
            all_sequences.append(sequence)
            
            if label == 1:
                if not target:
                    missing_targets.append(row_num)
                
                pair = (target, sequence)
                if pair in interacting_pairs:
                    duplicate_interacting_pairs.append((row_num, pair))
                interacting_pairs.add(pair)
                interacting_sequences.add(sequence)
                interacting_lengths.append(seq_len)
            else:
                if target:
                    unexpected_targets.append((row_num, target))
                
                if sequence in background_sequences:
                    duplicate_background_seqs.append((row_num, sequence))
                background_sequences.add(sequence)
                background_lengths.append(seq_len)
    
    # Check overlap between background and interacting
    overlap = background_sequences & interacting_sequences
    
    # Check global sequence uniqueness
    seq_counts = Counter(all_sequences)
    duplicates = {seq: count for seq, count in seq_counts.items() if count > 1}
    
    invalid_duplicates = {}
    for seq, count in duplicates.items():
        if seq in background_sequences:
            invalid_duplicates[seq] = count
        else:
            pairs_with_seq = sum(1 for t, s in interacting_pairs if s == seq)
            if count != pairs_with_seq:
                invalid_duplicates[seq] = count
    
    # Report each constraint
    print("\n🔍 Checking constraints...\n", file=sys.stderr)
    
    all_passed = True
    
    # Constraint 1: Labels
    if invalid_labels:
        print(f"❌ Labels valid (0 or 1): FAILED - {len(invalid_labels)} invalid", file=sys.stderr)
        if verbose:
            for row_num, label in invalid_labels[:3]:
                print(f"     Row {row_num}: '{label}'", file=sys.stderr)
        all_passed = False
    else:
        print(f"✅ Labels valid (0 or 1)", file=sys.stderr)
    
    # Constraint 2: Sequence lengths
    if invalid_lengths:
        print(f"❌ Sequence lengths [{MIN_PEPTIDE_LENGTH}-{MAX_PEPTIDE_LENGTH}]: FAILED - {len(invalid_lengths)} invalid", file=sys.stderr)
        if verbose:
            for row_num, length in invalid_lengths[:3]:
                print(f"     Row {row_num}: length {length}", file=sys.stderr)
        all_passed = False
    else:
        print(f"✅ Sequence lengths [{MIN_PEPTIDE_LENGTH}-{MAX_PEPTIDE_LENGTH}]", file=sys.stderr)
    
    # Constraint 3: Interacting rows have target
    if missing_targets:
        print(f"❌ Interacting rows have target: FAILED - {len(missing_targets)} missing", file=sys.stderr)
        all_passed = False
    else:
        print(f"✅ Interacting rows have target", file=sys.stderr)
    
    # Constraint 4: Background rows have empty target
    if unexpected_targets:
        print(f"⚠️  Background rows have empty target: {len(unexpected_targets)} have values", file=sys.stderr)
    else:
        print(f"✅ Background rows have empty target", file=sys.stderr)
    
    # Constraint 5: Interacting pairs unique
    if duplicate_interacting_pairs:
        print(f"❌ Interacting pairs (target, sequence) unique: FAILED - {len(duplicate_interacting_pairs)} duplicates", file=sys.stderr)
        all_passed = False
    else:
        print(f"✅ Interacting pairs (target, sequence) unique", file=sys.stderr)
    
    # Constraint 6: Background sequences unique
    if duplicate_background_seqs:
        print(f"❌ Background sequences unique: FAILED - {len(duplicate_background_seqs)} duplicates", file=sys.stderr)
        all_passed = False
    else:
        print(f"✅ Background sequences unique", file=sys.stderr)
    
    # Constraint 7: No overlap between background and interacting
    if overlap:
        print(f"❌ Background not in interacting set: FAILED - {len(overlap)} overlaps", file=sys.stderr)
        if verbose:
            for seq in list(overlap)[:3]:
                print(f"     {seq}", file=sys.stderr)
        all_passed = False
    else:
        print(f"✅ Background not in interacting set", file=sys.stderr)
    
    # Constraint 8: Global sequence uniqueness (valid duplicates allowed)
    if invalid_duplicates:
        print(f"❌ Sequences unique (valid duplicates allowed): FAILED - {len(invalid_duplicates)} invalid", file=sys.stderr)
        if verbose:
            for seq, count in list(invalid_duplicates.items())[:3]:
                print(f"     {seq} appears {count} times", file=sys.stderr)
        all_passed = False
    else:
        valid_dups = len(duplicates) - len(invalid_duplicates)
        print(f"✅ Sequences unique ({valid_dups} valid duplicates: same peptide, different targets)", file=sys.stderr)
    
    # Statistics
    if verbose:
        print("\n📊 Statistics:", file=sys.stderr)
        print(f"   Interacting pairs: {len(interacting_pairs)}", file=sys.stderr)
        print(f"   Unique interacting sequences: {len(interacting_sequences)}", file=sys.stderr)
        print(f"   Background sequences: {len(background_sequences)}", file=sys.stderr)
        print(f"   Total rows: {len(all_sequences)}", file=sys.stderr)
        
        print("\n📏 Length distribution:", file=sys.stderr)
        interacting_dist = Counter(interacting_lengths)
        background_dist = Counter(background_lengths)
        
        all_lengths = sorted(set(interacting_dist.keys()) | set(background_dist.keys()))
        
        print(f"   {'Len':<4} {'Interacting':<18} {'Background':<18} {'Diff':<8}", file=sys.stderr)
        for length in all_lengths:
            int_count = interacting_dist.get(length, 0)
            bg_count = background_dist.get(length, 0)
            int_pct = int_count / len(interacting_lengths) * 100 if interacting_lengths else 0
            bg_pct = bg_count / len(background_lengths) * 100 if background_lengths else 0
            diff = bg_pct - int_pct
            print(f"   {length:<4} {int_pct:>5.1f}% ({int_count:<6})  {bg_pct:>5.1f}% ({bg_count:<7}) {diff:>+5.1f}%", file=sys.stderr)
    
    # Final result
    if all_passed:
        print("\n✅ All constraints passed", file=sys.stderr)
    else:
        print("\n❌ Some constraints failed", file=sys.stderr)
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify dataset constraints")
    parser.add_argument("dataset", nargs="?", help="Path to dataset CSV")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed statistics")
    args = parser.parse_args()
    
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = Path(__file__).parent.parent / "data" / "dataset.csv"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"📁 Verifying {dataset_path}", file=sys.stderr)
    success = verify_dataset(dataset_path, verbose=args.verbose)
    sys.exit(0 if success else 1)
