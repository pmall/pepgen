"""
Amino acid tokenizer for peptide sequences.

Handles variable-length peptides (4-20 aa) with special tokens:
- [PAD]: Padding for variable-length batching
- [MASK]: Masking for masked diffusion training

Vocabulary is built from the dataset to handle non-standard amino acids.
"""

import csv
from pathlib import Path

# Special tokens for masked diffusion
PAD_TOKEN = "<PAD>"   # Index 0: padding for variable-length sequences
MASK_TOKEN = "<MASK>"  # Index 1: masking for diffusion training


class AminoAcidTokenizer:
    """
    Tokenizer for amino acid sequences with padding and masking support.
    
    Token layout:
        0: [PAD] - padding token for variable-length batching
        1: [MASK] - mask token for masked diffusion training
        2+: amino acids (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y, ...)
    """

    def __init__(self, vocab: list[str] | None = None):
        """
        Initialize tokenizer with a vocabulary.

        Args:
            vocab: List of amino acid characters. If None, must call build_vocab().
        """
        if vocab is not None:
            self._build_from_vocab(vocab)
        else:
            self.aa_to_idx = {}
            self.idx_to_aa = {}
            self.vocab_size = 0
            self.pad_idx = 0
            self.mask_idx = 1

    def _build_from_vocab(self, vocab: list[str]):
        """
        Build token mappings from vocabulary list.
        
        Token indices:
            0: [PAD]
            1: [MASK]  
            2+: amino acids (sorted alphabetically)
        """
        # Special tokens first
        self.aa_to_idx = {
            PAD_TOKEN: 0,
            MASK_TOKEN: 1,
        }
        
        # Amino acids start at index 2
        for i, aa in enumerate(sorted(vocab)):
            self.aa_to_idx[aa] = i + 2

        self.idx_to_aa = {v: k for k, v in self.aa_to_idx.items()}
        self.vocab_size = len(self.aa_to_idx)
        self.pad_idx = 0
        self.mask_idx = 1

    @classmethod
    def from_dataset(cls, dataset_path: str | Path) -> "AminoAcidTokenizer":
        """
        Build tokenizer vocabulary from a dataset CSV file.

        Args:
            dataset_path: Path to dataset.csv with 'sequence' column.

        Returns:
            AminoAcidTokenizer with vocabulary built from all sequences.
        """
        vocab = set()
        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vocab.update(row["sequence"])

        return cls(vocab=list(vocab))

    def encode(self, sequence: str, max_length: int = 20) -> list[int]:
        """
        Encode an amino acid sequence to token indices.

        Args:
            sequence: Amino acid sequence string.
            max_length: Maximum sequence length (will pad to this length).

        Returns:
            List of token indices, padded to max_length.
        """
        tokens = [self.aa_to_idx[aa] for aa in sequence]

        if len(tokens) < max_length:
            tokens = tokens + [self.pad_idx] * (max_length - len(tokens))
        elif len(tokens) > max_length:
            tokens = tokens[:max_length]

        return tokens

    def decode(self, tokens: list[int], strip_special: bool = True) -> str:
        """
        Decode token indices back to amino acid sequence.

        Args:
            tokens: List of token indices.
            strip_special: If True, remove PAD and MASK tokens from output.

        Returns:
            Amino acid sequence string.
        """
        sequence = []
        for idx in tokens:
            aa = self.idx_to_aa[idx]
            # Skip special tokens if requested
            if strip_special and aa in (PAD_TOKEN, MASK_TOKEN):
                if aa == PAD_TOKEN:
                    break  # PAD marks end of sequence
                continue   # Skip MASK tokens but continue
            sequence.append(aa)

        return "".join(sequence)

    def save(self, path: str | Path):
        """Save tokenizer vocabulary to file."""
        vocab = [self.idx_to_aa[i] for i in range(self.vocab_size)]
        with open(path, "w") as f:
            for aa in vocab:
                f.write(f"{aa}\n")

    @classmethod
    def load(cls, path: str | Path) -> "AminoAcidTokenizer":
        """Load tokenizer vocabulary from file."""
        with open(path) as f:
            vocab = [line.strip() for line in f]

        tokenizer = cls()
        tokenizer.aa_to_idx = {aa: i for i, aa in enumerate(vocab)}
        tokenizer.idx_to_aa = {i: aa for i, aa in enumerate(vocab)}
        tokenizer.vocab_size = len(vocab)
        tokenizer.pad_idx = tokenizer.aa_to_idx[PAD_TOKEN]
        tokenizer.mask_idx = tokenizer.aa_to_idx[MASK_TOKEN]
        return tokenizer
