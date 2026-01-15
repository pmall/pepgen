"""
PyTorch Dataset for peptide sequences.

This module handles variable-length peptides (4-20 amino acids) with two strategies:
1. Static padding: Pad all sequences to max_length (20) - simple but wastes compute
2. Dynamic padding: Pad to longest sequence in batch - efficient GPU utilization

The dynamic_collate_fn implements strategy #2, which is recommended for training.
"""

import csv
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from .tokenizer import AminoAcidTokenizer


def create_dynamic_collate_fn(pad_idx: int = 0) -> Callable:
    """
    Create a collate function that pads sequences dynamically to batch-max length.
    
    WHY DYNAMIC PADDING?
    --------------------
    Instead of padding every sequence to the global max (20 amino acids), we pad 
    only to the longest sequence in the current batch. This optimization:
    
    1. Reduces memory usage: A batch of 8-AA peptides uses 40% less memory
    2. Speeds up attention: Transformer attention is O(L²), so shorter = faster
    3. Reduces padding noise: Fewer PAD tokens means cleaner gradient signals
    
    HOW IT WORKS
    ------------
    1. Find L_max = max length in the current batch
    2. Trim all pre-padded sequences to L_max
    3. Build pad_mask where True indicates PAD positions (for attention masking)
    
    Note: The PeptideDataset already pads to max_length=20. This function RE-TRIMS
    tokens to the actual needed length based on the 'length' field.
    
    Args:
        pad_idx: The token index for PAD (default 0, matching tokenizer.pad_idx)
    
    Returns:
        A collate_fn compatible with torch DataLoader
    
    Example:
        >>> collate_fn = create_dynamic_collate_fn(pad_idx=0)
        >>> dataloader = DataLoader(dataset, collate_fn=collate_fn)
    """
    def dynamic_collate_fn(batch: list[dict]) -> dict:
        """
        Collate samples with dynamic padding to the longest sequence in batch.
        
        Input batch items have:
            - tokens: [max_length] pre-padded to 20
            - length: scalar, actual sequence length
            - label: scalar, 0=background, 1=interacting  
            - target_id: scalar, target protein index or -1
        
        Output dict has:
            - tokens: [batch, L_max] trimmed to batch-max length
            - length: [batch] original lengths
            - label: [batch] interaction labels
            - target_id: [batch] target protein indices
            - pad_mask: [batch, L_max] boolean mask, True = PAD position
        """
        # Extract lengths and find batch maximum
        lengths = torch.stack([item["length"] for item in batch])  # [batch]
        max_len = lengths.max().item()
        
        # Trim tokens to L_max (they were pre-padded to 20)
        # This is the key optimization: we only keep what we need
        tokens = torch.stack([item["tokens"][:max_len] for item in batch])  # [batch, L_max]
        
        # Build padding mask for attention: True where position >= actual length
        # Example: length=5, max_len=8 → mask = [F, F, F, F, F, T, T, T]
        positions = torch.arange(max_len).unsqueeze(0)  # [1, L_max]
        pad_mask = positions >= lengths.unsqueeze(1)    # [batch, L_max] broadcast
        
        return {
            "tokens": tokens,
            "length": lengths,
            "label": torch.stack([item["label"] for item in batch]),
            "target_id": torch.stack([item["target_id"] for item in batch]),
            "pad_mask": pad_mask,  # Critical for masked attention and loss
        }
    
    return dynamic_collate_fn


class PeptideDataset(Dataset):
    """
    Dataset for peptide sequences with interaction labels and optional targets.

    Each sample contains:
        - sequence: Tokenized peptide sequence (padded to max_length)
        - length: Original sequence length (before padding)
        - label: 1 for interacting, 0 for non-interacting
        - target_id: Integer ID of target protein (-1 if no target)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer: AminoAcidTokenizer,
        max_length: int = 20,
        interacting_only: bool = False,
    ):
        """
        Load dataset from CSV file.

        Args:
            dataset_path: Path to dataset.csv
            tokenizer: AminoAcidTokenizer instance
            max_length: Maximum sequence length for padding
            interacting_only: If True, only load interacting samples (label=1)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.target_to_id = {}

        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row["label"])

                if interacting_only and label == 0:
                    continue

                target = row["target"]
                if target and target not in self.target_to_id:
                    self.target_to_id[target] = len(self.target_to_id)

                self.samples.append(
                    {
                        "sequence": row["sequence"],
                        "label": label,
                        "target": target,
                    }
                )

        self.id_to_target = {v: k for k, v in self.target_to_id.items()}
        self.num_targets = len(self.target_to_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        sequence = sample["sequence"]

        tokens = self.tokenizer.encode(sequence, self.max_length)

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "length": torch.tensor(len(sequence), dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "target_id": torch.tensor(
                self.target_to_id.get(sample["target"], -1), dtype=torch.long
            ),
        }


class BalancedPeptideDataset(Dataset):
    """
    Wrapper that provides balanced sampling between interacting and non-interacting.

    Oversamples the minority class (interacting) to achieve balance.
    """

    def __init__(self, base_dataset: PeptideDataset, oversample_ratio: float = 1.0):
        """
        Args:
            base_dataset: The underlying PeptideDataset
            oversample_ratio: Ratio of positive to negative samples (1.0 = balanced)
        """
        self._base_dataset = base_dataset
        
        # Forward key attributes from base dataset
        self.tokenizer = base_dataset.tokenizer
        self.max_length = base_dataset.max_length
        self.samples = base_dataset.samples
        self.target_to_id = base_dataset.target_to_id
        self.id_to_target = base_dataset.id_to_target
        self.num_targets = base_dataset.num_targets

        self.positive_indices = []
        self.negative_indices = []

        for i, sample in enumerate(base_dataset.samples):
            if sample["label"] == 1:
                self.positive_indices.append(i)
            else:
                self.negative_indices.append(i)

        num_neg = len(self.negative_indices)
        num_pos = len(self.positive_indices)

        target_pos = int(num_neg * oversample_ratio)
        repeat_factor = (target_pos // num_pos) + 1
        self.oversampled_positives = (self.positive_indices * repeat_factor)[
            :target_pos
        ]

        self.indices = self.negative_indices + self.oversampled_positives

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self._base_dataset[self.indices[idx]]
