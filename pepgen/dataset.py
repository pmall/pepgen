"""
PyTorch Dataset for peptide sequences.
"""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import AminoAcidTokenizer


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
