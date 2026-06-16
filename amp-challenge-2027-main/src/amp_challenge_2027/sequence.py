"""Sequence encoding/decoding utilities — PyTorch version."""

from typing import List

import numpy as np
import torch

from amp_challenge_2027.config import STD_AMINO_ACIDS

# Mapping: amino acid char → integer (1–20).  0 is reserved for padding.
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(STD_AMINO_ACIDS)}
IDX_TO_AA = {i + 1: aa for i, aa in enumerate(STD_AMINO_ACIDS)}


def to_one_hot(sequences: List[str]) -> List[List[int]]:
    """Encode amino-acid strings as lists of integers 1–20."""
    return [[AA_TO_IDX[aa] for aa in seq] for seq in sequences]


def pad(encoded: List[List[int]], max_length: int = 25) -> np.ndarray:
    """Post-pad integer-encoded sequences to *max_length* with zeros."""
    result = np.zeros((len(encoded), max_length), dtype=np.int64)
    for i, seq in enumerate(encoded):
        length = min(len(seq), max_length)
        result[i, :length] = seq[:length]
    return result


def encode_sequences(sequences: List[str], max_length: int = 25) -> np.ndarray:
    """One-shot helper: string sequences → padded integer array."""
    return pad(to_one_hot(sequences), max_length)


def decode_indices(indices: np.ndarray) -> List[str]:
    """Convert integer-encoded (padded) array back to amino-acid strings."""
    result = []
    for row in indices:
        chars = []
        for idx in row:
            idx = int(idx)
            if idx == 0:
                break  # padding → end of sequence
            if idx in IDX_TO_AA:
                chars.append(IDX_TO_AA[idx])
        result.append("".join(chars))
    return result


def translate_peptide(encoded_peptide) -> str:
    """Translate a single integer-encoded peptide (1-D) to string."""
    alphabet = STD_AMINO_ACIDS
    return "".join(
        alphabet[int(el) - 1] if int(el) != 0 else "" for el in encoded_peptide
    )
