#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Peptide graph dataset for PyTorch Geometric.
Converts peptide sequences into molecular graph representations with
physicochemical node features and spatial/bond edge features.
"""

import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data, Batch
    _PYG_AVAILABLE = True
except Exception:
    _PYG_AVAILABLE = False
    Data = None  # type: ignore
    Batch = None  # type: ignore

from ..models.graph_encoder import PeptideGraphBuilder


class PeptideGraphDataset(Dataset):
    """Dataset that converts peptide sequences to PyG graph Data objects."""

    def __init__(self, sequences, labels=None, mic_values=None, k_neighbors=5):
        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for PeptideGraphDataset. "
                "Install with: pip install torch-geometric"
            )
        self.sequences = sequences
        self.labels = labels
        self.mic_values = mic_values
        self.graph_builder = PeptideGraphBuilder(k_neighbors=k_neighbors)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        node_features, edge_index, edge_attr = self.graph_builder.sequence_to_graph(seq)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            seq_len=torch.tensor([len(seq)]),
        )

        if self.labels is not None:
            data.y = torch.tensor([self.labels[idx]], dtype=torch.float32)

        if self.mic_values is not None:
            data.mic = torch.tensor([self.mic_values[idx]], dtype=torch.float32)

        return data

    @staticmethod
    def collate_fn(batch):
        """Custom collate for PyG batching."""
        return Batch.from_data_list(batch)
