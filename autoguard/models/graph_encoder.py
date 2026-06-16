#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph-based peptide encoder with GG-FiLM (Gated Graph Feature-wise Linear Modulation).

Uses graph neural networks to represent peptides as molecular graphs with
physicochemical node/edge features, modulated by FiLM conditioning layers.

References:
- Brockschmidt, M. "GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation."
  ICML 2020.
- Perez, E. et al. "FiLM: Visual Reasoning with a General Conditioning Layer."
  AAAI 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
    from torch_geometric.utils import add_self_loops
    _PYG_AVAILABLE = True
except Exception:  # pragma: no cover
    _PYG_AVAILABLE = False

    class MessagePassing(nn.Module):  # type: ignore
        """Stub used when torch_geometric is not installed."""

        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError(
                "torch_geometric is required for GG-FiLM graph encoding. "
                "Install with: pip install torch-geometric"
            )

    def global_mean_pool(*args, **kwargs):  # type: ignore
        raise ImportError("torch_geometric is required for GG-FiLM graph encoding.")

    def global_max_pool(*args, **kwargs):  # type: ignore
        raise ImportError("torch_geometric is required for GG-FiLM graph encoding.")

    def add_self_loops(*args, **kwargs):  # type: ignore
        raise ImportError("torch_geometric is required for GG-FiLM graph encoding.")


def is_pyg_available() -> bool:
    """Return True if torch_geometric is importable (graph encoder fully usable)."""
    return _PYG_AVAILABLE


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, feature_dim, conditioning_dim):
        super().__init__()
        self.gamma_net = nn.Linear(conditioning_dim, feature_dim)
        self.beta_net = nn.Linear(conditioning_dim, feature_dim)

    def forward(self, x, conditioning):
        gamma = self.gamma_net(conditioning)
        beta = self.beta_net(conditioning)
        return gamma * x + beta


class GGFiLMConv(MessagePassing):
    """Gated Graph convolution with FiLM conditioning.
    Each message passing step applies FiLM modulation conditioned on
    edge features and a global conditioning vector.
    """

    def __init__(self, in_channels, out_channels, edge_dim, conditioning_dim):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Sigmoid(),
        )

        # FiLM modulation
        self.film = FiLMLayer(out_channels, conditioning_dim)

        # Update GRU
        self.gru = nn.GRUCell(out_channels, out_channels)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr, conditioning):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        # Message passing
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # FiLM modulation with conditioning
        if conditioning.dim() == 2 and conditioning.size(0) != x.size(0):
            conditioning_expanded = conditioning.repeat_interleave(
                x.size(0) // conditioning.size(0), dim=0
            )
        else:
            conditioning_expanded = conditioning

        modulated = self.film(aggr_out, conditioning_expanded)

        # Gating
        gate_input = torch.cat([x[:, :self.out_channels] if x.size(-1) != self.out_channels
                                else x, modulated], dim=-1)
        gate_val = self.gate(gate_input)
        gated_output = gate_val * modulated

        # GRU update
        if x.size(-1) == self.out_channels:
            updated = self.gru(gated_output, x)
        else:
            updated = gated_output

        return self.layer_norm(updated)

    def message(self, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))


class GGFiLMEncoder(nn.Module):
    """Full GG-FiLM encoder for peptide graphs.
    Converts peptide molecular graphs (with physicochemical node features
    and bond/distance edge features) into a latent representation using
    gated graph convolutions with FiLM conditioning.
    """

    def __init__(self, node_feature_dim=32, edge_feature_dim=16,
                 hidden_dim=128, num_layers=4, conditioning_dim=64,
                 output_dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.conditioning_dim = conditioning_dim

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GG-FiLM layers
        self.conv_layers = nn.ModuleList([
            GGFiLMConv(hidden_dim, hidden_dim, edge_feature_dim, conditioning_dim)
            for _ in range(num_layers)
        ])

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Output mean/logvar for VAE
        self.mean_head = nn.Linear(output_dim, output_dim)
        self.logvar_head = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch, conditioning=None):
        """
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment vector [num_nodes]
            conditioning: Optional conditioning vector [batch_size, conditioning_dim]
        """
        if conditioning is None:
            num_graphs = int(batch.max().item()) + 1
            conditioning = torch.zeros(
                num_graphs, self.conditioning_dim, device=x.device, dtype=x.dtype
            )

        h = self.input_proj(x)

        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_attr, conditioning[batch])

        # Global readout (mean + max pooling)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_global = torch.cat([h_mean, h_max], dim=-1)
        output = self.readout(h_global)
        mean = self.mean_head(output)
        logvar = self.logvar_head(output)

        return mean, logvar, h  # Also return node-level features


class PeptideGraphBuilder:
    """Builds molecular graph representations of peptides with physicochemical features."""

    # Physicochemical properties for 20 amino acids (normalized)
    AA_PROPERTIES = {
        'A': {'hydrophobicity': 0.62, 'charge': 0.0, 'mass': 0.36, 'volume': 0.28,
              'polarity': 0.0, 'hbond_donor': 0.0, 'hbond_acceptor': 0.0, 'aromatic': 0.0},
        'R': {'hydrophobicity': 0.0, 'charge': 1.0, 'mass': 0.78, 'volume': 0.72,
              'polarity': 1.0, 'hbond_donor': 1.0, 'hbond_acceptor': 0.0, 'aromatic': 0.0},
        'N': {'hydrophobicity': 0.23, 'charge': 0.0, 'mass': 0.50, 'volume': 0.44,
              'polarity': 0.7, 'hbond_donor': 0.5, 'hbond_acceptor': 0.5, 'aromatic': 0.0},
        'D': {'hydrophobicity': 0.23, 'charge': -1.0, 'mass': 0.50, 'volume': 0.40,
              'polarity': 0.8, 'hbond_donor': 0.0, 'hbond_acceptor': 1.0, 'aromatic': 0.0},
        'C': {'hydrophobicity': 0.68, 'charge': 0.0, 'mass': 0.47, 'volume': 0.36,
              'polarity': 0.1, 'hbond_donor': 0.25, 'hbond_acceptor': 0.0, 'aromatic': 0.0},
        'E': {'hydrophobicity': 0.31, 'charge': -1.0, 'mass': 0.56, 'volume': 0.52,
              'polarity': 0.8, 'hbond_donor': 0.0, 'hbond_acceptor': 1.0, 'aromatic': 0.0},
        'Q': {'hydrophobicity': 0.31, 'charge': 0.0, 'mass': 0.56, 'volume': 0.52,
              'polarity': 0.7, 'hbond_donor': 0.5, 'hbond_acceptor': 0.5, 'aromatic': 0.0},
        'G': {'hydrophobicity': 0.50, 'charge': 0.0, 'mass': 0.22, 'volume': 0.16,
              'polarity': 0.0, 'hbond_donor': 0.0, 'hbond_acceptor': 0.0, 'aromatic': 0.0},
        'H': {'hydrophobicity': 0.46, 'charge': 0.5, 'mass': 0.58, 'volume': 0.56,
              'polarity': 0.6, 'hbond_donor': 0.5, 'hbond_acceptor': 0.5, 'aromatic': 0.5},
        'I': {'hydrophobicity': 1.0, 'charge': 0.0, 'mass': 0.50, 'volume': 0.60,
              'polarity': 0.0, 'hbond_donor': 0.0, 'hbond_acceptor': 0.0, 'aromatic': 0.0},
        'L': {'hydrophobicity': 0.94, 'charge': 0.0, 'mass': 0.50, 'volume': 0.60,
              'polarity': 0.0, 'hbond_donor': 0.0, 'hbond_acceptor': 0.0, 'aromatic': 0.0},
        'K': {'hydrophobicity': 0.15, 'charge': 1.0, 'mass': 0.56, 'volume': 0.64,
              'polarity': 0.9, 'hbond_donor': 0.75, 'hbond_acceptor': 0.0, 'aromatic': 0.0},
        'M': {'hydrophobicity': 0.74, 'charge': 0.0, 'mass': 0.56, 'volume': 0.56,
              'polarity': 0.1, 'hbond_donor': 0.0, 'hbond_acceptor': 0.25, 'aromatic': 0.0},
        'F': {'hydrophobicity': 1.0, 'charge': 0.0, 'mass': 0.64, 'volume': 0.72,
              'polarity': 0.0, 'hbond_donor': 0.0, 'hbond_acceptor': 0.0, 'aromatic': 1.0},
        'P': {'hydrophobicity': 0.68, 'charge': 0.0, 'mass': 0.44, 'volume': 0.40,
              'polarity': 0.0, 'hbond_donor': 0.0, 'hbond_acceptor': 0.0, 'aromatic': 0.0},
        'S': {'hydrophobicity': 0.38, 'charge': 0.0, 'mass': 0.39, 'volume': 0.32,
              'polarity': 0.4, 'hbond_donor': 0.5, 'hbond_acceptor': 0.5, 'aromatic': 0.0},
        'T': {'hydrophobicity': 0.46, 'charge': 0.0, 'mass': 0.44, 'volume': 0.40,
              'polarity': 0.4, 'hbond_donor': 0.5, 'hbond_acceptor': 0.5, 'aromatic': 0.0},
        'W': {'hydrophobicity': 0.85, 'charge': 0.0, 'mass': 0.83, 'volume': 0.88,
              'polarity': 0.1, 'hbond_donor': 0.25, 'hbond_acceptor': 0.0, 'aromatic': 1.0},
        'Y': {'hydrophobicity': 0.77, 'charge': 0.0, 'mass': 0.72, 'volume': 0.76,
              'polarity': 0.3, 'hbond_donor': 0.5, 'hbond_acceptor': 0.25, 'aromatic': 1.0},
        'V': {'hydrophobicity': 0.88, 'charge': 0.0, 'mass': 0.44, 'volume': 0.48,
              'polarity': 0.0, 'hbond_donor': 0.0, 'hbond_acceptor': 0.0, 'aromatic': 0.0},
    }

    # Additional computed features
    POSITIONAL_ENCODING_DIM = 8
    SECONDARY_STRUCTURE_DIM = 3

    def __init__(self, k_neighbors=5, include_positional=True):
        self.k_neighbors = k_neighbors
        self.include_positional = include_positional
        self.aa_to_idx = {aa: i for i, aa in enumerate(sorted(self.AA_PROPERTIES.keys()))}

    def sequence_to_graph(self, sequence: str):
        """Convert peptide sequence to graph with physicochemical features.

        Returns:
            node_features: [seq_len, node_feature_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_feature_dim]
        """
        import numpy as np

        seq_len = len(sequence)
        node_features = []

        for i, aa in enumerate(sequence):
            if aa not in self.AA_PROPERTIES:
                props = {k: 0.0 for k in list(self.AA_PROPERTIES.values())[0].keys()}
            else:
                props = self.AA_PROPERTIES[aa]

            features = list(props.values())

            # Add one-hot amino acid identity
            one_hot = [0.0] * 20
            if aa in self.aa_to_idx:
                one_hot[self.aa_to_idx[aa]] = 1.0
            features.extend(one_hot)

            # Add positional encoding
            if self.include_positional:
                pos_enc = self._positional_encoding(i, seq_len)
                features.extend(pos_enc)

            node_features.append(features)

        node_features = torch.tensor(node_features, dtype=torch.float32)

        # Build edges: sequential bonds + k-nearest neighbors in sequence space
        edge_index = []
        edge_attr = []

        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                dist = abs(i - j)
                if dist <= self.k_neighbors:
                    edge_index.append([i, j])
                    # Edge features: distance, relative position, bond type
                    edge_feat = [
                        1.0 / (dist + 1),           # inverse distance
                        (j - i) / seq_len,          # relative position
                        1.0 if dist == 1 else 0.0,  # peptide bond
                        1.0 if dist == 2 else 0.0,  # i, i+2 interaction
                        1.0 if dist == 3 else 0.0,  # i, i+3 (helix)
                        1.0 if dist == 4 else 0.0,  # i, i+4 (helix)
                    ]
                    # Pad to edge_feature_dim
                    edge_feat.extend([0.0] * (16 - len(edge_feat)))
                    edge_attr.append(edge_feat[:16])

        if not edge_index:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 16, dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        return node_features, edge_index, edge_attr

    def _positional_encoding(self, pos, max_len):
        """Sinusoidal positional encoding."""
        import math
        pe = []

        for i in range(self.POSITIONAL_ENCODING_DIM // 2):
            freq = 1.0 / (10000 ** (2 * i / self.POSITIONAL_ENCODING_DIM))
            pe.append(math.sin(pos * freq))
            pe.append(math.cos(pos * freq))

        return pe[:self.POSITIONAL_ENCODING_DIM]
