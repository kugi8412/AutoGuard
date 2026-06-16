#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight GNN baseline for AMP generation, used as a comparison method.

This implements a pure-PyTorch graph neural network (no torch_geometric
dependency) over the per-residue peptide graph. The model is a small
GraphVAE-style encoder/decoder that operates on the same residue graph
used by AutoGuard's GG-FiLM encoder, but without the discrete codebook,
phylogenetic conditioning, mimicry detector or safety module. It exposes
the same train()/generate() surface as the HydrAMP baseline so that all
three models can be compared head-to-head in compare_models.py.

The graph is the residue chain with k-nearest sequence-distance edges
(matching PeptideGraphBuilder). Message passing is implemented with a
dense per-batch adjacency to avoid torch-geometric.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard AA alphabet (matches autoguard.data.datasets.AA_ALPHABET).
_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(_AA)}  # 0 = padding
_IDX_TO_AA = {v: k for k, v in _AA_TO_IDX.items()}

# Kyte-Doolittle hydrophobicity (normalized to [-1, 1]).
_HYDRO = {
    'A': 0.18, 'R': -0.45, 'N': -0.35, 'D': -0.35, 'C': 0.25,
    'E': -0.35, 'Q': -0.35, 'G': -0.04, 'H': -0.32, 'I': 0.45,
    'L': 0.38, 'K': -0.39, 'M': 0.19, 'F': 0.28, 'P': -0.16,
    'S': -0.08, 'T': -0.07, 'W': -0.09, 'Y': -0.13, 'V': 0.42,
}

_CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'E': -1, 'Q': 0, 'G': 0,
    'H': 0.5, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}


def _residue_features(aa: str) -> List[float]:
    """4-dim physicochemical feature vector per residue."""
    return [
        _HYDRO.get(aa, 0.0),
        _CHARGE.get(aa, 0.0),
        1.0 if aa in "FWY" else 0.0,                # aromatic
        1.0 if aa in "AILMFVWP" else 0.0,           # hydrophobic
    ]


def build_adjacency(max_len: int, k_neighbors: int = 3) -> torch.Tensor:
    """Symmetric, self-loop-augmented graph adjacency on a length-`max_len` chain.

    Returns the row-normalised adjacency matrix of shape ``[max_len, max_len]``
    used for message passing.
    """
    a = torch.zeros(max_len, max_len)
    for i in range(max_len):
        a[i, i] = 1.0
        for d in range(1, k_neighbors + 1):
            if i + d < max_len:
                a[i, i + d] = 1.0
                a[i + d, i] = 1.0
    deg = a.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return a / deg


class _GraphConv(nn.Module):
    """Single dense message-passing layer: h' = sigma(A · h · W + b)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: [B, L, in_dim]; adj: [L, L] (broadcast across batch)
        m = torch.einsum("ij,bjf->bif", adj, h)
        return F.relu(self.norm(self.lin(m)))


class GNNGenerator(nn.Module):
    """Compact graph-VAE that generates peptide sequences.

    Encoder:
        embed -> [GraphConv]*N -> mean-pool -> (mu, logvar)
    Decoder:
        z -> GRU unroll -> per-position logits over the 21-token vocabulary
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        max_len: int = 25,
        num_gnn_layers: int = 3,
        k_neighbors: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Per-residue physicochemical features
        feats = torch.zeros(vocab_size, 4)

        for aa, idx in _AA_TO_IDX.items():
            feats[idx] = torch.tensor(_residue_features(aa))

        self.register_buffer("residue_features", feats, persistent=False)

        self.input_proj = nn.Linear(embed_dim + 4, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [_GraphConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder_gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.decoder_head = nn.Linear(hidden_dim, vocab_size)

        # AMP head used both as a training auxiliary signal and at generation
        self.amp_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        adj = build_adjacency(max_len, k_neighbors=k_neighbors)
        self.register_buffer("adjacency", adj, persistent=False)

    def encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.token_embed(tokens)                                # [B, L, E]
        phys = self.residue_features[tokens]                          # [B, L, 4]
        h = self.input_proj(torch.cat([emb, phys], dim=-1))           # [B, L, H]
        for layer in self.gnn_layers:
            h = layer(h, self.adjacency[: tokens.size(1), : tokens.size(1)])
        pooled = h.mean(dim=1)                                        # [B, H]
        return self.mu_head(pooled), self.logvar_head(pooled)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Tile z across length and unroll the GRU.
        z_tiled = z.unsqueeze(1).expand(-1, self.max_len, -1)         # [B, L, Z]
        out, _ = self.decoder_gru(z_tiled)                            # [B, L, H]
        return self.decoder_head(out)                                 # [B, L, V]

    def forward(self, tokens: torch.Tensor):
        mu, logvar = self.encode(tokens)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        amp_pred = self.amp_head(z)
        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "amp_prediction": amp_pred,
        }

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        device: torch.device | str = "cpu",
        temperature: float = 1.0,
    ) -> Tuple[List[str], List[float]]:
        """Sample sequences from the prior and return decoded strings + AMP scores."""
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        logits = self.decode(z)

        if temperature <= 0:
            tokens = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(
                probs.reshape(-1, self.vocab_size), 1
            ).reshape(num_samples, self.max_len)

        amp_scores = self.amp_head(z).squeeze(-1).cpu().tolist()
        sequences = []

        for row in tokens.cpu().tolist():
            seq_chars = []

            for idx in row:
                if idx == 0:
                    break

                seq_chars.append(_IDX_TO_AA.get(int(idx), "X"))

            sequences.append("".join(seq_chars))

        return sequences, amp_scores
