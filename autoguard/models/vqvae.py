#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vector Quantized Variational Autoencoder (VQ-VAE) for discrete peptide representation.
Replaces continuous latent space with a discrete codebook, better representing
the multimodal biophysical forces governing protein folding.

References:
- van den Oord, A. et al. "Neural Discrete Representation Learning." NeurIPS 2017.
- Razavi, A. et al. "Generating Diverse High-Fidelity Images with VQ-VAE-2." NeurIPS 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates and dead-code restart.

    Maps continuous latent vectors to nearest codebook entries,
    providing a discrete bottleneck for structured representation.

    Anti-collapse measures:
    - EMA codebook updates (no codebook gradients needed)
    - Laplace smoothing of cluster sizes
    - Dead-code restart: replaces unused vectors with randomly sampled encoder outputs
    - Commitment cost annealing support
    """

    def __init__(self, num_embeddings=512, embedding_dim=64,
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5,
                 dead_code_threshold=2, restart_prob=0.5,
                 normalize=True, restart_interval=25, restart_warmup=10):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.dead_code_threshold = dead_code_threshold
        self.restart_prob = restart_prob
        # Cosine/L2-normalized codebook keeps all codes on the unit sphere so
        # unused codes cannot drift to large magnitudes and become unreachable.
        self.normalize = normalize
        self.restart_interval = restart_interval
        self.restart_warmup = restart_warmup

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        if normalize:
            with torch.no_grad():
                self.embedding.weight.data.copy_(
                    F.normalize(self.embedding.weight.data, dim=1))

        # EMA tracking
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.clone())
        self.register_buffer('_usage_count', torch.zeros(num_embeddings))
        self.register_buffer('_steps', torch.tensor(0, dtype=torch.long))

    def _restart_dead_codes(self, flat_z):
        """Replace dead codebook entries with randomly sampled encoder outputs."""
        if self._steps < self.restart_warmup:
            return  # Wait for EMA to warm up

        dead_mask = self._usage_count < self.dead_code_threshold
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return

        # Sample random encoder outputs to replace dead codes
        n_samples = min(num_dead, flat_z.shape[0])
        if n_samples == 0:
            return None

        # Stochastic restart
        if self.restart_prob < 1.0:
            restart_mask = torch.rand(num_dead, device=flat_z.device) < self.restart_prob
            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            dead_indices = dead_indices[restart_mask[:len(dead_indices)]]
            num_dead = len(dead_indices)
        else:
            dead_indices = dead_mask.nonzero(as_tuple=True)[0]

        if num_dead == 0:
            return

        # Sample from encoder outputs + small noise
        sample_indices = torch.randint(0, flat_z.shape[0], (num_dead,), device=flat_z.device)
        new_vectors = flat_z[sample_indices].detach()
        noise = torch.randn_like(new_vectors) * 0.02
        new_vectors = new_vectors + noise
        if self.normalize:
            new_vectors = F.normalize(new_vectors, dim=1)

        # Replace dead codes
        self.embedding.weight.data[dead_indices] = new_vectors
        self._ema_w[dead_indices] = new_vectors
        self._ema_cluster_size[dead_indices] = 1.0
        self._usage_count[dead_indices] = 0

    def forward(self, z):
        """
        Args:
            z: [batch, ..., embedding_dim] - continuous latent vectors

        Returns:
            quantized: [batch, ..., embedding_dim] - quantized vectors
            loss: VQ loss (commitment + codebook)
            encoding_indices: [batch, ...] - codebook indices
            perplexity: codebook usage metric
        """
        input_shape = z.shape
        flat_z = z.reshape(-1, self.embedding_dim)

        if self.normalize:
            flat_z_n = F.normalize(flat_z, dim=1)
            weight_n = F.normalize(self.embedding.weight, dim=1)
            distances = -torch.matmul(flat_z_n, weight_n.t())  # 1 - cos(drop const)
        else:
            distances = (
                torch.sum(flat_z ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_z, self.embedding.weight.t())
            )

        # Find nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize
        quantized = self.embedding(encoding_indices).view(input_shape)

        # EMA update (training only)
        if self.training:
            with torch.no_grad():
                self._steps += 1
                flat_z_d = flat_z.detach()

                self._ema_cluster_size = (
                    self.decay * self._ema_cluster_size
                    + (1 - self.decay) * encodings.sum(0)
                )

                # Laplace smoothing
                n = self._ema_cluster_size.sum()
                self._ema_cluster_size = (
                    (self._ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n
                )

                dw = torch.matmul(encodings.t(), flat_z_d)
                self._ema_w = self.decay * self._ema_w + (1 - self.decay) * dw
                new_weight = self._ema_w / self._ema_cluster_size.unsqueeze(1)

                # Keep codes on the unit sphere to prevent magnitude drift/collapse.
                if self.normalize:
                    new_weight = F.normalize(new_weight, dim=1)

                self.embedding.weight.data.copy_(new_weight)

                # Track usage for dead-code detection
                used = encodings.sum(0) > 0
                self._usage_count = self._usage_count * 0.99 + used.float()

                # Periodically revive dead codes from current encoder outputs
                if self._steps % self.restart_interval == 0:
                    self._restart_dead_codes(flat_z_d)

        # Losses
        commitment_loss = F.mse_loss(quantized.detach(), z)
        loss = self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Perplexity (codebook usage)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = encoding_indices.view(input_shape[:-1])

        return quantized, loss, encoding_indices, perplexity

    def get_codebook_entry(self, indices):
        """Retrieve codebook vectors by index."""
        return self.embedding(indices)

    @torch.no_grad()
    def get_usage_stats(self):
        """Return codebook utilization statistics."""
        dead = (self._usage_count < self.dead_code_threshold).sum().item()
        alive = self.num_embeddings - dead
        return {
            'alive_codes': alive,
            'dead_codes': dead,
            'utilization': alive / self.num_embeddings,
            'avg_cluster_size': self._ema_cluster_size.mean().item(),
        }


class ResidualVQ(nn.Module):
    """Residual Vector Quantization for hierarchical discrete representation.
    Uses multiple VQ layers in sequence, each quantizing the residual
    from the previous layer, enabling finer-grained representation.
    """

    def __init__(self, num_quantizers=4, num_embeddings=512,
                 embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])

    def forward(self, z):
        """
        Args:
            z: [batch, ..., embedding_dim]

        Returns:
            quantized: [batch, ..., embedding_dim] - sum of quantized residuals
            total_loss: combined VQ loss
            all_indices: List of encoding indices per level
            all_perplexities: List of perplexities per level
        """
        residual = z
        quantized_sum = torch.zeros_like(z)
        total_loss = 0.0
        all_indices = []
        all_perplexities = []

        for quantizer in self.quantizers:
            quantized, loss, indices, perplexity = quantizer(residual)
            residual = residual - quantized.detach()
            quantized_sum = quantized_sum + quantized
            total_loss = total_loss + loss
            all_indices.append(indices)
            all_perplexities.append(perplexity)

        return quantized_sum, total_loss, all_indices, all_perplexities
