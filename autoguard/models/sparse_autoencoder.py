#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sparse Autoencoder for mechanistic interpretability and XAI.

Trains sparse autoencoders on model activations to discover interpretable
features/concepts that drive peptide generation decisions.

References:
- Cunningham, H. et al. "Sparse Autoencoders Find Highly Interpretable Features
  in Language Models." ICLR 2024.
- Bricken, T. et al. "Towards Monosemanticity: Decomposing Language Models With
  Dictionary Learning." Anthropic 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for discovering interpretable latent features.

    Trained on intermediate activations of the AutoGuard model to decompose
    representations into monosemantic, interpretable features related to:
    - Physicochemical properties
    - Evolutionary conservation
    - Safety-related patterns
    - Antimicrobial activity mechanisms
    """

    def __init__(self, input_dim=256, hidden_dim=512, sparsity_lambda=1e-3,
                 top_k=32, dead_feature_threshold=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_lambda = sparsity_lambda
        self.top_k = top_k
        self.dead_feature_threshold = dead_feature_threshold

        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        # Feature activation tracking
        self.register_buffer('feature_activation_count', torch.zeros(hidden_dim))
        self.register_buffer('total_steps', torch.tensor(0.0))

        # Initialize with unit norm decoder columns
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.xavier_normal_(self.decoder.weight)
        self._normalize_decoder()

    def _normalize_decoder(self):
        """Normalize decoder weight columns to unit norm."""
        with torch.no_grad():
            norms = torch.norm(self.decoder.weight, dim=0, keepdim=True)
            self.decoder.weight.data /= norms.clamp(min=1e-8)

    def encode(self, x):
        """Encode activations to sparse feature space.

        Args:
            x: [batch, input_dim] - model activations

        Returns:
            features: [batch, hidden_dim] - sparse feature activations
            top_k_mask: [batch, hidden_dim] - binary mask of active features
        """
        # Center input
        x_centered = x - self.pre_bias

        # Encode
        pre_activation = self.encoder(x_centered) + self.encoder_bias
        features = F.relu(pre_activation)

        # Top-K sparsity
        if self.top_k > 0 and self.top_k < self.hidden_dim:
            top_k_values, top_k_indices = torch.topk(features, self.top_k, dim=-1)
            top_k_mask = torch.zeros_like(features)
            top_k_mask.scatter_(1, top_k_indices, 1.0)
            features = features * top_k_mask
        else:
            top_k_mask = (features > 0).float()

        return features, top_k_mask

    def decode(self, features):
        """Decode from sparse features back to activation space."""
        return self.decoder(features) + self.pre_bias

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] - model activations to decompose

        Returns:
            reconstructed: [batch, input_dim] - reconstructed activations
            features: [batch, hidden_dim] - sparse feature activations
            loss_dict: Dict with reconstruction and sparsity losses
        """
        features, top_k_mask = self.encode(x)
        reconstructed = self.decode(features)

        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, x)

        # L1 sparsity loss
        sparsity_loss = self.sparsity_lambda * features.abs().mean()

        # Track feature activations
        if self.training:
            self.total_steps += 1
            active_features = (features > 0).float().sum(dim=0)
            self.feature_activation_count += active_features

        # Auxiliary dead feature loss (encourage unused features)
        dead_features = self.get_dead_features()
        dead_feature_loss = 0.0

        if dead_features.any() and self.training:
            dead_indices = dead_features.nonzero(as_tuple=True)[0]
            dead_feature_loss = 0.01 * F.mse_loss(
                self.encoder.weight[dead_indices],
                x.detach().mean(dim=0).unsqueeze(0).expand(len(dead_indices), -1)
            )

        total_loss = reconstruction_loss + sparsity_loss + dead_feature_loss

        loss_dict = {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'sparsity': sparsity_loss,
            'dead_feature': dead_feature_loss if isinstance(dead_feature_loss, torch.Tensor) else torch.tensor(0.0),
            'num_dead_features': dead_features.sum().item(),
            'mean_active_features': top_k_mask.sum(dim=-1).mean().item(),
        }

        return reconstructed, features, loss_dict

    def get_dead_features(self):
        """Identify features that haven't activated recently."""
        if self.total_steps < 100:
            return torch.zeros(self.hidden_dim, dtype=torch.bool, device=self.encoder.weight.device)
        activation_rate = self.feature_activation_count / self.total_steps.clamp(min=1)
        return activation_rate < self.dead_feature_threshold

    def get_feature_importance(self, x, target_property):
        """Compute importance of each sparse feature for a target property.

        Args:
            x: [batch, input_dim] - model activations
            target_property: [batch, 1] - property to explain (e.g., AMP activity)

        Returns:
            importance: [hidden_dim] - feature importance scores
        """
        features, _ = self.encode(x)

        # Correlation-based importance
        feature_means = features.mean(dim=0)
        target_mean = target_property.mean()

        feature_centered = features - feature_means
        target_centered = target_property - target_mean

        correlations = (feature_centered * target_centered).mean(dim=0)
        feature_std = feature_centered.std(dim=0).clamp(min=1e-8)
        target_std = target_centered.std().clamp(min=1e-8)

        importance = correlations / (feature_std * target_std)
        return importance

    def interpret_features(self, feature_indices, top_n=10):
        """Get decoder weight patterns for interpretation of specific features.

        Args:
            feature_indices: List[int] - which features to interpret
            top_n: int - number of top contributing input dimensions

        Returns:
            interpretations: Dict mapping feature_idx -> top contributing dimensions
        """
        interpretations = {}
        for idx in feature_indices:
            weights = self.decoder.weight[:, idx]  # [input_dim]
            top_dims = torch.topk(weights.abs(), top_n)
            interpretations[idx] = {
                'top_dimensions': top_dims.indices.tolist(),
                'top_weights': top_dims.values.tolist(),
                'weight_pattern': weights.detach().cpu(),
            }

        return interpretations


class MultiLayerSAE(nn.Module):
    """Collection of SAEs trained on different model layers for full interpretability."""

    def __init__(self, layer_dims, hidden_dim=512, sparsity_lambda=1e-3, top_k=32):
        super().__init__()
        self.saes = nn.ModuleDict({
            name: SparseAutoencoder(dim, hidden_dim, sparsity_lambda, top_k)
            for name, dim in layer_dims.items()
        })

    def forward(self, activations_dict):
        """
        Args:
            activations_dict: Dict[str, Tensor] - activations from each model layer

        Returns:
            results: Dict with reconstructions, features, and losses per layer
        """
        results = {}
        for name, activation in activations_dict.items():
            if name in self.saes:
                reconstructed, features, loss_dict = self.saes[name](activation)
                results[name] = {
                    'reconstructed': reconstructed,
                    'features': features,
                    'losses': loss_dict,
                }
        return results
