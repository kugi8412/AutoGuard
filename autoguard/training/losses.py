#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loss functions for AutoGuard training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from ..config import LossWeights


class AutoGuardLoss(nn.Module):
    """Combined loss function for AutoGuard model training.

    Integrates:
    - Sequence reconstruction (cross-entropy)
    - VQ-VAE codebook loss (commitment + codebook)
    - KL divergence (for encoder regularization)
    - AMP classification loss
    - MIC regression loss
    - Phylogenetic conditioning loss
    - Mimicry penalty
    - Safety penalty
    """

    def __init__(self, weights: LossWeights):
        super().__init__()
        self.weights = weights

    def forward(self, output: Dict, targets: Dict, kl_weight: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Args:
            output: Dict from AutoGuardModel.forward()
            targets: Dict with 'tokens', 'label', optional 'toxic'/'hemolytic'
            kl_weight: unused (deterministic VQ-VAE encoder); kept for API compat

        Returns:
            Dict with individual and total losses
        """
        losses = {}

        # Reconstruction loss
        logits = output['logits']  # [batch, seq_len, vocab_size]
        target_tokens = targets['tokens']  # [batch, seq_len]
        batch_sz, seq_len, vocab = logits.shape
        lengths = (target_tokens != 0).sum(dim=1)  # [batch] real-residue counts
        positions = torch.arange(seq_len, device=logits.device).unsqueeze(0)
        supervise = positions <= lengths.clamp(max=seq_len - 1).unsqueeze(1)
        per_token = F.cross_entropy(
            logits.reshape(-1, vocab),
            target_tokens.reshape(-1),
            reduction='none',
        ).view(batch_sz, seq_len)
        reconstruction_loss = (per_token * supervise).sum() / supervise.sum().clamp_min(1)
        losses['reconstruction'] = self.weights.reconstruction * reconstruction_loss

        # VQ-VAE commitment loss
        vq_loss = output['vq_loss']
        losses['vq'] = self.weights.vq_commitment * vq_loss

        # Binary AMP classification (BCEWithLogits)
        if targets.get('label') is not None:
            amp_logits = output['amp_logits']
            amp_loss = F.binary_cross_entropy_with_logits(amp_logits, targets['label'])
            losses['amp_classification'] = self.weights.antimicrobial_activity * amp_loss

        # Supervised safety loss when toxicity/hemolysis labels are available;
        if output.get('safety') is not None:
            safety = output['safety']
            sup_terms = []
            if targets.get('toxic') is not None:
                sup_terms.append(F.binary_cross_entropy(
                    safety['toxicity'].clamp(1e-6, 1 - 1e-6), targets['toxic']))
            if targets.get('hemolytic') is not None:
                sup_terms.append(F.binary_cross_entropy(
                    safety['hemolysis'].clamp(1e-6, 1 - 1e-6), targets['hemolytic']))
            if sup_terms:
                losses['safety'] = self.weights.safety_penalty * torch.stack(sup_terms).mean()
            else:
                losses['safety'] = self.weights.safety_penalty * safety['safety_score'].mean()

        # Mimicry penalty (only when ESM features were provided)
        if output.get('mimicry_risk') is not None:
            losses['mimicry'] = self.weights.mimicry_penalty * output['mimicry_risk'].mean()

        # Codebook perplexity tracking (not a loss, for logging)
        encoding_info = output['encoding_info']
        losses['perplexity'] = encoding_info.get('perplexity', torch.tensor(0.0))

        # Total loss
        total = sum(v for k, v in losses.items()
                    if k != 'perplexity' and isinstance(v, torch.Tensor) and v.dim() == 0
                    and k != 'total')
        losses['total'] = total

        return losses


class PoincareEmbeddingLoss(nn.Module):
    """Loss for training Poincaré embeddings on phylogenetic trees."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pos_distances, neg_distances, target_distances=None):
        """
        Args:
            pos_distances: [batch] - distances between connected nodes
            neg_distances: [batch, num_neg] - distances to negative samples
            target_distances: Optional [batch] - target tree distances
        """
        # Ranking loss: positive pairs should be closer than negative
        loss = torch.relu(
            pos_distances.unsqueeze(1) - neg_distances + self.margin
        ).mean()

        # Optional distance regression
        if target_distances is not None:
            dist_loss = F.mse_loss(pos_distances.squeeze(), target_distances)
            loss = loss + 0.1 * dist_loss

        return loss
