#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contrastive learning trainer for molecular mimicry module.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import logging

from ..models.mimicry_module import MolecularMimicryDetector, ContrastiveMimicryLoss, ESMFeatureExtractor

logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    """Trainer for the molecular mimicry contrastive learning module.
    Trains the mimicry detector to:
    1. Identify sequences similar to autoantigens (dangerous mimicry)
    2. Encourage similarity to natural host defense peptides (beneficial mimicry)
    """

    def __init__(self, mimicry_detector: MolecularMimicryDetector,
                 esm_extractor: ESMFeatureExtractor = None,
                 learning_rate: float = 1e-4, temperature: float = 0.07,
                 margin: float = 0.5, device: str = 'cuda'):
        self.detector = mimicry_detector.to(device)
        self.device = device
        self.temperature = temperature

        if esm_extractor is None:
            self.esm = ESMFeatureExtractor()
        else:
            self.esm = esm_extractor

        self.contrastive_loss = ContrastiveMimicryLoss(temperature, margin)

        self.optimizer = optim.AdamW(
            self.detector.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train mimicry detector for one epoch."""
        self.detector.train()
        epoch_losses = {'contrastive': 0, 'risk_accuracy': 0}

        for batch in dataloader:
            peptide_seqs = batch['peptide_seq']
            positive_seqs = batch['positive_seq']
            negative_seqs = batch['negative_seq']

            # Extract ESM features
            with torch.no_grad():
                pep_features = self.esm(peptide_seqs).to(self.device)
                pos_features = self.esm(positive_seqs).to(self.device).unsqueeze(1)
                neg_features = self.esm(negative_seqs).to(self.device).unsqueeze(1)

            # Forward through mimicry detector
            _, _ = self.detector(
                pep_features, neg_features, pos_features
            )

            # Project for contrastive loss
            pep_proj = self.detector.peptide_projector(pep_features)
            pos_proj = self.detector.defense_projector(pos_features)
            neg_proj = self.detector.antigen_projector(neg_features)

            # Contrastive loss
            loss = self.contrastive_loss(pep_proj, pos_proj, neg_proj)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.detector.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses['contrastive'] += loss.item()

        num_batches = len(dataloader)
        return {k: v / num_batches for k, v in epoch_losses.items()}

    def evaluate_mimicry_risk(self, sequences, autoantigens, threshold=0.5):
        """Evaluate mimicry risk for a set of sequences.

        Args:
            sequences: List[str] - peptide sequences to evaluate
            autoantigens: List[str] - autoantigen sequences to compare against
            threshold: Risk threshold for flagging

        Returns:
            Dict with risk scores and flagged sequences
        """
        self.detector.eval()

        with torch.no_grad():
            pep_features = self.esm(sequences).to(self.device)
            antigen_features = self.esm(autoantigens).to(self.device)
            antigen_features = antigen_features.unsqueeze(0).expand(
                len(sequences), -1, -1
            )

            risks, _ = self.detector(pep_features, antigen_features)

        risk_scores = risks.squeeze(-1).cpu().numpy()
        flagged = [seq for seq, risk in zip(sequences, risk_scores) if risk > threshold]

        return {
            'risk_scores': risk_scores,
            'flagged_sequences': flagged,
            'mean_risk': risk_scores.mean(),
            'max_risk': risk_scores.max(),
        }
