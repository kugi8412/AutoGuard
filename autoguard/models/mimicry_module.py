#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Molecular mimicry detection module with contrastive learning.
Detects structural and sequence similarity between generated AMPs and human
autoantigens to prevent autoimmune cross-reactivity. Uses ESM-2 embeddings
and contrastive learning to enforce dissimilarity from host proteins.

References:
- Maoz-Segal R, Andrade P. "Molecular Mimicry and Autoimmunity." 2015.
- Lin, Z. et al. "Evolutionary-scale prediction of atomic-level protein structure
  with a language model." Science 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularMimicryDetector(nn.Module):
    """Detects molecular mimicry risk between generated peptides and host proteins.
    Uses pre-trained ESM-2 representations to compute structural/functional
    similarity, with learned projections for mimicry-specific features.
    """

    def __init__(self, esm_dim=1280, hidden_dim=256, output_dim=64,
                 temperature=0.07):
        super().__init__()
        self.temperature = temperature

        # Projection head for peptide embeddings
        self.peptide_projector = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Projection head for autoantigen embeddings
        self.antigen_projector = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Projection head for natural host defense peptides (positive mimicry)
        self.defense_projector = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Mimicry risk scorer
        self.risk_scorer = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Mimicry conditioning vector generator
        self.conditioning_head = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, peptide_esm, autoantigen_esm, defense_esm=None):
        """
        Args:
            peptide_esm: [batch, esm_dim] - ESM-2 embeddings of generated peptides
            autoantigen_esm: [batch, num_antigens, esm_dim] - ESM-2 embeddings of autoantigens
            defense_esm: [batch, num_defense, esm_dim] - ESM-2 embeddings of host defense peptides

        Returns:
            mimicry_risk: [batch, 1] - risk score (0=safe, 1=high mimicry risk)
            conditioning: [batch, output_dim] - conditioning vector for decoder
        """
        # Project peptide
        pep_proj = self.peptide_projector(peptide_esm)
        pep_proj = F.normalize(pep_proj, dim=-1)

        # Project autoantigens and compute max similarity
        antigen_proj = self.antigen_projector(autoantigen_esm)
        antigen_proj = F.normalize(antigen_proj, dim=-1)

        # Cosine similarity with autoantigens
        sim_antigens = torch.bmm(
            antigen_proj, pep_proj.unsqueeze(-1)
        ).squeeze(-1)  # [batch, num_antigens]
        max_antigen_sim = sim_antigens.max(dim=-1, keepdim=True)[0]  # [batch, 1]
        mean_antigen_sim = sim_antigens.mean(dim=-1, keepdim=True)

        # Project defense peptides
        if defense_esm is not None:
            defense_proj = self.defense_projector(defense_esm)
            defense_proj = F.normalize(defense_proj, dim=-1)
            sim_defense = torch.bmm(
                defense_proj, pep_proj.unsqueeze(-1)
            ).squeeze(-1)
            max_defense_sim = sim_defense.max(dim=-1, keepdim=True)[0]
        else:
            max_defense_sim = torch.zeros_like(max_antigen_sim)
            defense_proj = torch.zeros_like(pep_proj).unsqueeze(1)

        # Risk scoring
        risk_features = torch.cat([
            pep_proj,
            pep_proj * max_antigen_sim,  # Interaction with antigens
            pep_proj * max_defense_sim,  # Interaction with defense
        ], dim=-1)
        mimicry_risk = self.risk_scorer(risk_features)

        # Conditioning vector (encode mimicry context for decoder)
        antigen_context = (antigen_proj * sim_antigens.unsqueeze(-1)).mean(dim=1)
        conditioning = self.conditioning_head(
            torch.cat([pep_proj, antigen_context], dim=-1)
        )

        return mimicry_risk, conditioning

    def compute_similarity_matrix(self, peptide_esm, reference_esm):
        """Compute pairwise similarity for visualization/analysis."""
        pep_proj = F.normalize(self.peptide_projector(peptide_esm), dim=-1)
        ref_proj = F.normalize(self.antigen_projector(reference_esm), dim=-1)
        return torch.mm(pep_proj, ref_proj.t())


class ContrastiveMimicryLoss(nn.Module):
    """Contrastive loss for molecular mimicry training.

    Encourages generated peptides to be:
    - DISSIMILAR to autoantigens (negative pairs)
    - SIMILAR to natural host defense peptides (positive pairs, functional mimicry)
    """

    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, peptide_proj, positive_proj, negative_proj):
        """
        Args:
            peptide_proj: [batch, dim] - projected generated peptides
            positive_proj: [batch, num_pos, dim] - natural defense peptides (attract)
            negative_proj: [batch, num_neg, dim] - autoantigens (repel)

        Returns:
            loss: scalar contrastive loss
        """
        peptide_proj = F.normalize(peptide_proj, dim=-1)

        # Positive similarities (maximize)
        positive_proj = F.normalize(positive_proj, dim=-1)
        pos_sim = torch.bmm(
            positive_proj, peptide_proj.unsqueeze(-1)
        ).squeeze(-1) / self.temperature  # [batch, num_pos]

        # Negative similarities (minimize)
        negative_proj = F.normalize(negative_proj, dim=-1)
        neg_sim = torch.bmm(
            negative_proj, peptide_proj.unsqueeze(-1)
        ).squeeze(-1) / self.temperature  # [batch, num_neg]

        # InfoNCE-style loss with margin
        repulsion_loss = F.relu(neg_sim - self.margin).mean()

        # Attraction to positives with UPPER BOUND
        pos_attraction_target = torch.clamp(pos_sim, max=0.8 / self.temperature)
        attraction_loss = -pos_attraction_target.mean()

        # Combined
        loss = attraction_loss + repulsion_loss
        return loss


class ESMFeatureExtractor(nn.Module):
    """Wrapper to extract features from pre-trained ESM-2 model."""

    def __init__(self, model_name="esm2_t33_650M_UR50D", freeze=True, device="cpu"):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.device = device
        self._model = None
        self._alphabet = None

    def _load_model(self):
        """Lazy-load ESM model."""
        if self._model is None:
            try:
                import esm
                self._model, self._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                if self.freeze:
                    for param in self._model.parameters():
                        param.requires_grad = False
                self._model.eval()
                self._model.to(self.device)
            except ImportError:
                raise ImportError(
                    "ESM package required. Install with: pip install fair-esm"
                )

    def forward(self, sequences):
        """Extract ESM-2 embeddings for a list of sequences.

        Args:
            sequences: List[str] - amino acid sequences

        Returns:
            embeddings: [batch, esm_dim] - per-sequence representations
        """
        self._load_model()
        import esm

        batch_converter = self._alphabet.get_batch_converter()
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(next(self._model.parameters()).device)

        with torch.no_grad():
            results = self._model(tokens, repr_layers=[33])

        # Mean pool over sequence length
        token_representations = results["representations"][33]
        embeddings = token_representations[:, 1:-1].mean(dim=1)

        return embeddings
