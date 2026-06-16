#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Safety module for toxicity, hemolysis, and immunogenicity prediction.
Multi-task safety classifier ensuring generated AMPs are non-toxic,
non-hemolytic, and have low immunogenic potential.

The module: a single instance conditions on one of the
AMP-Challenge-2027 categories via a learnable category embedding, so the same
toxicity/safety head can steer generation toward any of the five categories
(broad-spectrum, Gram-positive, Gram-negative, MDR, optimal-selectivity)
instead of needing a separate module per category.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# The five AMP-Challenge-2027 categories, in a fixed canonical order.
CHALLENGE_CATEGORIES = (
    "broad_spectrum",   # 0 -> active across the whole strain panel
    "gram_pos",         # 1 -> Gram-positive activity
    "gram_neg",         # 2 -> Gram-negative activity
    "mdr",              # 3 -> multi-drug-resistant (ESKAPE) activity
    "therapeutic",      # 4 -> optimal selectivity (safety window)
)
CATEGORY_TO_IDX = {name: i for i, name in enumerate(CHALLENGE_CATEGORIES)}
NUM_CATEGORIES = len(CHALLENGE_CATEGORIES)


def resolve_category_index(category) -> int:
    """Map a category name, "generate_<name>" entry point, or int to its index."""

    if isinstance(category, int):
        return category

    name = str(category)

    if name.startswith("generate_"):
        name = name[len("generate_"):]
    if name not in CATEGORY_TO_IDX:
        raise ValueError(
            f"Unknown category {category!r}; expected one of {CHALLENGE_CATEGORIES}"
        )

    return CATEGORY_TO_IDX[name]


class SafetyModule(nn.Module):
    """Multi-task, category-general safety assessment for generated peptides.

    Predicts:
    - Toxicity probability
    - Hemolytic activity probability
    - Immunogenicity risk
    - Cell selectivity index (therapeutic index proxy)

    A single module serves all AMP-Challenge categories: pass "category" to
    "forward" to add a learnable per-category embedding to the shared
    representation. "category=None" reproduces the original (category-agnostic)
    behaviour, and the embedding is zero-initialised so freshly added category
    parameters do not perturb a model loaded from an older checkpoint.
    """

    def __init__(self, input_dim=64, hidden_dim=128, esm_dim=1280,
                 num_categories=NUM_CATEGORIES):
        super().__init__()

        self.num_categories = num_categories

        # Shared backbone from latent representation
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Category conditioning: one generalized module steered per challenge
        self.category_embedding = nn.Embedding(num_categories, hidden_dim)
        nn.init.zeros_(self.category_embedding.weight)

        # Optional ESM feature integration
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Task-specific heads
        self.toxicity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.hemolysis_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.immunogenicity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.selectivity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Positive selectivity index
        )

    def forward(self, latent_repr, esm_features=None, category=None):
        """
        Args:
            latent_repr: [batch, input_dim] - peptide latent representation
            esm_features: Optional [batch, esm_dim] - ESM-2 features
            category: Optional category conditioning, one of:
                - None: category-agnostic (original behaviour)
                - int / category name / ``generate_<name>`` entry-point string:
                  applies that single category to the whole batch
                - LongTensor [batch]: a per-sample category index

        Returns:
            Dict with safety predictions and composite safety score
        """
        shared = self.shared_encoder(latent_repr)

        # Optionally integrate ESM features
        if esm_features is not None:
            esm_proj = self.esm_proj(esm_features)
            gate = self.fusion_gate(torch.cat([shared, esm_proj], dim=-1))
            shared = gate * shared + (1 - gate) * esm_proj

        # Optionally steer toward an AMP-Challenge category (one general module).
        if category is not None:
            cat_idx = self._category_index_tensor(category, shared)
            shared = shared + self.category_embedding(cat_idx)

        # Task predictions
        toxicity = self.toxicity_head(shared)
        hemolysis = self.hemolysis_head(shared)
        immunogenicity = self.immunogenicity_head(shared)
        selectivity = self.selectivity_head(shared)

        # Composite safety score (lower = safer)
        safety_score = (
            0.4 * toxicity +
            0.3 * hemolysis +
            0.3 * immunogenicity
        )

        return {
            'toxicity': toxicity,
            'hemolysis': hemolysis,
            'immunogenicity': immunogenicity,
            'selectivity': selectivity,
            'safety_score': safety_score,
            'is_safe': (safety_score < 0.3).float(),
        }

    def _category_index_tensor(self, category, ref: torch.Tensor) -> torch.Tensor:
        """Broadcast a category spec to a LongTensor of shape [batch] on the right device."""
        batch = ref.shape[0]
        if torch.is_tensor(category):
            cat = category.to(device=ref.device, dtype=torch.long).view(-1)
            if cat.numel() == 1:
                cat = cat.expand(batch)
            return cat
        idx = resolve_category_index(category)
        return torch.full((batch,), idx, dtype=torch.long, device=ref.device)


class SafetyLoss(nn.Module):
    """Combined safety loss for training."""

    def __init__(self, toxicity_weight=0.4, hemolysis_weight=0.3,
                 immunogenicity_weight=0.3):
        super().__init__()
        self.weights = {
            'toxicity': toxicity_weight,
            'hemolysis': hemolysis_weight,
            'immunogenicity': immunogenicity_weight,
        }

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict from SafetyModule.forward()
            targets: Dict with 'toxicity', 'hemolysis', 'immunogenicity' binary labels
        """
        loss = 0.0

        for key, weight in self.weights.items():
            if key in targets:
                loss += weight * F.binary_cross_entropy(
                    predictions[key], targets[key]
                )

        return loss
