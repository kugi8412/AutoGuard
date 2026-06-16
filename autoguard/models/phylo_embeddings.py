
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phylogenetic context via Poincaré (hyperbolic) embeddings for evolutionary conditioning.
Embeds species phylogenetic trees into hyperbolic space to capture hierarchical
evolutionary relationships. These embeddings condition peptide generation to produce
AMPs targeting evolutionary conserved (mutation-resistant) bacterial features.

References:
- Nickel, M. & Kiela, D. "Poincaré Embeddings for Learning Hierarchical Representations."
  NeurIPS 2017.
- Sala, F. et al. "Representation Tradeoffs for Hyperbolic Embeddings."
  ICML 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PoincareDistance(torch.autograd.Function):
    """Poincaré ball distance with numerically stable gradient."""

    @staticmethod
    def forward(ctx, u, v, curvature=-1.0):
        c = abs(curvature)
        sqrt_c = math.sqrt(c)

        diff = u - v
        diff_norm_sq = torch.sum(diff * diff, dim=-1, keepdim=True).clamp(min=1e-10)
        u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)

        num = 2 * c * diff_norm_sq
        denom = (1 - c * u_norm_sq) * (1 - c * v_norm_sq)
        denom = denom.clamp(min=1e-10)

        arg = 1 + num / denom
        dist = (1.0 / sqrt_c) * torch.acosh(arg.clamp(min=1.0 + 1e-7))

        ctx.save_for_backward(u, v, dist, arg)
        ctx.curvature = curvature
        return dist

    @staticmethod
    def backward(ctx, grad_output):
        u, v, _, arg = ctx.saved_tensors
        c = abs(ctx.curvature)

        # Gradient through acosh
        grad_arg = grad_output / (torch.sqrt(arg * arg - 1).clamp(min=1e-10) * math.sqrt(c))

        u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        lambda_u = 1.0 / (1 - c * u_norm_sq).clamp(min=1e-10)
        lambda_v = 1.0 / (1 - c * v_norm_sq).clamp(min=1e-10)

        grad_u = grad_arg * 4 * c * lambda_u * lambda_v * (u - v)
        grad_v = -grad_arg * 4 * c * lambda_u * lambda_v * (u - v)

        return grad_u, grad_v, None


def poincare_distance(u, v, curvature=-1.0):
    return PoincareDistance.apply(u, v, curvature)


class ExponentialMap(nn.Module):
    """Maps from tangent space to Poincaré ball."""

    def __init__(self, curvature=-1.0):
        super().__init__()
        self.c = abs(curvature)

    def forward(self, x, v):
        """Map tangent vector v at point x to the manifold."""
        sqrt_c = math.sqrt(self.c)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-10)
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        lambda_x = 2.0 / (1 - self.c * x_norm_sq).clamp(min=1e-10)

        second_term = torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm)

        # Mobius addition
        return self._mobius_add(x, second_term)

    def _mobius_add(self, u, v):
        c = self.c
        u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        uv = torch.sum(u * v, dim=-1, keepdim=True)

        num = (1 + 2 * c * uv + c * v_norm_sq) * u + (1 - c * u_norm_sq) * v
        denom = (1 + 2 * c * uv + c * c * u_norm_sq * v_norm_sq).clamp(min=1e-10)
        result = num / denom

        # Project back to ball
        result_norm = torch.norm(result, dim=-1, keepdim=True)
        max_norm = (1.0 - 1e-5) / math.sqrt(c)
        result = torch.where(result_norm > max_norm, result * max_norm / result_norm, result)
        return result


class PoincareEmbedding(nn.Module):
    """Learnable Poincaré embeddings for phylogenetic tree nodes.
    Maps species/taxa into hyperbolic space preserving tree structure.
    """

    def __init__(self, num_entities, embed_dim=64, curvature=-1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.c = abs(curvature)

        # Initialize embeddings near origin
        max_norm = (1.0 - 1e-2) / math.sqrt(self.c)
        self.embeddings = nn.Parameter(
            torch.randn(num_entities, embed_dim) * 0.001
        )
        self.exp_map = ExponentialMap(curvature)

    def forward(self, indices):
        """Get hyperbolic embeddings for given entity indices."""
        embeds = self.embeddings[indices]
        # Project to Poincaré ball
        norms = torch.norm(embeds, dim=-1, keepdim=True)
        max_norm = (1.0 - 1e-5) / math.sqrt(self.c)
        embeds = torch.where(norms > max_norm, embeds * max_norm / norms, embeds)
        return embeds

    def distance(self, u_idx, v_idx):
        """Compute pairwise hyperbolic distances."""
        u = self.forward(u_idx)
        v = self.forward(v_idx)
        return poincare_distance(u, v, self.curvature)


class PhylogeneticConditioner(nn.Module):
    """Produces evolutionary conditioning vectors from Poincaré embeddings.
    Each species hyperbolic coordinate (a low-dimensional point on the Poincaré
    ball 10-D, since 2-D cannot embed the species tree without large metric
    distortion) is lifted to the conditioning width by a small MLP, aggregated
    by attention over the species panel, and projected to the final conditioning
    vector. This keeps the conditioner tiny while making the geometry usable by
    the rest of the model.
    """

    def __init__(self, embed_dim=10, output_dim=48, num_perturbations=4,
                 curvature=-1.0, hidden_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_perturbations = num_perturbations
        self.curvature = curvature
        self.c = abs(curvature)

        # Lift 2D hyperbolic coordinate -> hidden width MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learned attention pooling over the species panel
        self.attn_score = nn.Linear(hidden_dim, 1)

        # Output projection to conditioning width
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, species_embeddings, target_mask=None):
        """
        Args:
            species_embeddings: [batch, num_species, embed_dim] - Poincaré coords
            target_mask: [batch, num_species] - which species to attend to (optional)

        Returns:
            conditioning: [batch, output_dim] - evolutionary conditioning vector
        """
        d = species_embeddings.shape[-1]

        if d > self.embed_dim:
            species_embeddings = species_embeddings[..., :self.embed_dim]
        elif d < self.embed_dim:
            pad = torch.zeros(*species_embeddings.shape[:-1],
                              self.embed_dim - d, device=species_embeddings.device)
            species_embeddings = torch.cat([species_embeddings, pad], dim=-1)

        # [batch, num_species, hidden]
        h = self.coord_mlp(species_embeddings)

        # Attention weights over species
        scores = self.attn_score(h).squeeze(-1)  # [batch, num_species]
        if target_mask is not None:
            scores = scores.masked_fill(target_mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [batch, num_species, 1]

        pooled = (h * weights).sum(dim=1)  # [batch, hidden]
        return self.output_proj(pooled)



class PhyloTreeProcessor:
    """Processes Newick phylogenetic trees into Poincaré embedding training pairs."""

    def __init__(self, curvature=-1.0):
        self.curvature = curvature

    def parse_newick_to_pairs(self, newick_str):
        """Extract ancestor-descendant pairs from Newick tree for Poincaré training.

        Returns:
            pairs: List of (ancestor_idx, descendant_idx, distance) tuples
            node_names: Dict mapping node names to indices
        """
        from io import StringIO
        try:
            from Bio import Phylo
            tree = Phylo.read(StringIO(newick_str), "newick")
        except ImportError:
            # Fallback: simple parser for basic Newick
            return self._simple_newick_parse(newick_str)

        node_names = {}
        idx = 0
        pairs = []

        for clade in tree.find_clades(order='level'):
            name = clade.name or f"internal_{idx}"
            if name not in node_names:
                node_names[name] = idx
                idx += 1

        for clade in tree.find_clades(order='level'):
            parent_name = clade.name or f"internal_{node_names.get(clade.name, 0)}"
            parent_idx = node_names.get(parent_name, 0)
            for child in clade.clades:
                child_name = child.name or f"internal_{idx}"

                if child_name not in node_names:
                    node_names[child_name] = idx
                    idx += 1

                child_idx = node_names[child_name]
                dist = child.branch_length or 1.0
                pairs.append((parent_idx, child_idx, dist))

        return pairs, node_names

    def _simple_newick_parse(self, newick_str):
        """Simplified Newick parser fallback."""
        pairs = []
        node_names = {"root": 0}
        return pairs, node_names
