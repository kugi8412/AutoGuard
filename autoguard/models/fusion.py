#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal fusion module with cross-attention.
Fuses discrete sequence tokens with continuous phylogenetic conditioning (Z_phylo)
and molecular mimicry vectors (C_mimic) via cross-attention for guided generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultimodalCrossAttention(nn.Module):
    """Cross-attention between sequence tokens and conditioning modalities."""

    def __init__(self, query_dim, kv_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert query_dim % num_heads == 0

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(query_dim)

    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: [batch, seq_len, query_dim] - sequence features
            key_value: [batch, cond_len, kv_dim] - conditioning features
            mask: Optional attention mask
        """
        batch_size, seq_len, _ = query.shape
        residual = query

        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.out_proj(attn_output)
        return self.layer_norm(output + residual), attn_weights


class MultimodalFusion(nn.Module):
    """Lightweight cross-attention fusion of conditioning modalities.

    The sequence latent (queries) attends to a small set of conditioning
    tokens (keys/values): phylogenetic, mimicry (ESM-derived), safety, MIC and
    ESM peptide context. All conditions are pre-projected by the parent model
    to a common ``cond_dim`` and stacked into ``[batch, num_cond, cond_dim]``.

    This replaces the previous 3.59M-param transformer stack (which mean-pooled
    away the latent signal) with a single cheap cross-attention layer.
    """

    def __init__(self, seq_dim=48, cond_dim=48, fusion_dim=96,
                 num_heads=4, num_layers=1, dropout=0.1, max_modalities=8):
        super().__init__()
        self.fusion_dim = fusion_dim

        self.seq_proj = nn.Linear(seq_dim, fusion_dim)
        self.cond_proj = nn.Linear(cond_dim, fusion_dim)
        self.modality_embeddings = nn.Embedding(max_modalities, fusion_dim)

        self.cross_attention_layers = nn.ModuleList([
            MultimodalCrossAttention(fusion_dim, fusion_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, seq_dim),
        )

    def forward(self, seq_features, conditions):
        """
        Args:
            seq_features: [batch, seq_len, seq_dim] - sequence/latent features (queries)
            conditions: [batch, num_cond, cond_dim] - stacked conditioning tokens

        Returns:
            fused: [batch, seq_len, seq_dim] - fused representation for decoder
            attention_weights: list of attention maps (for XAI)
        """
        seq_proj = self.seq_proj(seq_features)
        cond = self.cond_proj(conditions)  # [batch, num_cond, fusion_dim]

        num_cond = cond.size(1)
        mod_ids = torch.arange(num_cond, device=conditions.device)
        cond = cond + self.modality_embeddings(mod_ids).unsqueeze(0)

        fused = seq_proj
        all_attention_weights = []
        for cross_attn in self.cross_attention_layers:
            fused, attn_w = cross_attn(fused, cond)
            all_attention_weights.append(attn_w)

        return self.output_proj(fused), all_attention_weights

