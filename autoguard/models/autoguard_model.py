#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoGuard: Full model integrating all components.

Evolutionary-Conditioned VQ-VAE with Graph-Based Encoding,
Phylogenetic Conditioning, Molecular Mimicry Detection, and Safety Modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hydramp_base import HydrAMPEncoder, HydrAMPDecoder
from .phylo_embeddings import PhylogeneticConditioner
from .mimicry_module import MolecularMimicryDetector
from .safety_module import SafetyModule
from .fusion import MultimodalFusion
from .vqvae import VectorQuantizer

# Graph encoder requires torch_geometric what is optional
try:
    from .graph_encoder import GGFiLMEncoder
    _GRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    GGFiLMEncoder = None  # type: ignore
    _GRAPH_AVAILABLE = False


class AutoGuardModel(nn.Module):
    """AutoGuard: End-to-end evolutionary-conditioned AMP generation model.

    Architecture:
    1. Dual encoder: HydrAMP GRU encoder + GG-FiLM graph encoder
    2. VQ-VAE discrete bottleneck (codebook quantization)
    3. Phylogenetic conditioning via Poincaré embeddings
    4. Molecular mimicry detection + safety module
    5. Multimodal cross-attention fusion
    6. Conditioned sequence decoder
    """

    def __init__(self, config, use_graph_encoder: bool = True):
        super().__init__()
        self.config = config
        # Graph encoder is optional (requires torch_geometric).
        self.use_graph_encoder = use_graph_encoder and _GRAPH_AVAILABLE

        # Sequence encoder (HydrAMP-based)
        self.seq_encoder = HydrAMPEncoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.gru_hidden_dim,
            latent_dim=config.latent_dim,
            max_len=config.max_seq_len,
        )

        # Graph encoder (GG-FiLM)
        if self.use_graph_encoder:
            self.graph_encoder = GGFiLMEncoder(
                node_feature_dim=config.node_feature_dim,
                edge_feature_dim=config.edge_feature_dim,
                hidden_dim=config.graph_hidden_dim,
                num_layers=config.graph_num_layers,
                conditioning_dim=config.film_hidden_dim,
                output_dim=config.latent_dim,
            )

            # Encoder fusion (combine sequence + graph encodings)
            self.encoder_fusion = nn.Sequential(
                nn.Linear(config.latent_dim * 2, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.ReLU(),
                nn.Linear(config.latent_dim, config.latent_dim),
            )
        else:
            self.graph_encoder = None
            self.encoder_fusion = None

        # Discrete Bottleneck
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=config.num_codebook_vectors,
            embedding_dim=config.codebook_dim,
            commitment_cost=config.commitment_cost,
            decay=getattr(config, 'ema_decay', 0.99),
        )

        # Phylogenetic conditioner
        self.phylo_conditioner = PhylogeneticConditioner(
            embed_dim=config.phylo_embed_dim,
            output_dim=config.latent_dim,
            num_perturbations=config.num_perturbations,
            curvature=config.hyperbolic_curvature,
        )

        # Molecular mimicry detector
        self.mimicry_detector = MolecularMimicryDetector(
            esm_dim=config.esm_embed_dim,
            hidden_dim=config.mimicry_hidden_dim,
            output_dim=config.latent_dim,
            temperature=config.temperature,
        )

        # Safety module
        self.safety_module = SafetyModule(
            input_dim=config.latent_dim,
            hidden_dim=config.safety_hidden_dim,
            esm_dim=config.esm_embed_dim,
        )

        # Multimodal Fusion (lightweight cross-attention)
        self.fusion = MultimodalFusion(
            seq_dim=config.latent_dim,
            cond_dim=config.latent_dim,
            fusion_dim=config.fusion_dim,
            num_heads=config.fusion_heads,
            num_layers=config.fusion_layers,
            dropout=config.dropout,
        )

        # === Decoder (autoregressive; z + condition injected every step) ===
        self.decoder = HydrAMPDecoder(
            latent_dim=config.latent_dim,
            condition_dim=config.latent_dim,  # Fused conditioning
            embed_dim=config.decoder_embed_dim,
            gru_dim=config.decoder_gru_dim,
            vocab_size=config.vocab_size,
            max_len=config.max_seq_len,
            pad_idx=config.pad_idx,
        )

        # === Conditioning projectors (all map to latent_dim cond tokens) ===
        # MIC as a conditional input: log10(MIC) -> conditioning token
        self.mic_proj = nn.Sequential(
            nn.Linear(1, config.mic_cond_dim),
            nn.GELU(),
            nn.Linear(config.mic_cond_dim, config.latent_dim),
        )
        # ESM peptide context -> conditioning token (cross-attention)
        self.esm_cond_proj = nn.Linear(config.esm_embed_dim, config.latent_dim)
        # Safety scores -> conditioning token
        self.safety_to_cond = nn.Linear(4, config.latent_dim)

        # Binary AMP classifier (returns logits; use BCEWithLogits)
        self.amp_classifier = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )


    def encode(self, sequences, graph_data=None, phylo_embeddings=None):
        """Encode input sequences through dual encoder + VQ bottleneck.

        Args:
            sequences: [batch, max_seq_len] - tokenized peptide sequences
            graph_data: Optional tuple (x, edge_index, edge_attr, batch) from PyG
            phylo_embeddings: Optional [batch, num_species, embed_dim]

        Returns:
            quantized: [batch, max_seq_len, latent_dim] - per-position quantized grid
            vq_loss: VQ-VAE loss
            encoding_info: Dict with auxiliary information
        """
        # Per-position sequence encoding (a latent *grid*, one vector per residue
        # position). VQ then assigns a sequence of codes instead of compressing
        # the whole peptide into a single code, which is far too lossy.
        seq_mean, seq_logvar = self.seq_encoder.encode_sequence(sequences)
        z_seq = seq_mean  # [batch, max_seq_len, latent_dim]

        # Graph encoding (if available and enabled)
        if graph_data is not None and self.use_graph_encoder:
            x, edge_index, edge_attr, batch_vec = graph_data
            # FiLM conditioning on phylogenetics makes graph features species-aware
            film_cond = None
            if phylo_embeddings is not None:
                film_cond = self.phylo_conditioner(phylo_embeddings)
            graph_mean, graph_logvar, node_features = self.graph_encoder(
                x, edge_index, edge_attr, batch_vec, film_cond
            )
            # Graph encoder yields one vector per peptide; broadcast it over the
            # positions and fuse with each position's sequence latent.
            z_graph = graph_mean.unsqueeze(1).expand(-1, z_seq.shape[1], -1)
            z_combined = self.encoder_fusion(torch.cat([z_seq, z_graph], dim=-1))
        else:
            z_combined = z_seq
            graph_mean, graph_logvar = None, None

        # Vector quantization over the latent grid: [batch, max_seq_len, latent_dim]
        # in -> per-position codes out ([batch, max_seq_len] indices).
        quantized, vq_loss, encoding_indices, perplexity = self.vector_quantizer(z_combined)

        encoding_info = {
            'seq_mean': seq_mean,
            'seq_logvar': seq_logvar,
            'graph_mean': graph_mean,
            'graph_logvar': graph_logvar,
            'z_combined': z_combined,
            'encoding_indices': encoding_indices,
            'perplexity': perplexity,
        }

        return quantized, vq_loss, encoding_info

    def _assemble_conditions(self, batch_size, device, phylo_cond=None,
                             mimicry_cond=None, safety_scores=None,
                             mic_cond=None, esm_cond=None):
        """Stack available conditioning vectors into [batch, num_cond, latent_dim].

        Missing modalities are filled with zeros so the set of conditioning
        tokens is fixed-size and order-stable across batches.
        """
        ld = self.config.latent_dim
        zeros = lambda: torch.zeros(batch_size, ld, device=device)
        tokens = [
            phylo_cond if phylo_cond is not None else zeros(),
            mimicry_cond if mimicry_cond is not None else zeros(),
            safety_scores if safety_scores is not None else zeros(),
            mic_cond if mic_cond is not None else zeros(),
            esm_cond if esm_cond is not None else zeros(),
        ]
        return torch.stack(tokens, dim=1)  # [batch, 5, latent_dim]

    def decode(self, quantized, conditions=None, temperature=1.0,
               target_tokens=None, sample=False):
        """Decode quantized representation with multimodal conditioning.

        Args:
            quantized: [batch, latent_dim] OR [batch, max_seq_len, latent_dim]
                - VQ-quantized latent (single vector or per-position grid)
            conditions: [batch, num_cond, latent_dim] - stacked conditioning tokens
            temperature: sampling temperature (generation only)
            target_tokens: Optional [batch, max_seq_len] - for teacher forcing
            sample: if True, sample tokens autoregressively (generation)

        Returns:
            logits: [batch, max_seq_len, vocab_size]
            attention_weights: cross-attention maps from fusion
        """
        batch_size = quantized.shape[0]
        device = quantized.device

        if conditions is None:
            conditions = self._assemble_conditions(batch_size, device)
        elif conditions.dim() == 2:
            # Legacy call: a single conditioning vector (treated as phylo cond)
            conditions = self._assemble_conditions(
                batch_size, device, phylo_cond=conditions)

        # Sequence query features. A latent grid is already per-position; a
        # single vector is broadcast over positions for backward compatibility.
        if quantized.dim() == 3:
            seq_features = quantized
        else:
            seq_features = quantized.unsqueeze(1).expand(
                -1, self.config.max_seq_len, -1)

        # Cross-attention fusion (sequence attends to conditioning tokens)
        fused, attention_weights = self.fusion(seq_features, conditions)
        fused_pooled = fused.mean(dim=1)  # [batch, latent_dim]

        logits = self.decoder(
            quantized, condition=fused_pooled, temperature=temperature,
            target_tokens=target_tokens,
            teacher_forcing_ratio=getattr(self.config, 'teacher_forcing_ratio', 0.5),
            sample=sample,
        )

        return logits, attention_weights


    def forward(self, sequences, graph_data=None, phylo_embeddings=None,
                mic=None, esm_peptide=None, esm_antigens=None, esm_defense=None,
                temperature=1.0):
        """Full forward pass.

        Args:
            sequences: [batch, max_seq_len] - input sequences
            graph_data: Optional graph data tuple (x, edge_index, edge_attr, batch)
            phylo_embeddings: Optional [batch, num_species, phylo_embed_dim] (hyperbolic coords)
            mic: Optional [batch, 1] - log10(MIC), conditional input (NaN where missing)
            esm_peptide: Optional [batch, esm_dim] - ESM-2 embeddings of peptides
            esm_antigens: Optional [batch, num_antigens, esm_dim]
            esm_defense: Optional [batch, num_defense, esm_dim]
            temperature: sampling temperature (generation only)

        Returns:
            output: Dict with all model outputs
        """
        batch_size = sequences.shape[0]
        device = sequences.device

        # Encode
        quantized, vq_loss, encoding_info = self.encode(
            sequences, graph_data, phylo_embeddings
        )
        # Pooled latent for whole-peptide heads
        quantized_pooled = quantized.mean(dim=1) if quantized.dim() == 3 else quantized

        # Conditioning tokens (each [batch, latent_dim])
        # Phylogenetic conditioning (2D Poincaré -> MLP)
        phylo_cond = None
        if phylo_embeddings is not None:
            phylo_cond = self.phylo_conditioner(phylo_embeddings)

        # Mimicry conditioning via ESM cross-attention module
        mimicry_cond = None
        mimicry_risk = None
        esm_cond = None
        if esm_peptide is not None:
            esm_cond = self.esm_cond_proj(esm_peptide)
            if esm_antigens is not None:
                mimicry_risk, mimicry_cond = self.mimicry_detector(
                    esm_peptide, esm_antigens, esm_defense
                )

        # MIC as a conditional input: log10(MIC); NaN -> 0 (no constraint)
        mic_cond = None

        if mic is not None:
            mic_clean = torch.nan_to_num(mic, nan=0.0)
            mic_cond = self.mic_proj(mic_clean)

        # Safety assessment
        safety_output = self.safety_module(quantized_pooled)
        safety_scores = torch.cat([
            safety_output['toxicity'],
            safety_output['hemolysis'],
            safety_output['immunogenicity'],
            safety_output['selectivity'],
        ], dim=-1)  # [batch, 4]
        safety_cond = self.safety_to_cond(safety_scores)

        conditions = self._assemble_conditions(
            batch_size, device,
            phylo_cond=phylo_cond, mimicry_cond=mimicry_cond,
            safety_scores=safety_cond, mic_cond=mic_cond, esm_cond=esm_cond,
        )

        # Decode (teacher forcing during training)
        logits, attention_weights = self.decode(
            quantized, conditions=conditions, temperature=temperature,
            target_tokens=sequences if self.training else None,
        )

        # Binary AMP classification (BCEWithLogits)
        amp_logits = self.amp_classifier(quantized_pooled)

        return {
            'logits': logits,
            'quantized': quantized_pooled,
            'vq_loss': vq_loss,
            'encoding_info': encoding_info,
            'phylo_cond': phylo_cond,
            'mimicry_cond': mimicry_cond,
            'mimicry_risk': mimicry_risk,
            'safety': safety_output,
            'amp_logits': amp_logits,
            'amp_prediction': torch.sigmoid(amp_logits),  # back-compat probability
            'attention_weights': attention_weights,
        }


    def generate(self,
                 num_samples=1,
                 phylo_embeddings=None,
                 esm_antigens=None,
                 safety_threshold=0.3,
                 temperature=0.5,
                 max_attempts=100):
        """Generate novel AMP sequences with evolutionary and safety constraints.

        Args:
            num_samples: Number of peptides to generate
            phylo_embeddings: Target species phylogenetic context
            esm_antigens: Autoantigen embeddings to avoid
            safety_threshold: Max acceptable safety score
            temperature: Sampling temperature
            max_attempts: Maximum generation attempts

        Returns:
            sequences: List of generated amino acid sequences
            metadata: Dict with scores and conditioning info
        """
        self.eval()
        device = next(self.parameters()).device

        generated_sequences = []
        generated_metadata = []

        with torch.no_grad():
            for _ in range(max_attempts):
                if len(generated_sequences) >= num_samples:
                    break

                # Sample a latent grid: one code per position
                indices = torch.randint(
                    0, self.config.num_codebook_vectors,
                    (1, self.config.max_seq_len), device=device
                )
                z = self.vector_quantizer.get_codebook_entry(indices)  # [1, L, latent]
                z_pool = z.mean(dim=1)

                # Get conditioning
                phylo_cond = None
                if phylo_embeddings is not None:
                    phylo_cond = self.phylo_conditioner(phylo_embeddings)

                # Safety check (pooled latent)
                safety = self.safety_module(z_pool)
                if safety['safety_score'].item() > safety_threshold:
                    continue

                # Build conditioning tokens and decode autoregressively (sampling)
                safety_scores = torch.cat([
                    safety['toxicity'], safety['hemolysis'],
                    safety['immunogenicity'], safety['selectivity'],
                ], dim=-1)
                conditions = self._assemble_conditions(
                    1, device, phylo_cond=phylo_cond,
                    safety_scores=self.safety_to_cond(safety_scores),
                )
                sampled = self.decoder(
                    z, condition=self.fusion(z, conditions)[0].mean(dim=1),
                    temperature=temperature, return_logits=False, sample=True,
                )
                sampled = sampled.view(1, self.config.max_seq_len)

                # AMP activity prediction (logits -> probability)
                amp_score = torch.sigmoid(self.amp_classifier(z_pool))

                if amp_score.item() > 0.5:
                    generated_sequences.append(sampled)
                    generated_metadata.append({
                        'amp_score': amp_score.item(),
                        'safety_score': safety['safety_score'].item(),
                        'toxicity': safety['toxicity'].item(),
                        'hemolysis': safety['hemolysis'].item(),
                        'codebook_index': indices.squeeze(0).tolist(),
                    })

        return generated_sequences, generated_metadata
