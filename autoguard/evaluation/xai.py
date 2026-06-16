#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Explainability and interpretability module for AutoGuard.

Integrates:
- Sparse Autoencoder feature discovery
- Attention weight visualization
- Integrated Gradients attribution
- SHAP-like feature importance
- Latent space exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List


class AutoGuardExplainer:
    """Comprehensive XAI toolkit for AutoGuard model interpretation."""

    def __init__(self, model, sparse_autoencoder=None, device=None):
        self.model = model
        self.sae = sparse_autoencoder

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = 'cpu'

        self.device = device

        if sparse_autoencoder is not None:
            self.sae = sparse_autoencoder.to(self.device)

    # =========================================================================
    # Sparse Autoencoder Analysis
    # =========================================================================

    def discover_features(self, dataloader, num_batches=100) -> Dict:
        """Run SAE on model activations to discover interpretable features.

        Returns:
            Dict with feature activations, importance scores, and dead features
        """
        if self.sae is None:
            raise ValueError("Sparse Autoencoder not provided")

        self.model.eval()
        self.sae.eval()

        all_features = []
        all_properties = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                tokens = batch['tokens'].to(self.device)
                output = self.model(tokens)

                # Get activations from the quantized representation
                activations = output['quantized']
                _, features, _ = self.sae(activations)
                all_features.append(features.cpu())

                if 'label' in batch:
                    all_properties.append(batch['label'])

        all_features = torch.cat(all_features, dim=0)

        result = {
            'feature_activations': all_features,
            'mean_activations': all_features.mean(dim=0),
            'activation_frequency': (all_features > 0).float().mean(dim=0),
            'num_dead_features': (all_features.mean(dim=0) < 1e-6).sum().item(),
        }

        # Feature-property correlations
        if all_properties:
            properties = torch.cat(all_properties, dim=0)
            importance = self.sae.get_feature_importance(
                all_features[:1000].to(self.device),
                properties[:1000].to(self.device)
            )
            result['feature_importance'] = importance.cpu()

        return result

    def interpret_top_features(self, feature_activations, target_property,
                               top_n=20) -> List[Dict]:
        """Identify and interpret the most important features for a property."""
        importance = self.sae.get_feature_importance(
            feature_activations.to(self.device),
            target_property.to(self.device)
        )

        top_indices = torch.topk(importance.abs(), top_n).indices.tolist()
        interpretations = self.sae.interpret_features(top_indices)

        results = []
        for idx in top_indices:
            results.append({
                'feature_idx': idx,
                'importance': importance[idx].item(),
                'activation_frequency': (feature_activations[:, idx] > 0).float().mean().item(),
                'decoder_pattern': interpretations[idx]['weight_pattern'],
            })

        return results

    # =========================================================================
    # Attention Analysis
    # =========================================================================

    def get_attention_maps(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Extract attention maps from multimodal fusion for a single sequence."""
        from ..data.datasets import tokenize_sequence

        self.model.eval()
        tokens = tokenize_sequence(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tokens)

        attention_weights = output.get('attention_weights', [])
        return {
            f'layer_{i}': attn.cpu()
            for i, attn in enumerate(attention_weights)
        }

    def modality_contribution(self, sequences: List[str]) -> Dict[str, float]:
        """Analyze which conditioning modalities contribute most to generation."""
        from ..data.datasets import tokenize_sequence

        self.model.eval()
        contributions = {'phylogenetic': [], 'mimicry': [], 'safety': []}

        with torch.no_grad():
            for seq in sequences:
                tokens = tokenize_sequence(seq).unsqueeze(0).to(self.device)
                output = self.model(tokens)

                attn_maps = output.get('attention_weights', [])
                if attn_maps:
                    # Last layer attention averaged over heads
                    last_attn = attn_maps[-1].mean(dim=1)  # [1, seq_len, 3]
                    # Three conditioning tokens: phylo, mimicry, safety
                    contributions['phylogenetic'].append(last_attn[0, :, 0].mean().item())
                    contributions['mimicry'].append(last_attn[0, :, 1].mean().item())
                    contributions['safety'].append(last_attn[0, :, 2].mean().item())

        return {k: np.mean(v) if v else 0.0 for k, v in contributions.items()}

    # =========================================================================
    # Integrated Gradients
    # =========================================================================

    def integrated_gradients(self,
                             sequence: str,
                             target: str = 'amp',
                             n_steps: int = 50
                             ) -> torch.Tensor:
        """Compute Integrated Gradients attribution for input residues.

        Args:
            sequence: Input peptide sequence
            target: Which output to attribute ('amp', 'safety', 'mic')
            n_steps: Number of interpolation steps

        Returns:
            attributions: [seq_len] - importance of each residue position
        """
        from ..data.datasets import tokenize_sequence

        self.model.eval()
        tokens = tokenize_sequence(sequence).unsqueeze(0).to(self.device)

        # Baseline: all padding
        baseline = torch.zeros_like(tokens)

        # Interpolate between baseline and input
        scaled_inputs = []
        for alpha in torch.linspace(0, 1, n_steps):
            scaled = baseline + alpha * (tokens - baseline)
            scaled_inputs.append(scaled)

        scaled_inputs = torch.cat(scaled_inputs, dim=0).requires_grad_(True)

        # Get embeddings and compute gradients
        embeddings = self.model.seq_encoder.embedding(scaled_inputs)
        embeddings.retain_grad()

        # Forward through encoder
        seq_mean, seq_logvar = self.model.seq_encoder(scaled_inputs)

        if target == 'amp':
            target_output = self.model.amp_classifier(seq_mean).sum()
        elif target == 'safety':
            safety = self.model.safety_module(seq_mean)
            target_output = safety['safety_score'].sum()
        else:
            # MIC is now a conditional input (no regressor); fall back to AMP logit
            target_output = self.model.amp_classifier(seq_mean).sum()

        target_output.backward()

        # Compute attributions
        grads = embeddings.grad  # [n_steps, seq_len, embed_dim]
        avg_grads = grads.mean(dim=0)  # [seq_len, embed_dim]

        # Input - baseline in embedding space
        input_embed = self.model.seq_encoder.embedding(tokens)
        baseline_embed = self.model.seq_encoder.embedding(baseline)
        diff = (input_embed - baseline_embed).squeeze(0)

        # Integrated gradients: (input - baseline) * avg_gradient
        attributions = (diff * avg_grads).sum(dim=-1)  # [seq_len]

        return attributions.detach().cpu()

    # =========================================================================
    # Latent Space Exploration
    # =========================================================================

    def latent_space_analysis(self, dataloader, num_samples=1000) -> Dict:
        """Analyze the learned latent/codebook space."""
        self.model.eval()
        latent_vectors = []
        codebook_indices = []
        labels = []

        with torch.no_grad():
            count = 0
            for batch in dataloader:
                if count >= num_samples:
                    break
                tokens = batch['tokens'].to(self.device)
                output = self.model(tokens)

                latent_vectors.append(output['quantized'].cpu())
                codebook_indices.append(output['encoding_info']['encoding_indices'].cpu())
                if 'label' in batch:
                    labels.append(batch['label'])
                count += len(tokens)

        latent_vectors = torch.cat(latent_vectors, dim=0)[:num_samples]
        codebook_indices = torch.cat(codebook_indices, dim=0)[:num_samples]

        # Codebook utilization
        unique_codes = codebook_indices.unique()
        total_codes = self.model.config.num_codebook_vectors

        # PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors.numpy())

        result = {
            'latent_2d': latent_2d,
            'codebook_utilization': len(unique_codes) / total_codes,
            'codebook_histogram': np.bincount(
                codebook_indices.numpy().flatten(),
                minlength=total_codes
            ),
            'explained_variance': pca.explained_variance_ratio_,
        }

        if labels:
            result['labels'] = torch.cat(labels, dim=0)[:num_samples].numpy()

        return result

    def generate_explanation_report(self, sequence: str) -> Dict:
        """Generate comprehensive explanation for a single peptide.

        Returns human-readable explanation of why the model scored
        this peptide the way it did.
        """
        from ..data.datasets import tokenize_sequence

        self.model.eval()
        tokens = tokenize_sequence(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tokens)

        # Gather all scores
        report = {
            'sequence': sequence,
            'length': len(sequence),
            'amp_score': output['amp_prediction'].item(),
            'safety_score': output['safety']['safety_score'].item(),
            'toxicity': output['safety']['toxicity'].item(),
            'hemolysis': output['safety']['hemolysis'].item(),
            'immunogenicity': output['safety']['immunogenicity'].item(),
            'codebook_index': output['encoding_info']['encoding_indices'].flatten().tolist(),
        }

        # Integrated gradients for key properties
        amp_attr = self.integrated_gradients(sequence, 'amp')
        safety_attr = self.integrated_gradients(sequence, 'safety')

        report['residue_importance_amp'] = {
            sequence[i]: amp_attr[i].item()
            for i in range(len(sequence))
        }
        report['residue_importance_safety'] = {
            sequence[i]: safety_attr[i].item()
            for i in range(len(sequence))
        }

        # Top contributing residues
        top_amp_positions = torch.topk(amp_attr[:len(sequence)].abs(), min(5, len(sequence)))
        report['top_amp_residues'] = [
            (sequence[i], amp_attr[i].item())
            for i in top_amp_positions.indices.tolist()
        ]

        return report
