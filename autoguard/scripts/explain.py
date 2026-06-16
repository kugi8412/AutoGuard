#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XAI & Interpretability CLI for AutoGuard.

Wraps all XAI analyses (SAE features, attention maps, integrated gradients,
codebook analysis) into bash-callable commands.

Usage:
    # Discover SAE features
    python -m autoguard.scripts.explain --mode features \
      --checkpoint checkpoints/best_model.pt \
      --sae_checkpoint checkpoints/sae/best_sae.pt \
      --data_dir data/ --output results/features.json

    # Attention visualization
    python -m autoguard.scripts.explain --mode attention \
      --checkpoint checkpoints/best_model.pt \
      --sequence "KFLKKLRKFLKK" --output results/attention.json

    # Integrated gradients attribution
    python -m autoguard.scripts.explain --mode attribution \
      --checkpoint checkpoints/best_model.pt \
      --sequence "KFLKKLRKFLKK" --target amp_prediction \
      --output results/attribution.json

    # Codebook utilization analysis
    python -m autoguard.scripts.explain --mode codebook \
      --checkpoint checkpoints/best_model.pt \
      --data_dir data/ --output results/codebook.json

    # Full report (all of the above)
    python -m autoguard.scripts.explain --mode full \
      --checkpoint checkpoints/best_model.pt \
      --sae_checkpoint checkpoints/sae/best_sae.pt \
      --data_dir data/ --output results/xai_report.json
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autoguard.config import ModelConfig
from autoguard.models.autoguard_model import AutoGuardModel
from autoguard.data.datasets import AMPDataset, tokenize_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, device='cpu'):
    """Load trained AutoGuard model."""
    config = ModelConfig()
    model = AutoGuardModel(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_sae(sae_path, device='cpu'):
    """Load trained SAE."""
    from autoguard.models.sparse_autoencoder import SparseAutoencoder

    state = torch.load(sae_path, map_location=device, weights_only=False)
    input_dim = state['pre_bias'].shape[0]
    hidden_dim = state['encoder_bias'].shape[0]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(state)
    sae = sae.to(device)
    sae.eval()
    return sae


def load_test_data(data_dir, max_seqs=1000):
    """Load test sequences."""
    path = Path(data_dir) / "processed" / "amp_test.csv"
    sequences, labels = [], []

    if path.exists():
        with open(path, encoding='utf-8') as f:

            for row in csv.DictReader(f):
                sequences.append(row['sequence'])
                labels.append(int(float(row.get('label', 1))))
                if len(sequences) >= max_seqs:
                    break

    return sequences, labels


def run_features(model, sae, data_dir, device, output):
    """Discover SAE features on test data."""
    logger.info("Discovering SAE features...")
    sequences, labels = load_test_data(data_dir)

    if not sequences:
        logger.error("No test data found.")
        return {}

    all_features = []
    all_labels = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            tokens = torch.stack([tokenize_sequence(s) for s in batch_seqs]).to(device)
            output_dict = model(tokens)
            activations = output_dict['quantized']

            if activations.dim() == 3:
                activations = activations.mean(dim=1)
            _, features, _ = sae(activations)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels[i:i+batch_size])

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)

    # Feature statistics
    mean_activation = all_features.mean(axis=0)
    active_count = (all_features > 0).sum(axis=0)
    dead_features = (active_count < 5).sum()

    # Correlation with AMP activity
    if len(np.unique(all_labels)) > 1:
        from scipy import stats as sp_stats

        correlations = []

        for j in range(all_features.shape[1]):
            r, _ = sp_stats.pearsonr(all_features[:, j], all_labels)
            correlations.append(float(r) if not np.isnan(r) else 0.0)
    else:
        correlations = [0.0] * all_features.shape[1]

    # Top features by activity correlation
    top_active = np.argsort(correlations)[-10:][::-1]
    top_safety = np.argsort(correlations)[:10]

    results = {
        'num_features': int(all_features.shape[1]),
        'dead_features': int(dead_features),
        'alive_features': int(all_features.shape[1] - dead_features),
        'mean_active_per_sample': float((all_features > 0).sum(axis=1).mean()),
        'top_activity_features': [int(i) for i in top_active],
        'top_safety_features': [int(i) for i in top_safety],
        'top_correlations': [correlations[i] for i in top_active],
    }
    logger.info(f"  Alive features: {results['alive_features']}/{results['num_features']}")
    logger.info(f"  Mean active per sample: {results['mean_active_per_sample']:.1f}")
    return results


def run_attention(model, sequence, device):
    """Get cross-modal attention weights for a sequence."""
    logger.info(f"Attention analysis for: {sequence}")
    tokens = tokenize_sequence(sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tokens)

    # Extract attention if available
    results = {'sequence': sequence, 'length': len(sequence)}

    if 'attention_weights' in output:
        attn = output['attention_weights'][0].cpu().numpy()  # [heads, seq, modalities]
        results['attention_shape'] = list(attn.shape)
        results['per_residue_attention'] = attn.mean(axis=0).tolist()
    else:
        # Fallback: report conditioning outputs
        results['mimicry_risk'] = float(output.get('mimicry_risk', torch.tensor(0)).mean())

        if output.get('safety') is not None:
            results['safety_score'] = float(output['safety']['safety_score'].mean())

        results['amp_prediction'] = float(output['amp_prediction'].mean())

    logger.info(f"  AMP prediction: {results.get('amp_prediction', 'N/A')}")
    return results


def run_attribution(model, sequence, target, device, n_steps=50):
    """Integrated gradients attribution."""
    logger.info(f"Integrated gradients for: {sequence} (target={target})")
    tokens = tokenize_sequence(sequence).unsqueeze(0).to(device)
    tokens.requires_grad = False

    # Baseline: all zeros (padding)
    baseline = torch.zeros_like(tokens)

    # Interpolate
    attributions = torch.zeros(len(sequence), device=device)
    model.eval()

    for step in range(n_steps):
        alpha = step / n_steps
        interpolated = baseline + alpha * (tokens - baseline)
        interpolated = interpolated.long()

        # Need gradients through embedding
        model.zero_grad()
        embedding = model.seq_encoder.embedding(interpolated)
        embedding.retain_grad()

        # Forward from embedding
        output = model(interpolated)

        if target == 'amp_prediction':
            score = output['amp_prediction'].mean()
        elif target == 'safety_score':
            score = output['safety']['safety_score'].mean()
        else:
            score = output['amp_prediction'].mean()

        score.backward(retain_graph=True)

        if embedding.grad is not None:
            grad = embedding.grad[0, :len(sequence)].norm(dim=-1)
            attributions += grad

    attributions /= n_steps
    attr_np = attributions.detach().cpu().numpy()

    results = {
        'sequence': sequence,
        'target': target,
        'per_residue_attribution': attr_np.tolist(),
        'top_residues': [int(i) for i in np.argsort(attr_np)[-5:][::-1]],
        'top_residue_chars': [sequence[i] for i in np.argsort(attr_np)[-5:][::-1]],
    }
    logger.info(f"  Top residues: {results['top_residue_chars']}")
    return results


def run_codebook(model, data_dir, device):
    """Analyze VQ-VAE codebook utilization."""
    logger.info("Analyzing codebook utilization...")
    sequences, _ = load_test_data(data_dir, max_seqs=2000)
    if not sequences:
        return {}

    index_counts = torch.zeros(model.vector_quantizer.num_embeddings)

    with torch.no_grad():
        for i in range(0, len(sequences), 64):
            batch = sequences[i:i+64]
            tokens = torch.stack([tokenize_sequence(s) for s in batch]).to(device)
            output = model(tokens)
            indices = output['encoding_info']['encoding_indices']
            for idx in indices.flatten():
                index_counts[idx.item()] += 1

    used = (index_counts > 0).sum().item()
    total = model.vector_quantizer.num_embeddings

    # Entropy
    probs = index_counts / index_counts.sum()
    probs = probs[probs > 0]
    entropy = -float((probs * probs.log()).sum())
    max_entropy = float(np.log(total))

    results = {
        'total_codes': total,
        'used_codes': used,
        'utilization': used / total,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'normalized_entropy': entropy / max_entropy,
        'top_10_codes': index_counts.topk(10).indices.tolist(),
        'top_10_counts': index_counts.topk(10).values.tolist(),
    }
    logger.info(f"  Codebook: {used}/{total} used ({100*used/total:.1f}%)")
    logger.info(f"  Entropy: {entropy:.2f} / {max_entropy:.2f} ({100*entropy/max_entropy:.1f}%)")
    return results


def main():
    parser = argparse.ArgumentParser(description='AutoGuard XAI & Interpretability')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['features', 'attention', 'attribution', 'codebook', 'full'],
                        help='Analysis mode')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--sae_checkpoint', type=str, default=None,
                        help='SAE checkpoint (for features mode)')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Data directory')
    parser.add_argument('--sequence', type=str, default='KFLKKLRKFLKK',
                        help='Sequence for attention/attribution')
    parser.add_argument('--target', type=str, default='amp_prediction',
                        choices=['amp_prediction', 'safety_score'],
                        help='Target for attribution')
    parser.add_argument('--output', type=str, default='results/xai.json',
                        help='Output path')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)
    logger.info(f"Loaded model from {args.checkpoint}")

    results = {}

    if args.mode in ('features', 'full'):
        if args.sae_checkpoint and os.path.exists(args.sae_checkpoint):
            sae = load_sae(args.sae_checkpoint, args.device)
            results['features'] = run_features(model, sae, args.data_dir, args.device, args.output)
        else:
            logger.warning("SAE checkpoint not provided/found. Skipping feature analysis.")

    if args.mode in ('attention', 'full'):
        results['attention'] = run_attention(model, args.sequence, args.device)

    if args.mode in ('attribution', 'full'):
        results['attribution'] = run_attribution(model, args.sequence, args.target, args.device)

    if args.mode in ('codebook', 'full'):
        results['codebook'] = run_codebook(model, args.data_dir, args.device)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
