#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train baseline models independently (GNN-VAE and HydrAMP).

Usage:
    # Train GNN-VAE baseline
    python -m autoguard.scripts.train_baselines --model gnn \
      --data_dir data/ --epochs 100 --device cuda --output checkpoints/gnn/

    # Train HydrAMP baseline
    python -m autoguard.scripts.train_baselines --model hydramp \
      --data_dir data/ --epochs 50 --device cuda --output checkpoints/hydramp/

    # Train both
    python -m autoguard.scripts.train_baselines --model all \
      --data_dir data/ --device cuda --output checkpoints/baselines/
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(data_dir: str, max_seqs: int = 5000):
    """Load training sequences from processed CSVs."""
    processed = Path(data_dir) / "processed" / "amp_train.csv"
    sequences = []
    labels = []

    if processed.exists():
        with open(processed, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                sequences.append(row['sequence'])
                labels.append(float(row.get('label', 1)))
    else:
        logger.error(f"No data at {processed}. Run prepare_data.py first.")
        sys.exit(1)

    if max_seqs and len(sequences) > max_seqs:
        sequences = sequences[:max_seqs]
        labels = labels[:max_seqs]

    return sequences, labels


def train_gnn(sequences, labels, args):
    """Train GNN-VAE baseline."""
    from autoguard.models.gnn_baseline import GNNGenerator

    logger.info("=" * 60)
    logger.info("Training GNN-VAE Baseline")
    logger.info("=" * 60)
    logger.info(f"  Sequences: {len(sequences)}")
    logger.info(f"  Epochs: {args.epochs}")

    # Filter to active AMPs only
    active_seqs = [s for s, l in zip(sequences, labels) if l >= 0.5]
    logger.info(f"  Active AMPs: {len(active_seqs)}")

    gnn = GNNGenerator(
        max_len=25, latent_dim=32, gru_dim=64, num_gnn_layers=3,
        device=args.device,
    )
    gnn.train_model(
        sequences=active_seqs,
        epochs=args.epochs if args.epochs != 200 else 100,
        batch_size=args.batch_size,
        lr=1e-3,
    )

    # Generate samples
    generated = gnn.generate(num_samples=100, temperature=0.7)
    logger.info(f"  Generated {len(generated)} sequences")
    logger.info(f"  Sample: {generated[:3]}")

    # Save
    output_dir = Path(args.output) / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gnn.state_dict(), str(output_dir / "gnn_model.pt"))

    with open(output_dir / "generated_samples.json", 'w') as f:
        json.dump(generated, f, indent=2)

    logger.info(f"  Saved to {output_dir}")
    return gnn


def train_hydramp(sequences, labels, args):
    """Train HydrAMP baseline."""
    from autoguard.evaluation.hydramp_adapter import HydrAMPBaseline

    logger.info("=" * 60)
    logger.info("Training HydrAMP Baseline")
    logger.info("=" * 60)
    logger.info(f"  Sequences: {len(sequences)}")
    logger.info(f"  Epochs: {args.epochs}")

    hydramp = HydrAMPBaseline(device=args.device, temperature=1.0)
    hydramp.train(
        sequences=sequences,
        amp_labels=labels,
        epochs=args.epochs if args.epochs != 200 else 50,
        batch_size=args.batch_size,
        lr=1e-3,
        classifier_epochs=10,
    )

    # Generate samples
    generated = hydramp.generate(num_samples=100, temperature=0.8)
    logger.info(f"  Generated {len(generated)} sequences")
    logger.info(f"  Sample: {generated[:3]}")

    # Save
    output_dir = Path(args.output) / "hydramp"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hydramp.state_dict(), str(output_dir / "hydramp_model.pt"))

    with open(output_dir / "generated_samples.json", 'w') as f:
        json.dump(generated, f, indent=2)

    logger.info(f"  Saved to {output_dir}")
    return hydramp


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--model', type=str, default='all', choices=['gnn', 'hydramp', 'all'],
                        help='Which baseline to train')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--output', type=str, default='checkpoints/baselines/',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_seqs', type=int, default=5000, help='Max training sequences')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    sequences, labels = load_training_data(args.data_dir, args.max_seqs)
    logger.info(f"Loaded {len(sequences)} sequences from {args.data_dir}")

    if args.model in ('gnn', 'all'):
        train_gnn(sequences, labels, args)

    if args.model in ('hydramp', 'all'):
        train_hydramp(sequences, labels, args)

    logger.info("Done.")


if __name__ == '__main__':
    main()
