#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generation script for AutoGuard.

Generates novel AMP sequences with evolutionary and safety constraints.

Usage:
    python -m autoguard.scripts.generate --checkpoint checkpoints/best_model.pt --num_samples 100
"""

import argparse
import logging
import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autoguard.config import ModelConfig
from autoguard.models.autoguard_model import AutoGuardModel
from autoguard.data.datasets import detokenize_sequence
from autoguard.utils.amino_acids import compute_peptide_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate AMPs with AutoGuard')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of sequences to generate')
    parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature')
    parser.add_argument('--safety_threshold', type=float, default=0.3, help='Safety threshold')
    parser.add_argument('--output', type=str, default='generated_amps.json', help='Output file')
    parser.add_argument('--output_fasta', type=str, default='generated_amps.fasta', help='FASTA output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint.get('config', ModelConfig())
    model = AutoGuardModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    logger.info(f"Loaded model from {args.checkpoint}")
    logger.info(f"Generating {args.num_samples} sequences with T={args.temperature}, "
                f"safety<{args.safety_threshold}")

    # Generate
    sequences, metadata = model.generate(
        num_samples=args.num_samples,
        temperature=args.temperature,
        safety_threshold=args.safety_threshold,
    )

    # Decode and compute properties
    results = []
    for i, (seq_tokens, meta) in enumerate(zip(sequences, metadata)):
        seq_str = detokenize_sequence(seq_tokens.squeeze())
        properties = compute_peptide_features(seq_str)

        result = {
            'id': f'AutoGuard_{i+1:04d}',
            'sequence': seq_str,
            'length': len(seq_str),
            **meta,
            **properties,
        }
        results.append(result)

    # Save JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} sequences to {args.output}")

    # Save FASTA
    with open(args.output_fasta, 'w') as f:
        for r in results:
            f.write(f">{r['id']} AMP_score={r['amp_score']:.3f} "
                    f"safety={r['safety_score']:.3f} "
                    f"MW={r.get('molecular_weight', 0):.1f}\n")
            f.write(f"{r['sequence']}\n")
    logger.info(f"Saved FASTA to {args.output_fasta}")

    # Summary statistics
    amp_scores = [r['amp_score'] for r in results]
    safety_scores = [r['safety_score'] for r in results]
    logger.info(f"\n--- Generation Summary ---")
    logger.info(f"Generated: {len(results)} sequences")
    logger.info(f"Mean AMP score: {sum(amp_scores)/len(amp_scores):.3f}")
    logger.info(f"Mean safety score: {sum(safety_scores)/len(safety_scores):.3f}")
    logger.info(f"Mean length: {sum(r['length'] for r in results)/len(results):.1f}")


if __name__ == '__main__':
    main()
