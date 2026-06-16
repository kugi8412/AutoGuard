#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation and comparison script for AutoGuard.

Evaluates generated sequences and compares against baselines.

Usage:
    python -m autoguard.scripts.evaluate --generated generated_amps.json --training_data data/
"""

import argparse
import logging
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autoguard.evaluation.metrics import AMPMetrics, compute_novelty, compute_diversity
from autoguard.evaluation.comparison import BaselineComparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate AutoGuard generated AMPs')
    parser.add_argument('--generated', type=str, required=True, help='Generated sequences JSON')
    parser.add_argument('--training_data', type=str, default='data/', help='Training data dir')
    parser.add_argument('--output_report', type=str, default='evaluation_report.md')
    args = parser.parse_args()

    # Load generated sequences
    with open(args.generated, 'r') as f:
        results = json.load(f)

    sequences = [r['sequence'] for r in results]
    amp_scores = [r.get('amp_score', 0) for r in results]
    safety_scores = [r.get('safety_score', 0) for r in results]

    logger.info(f"Evaluating {len(sequences)} generated sequences")

    # Metrics calculation
    metrics_calc = AMPMetrics()
    metrics = metrics_calc.evaluate_batch(
        sequences, amp_scores, safety_scores
    )

    logger.info("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            logger.info(f"  {key}: {value}")

    # Comparison setup
    comparison = BaselineComparison(metrics_calc)
    comparison.add_baseline('AutoGuard', sequences, amp_scores, safety_scores)

    # Generate report
    report = comparison.generate_report()

    with open(args.output_report, 'w') as f:
        f.write(report)

    logger.info(f"\nReport saved to {args.output_report}")


if __name__ == '__main__':
    main()
