#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline comparison module for evaluating AutoGuard against existing methods.

Compares against:
- HydrAMP (original)
- PepCVAE
- Basic VAE
- Random sampling
"""

import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class BaselineResult:
    name: str
    sequences: List[str]
    amp_scores: List[float]
    safety_scores: List[float]
    novelty: float
    diversity: float
    mean_mic: float


class BaselineComparison:
    """Compare AutoGuard against baseline generation methods."""

    def __init__(self, metrics_calculator):
        self.metrics = metrics_calculator
        self.results = {}

    def add_baseline(self, name: str, sequences: List[str],
                     amp_scores: List[float] = None,
                     safety_scores: List[float] = None,
                     mic_predictions: List[float] = None):
        """Register a baseline result."""
        metrics = self.metrics.evaluate_batch(
            sequences, amp_scores, safety_scores, mic_predictions
        )
        self.results[name] = metrics

    def compare(self) -> Dict:
        """Generate comparison table across all registered methods."""
        comparison = {}
        metric_keys = [
            'novelty', 'diversity', 'mean_amp_score', 'amp_hit_rate',
            'safe_fraction', 'mean_predicted_mic', 'potent_fraction',
            'quality_score', 'hydrophobicity_ratio',
        ]

        for key in metric_keys:
            comparison[key] = {
                name: metrics.get(key, None)
                for name, metrics in self.results.items()
            }

        # Rank methods by quality score
        quality_ranking = sorted(
            self.results.items(),
            key=lambda x: x[1].get('quality_score', 0),
            reverse=True,
        )
        comparison['ranking'] = [name for name, _ in quality_ranking]

        return comparison

    def generate_report(self) -> str:
        """Generate human-readable comparison report."""
        comparison = self.compare()
        lines = [
            "# AutoGuard Baseline Comparison Report",
            "",
            "## Method Ranking (by Quality Score)",
            "",
        ]

        for rank, name in enumerate(comparison['ranking'], 1):
            score = self.results[name].get('quality_score', 0)
            lines.append(f"{rank}. **{name}** — Quality Score: {score:.4f}")

        lines.extend(["", "## Detailed Metrics", "", "| Metric | " +
                      " | ".join(comparison['ranking']) + " |"])
        lines.append("|" + "---|" * (len(comparison['ranking']) + 1))

        for key in ['novelty', 'diversity', 'amp_hit_rate', 'safe_fraction',
                    'potent_fraction', 'quality_score']:
            row = f"| {key} |"
            for name in comparison['ranking']:
                val = comparison[key].get(name, None)
                row += f" {val:.4f} |" if val is not None else " N/A |"
            lines.append(row)

        return "\n".join(lines)


class AblationStudy:
    """Ablation study for AutoGuard components."""

    COMPONENTS = [
        'graph_encoder',
        'vq_vae',
        'phylo_conditioning',
        'mimicry_detection',
        'safety_module',
        'multimodal_fusion',
    ]

    def __init__(self, full_model, config, test_data):
        self.full_model = full_model
        self.config = config
        self.test_data = test_data
        self.ablation_results = {}

    def run_ablation(self, component_to_remove: str) -> Dict:
        """Run model with one component disabled/zeroed out.
        Returns metrics for the ablated model.
        """
        # This would be implemented by zeroing out specific conditioning
        # or using the model without certain modules
        pass

    def full_ablation_study(self) -> Dict:
        """Run complete ablation study across all components."""
        results = {}

        for component in self.COMPONENTS:
            results[component] = self.run_ablation(component)

        self.ablation_results = results
        return results
