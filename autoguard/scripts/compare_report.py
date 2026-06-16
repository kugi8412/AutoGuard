#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build the final AutoGuard-vs-HydrAMP comparison report.
Reads per-model ``metrics.json`` files (from ``evaluate_model``) and the
AutoGuard SAE ``sae_stats.json`` (from ``train --stage sae``) and writes a
side-by-side Markdown + JSON report.

Usage:
    python -m autoguard.scripts.compare_report \
        --metrics results/autoguard/metrics.json results/hydramp/metrics.json \
        --sae_stats checkpoints/sae/sae_stats.json \
        --out_dir results/comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Metrics where a higher value is better.
HIGHER_BETTER = {
    "novelty", "diversity", "mean_amp_score", "amp_hit_rate", "quality_score",
    "unique_fraction", "length_valid_fraction", "canonical_fraction",
    "challenge_valid_fraction", "predicted_success_rate",
    "predicted_safety_window", "predicted_non_hemolytic_fraction",
}

# Metrics where a lower value is better.
LOWER_BETTER = {
    "predicted_mic50", "predicted_mic90", "num_overlap_reference",
    "top_k_max_identity", "top_k_identity_violations",
}
METRIC_ORDER = [
    "num_generated", "mean_length", "novelty", "diversity",
    "mean_amp_score", "amp_hit_rate", "hydrophobicity_ratio", "quality_score",
    "unique_fraction", "length_valid_fraction", "canonical_fraction",
    "challenge_valid_fraction", "num_overlap_reference",
    "predicted_success_rate", "predicted_mic50", "predicted_mic90",
    "predicted_safety_window", "predicted_non_hemolytic_fraction",
    "top_k_max_identity", "top_k_identity_violations",
]


def _load_json(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_nan(v) -> bool:
    return isinstance(v, float) and v != v


def _fmt(v) -> str:
    if isinstance(v, float):
        if _is_nan(v):
            return "N/A"
        return f"{v:.4f}"
    return str(v)


# Metrics shown in the side-by-side bar chart (all in [0, 1], higher = better).
_BAR_METRICS = [
    "novelty", "diversity", "mean_amp_score", "amp_hit_rate", "quality_score",
    "challenge_valid_fraction", "predicted_success_rate",
    "predicted_non_hemolytic_fraction",
]


def _load_generated_csv(path: str):
    """Load (amp_scores, lengths, sequences) from an evaluate_model generated.csv."""
    import csv as _csv

    scores: List[float] = []
    lengths: List[int] = []
    seqs: List[str] = []

    if not path or not Path(path).exists():
        return scores, lengths, seqs

    with open(path, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            s = (row.get("sequence") or "").strip()
            if not s:
                continue
            seqs.append(s)
            try:
                scores.append(float(row.get("amp_score", "nan")))
            except ValueError:
                scores.append(float("nan"))
            try:
                lengths.append(int(row.get("length", len(s))))
            except ValueError:
                lengths.append(len(s))

    return scores, lengths, seqs


def make_comparison_plots(out_dir: Path,
                          names: List[str],
                          models: List[Dict],
                          generated_paths: List[str]
                          ) -> List[str]:
    """Render side-by-side comparison figures for a fair two-model comparison.
    Produces (when matplotlib is available):
      * metrics_comparison.png        — grouped bar chart of key quality metrics
      * amp_score_distribution.png    — per-model AMP-score histograms
      * length_distribution.png       — per-model sequence-length histograms
      * physicochemical_comparison.png — per-model net-charge & hydrophobicity
    Returns the list of file names that were written (empty if plotting was
    skipped, e.g. matplotlib not installed).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:  # pragma: no cover - depends on env
        logger.warning(f"matplotlib/numpy not available, skipping plots: {e}")
        return []

    from autoguard.scripts.full_experiment import compute_physicochemical

    written: List[str] = []
    colors = ["#2c7fb8", "#de2d26", "#31a354", "#756bb1"]

    # Grouped bar chart of headline metrics
    keys = [k for k in _BAR_METRICS
            if any(isinstance(m.get(k), (int, float)) and not _is_nan(m.get(k))
                   for m in models)]
    if keys:
        x = np.arange(len(keys))
        n_models = len(models)
        width = 0.8 / max(n_models, 1)
        fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.3), 6))
        for i, (name, m) in enumerate(zip(names, models)):
            vals = [m.get(k) if isinstance(m.get(k), (int, float))
                    and not _is_nan(m.get(k)) else 0.0 for k in keys]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=name, color=colors[i % len(colors)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_ylabel("Score (higher is better)")
        ax.set_title("AutoGuard vs HydrAMP — headline metrics")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_dir / "metrics_comparison.png"), dpi=150)
        plt.close(fig)
        written.append("metrics_comparison.png")

    # Load per-model generated sequences for distribution plots.
    per_model = [_load_generated_csv(p) for p in generated_paths]

    # AMP-score distributions
    if any(scores for scores, _, _ in per_model):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (name, (scores, _, _)) in enumerate(zip(names, per_model)):
            clean = [s for s in scores if s == s]  # drop NaN
            if clean:
                ax.hist(clean, bins=20, alpha=0.55, label=name,
                        color=colors[i % len(colors)])
        ax.set_xlabel("AMP score")
        ax.set_ylabel("Count")
        ax.set_title("Generated AMP-score distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_dir / "amp_score_distribution.png"), dpi=150)
        plt.close(fig)
        written.append("amp_score_distribution.png")

    # Length distributions
    if any(lengths for _, lengths, _ in per_model):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (name, (_, lengths, _)) in enumerate(zip(names, per_model)):
            if lengths:
                ax.hist(lengths, bins=range(min(lengths), max(lengths) + 2),
                        alpha=0.55, label=name, color=colors[i % len(colors)])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Count")
        ax.set_title("Generated sequence-length distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_dir / "length_distribution.png"), dpi=150)
        plt.close(fig)
        written.append("length_distribution.png")

    # Physicochemical comparison (net charge + hydrophobicity)
    charges, hydros, labels = [], [], []
    for name, (_, _, seqs) in zip(names, per_model):
        if not seqs:
            continue
        feats = [compute_physicochemical(s) for s in seqs if s]
        feats = [f for f in feats if f]
        if not feats:
            continue
        charges.append([f["net_charge"] for f in feats])
        hydros.append([f["mean_hydrophobicity"] for f in feats])
        labels.append(name)

    if labels:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        bp0 = axes[0].boxplot(charges, labels=labels, patch_artist=True)
        axes[0].set_title("Net charge")
        axes[0].set_ylabel("Net charge")
        axes[0].grid(axis="y", alpha=0.3)
        bp1 = axes[1].boxplot(hydros, labels=labels, patch_artist=True)
        axes[1].set_title("Mean hydrophobicity (Kyte-Doolittle)")
        axes[1].set_ylabel("Mean hydrophobicity")
        axes[1].grid(axis="y", alpha=0.3)

        for bp in (bp0, bp1):
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor(colors[i % len(colors)])
                box.set_alpha(0.6)

        fig.suptitle("Physicochemical properties of generated peptides")
        fig.tight_layout()
        fig.savefig(str(out_dir / "physicochemical_comparison.png"), dpi=150)
        plt.close(fig)
        written.append("physicochemical_comparison.png")

    return written


def main():
    parser = argparse.ArgumentParser(description="Compare models + SAE into one report.")
    parser.add_argument("--metrics", nargs="+", required=True,
                        help="metrics.json files (one per model)")
    parser.add_argument("--generated", nargs="*", default=[],
                        help="generated.csv files (one per model, same order as "
                             "--metrics) used for distribution/physicochemical plots")
    parser.add_argument("--sae_stats", type=str, default="",
                        help="AutoGuard SAE sae_stats.json (optional)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable PNG comparison plots.")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models: List[Dict] = []

    for p in args.metrics:
        if not Path(p).exists():
            logger.warning(f"  Missing metrics file: {p}")
            continue
        models.append(_load_json(p))

    if not models:
        logger.error("No metrics files could be loaded.")
        sys.exit(1)

    names = [m.get("model", f"model{i}") for i, m in enumerate(models)]

    # Markdown Report
    lines: List[str] = [
        "# Model Comparison Report — AutoGuard vs HydrAMP",
        "",
        f"Models compared: {', '.join(names)}",
        "",
        "## Generated-sequence metrics",
        "",
        "| Metric | " + " | ".join(names) + " | Winner |",
        "|" + "---|" * (len(names) + 2),
    ]
    for key in METRIC_ORDER:
        vals = [m.get(key) for m in models]
        if all(v is None for v in vals):
            continue

        row = f"| {key} |"

        for v in vals:
            row += f" {_fmt(v) if v is not None else 'N/A'} |"

        winner = "—"
        numeric = [(n, v) for n, v in zip(names, vals)
                   if isinstance(v, (int, float)) and not _is_nan(v)]

        if key in HIGHER_BETTER and numeric:
            winner = max(numeric, key=lambda x: x[1])[0]
        elif key in LOWER_BETTER and numeric:
            winner = min(numeric, key=lambda x: x[1])[0]

        row += f" {winner} |"
        lines.append(row)

    # Overall winner by quality_score.
    q = [(m.get("model", "?"), m.get("quality_score")) for m in models
         if isinstance(m.get("quality_score"), (int, float))]
    overall = max(q, key=lambda x: x[1])[0] if q else "N/A"
    lines += ["", f"**Overall winner (by quality_score): {overall}**", ""]

    # Comparison plots
    plot_files: List[str] = []
    if not args.no_plots:
        plot_files = make_comparison_plots(out_dir, names, models, args.generated)
        if plot_files:
            lines += ["## Comparison plots", ""]
            for fn in plot_files:
                title = fn.replace("_", " ").replace(".png", "").capitalize()
                lines += [f"### {title}", "", f"![{title}]({fn})", ""]

    # SAE section
    sae = None
    if args.sae_stats and Path(args.sae_stats).exists():
        sae = _load_json(args.sae_stats)
        lines += [
            "## Sparse Autoencoder (AutoGuard interpretability)",
            "",
            "The SAE is an AutoGuard-only interpretability module (HydrAMP has no",
            "equivalent). It decomposes the fused latent into sparse, monosemantic",
            "features.",
            "",
            "| SAE statistic | Value |",
            "|---|---|",
            f"| input_dim | {sae.get('input_dim')} |",
            f"| hidden_dim (features) | {sae.get('hidden_dim')} |",
            f"| alive_features | {sae.get('alive_features')} |",
            f"| dead_features | {sae.get('dead_features')} |",
            f"| best_loss | {_fmt(sae.get('best_loss'))} |",
            f"| sparsity_lambda | {sae.get('sparsity_lambda')} |",
            f"| top_k | {sae.get('top_k')} |",
            f"| num_activations | {sae.get('num_activations')} |",
            "",
        ]
    else:
        lines += ["## Sparse Autoencoder", "", "_No SAE stats provided._", ""]

    report_md = "\n".join(lines)
    (out_dir / "comparison_report.md").write_text(report_md, encoding="utf-8")

    # JSON report
    json_report = {
        "models": models,
        "overall_winner": overall,
        "sae": sae,
        "plots": plot_files,
    }
    with open(out_dir / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    logger.info("\n" + report_md)
    logger.info(f"Saved: {out_dir / 'comparison_report.md'} and comparison_report.json")


if __name__ == "__main__":
    main()
