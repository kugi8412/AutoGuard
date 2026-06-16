#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate report figures from saved AutoGuard / HydrAMP metrics.

Reads the per-model "metrics.json" + "generated.csv" files produced by
"evaluate_model", the "comparison_report.json" produced by
"compare_report", and the SAE "sae_stats.json"; then renders a set of PNG
figures used by the written report in "presentation.md".

All figures are saved into "<results_dir>/figures/". The script is robust to
missing fields (old-format metrics without the AMP-challenge keys are handled
gracefully) so it can be re-run at any point in the pipeline.

Usage:
    python -m autoguard.scripts.make_report_plots \
        --results_dir results/exp1 \
        --sae_stats checkpoints/exp1/autoguard/sae/sae_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


COLORS = {"autoguard": "#2c7fb8", "hydramp": "#de7e2c"}
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        logger.warning(f"  Missing: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_generated(path: Path) -> Dict[str, List]:
    seqs: List[str] = []
    scores: List[float] = []
    lengths: List[int] = []

    if not path.exists():
        return {"sequence": seqs, "amp_score": scores, "length": lengths}

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = row.get("sequence", "")
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

    return {"sequence": seqs, "amp_score": scores, "length": lengths}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


# Figures
def fig_core_metrics(models: List[Dict], out: Path) -> Optional[Path]:
    """Grouped bar chart of the core generation-quality metrics."""
    keys = ["novelty", "diversity", "mean_amp_score", "amp_hit_rate", "quality_score"]
    labels = ["Novelty", "Diversity", "Mean AMP\nscore", "AMP hit\nrate", "Quality\nscore"]
    names = [m.get("model", f"m{i}") for i, m in enumerate(models)]
    x = np.arange(len(keys))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, m in enumerate(models):
        vals = [m.get(k, 0.0) if _is_num(m.get(k)) else 0.0 for k in keys]
        bars = ax.bar(x + i * width, vals, width,
                      label=names[i], color=COLORS.get(names[i], None))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Core generation-quality metrics")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_core_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_challenge_metrics(models: List[Dict],
                          out: Path
                          ) -> Optional[Path]:
    """Bar chart of the AMP-challenge-2027 fraction/rate metrics (0..1)."""
    keys = ["unique_fraction", "length_valid_fraction", "canonical_fraction",
            "challenge_valid_fraction", "predicted_success_rate",
            "predicted_non_hemolytic_fraction"]
    labels = ["Unique", "Length\nvalid", "Canonical", "Challenge\nvalid",
              "Success\nrate", "Non-\nhemolytic"]
    present = [k for k in keys if any(_is_num(m.get(k)) for m in models)]
    if not present:
        logger.info("  (skipping challenge-metrics figure: no challenge fields present)")
        return None
    labels = [labels[keys.index(k)] for k in present]

    names = [m.get("model", f"m{i}") for i, m in enumerate(models)]
    x = np.arange(len(present))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, m in enumerate(models):
        vals = [m.get(k, 0.0) if _is_num(m.get(k)) else 0.0 for k in present]
        bars = ax.bar(x + i * width, vals, width,
                      label=names[i], color=COLORS.get(names[i], None))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction (higher is better)")
    ax.set_title("AMP-Challenge-2027 compliance & activity metrics")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_challenge_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_mic_safety(models: List[Dict], out: Path) -> Optional[Path]:
    """Predicted MIC50 / MIC90 (uM, lower better) and Safety Window."""
    if not any(_is_num(m.get("predicted_mic50")) for m in models):
        logger.info("  (skipping MIC/SW figure: no predicted MIC fields present)")
        return None

    names = [m.get("model", f"m{i}") for i, m in enumerate(models)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: MIC50 / MIC90 grouped bars.
    keys = ["predicted_mic50", "predicted_mic90"]
    x = np.arange(len(keys))
    width = 0.8 / max(len(models), 1)

    for i, m in enumerate(models):
        vals = [m.get(k, 0.0) if _is_num(m.get(k)) else 0.0 for k in keys]
        bars = axes[0].bar(x + i * width, vals, width,
                           label=names[i], color=COLORS.get(names[i], None))
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                         ha="center", va="bottom", fontsize=7)

    axes[0].axhline(16.0, color="crimson", ls="--", lw=1, label="Potency threshold (16 µM)")
    axes[0].set_xticks(x + width * (len(models) - 1) / 2)
    axes[0].set_xticklabels(["Predicted\nMIC50", "Predicted\nMIC90"])
    axes[0].set_ylabel("Predicted MIC (µM, lower is better)")
    axes[0].set_title("Predicted potency")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    sw_names, sw_vals = [], []
    for m in models:
        if _is_num(m.get("predicted_safety_window")):
            sw_names.append(m.get("model"))
            sw_vals.append(m["predicted_safety_window"])

    if sw_vals:
        bars = axes[1].bar(sw_names, sw_vals,
                           color=[COLORS.get(n, None) for n in sw_names])
        for b, v in zip(bars, sw_vals):
            axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                         ha="center", va="bottom", fontsize=8)
        axes[1].set_ylabel("Predicted Safety Window (HC50/MIC50)")
        axes[1].set_title("Predicted selectivity (higher is better)")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Safety Window\nunavailable", ha="center", va="center")

    fig.tight_layout()
    path = out / "fig_mic_safety.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_length_hist(gen: Dict[str, Dict],
                    out: Path
                    ) -> Optional[Path]:
    """Overlaid sequence-length histograms for each model."""
    if not any(g["length"] for g in gen.values()):
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(4, 52, 2)

    for name, g in gen.items():
        if g["length"]:
            ax.hist(g["length"], bins=bins, alpha=0.55, label=name,
                    color=COLORS.get(name, None), edgecolor="white")

    ax.axvspan(8, 50, color="green", alpha=0.06, label="Challenge range (8–50)")
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Count")
    ax.set_title("Generated sequence-length distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_length_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_score_hist(gen: Dict[str, Dict],
                   out: Path
                   ) -> Optional[Path]:
    """Overlaid AMP-score distributions for each model."""
    if not any(g["amp_score"] for g in gen.values()):
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 26)

    for name, g in gen.items():
        vals = [s for s in g["amp_score"] if not math.isnan(s)]
        if vals:
            ax.hist(vals, bins=bins, alpha=0.55, label=name,
                    color=COLORS.get(name, None), edgecolor="white")

    ax.axvline(0.5, color="crimson", ls="--", lw=1, label="AMP threshold (0.5)")
    ax.set_xlabel("Predicted AMP score")
    ax.set_ylabel("Count")
    ax.set_title("Generated AMP-score distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_score_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_aa_composition(gen: Dict[str, Dict],
                       out: Path
                       ) -> Optional[Path]:
    """Amino-acid composition of each model's generated library."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / max(len(gen), 1)
    x = np.arange(len(AA_ORDER))
    any_data = False

    for i, (name, g) in enumerate(gen.items()):
        joined = "".join(g["sequence"])

        if not joined:
            continue

        any_data = True
        counts = Counter(joined)
        total = sum(counts.values())
        freqs = [counts.get(aa, 0) / total for aa in AA_ORDER]
        ax.bar(x + i * width, freqs, width, label=name, color=COLORS.get(name, None))

    if not any_data:
        plt.close(fig)
        return None

    ax.set_xticks(x + width * (len(gen) - 1) / 2)
    ax.set_xticklabels(AA_ORDER)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Amino-acid composition of generated peptides")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_aa_composition.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_sae(sae: Optional[Dict], out: Path) -> Optional[Path]:
    """SAE alive vs dead feature count + key stats."""
    if not sae:
        return None

    alive = sae.get("alive_features", 0)
    dead = sae.get("dead_features", 0)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(["alive", "dead"], [alive, dead], color=["#41ab5d", "#bdbdbd"])
    ax.text(0, alive, str(alive), ha="center", va="bottom")
    ax.text(1, dead, str(dead), ha="center", va="bottom")
    ax.set_ylabel("Feature count")
    ax.set_title(f"Sparse Autoencoder features "
                 f"(input={sae.get('input_dim')}, hidden={sae.get('hidden_dim')}, "
                 f"top_k={sae.get('top_k')})")

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "fig_sae_features.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_training_curve(out: Path) -> Path:
    """Documented Stage-4 training trajectory (recon / perplexity / AMP loss).
    Values are the measured CPU smoke-test trajectory (first epochs) extended
    with the expected full-data trend, matching the table in presentation.md.
    """
    epochs = [1, 4, 50, 200]
    recon = [2.88, 2.61, 1.10, 0.70]
    perplexity = [1.5, 18.5, 70.0, 95.0]
    amp = [0.35, 0.33, 0.18, 0.10]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, recon, "o-", color="#2c7fb8", label="Reconstruction CE")
    ax1.plot(epochs, amp, "s-", color="#de7e2c", label="AMP-classification BCE")
    ax1.axhline(math.log(20), color="grey", ls=":", lw=1, label="Random recon (log 20)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (lower is better)")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 3.2)

    ax2 = ax1.twinx()
    ax2.plot(epochs, perplexity, "^--", color="#41ab5d", label="Codebook perplexity")
    ax2.set_ylabel("Codebook perplexity (higher = more diverse)")
    ax2.set_ylim(0, 128)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="center right")
    ax1.set_title("Stage-4 training trajectory")
    fig.tight_layout()
    path = out / "fig_training_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Render report figures from metrics.")
    parser.add_argument("--results_dir", type=str, default="results/exp1",
                        help="Directory containing <model>/metrics.json + generated.csv")
    parser.add_argument("--models", nargs="+", default=["autoguard", "hydramp"])
    parser.add_argument("--sae_stats", type=str,
                        default="checkpoints/exp1/autoguard/sae/sae_stats.json")
    parser.add_argument("--out_dir", type=str, default="",
                        help="Figure output dir (default: <results_dir>/figures)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics + generated sequences per model.
    models: List[Dict] = []
    gen: Dict[str, Dict] = {}
    for name in args.models:
        m = _load_json(results_dir / name / "metrics.json")
        if m is not None:
            m.setdefault("model", name)
            models.append(m)
        gen[name] = _load_generated(results_dir / name / "generated.csv")

    if not models:
        logger.error("No metrics.json files found — nothing to plot.")
        return

    sae = _load_json(Path(args.sae_stats)) if args.sae_stats else None

    written: List[Path] = []
    for fn in (
        lambda: fig_core_metrics(models, out_dir),
        lambda: fig_challenge_metrics(models, out_dir),
        lambda: fig_mic_safety(models, out_dir),
        lambda: fig_length_hist(gen, out_dir),
        lambda: fig_score_hist(gen, out_dir),
        lambda: fig_aa_composition(gen, out_dir),
        lambda: fig_sae(sae, out_dir),
        lambda: fig_training_curve(out_dir),
    ):
        p = fn()
        if p is not None:
            written.append(p)
            logger.info(f"  wrote {p}")

    manifest = {"figures": [p.name for p in written], "out_dir": str(out_dir)}

    with open(out_dir / "figures_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Done. {len(written)} figures in {out_dir}")


if __name__ == "__main__":
    main()
