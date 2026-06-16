#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate 3 x 100 peptide sets and produce analysis + plots.
Works with any available checkpoint (even minimally trained).

Usage:
    python -m autoguard.scripts.generate_and_analyze --results_dir results/
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Import physicochemical computation from full_experiment
from autoguard.scripts.full_experiment import compute_physicochemical, create_plots


AA_VOCAB = "_ACDEFGHIKLMNPQRSTVWY"


def generate_peptides(model, species_embeds, device, num_per_set=100, max_attempts=5000):
    """Generate 3 sets of peptides: AMP-safe, AMP-SLE, Non-AMP."""
    model = model.to(device)
    model.eval()

    amp_safe = []    # High AMP, low safety risk
    amp_sle = []     # High AMP, high safety risk (mimicry concern)
    non_amp = []     # Low AMP score

    # Pre-compute phylo conditioning once
    if species_embeds is not None:
        phylo_input = species_embeds.unsqueeze(0).to(device)
        with torch.no_grad():
            phylo_cond = model.phylo_conditioner(phylo_input)
    else:
        phylo_cond = torch.zeros(1, model.config.latent_dim, device=device)

    all_seqs = []
    all_amp_scores = []
    all_safety_scores = []

    n_codes = model.config.num_codebook_vectors
    seq_len = model.config.max_seq_len
    num_grids = max(n_codes, num_per_set * 8)
    logger.info(f"  Sampling {num_grids} per-position latent grids ({seq_len} codes each)...")
    with torch.no_grad():
        for _ in range(num_grids):
            idx = torch.randint(0, n_codes, (1, seq_len), device=device)
            z = model.vector_quantizer.get_codebook_entry(idx)  # [1, seq_len, latent_dim]
            # Whole-peptide heads (AMP/safety) expect one vector per peptide.
            z_pool = z.mean(dim=1)

            amp_score = torch.sigmoid(model.amp_classifier(z_pool)).item()
            safety_out = model.safety_module(z_pool)
            safety_score = safety_out['safety_score'].item()

            # Decode with multiple temperatures for diversity
            for temp in [0.5, 0.8, 1.0, 1.2]:
                logits, _ = model.decode(z, phylo_cond, temperature=temp, sample=True)
                tokens = torch.argmax(logits.squeeze(0), dim=-1)
                seq = "".join(AA_VOCAB[t.item()] for t in tokens if 0 < t.item() < len(AA_VOCAB))

                if 5 <= len(seq) <= 25:
                    all_seqs.append(seq)
                    all_amp_scores.append(amp_score)
                    all_safety_scores.append(safety_score)

    logger.info(f"  Generated {len(all_seqs)} valid candidates from latent grids")

    if not all_seqs:
        logger.error("  No valid sequences generated!")
        return [], [], []

    # Sort into categories based on score percentiles
    amp_scores = np.array(all_amp_scores)
    safety_scores = np.array(all_safety_scores)

    # Determine thresholds from actual distribution
    amp_median = np.median(amp_scores)
    safety_median = np.median(safety_scores)
    logger.info(f"  AMP score range: [{amp_scores.min():.3f}, {amp_scores.max():.3f}], median={amp_median:.3f}")
    logger.info(f"  Safety score range: [{safety_scores.min():.3f}, {safety_scores.max():.3f}], median={safety_median:.3f}")

    # Classify: use relative thresholds
    # AMP-safe: high AMP + low safety risk
    # AMP-SLE: high AMP + high safety risk
    # Non-AMP: low AMP score
    indices = np.arange(len(all_seqs))
    np.random.seed(42)
    np.random.shuffle(indices)

    for i in indices:
        amp_s = amp_scores[i]
        safety_s = safety_scores[i]
        seq = all_seqs[i]

        if amp_s >= amp_median and safety_s < safety_median and len(amp_safe) < num_per_set:
            amp_safe.append(seq)
        elif amp_s >= amp_median and safety_s >= safety_median and len(amp_sle) < num_per_set:
            amp_sle.append(seq)
        elif amp_s < amp_median and len(non_amp) < num_per_set:
            non_amp.append(seq)

    # Fill any remaining from candidates (relax constraints fully)
    remaining = [all_seqs[i] for i in indices
                 if all_seqs[i] not in set(amp_safe + amp_sle + non_amp)]
    for seq in remaining:
        if len(amp_safe) < num_per_set:
            amp_safe.append(seq)
        elif len(amp_sle) < num_per_set:
            amp_sle.append(seq)
        elif len(non_amp) < num_per_set:
            non_amp.append(seq)
        else:
            break

    return amp_safe[:num_per_set], amp_sle[:num_per_set], non_amp[:num_per_set]


def save_peptides(results_dir, amp_safe, amp_sle, non_amp):
    """Save peptide sets with physicochemical features to CSV."""
    for name, peptides in [("amp_safe", amp_safe), ("amp_sle", amp_sle), ("non_amp", non_amp)]:
        path = results_dir / f"{name}_peptides.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "length", "net_charge", "mean_hydrophobicity",
                           "total_mass", "frac_cationic", "frac_hydrophobic",
                           "frac_aromatic", "amphipathicity", "boman_index"])
            for seq in peptides:
                feats = compute_physicochemical(seq)
                writer.writerow([seq, feats["length"], feats["net_charge"],
                               feats["mean_hydrophobicity"], feats["total_mass"],
                               feats["frac_cationic"], feats["frac_hydrophobic"],
                               feats["frac_aromatic"], feats["amphipathicity"],
                               feats["boman_index"]])

        logger.info(f"  Saved: {path} ({len(peptides)} peptides)")


def compute_summary(amp_safe, amp_sle, non_amp, results_dir):
    """Compute and save summary statistics."""
    summary = {}
    for name, peptides in [("amp_safe", amp_safe), ("amp_sle", amp_sle), ("non_amp", non_amp)]:
        feats = [compute_physicochemical(p) for p in peptides]

        if not feats:
            continue

        summary[name] = {
            "count": len(peptides),
            "unique": len(set(peptides)),
            "mean_length": round(np.mean([f["length"] for f in feats]), 1),
            "mean_charge": round(np.mean([f["net_charge"] for f in feats]), 2),
            "mean_hydrophobicity": round(np.mean([f["mean_hydrophobicity"] for f in feats]), 3),
            "mean_amphipathicity": round(np.mean([f["amphipathicity"] for f in feats]), 3),
            "mean_frac_cationic": round(np.mean([f["frac_cationic"] for f in feats]), 3),
            "mean_frac_hydrophobic": round(np.mean([f["frac_hydrophobic"] for f in feats]), 3),
            "mean_boman": round(np.mean([f["boman_index"] for f in feats]), 3),
            "examples": peptides[:5],
        }

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    hdr = f"{'Set':<12} {'N':>4} {'Uniq':>5} {'Len':>5} {'Charge':>7} {'Hydro':>7} {'Cat%':>6} {'HPhob%':>6} {'Amp':>6} {'Boman':>6}"
    logger.info(hdr)
    logger.info("-" * 80)

    for name, s in summary.items():
        logger.info(f"{name:<12} {s['count']:>4} {s['unique']:>5} {s['mean_length']:>5.1f} "
                   f"{s['mean_charge']:>7.2f} {s['mean_hydrophobicity']:>7.3f} "
                   f"{s['mean_frac_cationic']:>6.3f} {s['mean_frac_hydrophobic']:>6.3f} "
                   f"{s['mean_amphipathicity']:>6.3f} {s['mean_boman']:>6.3f}")

    logger.info("\nExamples:")

    for name, s in summary.items():
        logger.info(f"  {name}: {', '.join(s['examples'][:3])}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate peptides and analyze")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--poincare', type=str, default='checkpoints/poincare_trained.npz')
    parser.add_argument('--trees_dir', type=str, default='data/species_trees')
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--num_per_set', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from autoguard.models.autoguard_model import AutoGuardModel
    from autoguard.config import ModelConfig

    config = ModelConfig()
    model = AutoGuardModel(config, use_graph_encoder=False)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict']
        # Filter out keys with shape mismatches
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
        skipped = set(state_dict.keys()) - set(filtered.keys())

        if skipped:
            logger.warning(f"  Skipped {len(skipped)} incompatible keys: {list(skipped)[:5]}")
        missing, unexpected = model.load_state_dict(filtered, strict=False)

        if missing:
            logger.info(f"  Using defaults for {len(missing)} new parameters")

        logger.info(f"  Loaded from epoch {ckpt.get('epoch', '?')}")
    else:
        logger.warning(f"No checkpoint at {ckpt_path}. Using untrained model.")

    # Load phylo embeddings
    species_embeds = None
    poincare_path = Path(args.poincare)

    if poincare_path.exists():
        from autoguard.data.phylo_data import SpeciesEmbeddingLookup
        try:
            lookup = SpeciesEmbeddingLookup(
                str(poincare_path), args.trees_dir, embed_dim=64
            )
            species_embeds = lookup.get_all_species_embeddings()
            logger.info(f"  Phylo embeddings: {species_embeds.shape}")
        except Exception as e:
            logger.warning(f"  Could not load phylo embeddings: {e}")

    # Generate
    logger.info("\nGenerating 3×100 peptide sets...")
    amp_safe, amp_sle, non_amp = generate_peptides(
        model, species_embeds, args.device, num_per_set=args.num_per_set
    )
    logger.info(f"  Generated: AMP-safe={len(amp_safe)}, AMP-SLE={len(amp_sle)}, Non-AMP={len(non_amp)}")

    # Save
    logger.info("\nSaving peptides:")
    save_peptides(results_dir, amp_safe, amp_sle, non_amp)

    # Analysis
    logger.info("\nComputing summary statistics:")
    compute_summary(amp_safe, amp_sle, non_amp, results_dir)

    # Plots
    logger.info("\nCreating plots:")
    create_plots(results_dir, amp_safe, amp_sle, non_amp)

    logger.info(f"\nAll outputs saved to: {results_dir}/")
    logger.info("Run `python -m autoguard.scripts.esm_comparison --results_dir results/` for ESM-2 analysis")


if __name__ == "__main__":
    main()
