#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoGuard: Step-by-step training, generation, and analysis pipeline.

Trains the model stage by stage, generates 3×100 peptide sets,
computes physicochemical features, and creates comparison plots.

Usage:
    python -m autoguard.scripts.full_experiment --data_dir data/ --save_dir checkpoints/
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Physicochemical feature computation
# ============================================================================

# Kyte-Doolittle hydrophobicity
HYDRO = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'E': -3.5,
    'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
    'Y': -1.3, 'V': 4.2,
}

CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'E': -1, 'Q': 0, 'G': 0,
    'H': 0.5, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

MASS = {
    'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121, 'E': 147, 'Q': 146,
    'G': 75, 'H': 155, 'I': 131, 'L': 131, 'K': 146, 'M': 149, 'F': 165,
    'P': 115, 'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117,
}

AROMATIC = set("FWY")
HYDROPHOBIC = set("AILMFVWP")
CATIONIC = set("KRH")
ANIONIC = set("DE")


def compute_physicochemical(seq: str) -> dict:
    """Compute physicochemical features for a peptide sequence."""
    seq = seq.upper().strip()
    n = len(seq)
    if n == 0:
        return {}

    hydro_vals = [HYDRO.get(aa, 0) for aa in seq]
    charge_vals = [CHARGE.get(aa, 0) for aa in seq]

    net_charge = sum(charge_vals)
    mean_hydro = np.mean(hydro_vals)
    mean_mass = np.mean([MASS.get(aa, 100) for aa in seq])
    total_mass = sum(MASS.get(aa, 100) for aa in seq)
    frac_aromatic = sum(1 for aa in seq if aa in AROMATIC) / n
    frac_hydrophobic = sum(1 for aa in seq if aa in HYDROPHOBIC) / n
    frac_cationic = sum(1 for aa in seq if aa in CATIONIC) / n
    frac_anionic = sum(1 for aa in seq if aa in ANIONIC) / n

    # Amphipathicity: variance of hydrophobicity along helix faces
    # Approximate by looking at i vs i+2 pattern
    if n >= 4:
        even_hydro = [hydro_vals[i] for i in range(0, n, 2)]
        odd_hydro = [hydro_vals[i] for i in range(1, n, 2)]
        amphipathicity = abs(np.mean(even_hydro) - np.mean(odd_hydro))
    else:
        amphipathicity = 0.0

    # Boman index (potential for protein interaction)
    boman = -mean_hydro

    return {
        "length": n,
        "net_charge": net_charge,
        "mean_hydrophobicity": round(mean_hydro, 3),
        "total_mass": total_mass,
        "mean_mass": round(mean_mass, 1),
        "frac_aromatic": round(frac_aromatic, 3),
        "frac_hydrophobic": round(frac_hydrophobic, 3),
        "frac_cationic": round(frac_cationic, 3),
        "frac_anionic": round(frac_anionic, 3),
        "amphipathicity": round(amphipathicity, 3),
        "boman_index": round(boman, 3),
    }


# ============================================================================
# Plotting
# ============================================================================

def create_plots(results_dir: Path, amp_safe, amp_sle, non_amp):
    """Create comparison plots for the three peptide sets."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plots")
        return

    # Compute features for each set
    sets = {
        "AMP-safe (no SLE)": amp_safe,
        "AMP-SLE risk": amp_sle,
        "Non-AMP": non_amp,
    }

    features_by_set = {}
    for name, peptides in sets.items():
        feats = [compute_physicochemical(p) for p in peptides]
        features_by_set[name] = feats

    # --- Plot 1: Physicochemical comparison (box plots) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Physicochemical Properties: AMP-safe vs AMP-SLE vs Non-AMP", fontsize=14)

    properties = [
        ("net_charge", "Net Charge"),
        ("mean_hydrophobicity", "Mean Hydrophobicity"),
        ("frac_cationic", "Fraction Cationic"),
        ("frac_hydrophobic", "Fraction Hydrophobic"),
        ("amphipathicity", "Amphipathicity"),
        ("boman_index", "Boman Index"),
    ]

    colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # green, red, gray
    for idx, (prop, title) in enumerate(properties):
        ax = axes[idx // 3, idx % 3]
        data = []
        labels = []
        for name, feats in features_by_set.items():
            vals = [f[prop] for f in feats if prop in f]
            data.append(vals)
            labels.append(name)
        bp = ax.boxplot(data, labels=["Safe", "SLE", "Non-AMP"],
                       patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(results_dir / "physicochemical_comparison.png"), dpi=300)
    plt.close()
    logger.info(f"  Saved: {results_dir / 'physicochemical_comparison.png'}")

    # Length distribution
    fig, ax = plt.subplots(figsize=(8, 5))

    for (name, feats), color in zip(features_by_set.items(), colors):
        lengths = [f["length"] for f in feats if "length" in f]
        ax.hist(lengths, bins=range(5, 26), alpha=0.5, label=name, color=color)

    ax.set_xlabel("Peptide Length")
    ax.set_ylabel("Count")
    ax.set_title("Length Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(results_dir / "length_distribution.png"), dpi=150)
    plt.close()
    logger.info(f"  Saved: {results_dir / 'length_distribution.png'}")

    # Charge vs Hydrophobicity scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, feats), color in zip(features_by_set.items(), colors):
        charges = [f["net_charge"] for f in feats]
        hydros = [f["mean_hydrophobicity"] for f in feats]
        ax.scatter(charges, hydros, alpha=0.5, label=name, color=color, s=30)

    ax.set_xlabel("Net Charge")
    ax.set_ylabel("Mean Hydrophobicity (Kyte-Doolittle)")
    ax.set_title("Charge vs Hydrophobicity: AMP Design Space")
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    # Typical AMP region
    ax.annotate("Typical AMP zone\n(cationic + amphipathic)",
                xy=(4, -0.5), fontsize=9, color='#2ecc71',
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(results_dir / "charge_vs_hydrophobicity.png"), dpi=150)
    plt.close()
    logger.info(f"  Saved: {results_dir / 'charge_vs_hydrophobicity.png'}")

    # Amino acid composition
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Amino Acid Composition", fontsize=14)
    aa_order = "ACDEFGHIKLMNPQRSTVWY"

    for ax, (name, peptides), color in zip(axes, sets.items(), colors):
        counts = {aa: 0 for aa in aa_order}
        total = 0
        for pep in peptides:
            for aa in pep.upper():
                if aa in counts:
                    counts[aa] += 1
                    total += 1

        if total > 0:
            freqs = [counts[aa] / total * 100 for aa in aa_order]
        else:
            freqs = [0] * len(aa_order)

        ax.bar(list(aa_order), freqs, color=color, alpha=0.7)
        ax.set_title(name)
        ax.set_ylabel("Frequency (%)")
        ax.set_ylim(0, max(freqs) * 1.3 if freqs and max(freqs) > 0 else 10)

    plt.tight_layout()
    plt.savefig(str(results_dir / "amino_acid_composition.png"), dpi=150)
    plt.close()
    logger.info(f"  Saved: {results_dir / 'amino_acid_composition.png'}")

    # Radar chart of mean properties
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    radar_props = ["frac_cationic", "frac_hydrophobic", "frac_aromatic",
                   "amphipathicity", "frac_anionic"]
    radar_labels = ["Cationic", "Hydrophobic", "Aromatic", "Amphipathic", "Anionic"]
    angles = np.linspace(0, 2 * np.pi, len(radar_props), endpoint=False).tolist()
    angles += angles[:1]

    for (name, feats), color in zip(features_by_set.items(), colors):
        values = [np.mean([f[p] for f in feats]) for p in radar_props]
        values += values[:1]
        ax.plot(angles, values, 'o-', label=name, color=color, linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
    ax.set_title("Mean Property Radar", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(str(results_dir / "property_radar.png"), dpi=150)
    plt.close()
    logger.info(f"  Saved: {results_dir / 'property_radar.png'}")


# ============================================================================
# Main experiment pipeline
# ============================================================================

def run_experiment(args):
    """Run full AutoGuard experiment: train → generate → analyze."""
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    from autoguard.models.autoguard_model import AutoGuardModel
    from autoguard.config import ModelConfig, LossWeights
    from autoguard.training.trainer import AutoGuardTrainer
    from autoguard.data.datasets import AMPDataset, MimicryDataset, tokenize_sequence

    # ====================================================================
    # Stage 1: Poincaré Embeddings
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Stage 1/6: Poincaré Embedding Pre-training")
    logger.info("=" * 70)

    from autoguard.data.phylo_data import PhylogeneticDataProcessor, PoincareEmbeddingTrainer
    trees_dir = data_dir / "species_trees"
    poincare_path = save_dir / "poincare_trained.npz"

    if poincare_path.exists() and not args.retrain:
        logger.info(f"  [SKIP] Already exists: {poincare_path}")
    else:
        processor = PhylogeneticDataProcessor(
            trees_dir=str(trees_dir), num_perturbations=10,
            perturbation_scale=0.1, tree_filter=args.tree_filter,
        )
        data = processor.prepare_training_data()
        logger.info(f"  Tree pairs: {len(data['parents'])}, entities: {data['num_entities']}")

        trainer = PoincareEmbeddingTrainer(
            num_entities=data['num_entities'], embed_dim=64,
            learning_rate=0.01, num_negatives=10,
        )
        trainer.train(data, epochs=args.poincare_epochs, verbose=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.save(str(poincare_path))
        logger.info(f"  Saved: {poincare_path}")

    # ====================================================================
    # Stage 2: Mimicry Contrastive Pre-training
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Stage 2/6: Mimicry Contrastive Pre-training")
    logger.info("=" * 70)

    model_config = ModelConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    model = AutoGuardModel(model_config, use_graph_encoder=False)

    mimicry_path = save_dir / "mimicry_pretrained.pt"
    processed_dir = data_dir / "processed"

    if mimicry_path.exists() and not args.retrain:
        logger.info(f"  [SKIP] Already exists: {mimicry_path}")
        mimicry_state = torch.load(str(mimicry_path), map_location='cpu', weights_only=False)
        model.mimicry_detector.load_state_dict(mimicry_state)
    else:
        try:
            import esm as _esm_check  # noqa: F401
        except ImportError:
            logger.warning("  [SKIP] ESM-2 not installed (pip install fair-esm). "
                          "Mimicry detector will train with random initialization.")
            logger.warning("  To enable mimicry pre-training: pip install fair-esm")
        else:
            from autoguard.data.datasets import parse_fasta, filter_sequences
            from autoguard.training.contrastive import ContrastiveTrainer

            pos_seqs = [s for _, s in parse_fasta(str(processed_dir / "mimicry_positives.fasta"))]
            neg_seqs = [s for _, s in parse_fasta(str(processed_dir / "mimicry_negatives.fasta"))]

            # Load anchors
            anchor_seqs = []
            with open(processed_dir / "amp_train.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    anchor_seqs.append(row["sequence"])
            anchors = anchor_seqs[:500]

            # Augment positives (32 defensins + natural 500 AMPs)
            import random
            rng = random.Random(42)
            augmented_pos = list(pos_seqs)
            natural = [s for s in anchor_seqs[:2000] if len(s) >= 10]
            rng.shuffle(natural)
            augmented_pos.extend(natural[:468])
            logger.info(f"  Positives: {len(augmented_pos)}, Negatives: {min(len(neg_seqs), 500)}")

            dataset = MimicryDataset(anchors, augmented_pos, neg_seqs[:500])
            loader = DataLoader(dataset, batch_size=min(args.batch_size, 32), shuffle=True)

            ct = ContrastiveTrainer(
                mimicry_detector=model.mimicry_detector,
                learning_rate=5e-4, temperature=0.07, margin=0.5, device=device,
            )

            for epoch in range(args.mimicry_epochs):
                losses = ct.train_epoch(loader)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"  Epoch {epoch+1}/{args.mimicry_epochs}: loss={losses['contrastive']:.4f}")

            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.mimicry_detector.state_dict(), str(mimicry_path))
            logger.info(f"  Saved: {mimicry_path}")

    # ====================================================================
    # Load training data (shared by Stage 3 and 4)
    # ====================================================================
    from torch.utils.data import DataLoader

    train_seqs, train_labels, train_mics, train_species = [], [], [], []
    with open(processed_dir / "amp_train.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train_seqs.append(row["sequence"])
            train_labels.append(int(float(row.get("label", 1))))
            mic_val = row.get("mic", "")
            train_mics.append(float(mic_val) if mic_val else None)
            train_species.append(row.get("target_species", ""))

    val_seqs, val_labels, val_mics = [], [], []
    with open(processed_dir / "amp_val.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val_seqs.append(row["sequence"])
            val_labels.append(int(float(row.get("label", 1))))
            mic_val = row.get("mic", "")
            val_mics.append(float(mic_val) if mic_val else None)

    logger.info(f"  Data loaded: train={len(train_seqs)}, val={len(val_seqs)}")

    # ====================================================================
    # Stage 3: Encoder Warmup
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Stage 3/6: Encoder Warm-up (reconstruction only)")
    logger.info("=" * 70)

    warmup_path = save_dir / "warmup" / "best_model.pt"

    if warmup_path.exists() and not args.retrain:
        logger.info(f"  [SKIP] Already exists: {warmup_path}")
        ckpt = torch.load(str(warmup_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        train_ds = AMPDataset(train_seqs, train_labels, train_mics)
        val_ds = AMPDataset(val_seqs, val_labels, val_mics)
        logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        warmup_weights = LossWeights()
        warmup_weights.antimicrobial_activity = 0.0
        warmup_weights.mimicry_penalty = 0.0
        warmup_weights.safety_penalty = 0.0

        warmup_trainer = AutoGuardTrainer(model, model_config, warmup_weights, device=device)

        warmup_epochs = args.warmup_epochs
        best_loss = float('inf')
        warmup_save = save_dir / "warmup"
        warmup_save.mkdir(parents=True, exist_ok=True)

        for epoch in range(warmup_epochs):
            t_loss = warmup_trainer.train_epoch(train_loader, epoch)
            v_loss = warmup_trainer.validate(val_loader)
            vq_stats = model.vector_quantizer.get_usage_stats()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1}/{warmup_epochs}: "
                           f"train={t_loss['total']:.4f}, val={v_loss['total']:.4f}, "
                           f"VQ alive={vq_stats['alive_codes']}/{model.vector_quantizer.num_embeddings}")
            if v_loss['total'] < best_loss:
                best_loss = v_loss['total']
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch},
                          str(warmup_path))

        logger.info(f"  Best val loss: {best_loss:.4f}")

    # ====================================================================
    # Stage 4: Full Multi-task Training
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Stage 4/6: Full Multi-task Training")
    logger.info("=" * 70)

    full_path = save_dir / "best_model.pt"
    best_loss = float('inf')

    if full_path.exists() and not args.retrain:
        logger.info(f"  [SKIP] Already exists: {full_path}")
        ckpt = torch.load(str(full_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        from autoguard.data.phylo_data import SpeciesEmbeddingLookup

        train_ds = AMPDataset(train_seqs, train_labels, train_mics, train_species)
        val_ds = AMPDataset(val_seqs, val_labels, val_mics)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        # Load phylo embeddings
        species_lookup = SpeciesEmbeddingLookup(
            str(poincare_path), str(trees_dir), embed_dim=64,
            tree_filter=args.tree_filter,
        )
        species_embeds = species_lookup.get_all_species_embeddings()
        logger.info(f"  Phylo: {species_embeds.shape[0]} species, dim={species_embeds.shape[1]}")

        n_species_labeled = sum(1 for s in train_species if s)
        logger.info(f"  Species-labeled: {n_species_labeled}/{len(train_species)}")

        loss_weights = LossWeights()
        trainer = AutoGuardTrainer(
            model, model_config, loss_weights, device=device,
            species_embeddings=species_embeds,
        )

        for epoch in range(args.full_epochs):
            t_loss = trainer.train_epoch(train_loader, epoch)
            v_loss = trainer.validate(val_loader)
            vq_stats = model.vector_quantizer.get_usage_stats()
            perp = t_loss.get('perplexity', 0)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                # Log total + all components
                components = []
                for key in ['reconstruction', 'vq', 'kl', 'amp_classification',
                           'mic_regression', 'mimicry', 'safety']:
                    if key in t_loss:
                        components.append(f"{key[:5]}={t_loss[key]:.4f}")
                logger.info(f"  Epoch {epoch+1}/{args.full_epochs}: "
                           f"TOTAL train={t_loss['total']:.4f} val={v_loss['total']:.4f} | "
                           f"{' '.join(components)} | "
                           f"perp={perp:.1f} alive={vq_stats['alive_codes']}/{model.vector_quantizer.num_embeddings}")
            if v_loss['total'] < best_loss:
                best_loss = v_loss['total']
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch},
                          str(full_path))

        logger.info(f"  Full training best val loss: {best_loss:.4f}")

    # ====================================================================
    # Stage 5: Generate 3×100 Peptide Sets
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Stage 5/6: Generating Peptide Sets")
    logger.info("=" * 70)

    model = model.to(device)
    model.eval()

    # Load phylo embeddings for generation
    from autoguard.data.phylo_data import SpeciesEmbeddingLookup
    species_lookup = SpeciesEmbeddingLookup(
        str(poincare_path), str(trees_dir), embed_dim=64,
    )
    species_embeds = species_lookup.get_all_species_embeddings().to(device)

    amp_safe = []     # High AMP, low mimicry
    amp_sle = []      # High AMP, high mimicry risk
    non_amp = []      # Low AMP

    logger.info("  Generating candidates (sampling codebook)...")
    max_attempts = 10000
    attempt = 0

    with torch.no_grad():
        while (len(amp_safe) < 100 or len(amp_sle) < 100 or len(non_amp) < 100) and attempt < max_attempts:
            attempt += 1

            # Sample from codebook
            idx = torch.randint(0, 512, (1,))
            z = model.vector_quantizer.get_codebook_entry(idx)

            # Get AMP score
            amp_score = torch.sigmoid(model.amp_classifier(z)).item()

            # Get safety score
            safety_out = model.safety_module(z)
            safety_score = safety_out.get('safety_score',
                                          safety_out.get('toxicity', torch.tensor([0.5]))).mean().item()

            # Phylo conditioning
            phylo_input = species_embeds.unsqueeze(0)  # [1, num_species, phylo_dim]
            phylo_cond = model.phylo_conditioner(phylo_input)

            # Decode sequence
            logits, _ = model.decode(z, phylo_cond, temperature=0.5, sample=True)
            tokens = logits.squeeze(0).argmax(dim=-1)

            # Convert tokens to sequence
            AA_VOCAB = "_ACDEFGHIKLMNPQRSTVWY"
            seq = ""
            for t in tokens:
                t_val = t.item()
                if 0 < t_val < len(AA_VOCAB):
                    seq += AA_VOCAB[t_val]

            if len(seq) < 5:
                continue

            # Classify into sets
            if amp_score > 0.7 and safety_score < 0.3 and len(amp_safe) < 100:
                amp_safe.append(seq)
            elif amp_score > 0.7 and safety_score > 0.5 and len(amp_sle) < 100:
                amp_sle.append(seq)
            elif amp_score < 0.3 and len(non_amp) < 100:
                non_amp.append(seq)

    # If we didn't get enough from strict thresholds, relax
    if len(amp_safe) < 100 or len(amp_sle) < 100 or len(non_amp) < 100:
        logger.info("  Relaxing thresholds for remaining candidates...")
        attempt = 0
        while (len(amp_safe) < 100 or len(amp_sle) < 100 or len(non_amp) < 100) and attempt < max_attempts:
            attempt += 1
            idx = torch.randint(0, model.config.num_codebook_vectors, (1,))
            z = model.vector_quantizer.get_codebook_entry(idx)
            amp_score = torch.sigmoid(model.amp_classifier(z)).item()
            safety_out = model.safety_module(z)
            safety_score = safety_out.get('safety_score',
                                          safety_out.get('toxicity', torch.tensor([0.5]))).mean().item()

            phylo_cond = model.phylo_conditioner(species_embeds.unsqueeze(0))
            logits, _ = model.decode(z, phylo_cond, temperature=0.8, sample=True)
            tokens = torch.argmax(logits.squeeze(0), dim=-1)
            seq = "".join(AA_VOCAB[t.item()] for t in tokens if 0 < t.item() < len(AA_VOCAB))

            if len(seq) < 5:
                continue

            if amp_score >= 0.5 and len(amp_safe) < 100:
                amp_safe.append(seq)
            elif amp_score >= 0.5 and len(amp_sle) < 100:
                amp_sle.append(seq)
            elif amp_score < 0.5 and len(non_amp) < 100:
                non_amp.append(seq)

    logger.info(f"  Generated: AMP-safe={len(amp_safe)}, AMP-SLE={len(amp_sle)}, Non-AMP={len(non_amp)}")

    # Save peptide sets
    for name, peptides in [("amp_safe", amp_safe), ("amp_sle", amp_sle), ("non_amp", non_amp)]:
        with open(results_dir / f"{name}_peptides.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "length", "net_charge", "mean_hydrophobicity",
                           "frac_cationic", "frac_hydrophobic", "amphipathicity", "boman_index"])
            for seq in peptides:
                feats = compute_physicochemical(seq)
                writer.writerow([seq, feats["length"], feats["net_charge"],
                               feats["mean_hydrophobicity"], feats["frac_cationic"],
                               feats["frac_hydrophobic"], feats["amphipathicity"],
                               feats["boman_index"]])
        logger.info(f"  Saved: {results_dir / f'{name}_peptides.csv'}")

    # ====================================================================
    # STEP 6: Analysis & Plots
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6/6: Analysis & Plots")
    logger.info("=" * 70)

    create_plots(results_dir, amp_safe, amp_sle, non_amp)

    # Summary statistics
    summary = {}
    for name, peptides in [("amp_safe", amp_safe), ("amp_sle", amp_sle), ("non_amp", non_amp)]:
        feats = [compute_physicochemical(p) for p in peptides]
        summary[name] = {
            "count": len(peptides),
            "mean_length": round(np.mean([f["length"] for f in feats]), 1),
            "mean_charge": round(np.mean([f["net_charge"] for f in feats]), 2),
            "mean_hydrophobicity": round(np.mean([f["mean_hydrophobicity"] for f in feats]), 3),
            "mean_amphipathicity": round(np.mean([f["amphipathicity"] for f in feats]), 3),
            "mean_frac_cationic": round(np.mean([f["frac_cationic"] for f in feats]), 3),
            "mean_boman": round(np.mean([f["boman_index"] for f in feats]), 3),
            "unique_sequences": len(set(peptides)),
            "example_sequences": peptides[:5],
        }

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: {results_dir / 'summary.json'}")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    header = f"{'Set':<15} {'Count':>5} {'Len':>5} {'Charge':>7} {'Hydro':>7} {'Cationic':>8} {'Amphip':>7}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, stats in summary.items():
        logger.info(f"{name:<15} {stats['count']:>5} {stats['mean_length']:>5.1f} "
                   f"{stats['mean_charge']:>7.2f} {stats['mean_hydrophobicity']:>7.3f} "
                   f"{stats['mean_frac_cationic']:>8.3f} {stats['mean_amphipathicity']:>7.3f}")

    logger.info(f"\nAll results saved to: {results_dir}/")
    logger.info("Files:")
    logger.info("  amp_safe_peptides.csv     — 100 AMPs safe from SLE")
    logger.info("  amp_sle_peptides.csv      — 100 AMPs with SLE risk")
    logger.info("  non_amp_peptides.csv      — 100 non-AMP sequences")
    logger.info("  physicochemical_comparison.png")
    logger.info("  length_distribution.png")
    logger.info("  charge_vs_hydrophobicity.png")
    logger.info("  amino_acid_composition.png")
    logger.info("  property_radar.png")
    logger.info("  summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGuard full experiment pipeline")
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--save_dir', type=str, default='checkpoints/')
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--poincare_epochs', type=int, default=100)
    parser.add_argument('--mimicry_epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('--full_epochs', type=int, default=200)
    parser.add_argument('--tree_filter', type=str, default='')
    parser.add_argument('--retrain', action='store_true', help='Force retrain even if checkpoints exist')
    args = parser.parse_args()

    run_experiment(args)
