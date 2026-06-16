#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESM-2 comparison: embed generated peptides with ESM-2 and compare
against physicochemical features.

Requires: pip install fair-esm
Usage: python -m autoguard.scripts.esm_comparison --results_dir results/
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_peptides(results_dir: Path):
    """Load generated peptide sets from CSV files."""
    sets = {}
    for name in ["amp_safe", "amp_sle", "non_amp"]:
        path = results_dir / f"{name}_peptides.csv"

        if not path.exists():
            logger.warning(f"  Missing: {path}")
            continue

        seqs = []

        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seqs.append(row["sequence"])

        sets[name] = seqs
        logger.info(f"  {name}: {len(seqs)} peptides")

    return sets


def compute_esm_embeddings(sequences, batch_size=8):
    """Compute ESM-2 embeddings for a list of sequences."""
    try:
        import torch
        import esm
    except ImportError:
        logger.error("ESM not installed. Run: pip install fair-esm")
        return None

    logger.info(f"  Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[33])
            embeddings = results["representations"][33]  # [batch, seq_len, 1280]
            # Mean pool over sequence length (exclude BOS/EOS)
            mask = tokens != alphabet.padding_idx
            embeddings = (embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())

        if (i + batch_size) % 50 == 0:
            logger.info(f"    Embedded {min(i+batch_size, len(sequences))}/{len(sequences)}")

    return np.concatenate(all_embeddings, axis=0)


def create_esm_plots(results_dir: Path, peptide_sets, embeddings_dict):
    """Create ESM-2 comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as e:
        logger.warning(f"Missing dependency for plots: {e}")
        return

    colors = {'amp_safe': '#2ecc71', 'amp_sle': '#e74c3c', 'non_amp': '#95a5a6'}
    labels = {'amp_safe': 'AMP-safe', 'amp_sle': 'AMP-SLE', 'non_amp': 'Non-AMP'}

    # PCA of ESM-2 embeddings
    all_emb = []
    all_labels = []
    for name in ["amp_safe", "amp_sle", "non_amp"]:
        if name in embeddings_dict:
            all_emb.append(embeddings_dict[name])
            all_labels.extend([name] * len(embeddings_dict[name]))
    all_emb = np.concatenate(all_emb, axis=0)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(all_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("ESM-2 Embedding Space: AMP-safe vs AMP-SLE vs Non-AMP", fontsize=14)

    # PCA
    ax = axes[0]
    offset = 0
    for name in ["amp_safe", "amp_sle", "non_amp"]:
        if name in embeddings_dict:
            n = len(embeddings_dict[name])
            ax.scatter(pca_coords[offset:offset+n, 0], pca_coords[offset:offset+n, 1],
                      c=colors[name], label=labels[name], alpha=0.6, s=20)
            offset += n
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of ESM-2 Embeddings")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # t-SNE
    ax = axes[1]
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_emb)-1))
    tsne_coords = tsne.fit_transform(all_emb)
    offset = 0
    for name in ["amp_safe", "amp_sle", "non_amp"]:
        if name in embeddings_dict:
            n = len(embeddings_dict[name])
            ax.scatter(tsne_coords[offset:offset+n, 0], tsne_coords[offset:offset+n, 1],
                      c=colors[name], label=labels[name], alpha=0.6, s=20)
            offset += n
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of ESM-2 Embeddings")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(results_dir / "esm2_embedding_space.png"), dpi=150)
    plt.close()
    logger.info(f"  Saved: {results_dir / 'esm2_embedding_space.png'}")

    # Cosine similarity heatmap
    from numpy.linalg import norm
    fig, ax = plt.subplots(figsize=(8, 6))

    set_names = [n for n in ["amp_safe", "amp_sle", "non_amp"] if n in embeddings_dict]
    centroids = {}
    for name in set_names:
        centroids[name] = embeddings_dict[name].mean(axis=0)

    sim_matrix = np.zeros((len(set_names), len(set_names)))
    for i, n1 in enumerate(set_names):
        for j, n2 in enumerate(set_names):
            c1, c2 = centroids[n1], centroids[n2]
            sim_matrix[i, j] = np.dot(c1, c2) / (norm(c1) * norm(c2))

    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(set_names)))
    ax.set_yticks(range(len(set_names)))
    ax.set_xticklabels([labels[n] for n in set_names], rotation=45, ha='right')
    ax.set_yticklabels([labels[n] for n in set_names])

    for i in range(len(set_names)):
        for j in range(len(set_names)):
            ax.text(j, i, f"{sim_matrix[i,j]:.3f}", ha='center', va='center', fontsize=12)

    plt.colorbar(im, label="Cosine Similarity")
    ax.set_title("ESM-2 Centroid Cosine Similarity")
    plt.tight_layout()
    plt.savefig(str(results_dir / "esm2_cosine_similarity.png"), dpi=150)
    plt.close()
    logger.info(f"  Saved: {results_dir / 'esm2_cosine_similarity.png'}")

    # Intra-set diversity
    fig, ax = plt.subplots(figsize=(8, 5))
    diversities = {}

    for name in set_names:
        emb = embeddings_dict[name]
        # Pairwise cosine similarity (sample for speed)
        n = min(50, len(emb))
        idx = np.random.choice(len(emb), n, replace=False)
        subset = emb[idx]
        norms = norm(subset, axis=1, keepdims=True)
        cos_sim = (subset @ subset.T) / (norms @ norms.T + 1e-8)
        # Upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(n, k=1)
        sims = cos_sim[triu_idx]
        diversities[name] = sims

    positions = range(len(set_names)) # Not use anymore

    bp = ax.boxplot([diversities[n] for n in set_names],
                   labels=[labels[n] for n in set_names],
                   patch_artist=True)

    for patch, name in zip(bp['boxes'], set_names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)

    ax.set_ylabel("Pairwise Cosine Similarity")
    ax.set_title("Intra-set Diversity (ESM-2 Space)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(results_dir / "esm2_diversity.png"), dpi=150)
    plt.close()
    logger.info(f"  Saved: {results_dir / 'esm2_diversity.png'}")


def main():
    parser = argparse.ArgumentParser(description="ESM-2 embedding comparison")
    parser.add_argument('--results_dir', type=str, default='results/')
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        logger.error("Run full_experiment.py first to generate peptides.")
        sys.exit(1)

    logger.info("Loading generated peptides...")
    peptide_sets = load_peptides(results_dir)

    if not peptide_sets:
        logger.error("No peptide files found. Run full_experiment.py first.")
        sys.exit(1)

    logger.info("\nComputing ESM-2 embeddings.")
    embeddings_dict = {}

    for name, seqs in peptide_sets.items():
        logger.info(f"  {name}: embedding {len(seqs)} peptides.")
        emb = compute_esm_embeddings(seqs)
        if emb is not None:
            embeddings_dict[name] = emb
            np.save(str(results_dir / f"{name}_esm2_embeddings.npy"), emb)

    if embeddings_dict:
        logger.info("\nCreating ESM-2 plots.")
        create_esm_plots(results_dir, peptide_sets, embeddings_dict)
    else:
        logger.warning("Could not compute ESM-2 embeddings.")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
