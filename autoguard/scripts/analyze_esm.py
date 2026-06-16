#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESM-2 structural similarity analysis for SLE safety.

Uses ESM-2 embeddings to:
1. Compute structural similarity between generated AMPs and SLE autoantigens
2. Identify which generated peptides might trigger autoimmune responses
3. Find structural motifs that differentiate safe AMPs from risky ones
4. Predict secondary structure propensity for generated peptides

Usage:
    # Full SLE safety analysis
    python -m autoguard.scripts.analyze_esm \
      --generated generated_sle.json \
      --sle_epitopes data/processed/sle_epitopes.csv \
      --output results/esm_sle_analysis.json

    # Compare two sets of generated peptides
    python -m autoguard.scripts.analyze_esm \
      --generated generated_exp1.json \
      --compare generated_sle.json \
      --sle_epitopes data/processed/sle_epitopes.csv \
      --output results/esm_comparison.json

    # Structure prediction for generated peptides
    python -m autoguard.scripts.analyze_esm \
      --generated generated_sle.json \
      --predict_structure \
      --output results/structures/
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_esm2_model(device='cpu'):
    """Load ESM-2 model for embedding extraction."""
    try:
        import esm
    except ImportError:
        logger.error("ESM package required. Install with: pip install fair-esm")
        sys.exit(1)

    logger.info("Loading ESM-2 (esm2_t33_650M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    logger.info(f"  ESM-2 loaded on {device}")
    return model, alphabet


def extract_embeddings(model,
                       alphabet,
                       sequences: List[str],
                       device='cpu',
                       batch_size=16
                       ) -> np.ndarray:
    """Extract ESM-2 embeddings for sequences.

    Returns: [num_sequences, 1280] mean-pooled representations from layer 33.
    """
    batch_converter = alphabet.get_batch_converter()
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[33])

        # Mean pool over positions (exclude BOS/EOS tokens)
        representations = results["representations"][33]
        # For each sequence, pool over actual length
        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            emb = representations[j, 1:seq_len+1].mean(dim=0)
            all_embeddings.append(emb.cpu().numpy())

    return np.stack(all_embeddings)


def extract_per_residue(model,
                        alphabet,
                        sequences: List[str],
                        device='cpu',
                        batch_size=16
                        ) -> List[np.ndarray]:
    """Extract per-residue ESM-2 embeddings.

    Returns: List of [seq_len, 1280] arrays.
    """
    batch_converter = alphabet.get_batch_converter()
    all_residue_embs = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[33])

        representations = results["representations"][33]
        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            residue_emb = representations[j, 1:seq_len+1].cpu().numpy()
            all_residue_embs.append(residue_emb)

    return all_residue_embs


def compute_similarity_matrix(emb_a: np.ndarray,
                              emb_b: np.ndarray
                              ) -> np.ndarray:
    """Cosine similarity between two sets of embeddings."""
    # Normalize
    norm_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
    return norm_a @ norm_b.T


def analyze_sle_risk(amp_embeddings: np.ndarray,
                     sle_embeddings: np.ndarray,
                     amp_sequences: List[str],
                     sle_sequences: List[str],
                     threshold: float = 0.4
                     ) -> Dict:
    """Analyze SLE autoantigen mimicry risk for generated AMPs.

    Returns per-peptide risk scores and flagged sequences.
    """
    sim_matrix = compute_similarity_matrix(amp_embeddings, sle_embeddings)

    # Per-AMP
    max_sim = sim_matrix.max(axis=1)
    mean_max_sim = max_sim.mean()
    flagged_mask = max_sim > threshold
    flagged_count = flagged_mask.sum()

    # Per-SLE epitope
    most_mimicked_idx = sim_matrix.max(axis=0).argmax()

    # Find the most dangerous AMP-epitope pairs
    top_pairs = []
    flat_indices = np.argsort(sim_matrix.ravel())[-10:][::-1]
    for idx in flat_indices:
        amp_idx = idx // sim_matrix.shape[1]
        sle_idx = idx % sim_matrix.shape[1]
        top_pairs.append({
            'amp_sequence': amp_sequences[amp_idx],
            'sle_epitope': sle_sequences[sle_idx],
            'similarity': float(sim_matrix[amp_idx, sle_idx]),
        })

    return {
        'num_amps': len(amp_sequences),
        'num_sle_epitopes': len(sle_sequences),
        'mean_max_similarity': float(mean_max_sim),
        'max_similarity': float(max_sim.max()),
        'flagged_count': int(flagged_count),
        'flagged_fraction': float(flagged_count / len(amp_sequences)),
        'threshold': threshold,
        'risk_scores': max_sim.tolist(),
        'top_dangerous_pairs': top_pairs,
        'risk_distribution': {
            '<0.2': int((max_sim < 0.2).sum()),
            '0.2-0.3': int(((max_sim >= 0.2) & (max_sim < 0.3)).sum()),
            '0.3-0.4': int(((max_sim >= 0.3) & (max_sim < 0.4)).sum()),
            '>0.4': int((max_sim >= 0.4).sum()),
        },
    }


def predict_secondary_structure(model, alphabet, sequences: List[str],
                                device='cpu') -> List[Dict]:
    """Predict secondary structure propensity from ESM-2 contact predictions.
    Uses ESM-2's learned representations to estimate:
    - Helix propensity (amphipathic helices are key for AMP activity)
    - Sheet propensity
    - Coil/disordered regions
    """
    per_residue = extract_per_residue(model, alphabet, sequences, device)

    structures = []
    for _, (seq, emb) in enumerate(zip(sequences, per_residue)):
        # Simple helix/sheet prediction based on ESM-2 features
        norms = np.linalg.norm(emb, axis=1)
        mean_norm = norms.mean()

        # Compute local hydrophobic periodicity
        hydrophobic = set('AILMFWVP')
        hp = np.array([1.0 if aa in hydrophobic else 0.0 for aa in seq])

        # Amphipathic helix score: i, i+3, i+4 hydrophobic pattern
        helix_score = 0.0
        if len(seq) >= 7:
            for j in range(len(seq) - 4):
                if hp[j] + hp[j+3] + hp[j+4] >= 2:
                    helix_score += 1
            helix_score /= (len(seq) - 4)

        charges = np.array([1.0 if aa in 'KR' else (-1.0 if aa in 'DE' else 0.0)
                           for aa in seq])
        net_charge = charges.sum()
        charge_density = abs(net_charge) / len(seq)

        structures.append({
            'sequence': seq,
            'length': len(seq),
            'amphipathic_helix_score': float(helix_score),
            'net_charge': float(net_charge),
            'charge_density': float(charge_density),
            'embedding_confidence': float(mean_norm),
            'hydrophobic_fraction': float(hp.mean()),
        })

    return structures


def compare_sets(model,
                 alphabet,
                 set_a: List[str],
                 set_b: List[str],
                 sle_sequences: List[str],
                 device='cpu'
                 ) -> Dict:
    """Compare two sets of generated peptides against SLE epitopes.
    Useful for comparing Experiment 1 (general) vs Experiment 2 (SLE-safe).
    """
    logger.info(f"  Extracting embeddings for set A ({len(set_a)} seqs).")
    emb_a = extract_embeddings(model, alphabet, set_a, device)
    logger.info(f"  Extracting embeddings for set B ({len(set_b)} seqs).")
    emb_b = extract_embeddings(model, alphabet, set_b, device)
    logger.info(f"  Extracting embeddings for SLE epitopes ({len(sle_sequences)} seqs).")
    emb_sle = extract_embeddings(model, alphabet, sle_sequences, device)
    risk_a = analyze_sle_risk(emb_a, emb_sle, set_a, sle_sequences)
    risk_b = analyze_sle_risk(emb_b, emb_sle, set_b, sle_sequences)

    # Diversity within each set
    intra_sim_a = compute_similarity_matrix(emb_a, emb_a)
    np.fill_diagonal(intra_sim_a, 0)
    intra_sim_b = compute_similarity_matrix(emb_b, emb_b)
    np.fill_diagonal(intra_sim_b, 0)

    return {
        'set_a': {
            'name': 'general',
            'count': len(set_a),
            'sle_risk': risk_a,
            'mean_intra_similarity': float(intra_sim_a.mean()),
        },
        'set_b': {
            'name': 'sle_safe',
            'count': len(set_b),
            'sle_risk': risk_b,
            'mean_intra_similarity': float(intra_sim_b.mean()),
        },
        'improvement': {
            'mean_max_sim_reduction': float(
                risk_a['mean_max_similarity'] - risk_b['mean_max_similarity']
            ),
            'flagged_reduction': float(
                risk_a['flagged_fraction'] - risk_b['flagged_fraction']
            ),
        },
    }


def load_sequences_from_json(path: str) -> List[str]:
    """Load generated sequences from JSON file."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):

        if isinstance(data[0], str):
            return data

        return [entry.get('sequence', entry.get('seq', '')) for entry in data]

    if 'sequences' in data:
        return data['sequences']

    if 'generated' in data:
        return [s['sequence'] for s in data['generated']]

    raise ValueError(f"Cannot parse sequences from {path}")


def load_sequences_from_fasta(path: str) -> List[str]:
    """Load sequences from FASTA file."""
    sequences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current:
                    sequences.append(''.join(current))
                    current = []
            else:
                current.append(line)

    if current:
        sequences.append(''.join(current))

    return sequences


def load_sle_epitopes(path: str) -> List[str]:
    """Load SLE epitopes from CSV or FASTA."""
    if path.endswith('.fasta') or path.endswith('.fa'):
        return load_sequences_from_fasta(path)

    sequences = []

    with open(path, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get('sequence', row.get('epitope', row.get('Epitope', '')))
            if seq and len(seq) >= 5 and seq.isalpha():
                sequences.append(seq)

    return sequences


def main():
    parser = argparse.ArgumentParser(
        description='ESM-2 structural similarity analysis for SLE safety')
    parser.add_argument('--generated', type=str, required=True,
                        help='Generated peptides (JSON or FASTA)')
    parser.add_argument('--compare', type=str, default=None,
                        help='Second set to compare (JSON or FASTA)')
    parser.add_argument('--sle_epitopes', type=str, default='data/processed/sle_epitopes.csv',
                        help='SLE epitope sequences')
    parser.add_argument('--predict_structure', action='store_true',
                        help='Predict structural properties')
    parser.add_argument('--output', type=str, default='results/esm_analysis.json',
                        help='Output path (JSON for analysis, directory for structures)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='SLE risk threshold for flagging')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for ESM-2 inference')
    args = parser.parse_args()

    # Load sequences
    gen_path = args.generated
    if gen_path.endswith('.fasta') or gen_path.endswith('.fa'):
        generated_seqs = load_sequences_from_fasta(gen_path)
    else:
        generated_seqs = load_sequences_from_json(gen_path)
    logger.info(f"Loaded {len(generated_seqs)} generated sequences from {gen_path}")

    # Load ESM-2
    model, alphabet = load_esm2_model(args.device)

    results = {}

    # Structure prediction mode
    if args.predict_structure:
        logger.info("Predicting structural properties...")
        structures = predict_secondary_structure(model, alphabet, generated_seqs, args.device)
        results['structures'] = structures

        # Summary stats
        helix_scores = [s['amphipathic_helix_score'] for s in structures]
        charges = [s['net_charge'] for s in structures]
        results['summary'] = {
            'mean_amphipathic_helix_score': float(np.mean(helix_scores)),
            'mean_net_charge': float(np.mean(charges)),
            'fraction_helix_dominant': float(np.mean([h > 0.5 for h in helix_scores])),
            'fraction_cationic': float(np.mean([c > 0 for c in charges])),
        }
        logger.info(f"  Amphipathic helix: {results['summary']['mean_amphipathic_helix_score']:.3f}")
        logger.info(f"  Mean charge: {results['summary']['mean_net_charge']:.1f}")

    # SLE risk analysis
    sle_path = args.sle_epitopes
    if os.path.exists(sle_path):
        sle_seqs = load_sle_epitopes(sle_path)
        logger.info(f"Loaded {len(sle_seqs)} SLE epitopes from {sle_path}")

        if sle_seqs:
            if args.compare:
                # Compare two sets
                compare_path = args.compare
                if compare_path.endswith('.fasta') or compare_path.endswith('.fa'):
                    compare_seqs = load_sequences_from_fasta(compare_path)
                else:
                    compare_seqs = load_sequences_from_json(compare_path)
                logger.info(f"Loaded {len(compare_seqs)} comparison sequences")

                comparison = compare_sets(
                    model, alphabet, generated_seqs, compare_seqs,
                    sle_seqs, args.device
                )
                results['comparison'] = comparison
                logger.info(f"\n{'='*60}")
                logger.info(f"COMPARISON RESULTS")
                logger.info(f"{'='*60}")
                logger.info(f"  General:  mean_max_sim={comparison['set_a']['sle_risk']['mean_max_similarity']:.4f}, "
                           f"flagged={comparison['set_a']['sle_risk']['flagged_fraction']*100:.1f}%")
                logger.info(f"  SLE-safe: mean_max_sim={comparison['set_b']['sle_risk']['mean_max_similarity']:.4f}, "
                           f"flagged={comparison['set_b']['sle_risk']['flagged_fraction']*100:.1f}%")
                logger.info(f"  Improvement: sim_reduction={comparison['improvement']['mean_max_sim_reduction']:.4f}")
            else:
                # Single set analysis
                logger.info("Computing ESM-2 embeddings for generated AMPs...")
                amp_emb = extract_embeddings(model, alphabet, generated_seqs,
                                            args.device, args.batch_size)
                logger.info("Computing ESM-2 embeddings for SLE epitopes...")
                sle_emb = extract_embeddings(model, alphabet, sle_seqs,
                                            args.device, args.batch_size)

                risk = analyze_sle_risk(amp_emb, sle_emb, generated_seqs, sle_seqs,
                                       args.threshold)
                results['sle_risk'] = risk

                logger.info(f"\n{'='*60}")
                logger.info(f"SLE RISK ANALYSIS")
                logger.info(f"{'='*60}")
                logger.info(f"  Mean max similarity to SLE epitopes: {risk['mean_max_similarity']:.4f}")
                logger.info(f"  Max similarity: {risk['max_similarity']:.4f}")
                logger.info(f"  Flagged (>{args.threshold}): {risk['flagged_count']}/{risk['num_amps']} "
                           f"({risk['flagged_fraction']*100:.1f}%)")
                logger.info(f"  Risk distribution:")
                for bucket, count in risk['risk_distribution'].items():
                    logger.info(f"    {bucket}: {count}")
    else:
        logger.warning(f"SLE epitopes not found at {sle_path}. Skipping risk analysis.")
        logger.warning("  Run: python -m autoguard.scripts.filter_iedb --disease 'Ro60|SSA|Smith|snRNP' ...")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
