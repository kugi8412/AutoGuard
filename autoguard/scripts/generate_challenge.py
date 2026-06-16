#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate AMP-Challenge-2027 submission libraries for all five categories.

The AMP Challenge 2027 (see "amp-challenge-2027-main/README.md") defines five
category entry points, each expecting a FASTA library of designed peptides:

    1. generate_broad_spectrum  — active across the whole strain panel
    2. generate_gram_pos        — Gram-positive activity
    3. generate_gram_neg        — Gram-negative activity
    4. generate_mdr             — multi-drug-resistant (ESKAPE) activity
    5. generate_therapeutic     — optimal selectivity (safety window)

This script samples a large candidate pool from a trained AutoGuard checkpoint,
enforces the challenge sequence rules (canonical 20 AAs, length 8..50, unique,
no exact match to the antibacterial reference set), then selects and ranks one
library per category using AutoGuard's predicted potency/charge/safety signals.

Heuristic note
--------------
AutoGuard's bundled phylogeny is the curated human-microbiome / autoimmune
``timetree.nwk``, not the challenge's 20-strain ESKAPE panel, so we do NOT claim
strain-specific MIC conditioning. Instead the five libraries differ by a
documented, biologically-motivated *post-hoc* ranking over model predictions:

    broad_spectrum : highest predicted AMP probability (potency proxy)
    gram_pos       : potency weighted toward amphipathic/hydrophobic peptides
                     (thick peptidoglycan target)
    gram_neg       : potency weighted toward high net positive charge
                     (LPS outer-membrane crossing)
    mdr            : strict potency — lowest predicted MIC
    therapeutic    : best predicted safety window (HC50/MIC), potency-gated and
                     non-hemolytic

Usage
-----
    # all five categories, small smoke run
    python -m autoguard.scripts.generate_challenge \
        --checkpoint checkpoints/exp1/autoguard/best_model.pt \
        --out_dir results/challenge --library_size 100

    # one category, full challenge-size library
    python -m autoguard.scripts.generate_challenge \
        --checkpoint checkpoints/exp1/autoguard/best_model.pt \
        --category generate_gram_neg --library_size 50000 --out_dir results/challenge
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autoguard.config import ModelConfig
from autoguard.models.autoguard_model import AutoGuardModel
from autoguard.data.datasets import detokenize_sequence, parse_fasta
from autoguard.utils.amino_acids import compute_peptide_features
from autoguard.evaluation.metrics import (
    CHALLENGE_MIN_LEN,
    CHALLENGE_MAX_LEN,
    POTENCY_THRESHOLD_UM,
    amp_score_to_predicted_mic,
    hemolysis_to_predicted_hc50,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

CATEGORIES = (
    "generate_broad_spectrum",
    "generate_gram_pos",
    "generate_gram_neg",
    "generate_mdr",
    "generate_therapeutic",
)

TOP_SIZE = 100


@torch.no_grad()
def _sample_batch(model: AutoGuardModel, batch: int, temperature: float,
                  device: torch.device, category=None) -> Tuple[List[str], List[float], List[float], List[float]]:
    """Sample one batch of candidates; return (seqs, amp_scores, hemolysis, selectivity).
    "category" (a challenge category name / index) steers the generalized
    safety module so the conditioning token — and thus the decoder — is biased
    toward that category. ``None`` keeps the original category-agnostic sampling.
    """
    cfg = model.config
    indices = torch.randint(0, cfg.num_codebook_vectors, (batch,), device=device)
    z = model.vector_quantizer.get_codebook_entry(indices)  # [B, latent]

    safety = model.safety_module(z, category=category)
    safety_scores = torch.cat([
        safety["toxicity"], safety["hemolysis"],
        safety["immunogenicity"], safety["selectivity"],
    ], dim=-1)  # [B, 4]

    conditions = model._assemble_conditions(
        batch, device, safety_scores=model.safety_to_cond(safety_scores),
    )
    seq_features = z.unsqueeze(1).expand(-1, cfg.max_seq_len, -1)
    fused, _ = model.fusion(seq_features, conditions)
    fused_pooled = fused.mean(dim=1)

    tokens = model.decoder(
        z, condition=fused_pooled, temperature=temperature,
        return_logits=False, sample=True,
    ).view(batch, cfg.max_seq_len)

    amp_score = torch.sigmoid(model.amp_classifier(z)).view(-1)  # [B]

    seqs = [detokenize_sequence(tokens[i].cpu()) for i in range(batch)]
    return (
        seqs,
        amp_score.cpu().tolist(),
        safety["hemolysis"].view(-1).cpu().tolist(),
        safety["selectivity"].view(-1).cpu().tolist(),
    )


def _is_challenge_valid(seq: str) -> bool:
    return (
        CHALLENGE_MIN_LEN <= len(seq) <= CHALLENGE_MAX_LEN
        and set(seq) <= CANONICAL_AA
    )


def generate_candidate_pool(model: AutoGuardModel, target_unique: int,
                            temperature: float, device: torch.device,
                            reference: set, batch: int = 256,
                            max_batches: int = 100000, category=None) -> List[Dict]:
    """Sample until "target_unique" valid, novel, unique candidates are collected.
    When "category" is given the generalized safety module conditions the pool
    toward that AMP-Challenge category; None samples a shared, category-
    agnostic pool (reused across categories for efficiency).
    """
    pool: Dict[str, Dict] = {}
    n_batches = 0
    while len(pool) < target_unique and n_batches < max_batches:
        seqs, amps, hemos, sels = _sample_batch(model, batch, temperature, device, category)
        for seq, amp, hemo, sel in zip(seqs, amps, hemos, sels):

            if not _is_challenge_valid(seq) or seq in pool or seq in reference:
                continue

            feats = compute_peptide_features(seq)
            pred_mic = amp_score_to_predicted_mic(amp)
            pred_hc50 = hemolysis_to_predicted_hc50(hemo)
            pool[seq] = {
                "sequence": seq,
                "amp_score": amp,
                "hemolysis": hemo,
                "selectivity": sel,
                "pred_mic": pred_mic,
                "pred_hc50": pred_hc50,
                "safety_window": pred_hc50 / max(pred_mic, 1e-6),
                "net_charge": feats.get("net_charge", 0.0),
                "fraction_positive": feats.get("fraction_positive", 0.0),
                "mean_hydrophobicity": feats.get("mean_hydrophobicity", 0.0),
                "amphipathicity": feats.get("amphipathicity", 0.0),
                "length": len(seq),
            }
        n_batches += 1

        if n_batches % 20 == 0:
            logger.info(f"  pool: {len(pool)}/{target_unique} unique candidates "
                        f"({n_batches} batches)")

    if len(pool) < target_unique:
        logger.warning(f"  Only collected {len(pool)}/{target_unique} unique candidates "
                       f"after {n_batches} batches (model diversity limited).")

    return list(pool.values())


# Higher = better
def _rank_key(category: str):
    if category == "generate_broad_spectrum":
        return lambda c: c["amp_score"]

    if category == "generate_gram_pos":
        # Thick peptidoglycan -> amphipathic / hydrophobic membrane activity.
        return lambda c: c["amp_score"] * (1.0 + c["amphipathicity"]
                                           + max(c["mean_hydrophobicity"], 0.0))
    if category == "generate_gram_neg":
        # LPS outer membrane -> reward net positive charge (cationicity).
        return lambda c: c["amp_score"] * (1.0 + c["fraction_positive"]
                                           + max(c["net_charge"], 0.0) / 10.0)
    if category == "generate_mdr":
        # Strict potency: lowest predicted MIC (== highest amp_score).
        return lambda c: -c["pred_mic"]

    if category == "generate_therapeutic":
        # Optimal selectivity: best safety window, potency-gated + non-hemolytic.
        def key(c):
            gated = c["pred_mic"] <= POTENCY_THRESHOLD_UM and c["hemolysis"] < 0.5
            return (1 if gated else 0, c["safety_window"])

        return key

    raise ValueError(f"Unknown category: {category}")


def select_library(pool: List[Dict], category: str, size: int) -> List[Dict]:
    """Rank the candidate pool for a category and take the top ``size``."""
    ranked = sorted(pool, key=_rank_key(category), reverse=True)
    return ranked[:size]


def write_fasta(records: List[Dict], path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            f.write(f">{tag}_{i:05d} amp={r['amp_score']:.3f} "
                    f"pred_mic={r['pred_mic']:.1f}uM sw={r['safety_window']:.2f}\n")
            f.write(f"{r['sequence']}\n")


def _load_reference(path: Path) -> set:
    if not path.exists():
        logger.warning(f"  Reference FASTA not found: {path} (overlap filter disabled)")
        return set()
    return {seq.upper() for _, seq in parse_fasta(str(path))}


def run_category(model, category, pool, out_dir, library_size) -> None:
    library = select_library(pool, category, library_size)
    tag = category.replace("generate_", "")
    write_fasta(library, out_dir / f"{tag}_library.fasta", tag)
    write_fasta(library[:TOP_SIZE], out_dir / f"{tag}_top100.fasta", f"{tag}_top")
    logger.info(f"[{category}] wrote {len(library)} sequences "
                f"(top {min(TOP_SIZE, len(library))}) to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate AMP-Challenge-2027 libraries.")
    parser.add_argument("--checkpoint", required=True, help="Trained AutoGuard checkpoint.")
    parser.add_argument("--out_dir", default="results/challenge", help="Output directory.")
    parser.add_argument("--category", choices=CATEGORIES + ("all",), default="all",
                        help="Which challenge category to generate (default: all five).")
    parser.add_argument("--library_size", type=int, default=100,
                        help="Sequences per category library (challenge target: 50000).")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (higher = more diverse pool).")
    parser.add_argument("--batch", type=int, default=256, help="Sampling batch size.")
    parser.add_argument("--reference", default="autoguard/data/processed/antibacterial_reference.fasta",
                        help="Antibacterial reference FASTA (exact-match novelty filter).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pool_factor", type=float, default=4.0,
                        help="Candidate-pool oversampling factor over library_size.")
    parser.add_argument("--conditioned_pool", action="store_true",
                        help="Sample a separate category-conditioned pool per category "
                             "through the generalized safety module (slower). Default: "
                             "one shared pool ranked per category.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", ModelConfig())
    model = AutoGuardModel(config, use_graph_encoder=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    logger.info(f"Loaded AutoGuard from {args.checkpoint} on {device}")

    reference = _load_reference(Path(args.reference))
    logger.info(f"Reference antibacterial set: {len(reference)} sequences")

    target_unique = max(int(args.library_size * args.pool_factor), args.library_size)
    out_dir = Path(args.out_dir)
    categories = CATEGORIES if args.category == "all" else (args.category,)

    if args.conditioned_pool:
        # Each category gets its own pool, generated through the generalized
        for category in categories:
            logger.info(f"[{category}] sampling conditioned pool "
                        f"(target {target_unique} unique, T={args.temperature})...")
            pool = generate_candidate_pool(
                model, target_unique, args.temperature, device, reference,
                batch=args.batch, category=category,
            )
            logger.info(f"[{category}] pool: {len(pool)} unique valid novel sequences")
            run_category(model, category, pool, out_dir, args.library_size)
    else:
        # One shared candidate pool serves every category (each ranks it differently).
        logger.info(f"Sampling shared candidate pool (target {target_unique} unique, "
                    f"T={args.temperature})...")
        pool = generate_candidate_pool(
            model, target_unique, args.temperature, device, reference, batch=args.batch,
        )
        logger.info(f"Candidate pool: {len(pool)} unique valid novel sequences")
        for category in categories:
            run_category(model, category, pool, out_dir, args.library_size)

    logger.info(f"Done. Libraries written under {out_dir}/")


if __name__ == "__main__":
    main()
