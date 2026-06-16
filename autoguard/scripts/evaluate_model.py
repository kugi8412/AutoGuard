#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate sequences from a trained model and compute evaluation metrics.
Supports both AutoGuard and the HydrAMP baseline so the Snakemake pipeline can
evaluate each model the same way. Writes:
  * "<out_dir>/generated.csv"  — generated sequences + AMP scores
  * "<out_dir>/metrics.json"   — novelty/diversity/quality/etc.

Usage:
    python -m autoguard.scripts.evaluate_model --model autoguard \
        --checkpoint checkpoints/autoguard/best_model.pt \
        --data_dir data/ --out_dir results/autoguard --num 200 --seed 42

    python -m autoguard.scripts.evaluate_model --model hydramp \
        --checkpoint checkpoints/hydramp/hydramp.pt \
        --data_dir data/ --out_dir results/hydramp --num 200 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autoguard.config import ModelConfig
from autoguard.evaluation.metrics import AMPMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_training_set(data_dir: Path) -> Set[str]:
    path = data_dir / "processed" / "amp_train.csv"
    seqs: Set[str] = set()
    if not path.exists():
        return seqs
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = row.get("sequence", "")
            if s:
                seqs.add(s)
    return seqs


def _load_reference_set(data_dir: Path) -> Set[str]:
    """Load known antibacterial peptides (challenge reference set).
    Looks for the FASTA copied by prepare_data, then falls back to the
    challenge repository's bundled reference.
    """
    candidates = [
        data_dir / "processed" / "antibacterial_reference.fasta",
        data_dir.parent / "amp-challenge-2027-main" / "data" / "antibacterial.fasta",
    ]
    seqs: Set[str] = set()

    for path in candidates:
        if path.exists():
            hdr = None
            parts: List[str] = []

            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith(">"):
                    if hdr is not None and parts:
                        seqs.add("".join(parts).upper())
                    hdr, parts = line, []
                elif line:
                    parts.append(line)

            if hdr is not None and parts:
                seqs.add("".join(parts).upper())

            break
    return seqs


def _generate_autoguard(checkpoint: str, num: int, device: str,
                        use_graph: bool, temperature: float
                        ) -> Tuple[List[str], List[float], List[float]]:
    from autoguard.models.autoguard_model import AutoGuardModel
    from autoguard.data.datasets import detokenize_sequence

    config = ModelConfig()
    model = AutoGuardModel(config, use_graph_encoder=use_graph).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    sequences, metadata = model.generate(
        num_samples=num, temperature=temperature,
        safety_threshold=0.9, max_attempts=num * 20,
    )
    seqs = [detokenize_sequence(s.squeeze()) for s in sequences]
    scores = [float(m.get("amp_score", 0.5)) for m in metadata]
    hemolysis = [float(m.get("hemolysis", 0.0)) for m in metadata]
    return seqs, scores, hemolysis


def _generate_hydramp(checkpoint: str, num: int, device: str,
                     temperature: float) -> Tuple[List[str], List[float], List[float]]:
    from autoguard.evaluation.hydramp_adapter import HydrAMPBaseline

    hamp = HydrAMPBaseline(device=device, temperature=temperature)
    hamp.load(checkpoint)
    seqs, scores = hamp.generate(num)

    return seqs, scores, []  # HydrAMP baseline has no hemolysis head


def main():
    parser = argparse.ArgumentParser(description="Generate + evaluate a trained model.")
    parser.add_argument("--model", choices=["autoguard", "hydramp"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--use_graph", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _set_global_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"Generating {args.num} sequences from {args.model} ({args.checkpoint})...")
    if args.model == "autoguard":
        seqs, scores, hemolysis = _generate_autoguard(
            args.checkpoint, args.num, args.device, args.use_graph, args.temperature)
    else:
        seqs, scores, hemolysis = _generate_hydramp(
            args.checkpoint, args.num, args.device, args.temperature)

    # Drop empties (keep hemolysis aligned when present).
    if hemolysis and len(hemolysis) == len(seqs):
        triples = [(s, sc, h) for s, sc, h in zip(seqs, scores, hemolysis) if s]
        seqs = [s for s, _, _ in triples]
        scores = [sc for _, sc, _ in triples]
        hemolysis = [h for _, _, h in triples]
    else:
        pairs = [(s, sc) for s, sc in zip(seqs, scores) if s]
        seqs = [s for s, _ in pairs]
        scores = [sc for _, sc in pairs]
        hemolysis = []
    logger.info(f"  Generated {len(seqs)} non-empty sequences")

    # Write generated sequences.
    with open(out_dir / "generated.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "amp_score", "length"])

        for s, sc in zip(seqs, scores):
            w.writerow([s, f"{sc:.4f}", len(s)])

    # Evaluate.
    data_dir = Path(args.data_dir)
    training_set = _load_training_set(data_dir)
    reference_set = _load_reference_set(data_dir)
    metrics = AMPMetrics(training_sequences=training_set,
                         reference_sequences=reference_set)
    result = metrics.evaluate_batch(seqs, amp_scores=scores)
    challenge = metrics.evaluate_challenge(
        seqs, amp_scores=scores,
        hemolysis_scores=hemolysis or None,
    )
    # Keep only JSON-serializable scalars + the AA distribution.
    report = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "num_generated": int(result.get("num_generated", len(seqs))),
        "mean_length": float(result.get("mean_length", 0.0)),
        "novelty": float(result.get("novelty", 0.0)),
        "diversity": float(result.get("diversity", 0.0)),
        "mean_amp_score": float(result.get("mean_amp_score", 0.0)),
        "amp_hit_rate": float(result.get("amp_hit_rate", 0.0)),
        "hydrophobicity_ratio": float(result.get("hydrophobicity_ratio", 0.0)),
        "quality_score": float(result.get("quality_score", 0.0)),
        "unique_fraction": float(challenge.get("unique_fraction", 0.0)),
        "length_valid_fraction": float(challenge.get("length_valid_fraction", 0.0)),
        "canonical_fraction": float(challenge.get("canonical_fraction", 0.0)),
        "challenge_valid_fraction": float(challenge.get("challenge_valid_fraction", 0.0)),
        "num_overlap_reference": int(challenge.get("num_overlap_reference", 0)),
        "predicted_success_rate": float(challenge.get("predicted_success_rate", 0.0)),
        "predicted_mic50": float(challenge.get("predicted_mic50", float("nan"))),
        "predicted_mic90": float(challenge.get("predicted_mic90", float("nan"))),
        "predicted_safety_window": float(challenge.get("predicted_safety_window", float("nan"))),
        "predicted_non_hemolytic_fraction": float(
            challenge.get("predicted_non_hemolytic_fraction", float("nan"))),
        "top_k_max_identity": float(challenge.get("top_k_max_identity", 0.0)),
        "top_k_identity_violations": int(challenge.get("top_k_identity_violations", 0)),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"  novelty={report['novelty']:.3f} diversity={report['diversity']:.3f} "
                f"amp_hit={report['amp_hit_rate']:.3f} quality={report['quality_score']:.3f}")
    logger.info(f"  [challenge] success_rate={report['predicted_success_rate']:.3f} "
                f"MIC50={report['predicted_mic50']:.2f} MIC90={report['predicted_mic90']:.2f} "
                f"valid={report['challenge_valid_fraction']:.3f} "
                f"overlap_ref={report['num_overlap_reference']}")
    logger.info(f"Saved: {out_dir / 'generated.csv'} and {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
