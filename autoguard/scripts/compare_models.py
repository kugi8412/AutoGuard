#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare AutoGuard, standard HydrAMP, and a lightweight GNN-VAE baseline.

Usage:
    python -m autoguard.scripts.compare_models --mode smoke      # quick CPU smoke-test
    python -m autoguard.scripts.compare_models --mode full       # full comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autoguard.config import ModelConfig, LossWeights
from autoguard.models.autoguard_model import AutoGuardModel
from autoguard.models.gnn_baseline import GNNGenerator, _AA_TO_IDX, _IDX_TO_AA
from autoguard.evaluation.metrics import AMPMetrics, compute_novelty, compute_diversity
from autoguard.evaluation.hydramp_adapter import HydrAMPBaseline
from autoguard.data.datasets import tokenize_sequence, detokenize_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Synthetic training data for smoke tests

_SAMPLE_AMPS = [
    "KLAKLAKKLAKLAK",
    "GIGKFLHSAKKFGKAFV",
    "RWLIVWRIIRKF",
    "FKLRAKIKVRLRAKIKL",
    "GRFKRFRKKFKK",
    "ILGKLLSTAAGLLSNL",
    "KLKLKLKLKLKLKL",
    "GWLKKIGKKIERVGQH",
    "RLYLRIGRR",
    "VGIGALPIGIGL",
    "LLGDFFRKSKEKI",
    "KWKLFKKIGAVLKVL",
    "GFKRIVQRIKDFLRNL",
    "RKRWCWQGI",
    "GILGAGKKIVGGLIELI",
]


def _load_real_data(data_dir: str = "data/") -> Optional[List[str]]:
    """Try to load real AMP sequences from prepared data."""
    import csv
    from pathlib import Path
    processed = Path(data_dir) / "processed" / "amp_train.csv"

    if not processed.exists():
        return None

    sequences = []

    with open(processed, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("sequence", "")
            if 5 <= len(seq) <= 25:
                sequences.append(seq)

    return sequences if sequences else None


# Training
def train_autoguard_mini(
    sequences: List[str],
    epochs: int = 3,
    device: str = "cpu",
) -> AutoGuardModel:
    """Train a minimal AutoGuard model (no graph encoder) on CPU."""
    config = ModelConfig(
        num_epochs=epochs,
        batch_size=min(16, len(sequences)),
    )
    model = AutoGuardModel(config, use_graph_encoder=False).to(device)
    model.train()

    from autoguard.training.losses import AutoGuardLoss

    loss_fn = AutoGuardLoss(LossWeights())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokens_list = [tokenize_sequence(s) for s in sequences]
    tokens = torch.stack(tokens_list).to(device)
    labels = torch.ones(len(sequences), 1, device=device)

    for epoch in range(epochs):
        output = model(tokens)
        targets = {"tokens": tokens, "label": labels}
        losses = loss_fn(output, targets)
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    return model


def generate_autoguard(model: AutoGuardModel,
                       n: int,
                       device: str = "cpu"
                       ) -> Tuple[List[str], List[float]]:
    """Generate n sequences from a trained AutoGuard model."""
    sequences, metadata = model.generate(
        num_samples=n, temperature=0.8, safety_threshold=0.9, max_attempts=n * 20
    )
    seqs_str = [detokenize_sequence(s.squeeze()) for s in sequences]
    amp_scores = [m.get("amp_score", 0.5) for m in metadata]
    return seqs_str, amp_scores


def train_gnn_mini(
    sequences: List[str],
    epochs: int = 3,
    device: str = "cpu",
) -> GNNGenerator:
    """Train the lightweight GNN-VAE baseline on CPU."""
    model = GNNGenerator(max_len=25, latent_dim=32, hidden_dim=64, num_gnn_layers=2).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokens = torch.stack([tokenize_sequence(s) for s in sequences]).to(device)
    labels = torch.ones(len(sequences), 1, device=device)

    for _ in range(epochs):
        output = model(tokens)
        logits = output["logits"]
        mu, logvar = output["mu"], output["logvar"]
        # Reconstruction
        recon_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tokens.reshape(-1), ignore_index=0
        )
        # KL
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        # AMP auxiliary
        amp_loss = torch.nn.functional.binary_cross_entropy(output["amp_prediction"], labels)
        loss = recon_loss + 0.001 * kl + 0.1 * amp_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


# Evaluation helper
def evaluate_generated(
    name: str,
    sequences: List[str],
    amp_scores: List[float],
    training_set: Set[str],
) -> Dict:

    if not sequences:
        return {"name": name, "error": "no sequences generated"}

    metrics = AMPMetrics(training_sequences=training_set)
    result = metrics.evaluate_batch(sequences, amp_scores=amp_scores)
    result["name"] = name
    return result


# Comparison
def run_comparison(mode: str = "smoke",
                   device: str = "cpu",
                   output: str = "comparison_report.json"):
    t0 = time.time()
    _set_seed(42)

    # Use real data if available, otherwise fall back to synthetic samples
    real_data = _load_real_data() if mode == "full" else None
    if real_data and len(real_data) >= 50:
        training_sequences = real_data[:2000]  # Cap for speed
        logger.info(f"Using real AMP data: {len(training_sequences)} sequences from data/processed/")
    else:
        training_sequences = _SAMPLE_AMPS
        if mode == "full":
            logger.info("No real data found (run prepare_data.py first). Using synthetic samples.")

    training_set = set(training_sequences)
    n_gen = 50 if mode == "smoke" else 1000
    epochs = 3 if mode == "smoke" else 30

    results = []

    # AutoGuard (no graph encoder)
    logger.info("[1/3] Training AutoGuard (sequence-only mode).")
    ag_model = train_autoguard_mini(training_sequences, epochs=epochs, device=device)
    ag_seqs, ag_scores = generate_autoguard(ag_model, n_gen, device=device)
    results.append(evaluate_generated("AutoGuard", ag_seqs, ag_scores, training_set))
    logger.info(f"  AutoGuard generated {len(ag_seqs)} sequences, quality={results[-1].get('quality_score', 0):.3f}")

    # HydrAMP baseline (from amp-challenge-2027)
    logger.info("[2/3] Training HydrAMP baseline (PyTorch).")
    try:
        hamp = HydrAMPBaseline(device=device)
        hamp.train(training_sequences, epochs=epochs, batch_size=min(8, len(training_sequences)))
        hamp_seqs, hamp_scores = hamp.generate(n_gen)
        results.append(evaluate_generated("HydrAMP", hamp_seqs, hamp_scores, training_set))
        logger.info(f"  HydrAMP generated {len(hamp_seqs)} sequences, quality={results[-1].get('quality_score', 0):.3f}")
    except Exception as e:
        logger.warning(f"  HydrAMP baseline failed: {e}")
        results.append({"name": "HydrAMP", "error": str(e)})

    # GNN baseline
    logger.info("[3/3] Training GNN-VAE baseline...")
    gnn_model = train_gnn_mini(training_sequences, epochs=epochs, device=device)
    gnn_seqs, gnn_scores = gnn_model.generate(n_gen, device=device, temperature=0.8)
    results.append(evaluate_generated("GNN-VAE", gnn_seqs, gnn_scores, training_set))
    logger.info(f"  GNN-VAE generated {len(gnn_seqs)} sequences, quality={results[-1].get('quality_score', 0):.3f}")

    elapsed = time.time() - t0
    logger.info(f"\nComparison finished in {elapsed:.1f}s")

    # Report
    _print_comparison_table(results)
    _save_report(results, output, elapsed)
    return results


def _print_comparison_table(results: List[Dict]):
    header_keys = ["novelty", "diversity", "amp_hit_rate", "quality_score"]
    header = f"{'Model':<14}" + "".join(f"{k:<14}" for k in header_keys)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['name']:<14} ERROR: {r['error']}")
            continue
        row = f"{r.get('name', '?'):<14}"
        for k in header_keys:
            v = r.get(k, None)
            row += f"{v:<14.4f}" if v is not None else f"{'N/A':<14}"
        print(row)

    print("=" * len(header) + "\n")


def _save_report(results: List[Dict], output: str, elapsed: float):
    # Remove non-serializable arrays
    serializable = []

    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                sr[k] = v
            elif isinstance(v, dict):
                sr[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
        serializable.append(sr)

    report = {"elapsed_seconds": elapsed, "results": serializable}

    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Compare AutoGuard vs baselines")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="comparison_report.json")
    args = parser.parse_args()
    run_comparison(args.mode, args.device, args.output)


if __name__ == "__main__":
    main()
