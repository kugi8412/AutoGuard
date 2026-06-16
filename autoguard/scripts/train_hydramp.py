#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone training entry point for the HydrAMP baseline.

Trains the PyTorch HydrAMP model (ported from Szymczak et al., Nat Commun 2023,
shipped under "amp-challenge-2027-main/") on the prepared AMP dataset and
saves a checkpoint that the comparison/evaluation step can load.

This mirrors "autoguard.scripts.train" (stage full) so both models are
trained from the same data with the same seed, enabling a fair comparison.

Usage:
    python -m autoguard.scripts.train_hydramp \
        --data_dir data/ --save_dir checkpoints/hydramp \
        --epochs 30 --batch_size 32 --seed 42
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
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autoguard.evaluation.hydramp_adapter import HydrAMPBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"  Global seed set to {seed}")


def _load_amp_csv(path: Path,
                  max_train: int = 0
                  ) -> Tuple[List[str], List[float], List[float]]:
    """Load sequences, AMP labels and (linear) MIC targets from a processed CSV."""
    seqs: List[str] = []
    amp: List[float] = []
    mic: List[float] = []

    if not path.exists():
        return seqs, amp, mic

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row.get("sequence", "")

            if not (5 <= len(s) <= 25):
                continue

            seqs.append(s)
            amp.append(float(row.get("label", 1) or 1))
            mic_val = row.get("mic", "")
            # HydrAMP's MIC classifier expects a probability-of-low-MIC target.

            try:
                mic.append(1.0 if (mic_val and float(mic_val) <= 1.0) else 0.0)
            except ValueError:
                mic.append(0.0)

    if max_train and max_train > 0:
        seqs, amp, mic = seqs[:max_train], amp[:max_train], mic[:max_train]

    return seqs, amp, mic


def main():
    parser = argparse.ArgumentParser(description="Train the HydrAMP baseline (PyTorch).")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--save_dir", type=str, default="checkpoints/hydramp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--classifier_epochs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=0, help="Cap training sequences (0 = all)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _set_global_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "processed" / "amp_train.csv"
    seqs, amp, mic = _load_amp_csv(train_csv, args.max_train)

    if not seqs:
        logger.error(f"No training data at {train_csv}. Run prepare_data.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(seqs)} sequences (positives={int(sum(amp))}) from {train_csv}")

    logger.info("Building HydrAMP baseline...")
    hamp = HydrAMPBaseline(device=args.device, temperature=args.temperature)
    n_params = sum(p.numel() for p in hamp.model.parameters())
    logger.info(f"HydrAMP parameters: {n_params:,}")

    logger.info(f"Training HydrAMP for {args.epochs} epochs (batch={args.batch_size})...")
    hamp.train(
        sequences=seqs,
        amp_labels=amp,
        mic_labels=mic,
        epochs=args.epochs,
        batch_size=min(args.batch_size, len(seqs)),
        lr=args.lr,
        classifier_epochs=args.classifier_epochs,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "hydramp.pt"
    hamp.save(str(ckpt_path))

    meta = {
        "model": "HydrAMP",
        "params": int(n_params),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "num_train": len(seqs),
        "temperature": args.temperature,
    }
    with open(save_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved checkpoint: {ckpt_path}")
    logger.info(f"Saved metadata:   {save_dir / 'train_meta.json'}")
    logger.info("HydrAMP training complete.")


if __name__ == "__main__":
    main()
