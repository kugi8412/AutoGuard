"""Training script for HydrAMP (PyTorch version).

Trains the full conditional VAE (encoder + decoder) with frozen
discriminators on AMP / MIC / UniProt data.

Usage:
    uv run train_hydramp --data-dir ./data --epochs 50 --batch-size 128
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from amp_challenge_2027.config import (
    HIDDEN_DIM,
    LATENT_DIM,
    MAX_KL,
    MAX_LENGTH,
    MAX_TEMPERATURE,
    MIN_KL,
    MIN_TEMPERATURE,
    KL_ANNEALRATE,
    TAU_ANNEALRATE,
    hydra as HYDRA_WEIGHTS,
)
from amp_challenge_2027.model import HydrAMP, Encoder, Decoder, AMPClassifier, MICClassifier
from amp_challenge_2027.sequence import encode_sequences, translate_peptide, STD_AMINO_ACIDS


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AMPDataset(Dataset):
    """Simple dataset that loads peptide sequences + labels from a CSV or FASTA."""

    def __init__(
        self,
        sequences: List[str],
        amp_labels: List[float],
        mic_labels: List[float],
        max_length: int = MAX_LENGTH,
    ):
        self.encoded = encode_sequences(sequences, max_length)
        self.amp_labels = np.array(amp_labels, dtype=np.float32).reshape(-1, 1)
        self.mic_labels = np.array(mic_labels, dtype=np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.encoded[idx]),
            torch.tensor(self.amp_labels[idx]),
            torch.tensor(self.mic_labels[idx]),
        )


def load_fasta_sequences(path: str) -> List[str]:
    """Read sequences from a FASTA file."""
    sequences = []
    current = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def load_csv_sequences(path: str) -> Tuple[List[str], List[float], List[float]]:
    """Load sequences + AMP/MIC labels from CSV (columns: Sequence, AMP, MIC)."""
    sequences, amp_labels, mic_labels = [], [], []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seq = row.get("Sequence", row.get("sequence", ""))
            amp = float(row.get("AMP", row.get("amp", row.get("Label", row.get("label", 1)))))
            mic = float(row.get("MIC", row.get("mic", 0)))
            # Filter to valid amino acids and length
            if all(aa in STD_AMINO_ACIDS for aa in seq) and 1 <= len(seq) <= MAX_LENGTH:
                sequences.append(seq)
                amp_labels.append(amp)
                mic_labels.append(mic)
    return sequences, amp_labels, mic_labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(
    model: HydrAMP,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    kl_weight: float,
) -> dict:
    model.train()
    model.kl_weight = kl_weight

    total_loss = 0.0
    total_rcl = 0.0
    total_kl = 0.0
    n_batches = 0

    for sequences, amp_label, mic_label in dataloader:
        sequences = sequences.to(device)
        amp_label = amp_label.to(device)
        mic_label = mic_label.to(device)

        batch_size = sequences.size(0)
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        sleep_amp = torch.randint(0, 2, (batch_size, 1), device=device).float()
        sleep_mic = sleep_amp.clone()

        optimizer.zero_grad()
        outputs = model(sequences, amp_label, mic_label, noise, sleep_amp, sleep_mic)
        losses = model.compute_loss(outputs, sequences, amp_label, mic_label, sleep_amp, sleep_mic)

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        total_rcl += losses["rcl"].item()
        total_kl += losses["kl"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "rcl": total_rcl / max(n_batches, 1),
        "kl": total_kl / max(n_batches, 1),
    }


def train_classifier(
    classifier: nn.Module,
    sequences: List[str],
    labels: List[float],
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
):
    """Pre-train a single classifier (AMP or MIC) on labelled data."""
    encoded = encode_sequences(sequences, MAX_LENGTH)
    labels_arr = np.array(labels, dtype=np.float32).reshape(-1, 1)

    classifier.to(device)
    classifier.train()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(encoded),
        torch.from_numpy(labels_arr),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = classifier(x_batch)
            loss = nn.functional.binary_cross_entropy(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Classifier epoch {epoch+1}/{epochs}  loss={epoch_loss/len(loader):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train HydrAMP (PyTorch)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory with training CSVs or FASTA files")
    parser.add_argument("--positive-csv", type=str, default=None,
                        help="CSV file with positive (AMP) sequences")
    parser.add_argument("--negative-csv", type=str, default=None,
                        help="CSV file with negative (non-AMP) sequences")
    parser.add_argument("--mic-csv", type=str, default=None,
                        help="CSV file with MIC-labelled sequences")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--classifier-epochs", type=int, default=10,
                        help="Pre-training epochs for discriminators")
    parser.add_argument("--save-dir", type=str, default="./checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- Load data ----
    # Try to load from CSV files if provided; otherwise look for FASTA
    all_sequences, all_amp, all_mic = [], [], []

    if args.positive_csv and os.path.exists(args.positive_csv):
        seqs, amp_l, mic_l = load_csv_sequences(args.positive_csv)
        all_sequences.extend(seqs)
        all_amp.extend(amp_l)
        all_mic.extend(mic_l)
        print(f"Loaded {len(seqs)} positive sequences")

    if args.negative_csv and os.path.exists(args.negative_csv):
        seqs, amp_l, mic_l = load_csv_sequences(args.negative_csv)
        all_sequences.extend(seqs)
        all_amp.extend(amp_l)
        all_mic.extend(mic_l)
        print(f"Loaded {len(seqs)} negative sequences")

    if args.mic_csv and os.path.exists(args.mic_csv):
        seqs, amp_l, mic_l = load_csv_sequences(args.mic_csv)
        all_sequences.extend(seqs)
        all_amp.extend(amp_l)
        all_mic.extend(mic_l)
        print(f"Loaded {len(seqs)} MIC sequences")

    # Fallback: load antibacterial.fasta from the challenge data
    fasta_path = os.path.join(args.data_dir, "antibacterial.fasta")
    if not all_sequences and os.path.exists(fasta_path):
        fasta_seqs = load_fasta_sequences(fasta_path)
        # Filter to valid length
        fasta_seqs = [s for s in fasta_seqs if all(aa in STD_AMINO_ACIDS for aa in s) and 1 <= len(s) <= MAX_LENGTH]
        all_sequences = fasta_seqs
        all_amp = [1.0] * len(fasta_seqs)    # known AMPs
        all_mic = [1.0] * len(fasta_seqs)    # assume active
        print(f"Loaded {len(fasta_seqs)} sequences from antibacterial.fasta (all positive)")

    if not all_sequences:
        raise RuntimeError("No training data found. Provide --positive-csv or put antibacterial.fasta in --data-dir")

    print(f"Total training sequences: {len(all_sequences)}")

    dataset = AMPDataset(all_sequences, all_amp, all_mic, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ---- Build model ----
    temperature = MAX_TEMPERATURE
    hydramp = HydrAMP.create_default(loss_weights=HYDRA_WEIGHTS, temperature=temperature)
    hydramp.to(device)

    # ---- Pre-train discriminators ----
    print("\n--- Pre-training AMP classifier ---")
    hydramp.amp_classifier.requires_grad_(True)
    train_classifier(hydramp.amp_classifier, all_sequences, all_amp,
                     epochs=args.classifier_epochs, batch_size=args.batch_size, device=device)
    hydramp.amp_classifier.requires_grad_(False)

    print("\n--- Pre-training MIC classifier ---")
    hydramp.mic_classifier.requires_grad_(True)
    train_classifier(hydramp.mic_classifier, all_sequences, all_mic,
                     epochs=args.classifier_epochs, batch_size=args.batch_size, device=device)
    hydramp.mic_classifier.requires_grad_(False)

    # ---- Train VAE ----
    print("\n--- Training HydrAMP cVAE ---")
    # Only optimize encoder + decoder parameters
    vae_params = list(hydramp.encoder.parameters()) + list(hydramp.decoder.parameters())
    optimizer = optim.Adam(vae_params, lr=args.lr)

    kl_weight = MIN_KL
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Anneal
        kl_weight = min(kl_weight * math.exp(KL_ANNEALRATE * epoch), MAX_KL)
        temperature = max(temperature * math.exp(-TAU_ANNEALRATE * epoch), MIN_TEMPERATURE)
        hydramp.decoder.temperature = temperature

        metrics = train_epoch(hydramp, dataloader, optimizer, device, kl_weight)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={metrics['loss']:.4f}  rcl={metrics['rcl']:.4f}  "
            f"kl={metrics['kl']:.4f}  temp={temperature:.4f}  kl_w={kl_weight:.6f}"
        )

        # Save checkpoint every 10 epochs and on last epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": hydramp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "kl_weight": kl_weight,
                "temperature": temperature,
            }
            ckpt_path = save_dir / f"hydramp_epoch_{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  → saved {ckpt_path}")

    # Save final model
    final_path = save_dir / "hydramp_final.pt"
    torch.save(hydramp.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
