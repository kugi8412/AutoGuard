"""HydrAMP-based peptide generation for AMP Challenge 2027.

Entry point for all five challenge categories:
  generate_broad_spectrum, generate_gram_pos, generate_gram_neg,
  generate_mdr, generate_therapeutic

Produces:
  {category}/library.fasta  — 50 000 unique peptides
  {category}/top.fasta      — top-100 ranked candidates
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from amp_challenge_2027.config import LATENT_DIM, MAX_LENGTH, STD_AMINO_ACIDS
from amp_challenge_2027.model import HydrAMP
from amp_challenge_2027.sequence import translate_peptide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_fasta(sequences: List[str], path: Path) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n{seq}\n")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_reference_sequences(fasta_path: str) -> set:
    """Load reference sequences to exclude from the library."""
    ref = set()
    current: list[str] = []
    if not os.path.exists(fasta_path):
        return ref
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    ref.add("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        ref.add("".join(current))
    return ref


def _is_valid_sequence(seq: str) -> bool:
    """Check sequence meets challenge requirements: 8-50 residues, std AA."""
    return 8 <= len(seq) <= 50 and all(aa in STD_AMINO_ACIDS for aa in seq)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model(checkpoint_dir: str, device: torch.device) -> HydrAMP:
    """Load trained HydrAMP model from checkpoint."""
    model = HydrAMP.create_default()
    final_path = os.path.join(checkpoint_dir, "hydramp_final.pt")
    if os.path.exists(final_path):
        state = torch.load(final_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded model from {final_path}")
    else:
        pt_files = sorted(Path(checkpoint_dir).glob("hydramp_epoch_*.pt"))
        if pt_files:
            ckpt = torch.load(str(pt_files[-1]), map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded model from {pt_files[-1]}")
        else:
            print("WARNING: No checkpoint found — using randomly initialised model.")
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_peptides(
    model: HydrAMP,
    n_sequences: int,
    device: torch.device,
    amp_condition: float = 1.0,
    mic_condition: float = 1.0,
    batch_size: int = 2048,
    reference_seqs: set | None = None,
) -> Tuple[List[str], List[float], List[float]]:
    """
    Generate *n_sequences* unique, valid peptides via latent-space sampling.
    Returns (sequences, amp_scores, mic_scores).
    """
    if reference_seqs is None:
        reference_seqs = set()

    accepted_seqs: list[str] = []
    accepted_amp: list[float] = []
    accepted_mic: list[float] = []
    seen: set[str] = set()

    model.eval()
    with torch.no_grad():
        while len(accepted_seqs) < n_sequences:
            remaining = n_sequences - len(accepted_seqs)
            n_sample = min(batch_size, remaining * 2)

            z = torch.randn(n_sample, LATENT_DIM, device=device)
            c_amp = torch.full((n_sample, 1), amp_condition, device=device)
            c_mic = torch.full((n_sample, 1), mic_condition, device=device)
            z_cond = torch.cat([z, c_amp, c_mic], dim=1)

            soft_output = model.decoder.generate(z_cond)
            indices = soft_output.argmax(dim=-1).cpu().numpy()

            idx_tensor = torch.from_numpy(indices).long().to(device)
            amp_scores = model.amp_classifier(idx_tensor).cpu().numpy().flatten()
            mic_scores = model.mic_classifier(idx_tensor).cpu().numpy().flatten()

            peptides = [translate_peptide(row) for row in indices]

            for pep, a_score, m_score in zip(peptides, amp_scores, mic_scores):
                if len(accepted_seqs) >= n_sequences:
                    break
                if not _is_valid_sequence(pep):
                    continue
                if pep in seen or pep in reference_seqs:
                    continue
                seen.add(pep)
                accepted_seqs.append(pep)
                accepted_amp.append(float(a_score))
                accepted_mic.append(float(m_score))

    return accepted_seqs, accepted_amp, accepted_mic


def score_sequences(
    amp_scores: List[float],
    mic_scores: List[float],
) -> List[float]:
    """Combined score: higher is better (AMP probability + MIC activity)."""
    return [a + m for a, m in zip(amp_scores, mic_scores)]


# ---------------------------------------------------------------------------
# Main entry point (all categories share the same logic)
# ---------------------------------------------------------------------------
def main():
    category = Path(sys.argv[0]).stem

    parser = argparse.ArgumentParser(description=f"HydrAMP generation — {category}")
    parser.add_argument("--n-sequences", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint")
    parser.add_argument("--reference-fasta", type=str, default="./data/antibacterial.fasta")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    _set_seed(args.seed)

    device = torch.device(args.device)
    model = _load_model(args.checkpoint_dir, device)

    ref_seqs = _load_reference_sequences(args.reference_fasta)
    print(f"Reference sequences to exclude: {len(ref_seqs)}")

    print(f"Generating {args.n_sequences} peptides for category '{category}'...")
    sequences, amp_scores, mic_scores = generate_peptides(
        model=model,
        n_sequences=args.n_sequences,
        device=device,
        reference_seqs=ref_seqs,
    )

    out_dir = Path(category)
    out_dir.mkdir(parents=True, exist_ok=True)

    library_path = out_dir / "library.fasta"
    _write_fasta(sequences, library_path)
    print(f"Generated {len(sequences)} sequences → {library_path}")

    scores = score_sequences(amp_scores, mic_scores)
    ranked = sorted(zip(scores, sequences), key=lambda x: x[0], reverse=True)
    top_sequences = [seq for _, seq in ranked[: args.top_k]]

    top_path = out_dir / "top.fasta"
    _write_fasta(top_sequences, top_path)
    print(f"Top {args.top_k} sequences → {top_path}")
