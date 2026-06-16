#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapter that exposes the *amp-challenge-2027* HydrAMP PyTorch implementation
as a baseline for AutoGuard's comparison harness.

The HydrAMP code (Szymczak et al., Nat Commun 2023) lives in a separate
project under "amp-challenge-2027-main/"
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def _locate_amp_challenge_src() -> Path:
    """Return the absolute path to "amp-challenge-2027-main/src".
    Searches upward from this file's location for the sibling project. Raises
    "FileNotFoundError" if the project is not present.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent.parent.parent, *here.parents]:
        candidate = parent / "amp-challenge-2027-main" / "src"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate amp-challenge-2027-main/src. Expected it to sit "
        "next to the autoguard/ folder."
    )


def _ensure_amp_challenge_on_path() -> None:
    src = _locate_amp_challenge_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


class HydrAMPBaseline:
    """Thin wrapper around the AMP Challenge HydrAMP PyTorch model."""

    def __init__(self, device: str = "cpu", temperature: float = 1.0):
        _ensure_amp_challenge_on_path()
        from amp_challenge_2027.model import HydrAMP
        from amp_challenge_2027.config import LATENT_DIM, MAX_LENGTH
        from amp_challenge_2027.config import hydra as HYDRA_WEIGHTS

        self._HydrAMP = HydrAMP
        self._LATENT_DIM = LATENT_DIM
        self._MAX_LENGTH = MAX_LENGTH
        self.device = torch.device(device)
        # Use the tuned 16-term HydrAMP loss-weight preset
        self.model = HydrAMP.create_default(
            loss_weights=list(HYDRA_WEIGHTS), temperature=temperature
        ).to(self.device)

    @staticmethod
    def _encode_sequences(sequences: Sequence[str], max_length: int) -> np.ndarray:
        _ensure_amp_challenge_on_path()
        from amp_challenge_2027.sequence import encode_sequences  # type: ignore

        return encode_sequences(list(sequences), max_length=max_length)

    @staticmethod
    def _decode_tokens(tokens: np.ndarray) -> List[str]:
        _ensure_amp_challenge_on_path()
        from amp_challenge_2027.sequence import decode_indices  # type: ignore

        return decode_indices(tokens)

    def train(
        self,
        sequences: Sequence[str],
        amp_labels: Sequence[float] | None = None,
        mic_labels: Sequence[float] | None = None,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 1e-3,
        classifier_epochs: int = 2,
    ) -> None:
        """Train discriminators briefly, then the VAE backbone.
        The training loop mirrors "amp_challenge_2027.train.main" but is
        kept compact so it can run inside smoke-tests on a CPU.
        """
        max_len = self._MAX_LENGTH
        encoded = self._encode_sequences(sequences, max_len)
        amp = np.asarray(amp_labels if amp_labels is not None else [1.0] * len(sequences),
                         dtype=np.float32).reshape(-1, 1)
        mic = np.asarray(mic_labels if mic_labels is not None else [1.0] * len(sequences),
                         dtype=np.float32).reshape(-1, 1)

        x = torch.from_numpy(encoded).long().to(self.device)
        amp_t = torch.from_numpy(amp).to(self.device)
        mic_t = torch.from_numpy(mic).to(self.device)

        for name, module, target in [
            ("amp", self.model.amp_classifier, amp_t),
            ("mic", self.model.mic_classifier, mic_t),
        ]:
            module.requires_grad_(True)
            opt = optim.Adam(module.parameters(), lr=lr)
            ds = TensorDataset(x, target)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            for ep in range(classifier_epochs):
                ep_loss, n_batches = 0.0, 0
                for xb, yb in loader:
                    pred = module(xb)
                    loss = nn.functional.binary_cross_entropy(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    ep_loss += loss.item()
                    n_batches += 1
                logger.info(
                    "  [%s classifier] epoch %d/%d  bce=%.4f",
                    name, ep + 1, classifier_epochs, ep_loss / max(n_batches, 1),
                )
            module.requires_grad_(False)

        # Train VAE backbone
        vae_params = (
            list(self.model.encoder.parameters())
            + list(self.model.decoder.parameters())
        )
        opt = optim.Adam(vae_params, lr=lr)
        ds = TensorDataset(x, amp_t, mic_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

        if len(loader) == 0:
            logger.warning(
                "VAE loader is empty (n=%d < batch_size=%d, drop_last=True); "
                "no backbone training will occur.", len(ds), batch_size,
            )

        # KL annealing: ramp the KL weight from MIN_KL up to MAX_KL so the
        # latent posterior is regularised toward N(0,1). Without this, samples
        # drawn from the prior at generation time are out-of-distribution and
        # the decoder collapses to a handful of near-identical sequences.
        _ensure_amp_challenge_on_path()
        from amp_challenge_2027.config import (  # type: ignore
            MIN_KL, MAX_KL, KL_ANNEALRATE,
        )

        for epoch in range(epochs):
            kl_weight = min(MAX_KL, MIN_KL + KL_ANNEALRATE * epoch * (MAX_KL - MIN_KL))
            self.model.kl_weight = kl_weight
            ep_total, ep_rcl, ep_kl, n_batches = 0.0, 0.0, 0.0, 0
            for xb, ab, mb in loader:
                bs = xb.size(0)
                noise = torch.randn(bs, self._LATENT_DIM, device=self.device)
                sleep_amp = torch.randint(0, 2, (bs, 1), device=self.device).float()
                sleep_mic = sleep_amp.clone()
                outputs = self.model(xb, ab, mb, noise, sleep_amp, sleep_mic)
                losses = self.model.compute_loss(
                    outputs, xb, ab, mb, sleep_amp, sleep_mic
                )
                opt.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(vae_params, max_norm=1.0)
                opt.step()
                ep_total += losses["total"].item()
                ep_rcl += losses["rcl"].item()
                ep_kl += losses["kl"].item()
                n_batches += 1
            d = max(n_batches, 1)
            logger.info(
                "  [VAE] epoch %d/%d  total=%.4f  rcl=%.4f  kl=%.4f  kl_w=%.2e",
                epoch + 1, epochs, ep_total / d, ep_rcl / d, ep_kl / d, kl_weight,
            )

    # Generate
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        amp_condition: float = 1.0,
        mic_condition: float = 1.0,
        batch_size: int = 256,
    ) -> Tuple[List[str], List[float]]:
        """Generate sequences, returning (sequences, amp_scores)."""
        self.model.eval()
        accepted: List[str] = []
        scores: List[float] = []
        attempts = 0
        max_attempts = max(num_samples * 10, num_samples + 1)

        while len(accepted) < num_samples and attempts < max_attempts:
            attempts += batch_size
            n = min(batch_size, num_samples * 3)
            z = torch.randn(n, self._LATENT_DIM, device=self.device)
            c_amp = torch.full((n, 1), amp_condition, device=self.device)
            c_mic = torch.full((n, 1), mic_condition, device=self.device)
            soft = self.model.decoder.generate(torch.cat([z, c_amp, c_mic], dim=1))
            idx = soft.argmax(dim=-1)
            amp_scores = self.model.amp_classifier(idx).squeeze(-1).cpu().tolist()
            decoded = self._decode_tokens(idx.cpu().numpy())

            for seq, s in zip(decoded, amp_scores):

                if not seq:
                    continue

                accepted.append(seq)
                scores.append(float(s))

                if len(accepted) >= num_samples:
                    break
        return accepted[:num_samples], scores[:num_samples]

    # Persistence
    def save(self, path: str) -> None:
        """Save the underlying HydrAMP weights to ``path``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        """Load HydrAMP weights previously saved with :meth:`save`."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
