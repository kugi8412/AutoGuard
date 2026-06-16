"""HydrAMP model components — PyTorch implementation.

Contains:
  - Encoder  (Bidirectional GRU → latent)
  - Decoder  (Autoregressive GRU + LSTM + Gumbel-Softmax)
  - AMP classifier  (Conv1D + LSTM, Veltri-style)
  - MIC classifier  (LSTM only, NoConv-style)
  - MasterHydrAMP  (full cVAE assembler for training)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from amp_challenge_2027.config import (
    HIDDEN_DIM,
    LATENT_DIM,
    MAX_LENGTH,
    RCL_WEIGHT,
    VOCAB_PAD_SIZE,
)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """Bidirectional 2-layer GRU encoder → (z_mean, z_sigma, z)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_PAD_SIZE,
        embed_dim: int = 100,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        max_length: int = MAX_LENGTH,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru1 = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dense_z_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.dense_z_sigma = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len) integer-encoded sequences.
        Returns (z_mean, z_sigma, z).
        """
        emb = self.embedding(x)                   # (B, L, embed_dim)
        h1, _ = self.gru1(emb)                    # (B, L, hidden*2)
        h2, _ = self.gru2(h1)                     # (B, L, hidden*2)
        last = h2[:, -1, :]                        # take last time-step
        z_mean = self.dense_z_mean(last)           # (B, latent_dim)
        z_sigma = self.dense_z_sigma(last)         # (B, latent_dim)
        z = self.reparameterize(z_mean, z_sigma)
        return z_mean, z_sigma, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only z_mean (used for inference encoding)."""
        z_mean, _, _ = self.forward(x)
        return z_mean

    @staticmethod
    def reparameterize(z_mean: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        std = torch.exp(z_sigma / 2)
        eps = torch.randn_like(std)
        return z_mean + std * eps

    def forward_dense(self, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accept one-hot / soft input (B, L, vocab_size) instead of indices."""
        weight = self.embedding.weight  # (vocab_size, embed_dim)
        emb = x_onehot @ weight         # (B, L, embed_dim)
        h1, _ = self.gru1(emb)
        h2, _ = self.gru2(h1)
        last = h2[:, -1, :]
        z_mean = self.dense_z_mean(last)
        z_sigma = self.dense_z_sigma(last)
        z = self.reparameterize(z_mean, z_sigma)
        return z_mean, z_sigma, z


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """Autoregressive GRU → LSTM → Dense → Gumbel-Softmax decoder."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        condition_dim: int = 2,
        gru_hidden: int = LATENT_DIM + 2,
        lstm_hidden: int = 100,
        output_vocab: int = VOCAB_PAD_SIZE,
        max_length: int = MAX_LENGTH,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.max_length = max_length
        self.gru_hidden = gru_hidden
        self.output_vocab = output_vocab

        # Input to GRU cell is (latent_dim+2)-dim (=gru_hidden in original)
        self.gru_cell = nn.GRUCell(gru_hidden, gru_hidden)
        # LSTM takes GRU output + condition (2)
        self.lstm = nn.LSTM(
            gru_hidden + condition_dim, lstm_hidden,
            batch_first=True, dropout=0.0,
        )
        self.dense = nn.Linear(lstm_hidden + condition_dim, output_vocab)
        self.temperature = temperature

    def forward(
        self,
        z_cond: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        z_cond: (batch, latent_dim + 2) — latent vector concatenated with
                AMP/MIC condition scalars.
        Returns: (batch, max_length, output_vocab) logits or soft samples.
        """
        batch_size = z_cond.size(0)
        condition = z_cond[:, -self.condition_dim:]  # (B, 2)
        device = z_cond.device

        # Autoregressive GRU unrolling
        gru_out = z_cond  # initial state = z_cond itself
        current_input = torch.zeros(batch_size, self.gru_hidden, device=device)
        gru_outputs = []
        for _ in range(self.max_length):
            gru_out = self.gru_cell(current_input, gru_out)
            gru_outputs.append(gru_out)
            current_input = gru_out

        gru_seq = torch.stack(gru_outputs, dim=1)  # (B, L, gru_hidden)

        # Concatenate condition at every position
        cond_expanded = condition.unsqueeze(1).expand(-1, self.max_length, -1)
        lstm_input = torch.cat([gru_seq, cond_expanded], dim=2)

        lstm_out, _ = self.lstm(lstm_input)  # (B, L, lstm_hidden)
        lstm_out = torch.cat([lstm_out, cond_expanded], dim=2)
        logits = self.dense(lstm_out)  # (B, L, vocab)

        if self.training and self.temperature > 0:
            return self._gumbel_softmax(logits, hard=hard)
        return F.softmax(logits, dim=-1)

    def _gumbel_softmax(self, logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
        return F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)

    def generate(self, z_cond: torch.Tensor) -> torch.Tensor:
        """Deterministic generation (softmax, no Gumbel noise)."""
        self.eval()
        with torch.no_grad():
            return self.forward(z_cond)


# ---------------------------------------------------------------------------
# Discriminators / Classifiers
# ---------------------------------------------------------------------------
class AMPClassifier(nn.Module):
    """Veltri-style AMP classifier: Embedding → Conv1D → MaxPool → LSTM → sigmoid."""

    def __init__(
        self,
        vocab_size: int = VOCAB_PAD_SIZE,
        embed_dim: int = 128,
        conv_filters: int = 64,
        conv_kernel: int = 16,
        lstm_hidden: int = 100,
        max_length: int = MAX_LENGTH,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, conv_filters, kernel_size=conv_kernel, padding="same")
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=5)
        pool_len = max_length // 5
        self.lstm = nn.LSTM(conv_filters, lstm_hidden, batch_first=True)
        self.dense = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) integer indices  →  (B, 1) probability."""
        emb = self.embedding(x)           # (B, L, embed_dim)
        # Conv1d expects (B, C, L)
        c = self.relu(self.conv(emb.transpose(1, 2)))  # (B, filters, L)
        c = self.pool(c)                                # (B, filters, L//5)
        c = c.transpose(1, 2)                           # (B, L//5, filters)
        lstm_out, _ = self.lstm(c)
        last = lstm_out[:, -1, :]
        return torch.sigmoid(self.dense(last))

    def forward_dense(self, x_soft: torch.Tensor) -> torch.Tensor:
        """Accept soft/one-hot input (B, L, vocab_size)."""
        weight = self.embedding.weight
        emb = x_soft @ weight
        c = self.relu(self.conv(emb.transpose(1, 2)))
        c = self.pool(c)
        c = c.transpose(1, 2)
        lstm_out, _ = self.lstm(c)
        last = lstm_out[:, -1, :]
        return torch.sigmoid(self.dense(last))


class MICClassifier(nn.Module):
    """NoConv MIC classifier: Embedding → LSTM → MaxPool → LSTM → sigmoid."""

    def __init__(
        self,
        vocab_size: int = VOCAB_PAD_SIZE,
        embed_dim: int = 128,
        lstm1_hidden: int = 64,
        lstm2_hidden: int = 100,
        max_length: int = MAX_LENGTH,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_dim, lstm1_hidden, batch_first=True)
        self.pool = nn.MaxPool1d(kernel_size=5)
        pool_len = max_length // 5
        self.lstm2 = nn.LSTM(lstm1_hidden, lstm2_hidden, batch_first=True)
        self.dense = nn.Linear(lstm2_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        lstm1_out, _ = self.lstm1(emb)           # (B, L, h1)
        pooled = self.pool(lstm1_out.transpose(1, 2)).transpose(1, 2)  # (B, L//5, h1)
        lstm2_out, _ = self.lstm2(pooled)
        last = lstm2_out[:, -1, :]
        return torch.sigmoid(self.dense(last))

    def forward_dense(self, x_soft: torch.Tensor) -> torch.Tensor:
        weight = self.embedding.weight
        emb = x_soft @ weight
        lstm1_out, _ = self.lstm1(emb)
        pooled = self.pool(lstm1_out.transpose(1, 2)).transpose(1, 2)
        lstm2_out, _ = self.lstm2(pooled)
        last = lstm2_out[:, -1, :]
        return torch.sigmoid(self.dense(last))


# ---------------------------------------------------------------------------
# Full HydrAMP cVAE  (for training)
# ---------------------------------------------------------------------------
class HydrAMP(nn.Module):
    """Complete HydrAMP conditional VAE with frozen discriminators."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        amp_classifier: AMPClassifier,
        mic_classifier: MICClassifier,
        kl_weight: float = 1e-4,
        rcl_weight: float = RCL_WEIGHT,
        loss_weights: Optional[list] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.amp_classifier = amp_classifier
        self.mic_classifier = mic_classifier
        self.kl_weight = kl_weight
        self.rcl_weight = rcl_weight
        self.loss_weights = loss_weights or [1.0] * 16

        # Freeze discriminators by default
        for p in self.amp_classifier.parameters():
            p.requires_grad = False
        for p in self.mic_classifier.parameters():
            p.requires_grad = False

    def forward(
        self,
        sequences: torch.Tensor,
        amp_label: torch.Tensor,
        mic_label: torch.Tensor,
        noise: torch.Tensor,
        sleep_amp: torch.Tensor,
        sleep_mic: torch.Tensor,
    ):
        """Full forward pass returning dict of individual losses."""
        # Encode
        z_mean, z_sigma, z = self.encoder(sequences)

        # Override z with explicit noise-based sampling for reproducibility
        z = z_mean + torch.exp(z_sigma / 2) * noise

        # Decode with true conditions
        z_cond = torch.cat([z, amp_label, mic_label], dim=1)
        reconstructed = self.decoder(z_cond)  # (B, L, V)

        # Classifier predictions on reconstructed
        amp_pred = self.amp_classifier.forward_dense(reconstructed)
        mic_pred = self.mic_classifier.forward_dense(reconstructed)

        # Re-encode reconstructed → z' for consistency
        z_recon, _, _ = self.encoder.forward_dense(reconstructed)

        # Sleep phase — decode with random conditions
        sleep_z_cond = torch.cat([z, sleep_amp, sleep_mic], dim=1)
        sleep_recon = self.decoder(sleep_z_cond)
        sleep_z_recon, _, _ = self.encoder.forward_dense(sleep_recon)

        sleep_amp_pred = self.amp_classifier.forward_dense(sleep_recon)
        sleep_mic_pred = self.mic_classifier.forward_dense(sleep_recon)

        # Unconstrained sleep — decode from pure noise
        unc_z_cond = torch.cat([noise, sleep_amp, sleep_mic], dim=1)
        unc_recon = self.decoder(unc_z_cond)
        unc_z_recon, _, _ = self.encoder.forward_dense(unc_recon)

        unc_amp_pred = self.amp_classifier.forward_dense(unc_recon)
        unc_mic_pred = self.mic_classifier.forward_dense(unc_recon)

        return {
            "z_mean": z_mean,
            "z_sigma": z_sigma,
            "z": z,
            "reconstructed": reconstructed,
            "amp_pred": amp_pred,
            "mic_pred": mic_pred,
            "z_recon": z_recon,
            "sleep_recon": sleep_recon,
            "sleep_z_recon": sleep_z_recon,
            "sleep_amp_pred": sleep_amp_pred,
            "sleep_mic_pred": sleep_mic_pred,
            "unc_recon": unc_recon,
            "unc_z_recon": unc_z_recon,
            "unc_amp_pred": unc_amp_pred,
            "unc_mic_pred": unc_mic_pred,
            "noise": noise,
        }

    def compute_loss(self, outputs, sequences, amp_label, mic_label, sleep_amp, sleep_mic):
        """Compute the multi-term HydrAMP loss."""
        w = self.loss_weights
        z_mean = outputs["z_mean"]
        z_sigma = outputs["z_sigma"]
        z = outputs["z"]
        reconstructed = outputs["reconstructed"]

        # Reconstruction loss (sparse categorical cross-entropy)
        rcl = self.rcl_weight * F.cross_entropy(
            reconstructed.reshape(-1, reconstructed.size(-1)),
            sequences.reshape(-1),
            reduction="mean",
        )
        # KL divergence
        kl = -0.5 * torch.sum(1 + z_sigma - z_mean.pow(2) - z_sigma.exp(), dim=-1).mean()
        vae_loss = rcl + self.kl_weight * kl

        def bce(pred, target):
            return F.binary_cross_entropy(pred, target, reduction="mean")

        def huber(a, b):
            return F.smooth_l1_loss(a, b, reduction="mean")

        def mse(a, b):
            return F.mse_loss(a, b, reduction="mean")

        # Classifier losses
        amp_cls_loss = bce(outputs["amp_pred"], amp_label)
        mic_cls_loss = bce(outputs["mic_pred"], mic_label)

        # z reconstruction consistency
        z_recon_err = mse(outputs["z_recon"], z.detach())

        # Sleep losses
        sleep_amp_loss = bce(outputs["sleep_amp_pred"], sleep_amp)
        sleep_mic_loss = bce(outputs["sleep_mic_pred"], sleep_mic)
        sleep_z_err = mse(outputs["sleep_z_recon"], z.detach())

        # Unconstrained sleep losses
        unc_amp_loss = bce(outputs["unc_amp_pred"], sleep_amp)
        unc_mic_loss = bce(outputs["unc_mic_pred"], sleep_mic)
        unc_z_err = mse(outputs["unc_z_recon"], outputs["noise"].detach())

        # Combine with loss weights
        total = (
            w[0] * amp_cls_loss
            + w[1] * mic_cls_loss
            + w[2] * vae_loss
            + w[9] * sleep_amp_loss
            + w[10] * sleep_mic_loss
            + w[11] * unc_amp_loss
            + w[12] * unc_mic_loss
            + w[13] * z_recon_err
            + w[14] * sleep_z_err
            + w[15] * unc_z_err
        )

        return {
            "total": total,
            "rcl": rcl.detach(),
            "kl": kl.detach(),
            "vae": vae_loss.detach(),
            "amp_cls": amp_cls_loss.detach(),
            "mic_cls": mic_cls_loss.detach(),
        }

    @staticmethod
    def create_default(loss_weights=None, temperature: float = 1.0):
        """Factory helper — builds a default-sized HydrAMP."""
        enc = Encoder()
        dec = Decoder(temperature=temperature)
        amp_cls = AMPClassifier()
        mic_cls = MICClassifier()
        return HydrAMP(
            encoder=enc,
            decoder=dec,
            amp_classifier=amp_cls,
            mic_classifier=mic_cls,
            loss_weights=loss_weights,
        )
