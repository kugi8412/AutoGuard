#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main training loop for AutoGuard model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging

from ..models.autoguard_model import AutoGuardModel
from ..config import ModelConfig, LossWeights
from .losses import AutoGuardLoss

logger = logging.getLogger(__name__)


class AutoGuardTrainer:
    """End-to-end trainer for AutoGuard model."""

    def __init__(self, model: AutoGuardModel, config: ModelConfig,
                 loss_weights: LossWeights, device: str = 'cuda',
                 species_embeddings: Optional[torch.Tensor] = None,
                 species_lookup=None, esm_cache: Optional[dict] = None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = AutoGuardLoss(loss_weights)

        # Pre-trained species embeddings [num_species, embed_dim]
        if species_embeddings is not None:
            species_embeddings = self._fit_phylo_dim(species_embeddings)
            self.species_embeddings = species_embeddings.to(device)
        else:
            self.species_embeddings = None

        # ESM cache
        self.species_lookup = species_lookup
        self.esm_cache = esm_cache or {}

        # Optimizer with differential learning rates
        encoder_params = list(model.seq_encoder.parameters())
        if model.graph_encoder is not None:
            encoder_params += list(model.graph_encoder.parameters())
        decoder_params = list(model.decoder.parameters()) + \
                         list(model.fusion.parameters())
        conditioning_params = list(model.phylo_conditioner.parameters()) + \
                              list(model.mimicry_detector.parameters()) + \
                              list(model.safety_module.parameters()) + \
                              list(model.mic_proj.parameters()) + \
                              list(model.esm_cond_proj.parameters()) + \
                              list(model.safety_to_cond.parameters()) + \
                              list(model.amp_classifier.parameters())
        if model.encoder_fusion is not None:
            encoder_params += list(model.encoder_fusion.parameters())

        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': config.learning_rate},
            {'params': decoder_params, 'lr': config.learning_rate},
            {'params': conditioning_params, 'lr': config.learning_rate * 0.5},
            {'params': model.vector_quantizer.parameters(), 'lr': config.learning_rate * 0.1},
        ], weight_decay=config.weight_decay)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )

        self.global_step = 0

    def _fit_phylo_dim(self, embeds: torch.Tensor) -> torch.Tensor:
        """Slice or zero-pad species embeddings to config.phylo_embed_dim."""
        target = self.config.phylo_embed_dim

        if embeds.shape[-1] == target:
            return embeds

        if embeds.shape[-1] > target:
            return embeds[..., :target]

        pad = torch.zeros(*embeds.shape[:-1], target - embeds.shape[-1])
        return torch.cat([embeds, pad], dim=-1)

    def _build_phylo_input(self,
                           batch,
                           batch_size
                           ) -> Optional[torch.Tensor]:
        """Build per-sample phylogenetic input [batch, num_species, embed_dim].
        Uses the sample's target_species (via species_lookup) when available so
        each sample gets a distinct conditioning signal; otherwise falls back to
        the shared full-species panel.
        """
        names = batch.get('target_species')
        if self.species_lookup is not None and names is not None:
            embeds = []
            for name in names:
                emb = self.species_lookup.get_embedding(name) if name else None
                if emb is None:
                    emb = torch.zeros(self.config.phylo_embed_dim)
                else:
                    emb = self._fit_phylo_dim(emb)
                embeds.append(emb)
            # [batch, 1, embed_dim] — single target species per sample
            return torch.stack(embeds, dim=0).unsqueeze(1).to(self.device)

        if self.species_embeddings is not None:
            return self.species_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return None

    def _build_mic_input(self, batch):
        """Convert raw MIC to log10(MIC) conditional input; NaN where missing."""
        if 'mic' not in batch:
            return None

        mic = batch['mic'].to(self.device).float()
        mic = torch.where(mic > 0, torch.log10(mic), torch.full_like(mic, float('nan')))
        return mic

    def _build_esm_input(self, batch):
        """Look up cached ESM embeddings for the batch sequences (or None)."""
        if not self.esm_cache:
            return None

        seqs = batch.get('sequence')

        if seqs is None:
            return None

        feats = []
        ok = True

        for s in seqs:
            v = self.esm_cache.get(s)
            if v is None:
                ok = False
                break
            feats.append(v)

        if not ok:
            return None

        return torch.stack(feats, dim=0).to(self.device)

    def _build_graph_input(self, batch):
        """Move the (x, edge_index, edge_attr, batch) graph tuple onto the model
        device so the GG-FiLM encoder never mixes CPU and GPU tensors."""
        graph_data = batch.get('graph_data')

        if graph_data is None:
            return None

        return tuple(
            t.to(self.device) if isinstance(t, torch.Tensor) else t
            for t in graph_data
        )


    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}

        # Scheduled sampling
        total = max(getattr(self, '_num_epochs', 1) - 1, 1)
        tf_floor = getattr(self, '_tf_floor', 0.5)
        tf_ratio = tf_floor + (1.0 - tf_floor) * (1.0 - epoch / total)
        self.config.teacher_forcing_ratio = float(min(1.0, max(tf_floor, tf_ratio)))

        for _, batch in enumerate(dataloader):
            self.global_step += 1

            # Move batch to device
            tokens = batch['tokens'].to(self.device)
            batch_size = tokens.shape[0]
            labels = batch.get('label')
            if labels is not None:
                labels = labels.to(self.device)

            phylo_input = self._build_phylo_input(batch, batch_size)
            mic_input = self._build_mic_input(batch)
            esm_input = self._build_esm_input(batch)
            graph_data = self._build_graph_input(batch)

            # Forward pass
            output = self.model(
                tokens, graph_data=graph_data, phylo_embeddings=phylo_input,
                mic=mic_input, esm_peptide=esm_input,
            )

            # Compute losses
            targets = {'tokens': tokens, 'label': labels}
            for key in ('toxic', 'hemolytic'):
                if key in batch:
                    targets[key] = batch[key].to(self.device)

            loss_dict = self.loss_fn(output, targets)

            # Backward pass
            total_loss = loss_dict['total']
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                epoch_losses[k] = epoch_losses.get(k, 0) + v

        # Average losses
        num_batches = len(dataloader)
        epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        self.scheduler.step()

        return epoch_losses

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = {}

        for batch in dataloader:
            tokens = batch['tokens'].to(self.device)
            batch_size = tokens.shape[0]
            labels = batch.get('label')
            if labels is not None:
                labels = labels.to(self.device)

            phylo_input = self._build_phylo_input(batch, batch_size)
            mic_input = self._build_mic_input(batch)
            esm_input = self._build_esm_input(batch)
            graph_data = self._build_graph_input(batch)

            output = self.model(
                tokens, graph_data=graph_data, phylo_embeddings=phylo_input,
                mic=mic_input, esm_peptide=esm_input,
            )
            targets = {'tokens': tokens, 'label': labels}
            for key in ('toxic', 'hemolytic'):
                if key in batch:
                    targets[key] = batch[key].to(self.device)
            loss_dict = self.loss_fn(output, targets)

            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                val_losses[k] = val_losses.get(k, 0) + v

        num_batches = len(dataloader)
        val_losses = {k: v / num_batches for k, v in val_losses.items()}
        return val_losses

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_dir: str = 'checkpoints'):
        """Full training loop."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        self._num_epochs = num_epochs
        self._tf_floor = float(getattr(self.config, 'teacher_forcing_ratio', 0.5))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(num_epochs, 1)
        )

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_losses = self.train_epoch(train_loader, epoch)
            val_losses = self.validate(val_loader)

            # Codebook usage stats
            vq_stats = self.model.vector_quantizer.get_usage_stats()
            perplexity = train_losses.get('perplexity', 0)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
                f"VQ Perplexity: {perplexity:.1f} | "
                f"Codebook: {vq_stats['alive_codes']}/{self.model.vector_quantizer.num_embeddings} alive"
            )

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': self.config,
                }, os.path.join(save_dir, 'best_model.pt'))

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        return best_val_loss
