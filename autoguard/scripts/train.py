#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main training script for AutoGuard.

Supports multi-stage training protocol:
  Stage 1: poincare   - Poincaré embedding pre-training on phylogenetic trees
  Stage 2: mimicry    - Contrastive mimicry pre-training (ESM-2 features)
  Stage 3: warmup     - Encoder warm-up (reconstruction only, no conditioning losses)
  Stage 4: full       - Full multi-task training (all losses, KL annealing)
  Stage 5: sae        - Sparse Autoencoder on frozen activations (post-hoc XAI)

Data sources:
  - AMP activity data      (from data/processed/amp_*.csv OR data/{grampa,apd,dramp,...})
  - Safety data            (from data/processed/safety_*.csv OR data/{toxinpred,hemopi})
  - Mimicry contrastive    (from data/processed/mimicry_*.fasta)
  - Phylogenetic trees     (from data/species_trees/)

Usage:
    # Run all 5 stages sequentially:
    python -m autoguard.scripts.train --data_dir data/ --stage all

    # Individual stages:
    python -m autoguard.scripts.train --data_dir data/ --stage poincare --epochs 100
    python -m autoguard.scripts.train --data_dir data/ --stage mimicry --epochs 50
    python -m autoguard.scripts.train --data_dir data/ --stage warmup --epochs 30
    python -m autoguard.scripts.train --data_dir data/ --stage full --epochs 200
    python -m autoguard.scripts.train --data_dir data/ --stage sae --epochs 50

    # Quick smoke-test (full stage, 2 epochs):
    python -m autoguard.scripts.train --data_dir data/ --stage full --epochs 2 --batch_size 8
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autoguard.config import ModelConfig, DataConfig, LossWeights, load_experiment_config
from autoguard.models.autoguard_model import AutoGuardModel
from autoguard.data.datasets import (
    AMPDataset, CombinedAMPDataset, MimicryDataset,
    tokenize_sequence, parse_fasta, filter_sequences, make_amp_collate,
)
from autoguard.training.trainer import AutoGuardTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STAGES = ('poincare', 'mimicry', 'warmup', 'full', 'sae', 'all')


def _build_configs(args, **model_cli):
    """Build (ModelConfig, LossWeights) for a stage.
    Loads architecture / loss-weight overrides from the optional ``--model_config``
    YAML/JSON file, then applies CLI overrides on top (training schedule from the
    Snakemake workflow always wins). ``model_cli`` are per-stage ModelConfig
    overrides such as batch_size / learning_rate / num_epochs.
    """
    loss_cli = {}

    if getattr(args, 'mimicry_weight', None) is not None:
        loss_cli['mimicry_penalty'] = args.mimicry_weight
    if getattr(args, 'safety_weight', None) is not None:
        loss_cli['safety_penalty'] = args.safety_weight

    model_config, loss_weights, _ = load_experiment_config(
        getattr(args, 'model_config', '') or None,
        model_overrides={k: v for k, v in model_cli.items() if v is not None},
        loss_overrides=loss_cli,
    )
    return model_config, loss_weights


def _set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"  Global seed set to {seed}")


def _load_processed_csv(path: Path):
    """Load a processed CSV into lists."""
    sequences, labels, mics, species = [], [], [], []

    if not path.exists():
        return sequences, labels, mics, species

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row["sequence"])
            labels.append(int(float(row.get("label", 1))))
            mic_val = row.get("mic", "")
            mics.append(float(mic_val) if mic_val else None)
            species.append(row.get("target_species", ""))

    return sequences, labels, mics, species


def _load_mimicry_fasta(path: Path) -> list:
    """Load sequences from a FASTA file."""
    if not path.exists():
        return []

    seqs = parse_fasta(str(path))
    return [s for _, s in filter_sequences(seqs, 5, 50)]


def _load_or_build_esm_cache(args, data_dir, sequences) -> dict:
    """Load a precomputed ESM embedding cache, or build it if requested.
    Returns a dict {sequence: tensor[esm_dim]}. Returns an empty dict (the ESM
    cross-attention branch is then skipped) when no cache exists and building is
    not requested — e.g. on a CPU smoke test where running ESM-2 is impractical.
    """
    cache_path = Path(getattr(args, 'esm_cache', '') or (data_dir / "embeddings" / "esm_cache.pt"))
    if cache_path.exists():
        try:
            cache = torch.load(str(cache_path), map_location='cpu', weights_only=False)
            if isinstance(cache, dict):
                return cache
        except Exception as e:  # pragma: no cover
            logger.warning(f"  Failed to load ESM cache {cache_path}: {e}")

    if not getattr(args, 'build_esm', False):
        return {}

    # Build cache with ESM-2
    try:
        from autoguard.models.mimicry_module import ESMFeatureExtractor
    except Exception as e:
        logger.warning(f"  Cannot build ESM cache: {e}")
        return {}

    extractor = ESMFeatureExtractor(device=args.device)
    uniq = sorted(set(s for s in sequences if s))
    logger.info(f"  Building ESM cache for {len(uniq)} unique sequences on {args.device}...")
    
    if str(args.device) == 'cpu':
        est_min = len(uniq) / 2.0 / 60.0
        logger.warning(
            f"  ESM-2 (650M) is running on CPU \u2014 this is slow "
            f"(~1-3 seq/s, est. ~{est_min:.0f} min for {len(uniq)} sequences). "
            f"Pass --device cuda for a ~10-50x speed-up."
        )
    cache = {}
    batch = 16
    start = time.time()

    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i + batch]
        embeds = extractor(chunk).cpu()

        for s, e in zip(chunk, embeds):
            cache[s] = e

        done = min(i + batch, len(uniq))

        if done % (batch * 20) == 0 or done == len(uniq):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (len(uniq) - done) / rate if rate > 0 else 0.0
            logger.info(
                f"    ESM cache {done}/{len(uniq)} "
                f"({rate:.1f} seq/s, ETA {eta/60:.1f} min)"
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, str(cache_path))
    logger.info(f"  Saved ESM cache: {cache_path} (took {(time.time()-start)/60:.1f} min)")
    return cache


def _run_stage_poincare(args, data_dir):
    """Stage 1: Train Poincaré embeddings on phylogenetic tree structure."""
    from autoguard.data.phylo_data import PhylogeneticDataProcessor, PoincareEmbeddingTrainer

    trees_dir = data_dir / "species_trees"
    if not trees_dir.exists() or not list(trees_dir.glob("*.*")):
        logger.error("No phylogenetic trees found in data/species_trees/")
        logger.error("Run download_data.py --auto first, or copy trees manually.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("STAGE 1: Poincaré Embedding Pre-training")
    logger.info("=" * 60)

    processor = PhylogeneticDataProcessor(
        trees_dir=str(trees_dir),
        num_perturbations=10,
        perturbation_scale=0.1,
        tree_filter=getattr(args, 'tree_filter', ''),
    )
    data = processor.prepare_training_data()
    logger.info(f"  Tree pairs: {len(data['parents'])}, entities: {data['num_entities']}")

    trainer = PoincareEmbeddingTrainer(
        num_entities=data['num_entities'],
        embed_dim=args.phylo_dim,
        # Adam fits the distance-distortion objective; 0.05 is its working LR.
        learning_rate=args.lr if args.lr != 1e-4 else 0.05,
        num_negatives=10,
    )

    epochs = args.epochs if args.epochs != 200 else 2000
    trainer.train(data, epochs=epochs, verbose=True)

    save_path = Path(args.save_dir) / "poincare_trained.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(save_path))
    logger.info(f"  Saved: {save_path}")
    return str(save_path)


def _run_stage_mimicry(args, data_dir, model=None):
    """Stage 2: Contrastive mimicry pre-training with ESM-2 features."""
    from autoguard.training.contrastive import ContrastiveTrainer

    processed_dir = data_dir / "processed"
    mimicry_pos = _load_mimicry_fasta(processed_dir / "mimicry_positives.fasta")
    mimicry_neg = _load_mimicry_fasta(processed_dir / "mimicry_negatives.fasta")

    if not mimicry_pos or not mimicry_neg:
        logger.warning("No mimicry data found. Skipping Stage 2.")
        logger.warning("  Expected: data/processed/mimicry_positives.fasta")
        return None

    logger.info("=" * 60)
    logger.info("STAGE 2: Contrastive Mimicry Pre-training")
    logger.info("=" * 60)
    logger.info(f"  Defense peptides (positives): {len(mimicry_pos)}")
    logger.info(f"  Autoantigens (negatives): {len(mimicry_neg)}")

    # Load some AMP sequences as anchors
    train_seqs, _, _, _ = _load_processed_csv(processed_dir / "amp_train.csv")

    if not train_seqs:
        logger.error("No AMP training data. Run prepare_data.py first.")
        sys.exit(1)

    anchor_seqs = train_seqs[:500]  # Use subset as anchors


    augmented_pos = list(mimicry_pos)
    natural_amps = [s for s in train_seqs[:2000] if len(s) >= 10]

    import random as _rng

    _rng.seed(42)
    _rng.shuffle(natural_amps)
    augmented_pos.extend(natural_amps[:468])  # Total positives = 500
    logger.info(f"  Augmented positives: {len(mimicry_pos)} defensins + "
                f"{len(augmented_pos) - len(mimicry_pos)} natural AMPs = {len(augmented_pos)} total")

    if model is None:
        model_config, _ = _build_configs(args)
        model = AutoGuardModel(model_config, use_graph_encoder=False)

    mimicry_dataset = MimicryDataset(
        peptides=anchor_seqs,
        defense_peptides=augmented_pos,
        autoantigens=mimicry_neg[:500],  # Balanced: 500 pos vs 500 neg
    )

    contrastive_trainer = ContrastiveTrainer(
        mimicry_detector=model.mimicry_detector,
        learning_rate=args.lr if args.lr != 1e-4 else 5e-4,
        temperature=0.07,
        margin=0.5,
        device=args.device,
    )

    mimicry_loader = DataLoader(
        mimicry_dataset, batch_size=min(args.batch_size, 32),
        shuffle=True, num_workers=0,
    )

    epochs = args.epochs if args.epochs != 200 else 50
    for epoch in range(epochs):
        losses = contrastive_trainer.train_epoch(mimicry_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: contrastive_loss={losses['contrastive']:.4f}")

    save_path = Path(args.save_dir) / "mimicry_pretrained.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.mimicry_detector.state_dict(), str(save_path))
    logger.info(f"  Saved: {save_path}")
    return model


def _run_stage_warmup(args, data_dir, model=None):
    """Stage 3: Encoder warm-up — reconstruction only, no conditioning losses."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Encoder Warm-up (reconstruction only)")
    logger.info("=" * 60)

    processed_dir = data_dir / "processed"
    train_seqs, train_labels, train_mics, _ = _load_processed_csv(processed_dir / "amp_train.csv")
    val_seqs, val_labels, val_mics, _ = _load_processed_csv(processed_dir / "amp_val.csv")

    if not train_seqs:
        logger.error("No AMP training data. Run prepare_data.py first.")
        sys.exit(1)

    cap = getattr(args, 'max_train', 0)

    if cap and cap > 0:
        train_seqs, train_labels, train_mics = train_seqs[:cap], train_labels[:cap], train_mics[:cap]
        logger.info(f"  Capped training set to {len(train_seqs)} sequences (--max_train)")

    train_dataset = AMPDataset(train_seqs, train_labels, train_mics)
    val_dataset = AMPDataset(val_seqs, val_labels, val_mics)

    collate = make_amp_collate(use_graph=args.use_graph)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate,
        shuffle=True, num_workers=0 if args.device == 'cpu' else 2,
        pin_memory=(args.device != 'cpu'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate,
        shuffle=False, num_workers=0 if args.device == 'cpu' else 2,
        pin_memory=(args.device != 'cpu'),
    )

    # Warm-up: reconstruction + VQ only (no conditioning losses)
    warmup_weights = LossWeights(
        reconstruction=1.0,
        vq_commitment=0.25,
        kl_divergence=0.0,
        antimicrobial_activity=0.0,
        phylo_conditioning=0.0,
        mimicry_penalty=0.0,
        safety_penalty=0.0,
        sparsity=0.0,
    )

    model_config, _ = _build_configs(
        args,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )

    if model is None:
        model = AutoGuardModel(model_config, use_graph_encoder=args.use_graph)

    logger.info(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    logger.info(f"  Losses: reconstruction + VQ only (mimicry=0, safety=0)")

    epochs = args.epochs if args.epochs != 200 else 30
    trainer = AutoGuardTrainer(model, model_config, warmup_weights, device=args.device)
    best_loss = trainer.train(train_loader, val_loader, epochs, args.save_dir + "/warmup")

    logger.info(f"  Warm-up complete. Best val loss: {best_loss:.4f}")
    return model


def _run_stage_full(args, data_dir, model=None):
    """Stage 4: Full multi-task training with all losses active."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Full Multi-task Training")
    logger.info("=" * 60)

    processed_dir = data_dir / "processed"

    # Load AMP data
    if (processed_dir / "amp_train.csv").exists():
        train_seqs, train_labels, train_mics, train_species = _load_processed_csv(processed_dir / "amp_train.csv")
        val_seqs, val_labels, val_mics, val_species = _load_processed_csv(processed_dir / "amp_val.csv")
        cap = getattr(args, 'max_train', 0)

        if cap and cap > 0:
            train_seqs, train_labels = train_seqs[:cap], train_labels[:cap]
            train_mics, train_species = train_mics[:cap], train_species[:cap]
            logger.info(f"  Capped training set to {len(train_seqs)} sequences (--max_train)")

        train_dataset = AMPDataset(train_seqs, train_labels, train_mics, train_species)
        val_dataset = AMPDataset(val_seqs, val_labels, val_mics, val_species)
        logger.info(f"  Loaded from processed CSVs: train={len(train_dataset)}, val={len(val_dataset)}")
        n_with_species = sum(1 for s in train_species if s)
        logger.info(f"  Species-labeled samples: {n_with_species}/{len(train_species)} "
                    f"({100*n_with_species/max(len(train_species),1):.1f}%)")
    else:
        train_dataset = CombinedAMPDataset(str(data_dir), split='train')
        val_dataset = CombinedAMPDataset(str(data_dir), split='val')
        logger.info(f"  Loaded from raw dirs: train={len(train_dataset)}, val={len(val_dataset)}")

    if len(train_dataset) == 0:
        logger.error("No training data found! Run prepare_data.py first.")
        sys.exit(1)

    collate = make_amp_collate(use_graph=args.use_graph)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate,
        shuffle=True, num_workers=0 if args.device == 'cpu' else 2,
        pin_memory=(args.device != 'cpu'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate,
        shuffle=False, num_workers=0 if args.device == 'cpu' else 2,
        pin_memory=(args.device != 'cpu'),
    )

    # Mimicry & phylo info
    mimicry_pos = _load_mimicry_fasta(processed_dir / "mimicry_positives.fasta")
    mimicry_neg = _load_mimicry_fasta(processed_dir / "mimicry_negatives.fasta")
    trees_dir = data_dir / "species_trees"
    n_trees = len(list(trees_dir.glob("*.*"))) if trees_dir.exists() else 0

    # Load Poincaré embeddings for species conditioning
    from autoguard.data.phylo_data import SpeciesEmbeddingLookup
    poincare_path = Path(args.save_dir) / "poincare_trained.npz"
    if not poincare_path.exists():
        # Try alternative locations
        alt_paths = [
            Path(args.save_dir) / "poincare_embeddings.pt",
            Path(args.save_dir) / "poincare" / "poincare_trained.npz",
            data_dir / "embeddings" / "poincare_embeddings.pt",
            data_dir / "embeddings" / "timetree_embeddings.npz",
        ]
        for ap in alt_paths:
            if ap.exists():
                poincare_path = ap
                break

    species_lookup = None
    species_embeds_all = None

    try:
        species_lookup = SpeciesEmbeddingLookup(
            poincare_checkpoint=str(poincare_path),
            trees_dir=str(trees_dir),
            embed_dim=args.phylo_dim,
            tree_filter=getattr(args, 'tree_filter', ''),
        )
        species_embeds_all = species_lookup.get_all_species_embeddings()
        logger.info(f"  Phylo: {n_trees} trees, {species_embeds_all.shape[0]} species embeddings "
                    f"(dim={species_embeds_all.shape[1]})")
    except Exception as e:
        logger.warning(f"  Phylo conditioning disabled (could not load embeddings: {e}). "
                       f"Run --stage poincare first for species-aware conditioning.")
        species_lookup = None
        species_embeds_all = None

    logger.info(f"  Mimicry: +{len(mimicry_pos)} / -{len(mimicry_neg)}")

    # Loss weights
    _, loss_weights = _build_configs(
        args,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )
    logger.info(f"  Loss weights: mimicry={loss_weights.mimicry_penalty}, safety={loss_weights.safety_penalty}")

    # Model
    model_config, _ = _build_configs(
        args,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )

    if model is None:
        model = AutoGuardModel(model_config, use_graph_encoder=args.use_graph)

    # Try loading warm-up checkpoint
    warmup_path = Path(args.save_dir) / "warmup" / "best_model.pt"
    if warmup_path.exists() and args.resume is None:
        checkpoint = torch.load(str(warmup_path), map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"  Loaded warm-up checkpoint: {warmup_path}")

    # Try loading mimicry pre-trained weights
    mimicry_path = Path(args.save_dir) / "mimicry_pretrained.pt"
    if mimicry_path.exists():
        mimicry_state = torch.load(str(mimicry_path), map_location=args.device, weights_only=False)
        model.mimicry_detector.load_state_dict(mimicry_state)
        logger.info(f"  Loaded mimicry weights: {mimicry_path}")

    # Resume from explicit checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"  Resumed from {args.resume}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model: {total_params:,} params ({trainable_params:,} trainable), "
                f"graph_encoder={'ON' if args.use_graph else 'OFF'}")

    # ESM embedding cache for the cross-attention branch
    esm_cache = _load_or_build_esm_cache(args, data_dir, train_seqs + val_seqs)
    logger.info(f"  ESM cache: {len(esm_cache)} sequences"
                f"{' (disabled)' if not esm_cache else ''}")

    trainer = AutoGuardTrainer(model, model_config, loss_weights, device=args.device,
                               species_embeddings=species_embeds_all,
                               species_lookup=species_lookup, esm_cache=esm_cache)
    best_loss = trainer.train(train_loader, val_loader, args.epochs, args.save_dir)

    logger.info(f"  Full training complete. Best val loss: {best_loss:.4f}")
    logger.info(f"  Checkpoint: {args.save_dir}/best_model.pt")
    return model


def _run_stage_sae(args, data_dir, model=None):
    """Stage 5: Train Sparse Autoencoder on frozen model activations."""
    from autoguard.models.sparse_autoencoder import SparseAutoencoder

    logger.info("=" * 60)
    logger.info("STAGE 5: Sparse Autoencoder Training (post-hoc XAI)")
    logger.info("=" * 60)

    processed_dir = data_dir / "processed"
    model_config, _ = _build_configs(args)

    # Load or create model
    if model is None:
        model = AutoGuardModel(model_config, use_graph_encoder=args.use_graph)
        model_path = Path(args.save_dir) / "best_model.pt"
        if model_path.exists():
            checkpoint = torch.load(str(model_path), map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"  Loaded model from {model_path}")
        else:
            logger.error(f"  No trained model found at {model_path}")
            logger.error("  Run --stage full first.")
            sys.exit(1)

    model = model.to(args.device)
    model.eval()

    # Collect activations from the fusion layer
    train_seqs, _, _, _ = _load_processed_csv(processed_dir / "amp_train.csv")
    if not train_seqs:
        logger.error("No training data. Run prepare_data.py first.")
        sys.exit(1)

    logger.info(f"  Collecting fusion activations from {min(len(train_seqs), 5000)} sequences...")
    activations = []
    batch_size = args.batch_size
    seqs_to_use = train_seqs[:5000]

    with torch.no_grad():
        for i in range(0, len(seqs_to_use), batch_size):
            batch_seqs = seqs_to_use[i:i+batch_size]
            tokens = torch.stack([tokenize_sequence(s) for s in batch_seqs]).to(args.device)
            output = model(tokens)
            # Extract fusion output (pre-decoder representation)
            if 'fused_representation' in output:
                activations.append(output['fused_representation'].cpu())
            else:
                # Fallback: use quantized representation
                activations.append(output['quantized'].cpu())

    all_activations = torch.cat(activations, dim=0)
    act_dim = all_activations.shape[-1]
    # Flatten if 3D [batch, seq, dim] -> [batch*seq, dim]

    if all_activations.dim() == 3:
        all_activations = all_activations.reshape(-1, act_dim)

    logger.info(f"  Collected {all_activations.shape[0]} activation vectors, dim={act_dim}")

    # Train SAE
    sae = SparseAutoencoder(
        input_dim=act_dim,
        hidden_dim=act_dim * 2,
        sparsity_lambda=model_config.sae_sparsity_lambda,
        top_k=model_config.sae_top_k,
    ).to(args.device)

    sae_optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr if args.lr != 1e-4 else 1e-3)
    sae_dataset = TensorDataset(all_activations)
    sae_loader = DataLoader(sae_dataset, batch_size=256, shuffle=True)

    epochs = args.epochs if args.epochs != 200 else 50
    best_sae_loss = float('inf')
    save_dir = Path(args.save_dir) / "sae"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        sae.train()
        epoch_loss = 0
        for (batch,) in sae_loader:
            batch = batch.to(args.device)
            _, _, loss_dict = sae(batch)
            loss = loss_dict['total']
            sae_optimizer.zero_grad()
            loss.backward()
            sae_optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(sae_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            dead = sae.get_dead_features().sum().item()
            logger.info(f"  SAE epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, "
                       f"dead_features={dead}/{sae.hidden_dim}")

        if epoch_loss < best_sae_loss:
            best_sae_loss = epoch_loss
            torch.save(sae.state_dict(), str(save_dir / "best_sae.pt"))

    logger.info(f"  SAE training complete. Best loss: {best_sae_loss:.4f}")
    logger.info(f"  Saved: {save_dir}/best_sae.pt")

    # Write SAE stats for the comparison/report step.
    dead = int(sae.get_dead_features().sum().item())
    sae_stats = {
        "best_loss": float(best_sae_loss),
        "hidden_dim": int(sae.hidden_dim),
        "input_dim": int(act_dim),
        "dead_features": dead,
        "alive_features": int(sae.hidden_dim) - dead,
        "sparsity_lambda": float(model_config.sae_sparsity_lambda),
        "top_k": int(model_config.sae_top_k),
        "num_activations": int(all_activations.shape[0]),
        "seed": int(getattr(args, "seed", 42)),
    }

    with open(save_dir / "sae_stats.json", "w", encoding="utf-8") as f:
        json.dump(sae_stats, f, indent=2)

    logger.info(f"  Saved: {save_dir}/sae_stats.json")
    return sae


def main():
    parser = argparse.ArgumentParser(description='Train AutoGuard model (multi-stage)')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help='Checkpoint directory')
    parser.add_argument('--stage', type=str, default='full', choices=STAGES,
                        help='Training stage: poincare, mimicry, warmup, full, sae, or all')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--phylo_dim', type=int, default=10, help='Poincaré embedding dimension (>=5 needed for low tree-distortion)')
    parser.add_argument('--tree_filter', type=str, default='', help='Glob to select trees, e.g. "timetree*" for single tree')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--use_graph', action='store_true', help='Enable GG-FiLM graph encoder')
    parser.add_argument('--esm_cache', type=str, default='', help='Path to precomputed ESM embedding cache (.pt dict)')
    parser.add_argument('--build_esm', action='store_true', help='Build ESM cache with ESM-2 (requires fair-esm + GPU)')
    parser.add_argument('--mimicry_weight', type=float, default=None, help='Override mimicry loss weight')
    parser.add_argument('--safety_weight', type=float, default=None, help='Override safety loss weight')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility')
    parser.add_argument('--max_train', type=int, default=0, help='Cap number of training sequences (0 = use all)')
    parser.add_argument('--model_config', type=str, default='',
                        help='Path to a YAML/JSON file with model:/loss_weights:/data: overrides '
                             '(full control of every ModelConfig/LossWeights field)')
    args = parser.parse_args()

    _set_global_seed(args.seed)

    data_dir = Path(args.data_dir)
    logger.info(f"AutoGuard Training | Stage: {args.stage} | Device: {args.device}")

    if args.stage == 'all':
        # Run all 5 stages sequentially
        logger.info("Running ALL stages sequentially:")
        _run_stage_poincare(args, data_dir)
        model = _run_stage_mimicry(args, data_dir)
        model = _run_stage_warmup(args, data_dir, model)

        # Reset epochs for full training
        orig_epochs = args.epochs
        if args.epochs == 200:
            pass

        model = _run_stage_full(args, data_dir, model)
        _run_stage_sae(args, data_dir, model)
        logger.info("=" * 60)
        logger.info("ALL STAGES COMPLETE")
        logger.info("=" * 60)

    elif args.stage == 'poincare':
        _run_stage_poincare(args, data_dir)

    elif args.stage == 'mimicry':
        _run_stage_mimicry(args, data_dir)

    elif args.stage == 'warmup':
        _run_stage_warmup(args, data_dir)

    elif args.stage == 'full':
        _run_stage_full(args, data_dir)

    elif args.stage == 'sae':
        _run_stage_sae(args, data_dir)

    logger.info("Done.")


if __name__ == '__main__':
    main()
