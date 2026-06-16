#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration for AutoGuard model and training.
"""

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    # Sequence parameters
    vocab_size: int = 21  # 20 amino acids + padding
    max_seq_len: int = 25
    pad_idx: int = 0

    # HydrAMP base encoder
    embedding_dim: int = 64
    gru_hidden_dim: int = 96
    latent_dim: int = 48  # shared conditioning width across all modalities

    # VQ-VAE codebook (larger codebook = more discrete motif archetypes)
    num_codebook_vectors: int = 1024
    codebook_dim: int = 48  # must equal latent_dim
    commitment_cost: float = 0.25
    ema_decay: float = 0.99  # EMA update rate for codebook (higher = slower update)

    # Graph encoder (GG-FiLM) FiLM conditioned on phylogenetic vector
    node_feature_dim: int = 36  # 8 physicochemical + 20 one-hot AA + 8 positional
    edge_feature_dim: int = 16
    graph_hidden_dim: int = 64
    graph_num_layers: int = 2
    film_hidden_dim: int = 48  # must equal phylo conditioning width (latent_dim)

    # Phylogenetic embeddings (Poincaré). A >=5-D hyperbolic space is needed to
    # embed the species tree with low distortion.
    phylo_embed_dim: int = 10
    phylo_cond_dim: int = 48  # MLP output (conditioning width = latent_dim)
    hyperbolic_curvature: float = -1.0
    num_perturbations: int = 4

    # MIC conditioning (log10 MIC fed as a conditional input)
    mic_cond_dim: int = 16

    # Molecular mimicry / ESM cross-attention module
    esm_embed_dim: int = 1280  # ESM-2 (650M) output dim
    mimicry_hidden_dim: int = 128
    mimicry_margin: float = 0.5
    temperature: float = 0.07

    # Safety module
    safety_hidden_dim: int = 64
    toxicity_threshold: float = 0.3
    hemolysis_threshold: float = 0.2

    # Multimodal fusion (lightweight cross-attention)
    fusion_heads: int = 4
    fusion_dim: int = 96
    fusion_layers: int = 1
    dropout: float = 0.1

    # Decoder (autoregressive GRU, z + condition injected every step)
    decoder_embed_dim: int = 64
    decoder_gru_dim: int = 128

    # Sparse Autoencoder (XAI)
    sae_hidden_dim: int = 256
    sae_sparsity_lambda: float = 1e-3
    sae_top_k: int = 16

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 200
    teacher_forcing_ratio: float = 0.5


@dataclass
class DataConfig:
    # Dataset paths
    grampa_path: str = "data/grampa/"
    dbaasp_path: str = "data/dbaasp/"
    dramp_path: str = "data/dramp/"
    apd_path: str = "data/apd/"
    ncbi_path: str = "data/ncbi_proteomes/"

    # Safety datasets
    toxinpred_path: str = "data/toxinpred/"
    hemopi_path: str = "data/hemopi/"

    # Mimicry datasets
    iedb_path: str = "data/iedb/"
    pdb_host_defense: str = "data/pdb_host_defense/"

    # Phylogenetic data
    species_trees_path: str = "data/species_trees/"
    tree_filter: str = "timetree*"  # glob selecting the static curated tree

    # Thresholds
    mic_threshold: float = 1.0  # ug/mL (MIC <= 1.0 = active, matches presentation)
    min_seq_len: int = 5
    max_seq_len: int = 25

    # Splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class LossWeights:
    reconstruction: float = 1.0
    vq_commitment: float = 0.25
    kl_divergence: float = 0.0  # deterministic VQ-VAE encoder
    antimicrobial_activity: float = 0.5  # binary AMP classification (supervised)
    phylo_conditioning: float = 0.15
    mimicry_penalty: float = 0.2
    safety_penalty: float = 0.15
    sparsity: float = 0.05


_SECTION_TYPES = {
    "model": ModelConfig,
    "loss_weights": LossWeights,
    "data": DataConfig,
}


def _read_config_file(path: str) -> Dict[str, Any]:
    """Read a YAML or JSON config-override file into a plain dict.
    The file may contain top-level ``model:``, ``loss_weights:`` and ``data:``
    sections, each mapping field names to values. Unknown sections/fields are
    ignored with a warning so typos do not silently change nothing.
    """
    p = Path(path)

    if not p.exists():
        logger.warning("Config override file not found: %s (using defaults)", p)
        return {}

    text = p.read_text(encoding="utf-8")

    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to read YAML config overrides. "
                "Install it or pass a .json file instead."
            ) from e
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text) if text.strip() else {}

    if not isinstance(data, dict):
        logger.warning("Config override file %s is not a mapping; ignoring.", p)
        return {}

    return data


def apply_overrides(obj: Any,
                    overrides: Optional[Dict[str, Any]],
                    section: str = "") -> Any:
    """Set valid dataclass fields on "obj" from "overrides" (in place)."""

    if not overrides:
        return obj

    valid = {f.name for f in dataclasses.fields(obj)}

    for key, value in overrides.items():
        if key in valid:
            setattr(obj, key, value)
        else:
            logger.warning(
                "Ignoring unknown %s override '%s' (not a field of %s)",
                section or type(obj).__name__, key, type(obj).__name__,
            )

    return obj


def load_experiment_config(
    path: Optional[str] = None,
    model_overrides: Optional[Dict[str, Any]] = None,
    loss_overrides: Optional[Dict[str, Any]] = None,
    data_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[ModelConfig, LossWeights, DataConfig]:
    """Build (ModelConfig, LossWeights, DataConfig) from an optional override file.
    Precedence (lowest to highest): dataclass defaults < file sections <
    explicit "*_overrides" dicts (e.g. CLI flags / hyperparameter search).
    A flat "model.<field>" / "loss_weights.<field>" / "data.<field>" key
    inside any override dict is also routed to the right section, which makes it
    convenient for hyperparameter search to use one flat namespace.
    """
    file_data: Dict[str, Any] = _read_config_file(path) if path else {}
    sections: Dict[str, Dict[str, Any]] = {name: {} for name in _SECTION_TYPES}

    for name in _SECTION_TYPES:
        sec = file_data.get(name)
        if isinstance(sec, dict):
            sections[name].update(sec)

    # Merge explicit override dicts, supporting dotted keys (model.latent_dim).
    explicit = {
        "model": dict(model_overrides or {}),
        "loss_weights": dict(loss_overrides or {}),
        "data": dict(data_overrides or {}),
    }

    for default_section, ov in explicit.items():
        for key, value in ov.items():
            if "." in key:
                sec_name, field_name = key.split(".", 1)
                if sec_name in sections:
                    sections[sec_name][field_name] = value
                    continue
            sections[default_section][key] = value

    model = apply_overrides(ModelConfig(), sections["model"], "model")
    loss = apply_overrides(LossWeights(), sections["loss_weights"], "loss_weights")
    data = apply_overrides(DataConfig(), sections["data"], "data")

    return model, loss, data
