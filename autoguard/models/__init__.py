#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .hydramp_base import HydrAMPEncoder, HydrAMPDecoder, HydrAMPGRU
from .phylo_embeddings import PoincareEmbedding, PhylogeneticConditioner
from .mimicry_module import MolecularMimicryDetector, ContrastiveMimicryLoss
from .safety_module import SafetyModule
from .fusion import MultimodalFusion
from .vqvae import VectorQuantizer
from .sparse_autoencoder import SparseAutoencoder
from .autoguard_model import AutoGuardModel

# Graph encoder requires torch_geometric
try:
    from .graph_encoder import GGFiLMEncoder, PeptideGraphBuilder, is_pyg_available
except ImportError:  # pragma: no cover
    GGFiLMEncoder = None  # type: ignore
    PeptideGraphBuilder = None  # type: ignore

    def is_pyg_available() -> bool:  # type: ignore
        return False
