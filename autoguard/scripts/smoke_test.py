#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end smoke test for AutoGuard.

Validates that the entire pipeline (data loading, model creation, training,
generation, evaluation, comparison) works correctly on a small synthetic
dataset. This test is designed to run on CPU in under 30 seconds.

Usage:
    python -m autoguard.scripts.smoke_test
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    t0 = time.time()
    device = "cpu"
    passed = 0
    failed = 0
    errors = []

    # ====================================================================
    # Config & data utilities
    # ====================================================================
    try:
        from autoguard.config import ModelConfig, DataConfig, LossWeights
        from autoguard.data.datasets import (
            tokenize_sequence,
            detokenize_sequence,
            parse_fasta,
            filter_sequences,
        )
        from autoguard.utils.amino_acids import compute_peptide_features

        cfg = ModelConfig()
        assert cfg.vocab_size == 21
        tok = tokenize_sequence("KLKLK")
        assert tok.shape == (25,)
        assert detokenize_sequence(tok) == "KLKLK"
        feats = compute_peptide_features("KLKLK")
        assert "molecular_weight" in feats
        passed += 1
        logger.info("[PASS] Config & data utilities")
    except Exception as e:
        failed += 1
        errors.append(f"Config & data utilities: {e}")
        logger.error(f"[FAIL] Config & data utilities: {e}")

    # ====================================================================
    # Encoder/Decoder (HydrAMP base)
    # ====================================================================
    try:
        from autoguard.models.hydramp_base import HydrAMPEncoder, HydrAMPDecoder

        enc = HydrAMPEncoder()
        dec = HydrAMPDecoder()
        x = torch.randint(1, 21, (2, 25))
        mu, _ = enc(x)
        assert mu.shape == (2, 64)
        condition = torch.zeros(2, 2)
        logits = dec(mu, condition=condition)
        assert logits.shape == (2, 25, 21)
        passed += 1
        logger.info("[PASS] HydrAMP Encoder/Decoder")
    except Exception as e:
        failed += 1
        errors.append(f"Encoder/Decoder: {e}")
        logger.error(f"[FAIL] HydrAMP Encoder/Decoder: {e}")

    # ====================================================================
    # VQ-VAE
    # ====================================================================
    try:
        from autoguard.models.vqvae import VectorQuantizer

        vq = VectorQuantizer(num_embeddings=64, embedding_dim=64)
        z = torch.randn(4, 64)
        quantized, loss, indices, perp = vq(z)
        assert quantized.shape == z.shape
        assert loss.item() >= 0
        assert indices.shape == (4,)
        passed += 1
        logger.info("[PASS] VQ-VAE")
    except Exception as e:
        failed += 1
        errors.append(f"VQ-VAE: {e}")
        logger.error(f"[FAIL] VQ-VAE: {e}")

    # ====================================================================
    # Phylogenetic embeddings
    # ====================================================================
    try:
        from autoguard.models.phylo_embeddings import (
            PhylogeneticConditioner,
            PoincareEmbedding,
            poincare_distance,
        )

        pe = PoincareEmbedding(num_entities=10, embed_dim=64)
        embeds = pe(torch.tensor([0, 1, 2]))
        assert embeds.shape == (3, 64)
        d = poincare_distance(embeds[0:1], embeds[1:2])
        assert d.shape == (1, 1)

        cond = PhylogeneticConditioner(embed_dim=64, output_dim=64)
        species = torch.randn(2, 5, 64) * 0.01
        out = cond(species)
        assert out.shape == (2, 64)
        passed += 1
        logger.info("[PASS] Phylogenetic embeddings")
    except Exception as e:
        failed += 1
        errors.append(f"Phylogenetic embeddings: {e}")
        logger.error(f"[FAIL] Phylogenetic embeddings: {e}")

    # ====================================================================
    # Safety module
    # ====================================================================
    try:
        from autoguard.models.safety_module import SafetyModule

        sm = SafetyModule(input_dim=64, hidden_dim=128)
        out = sm(torch.randn(4, 64))
        assert "safety_score" in out
        assert out["toxicity"].shape == (4, 1)
        passed += 1
        logger.info("[PASS] Safety module")
    except Exception as e:
        failed += 1
        errors.append(f"Safety module: {e}")
        logger.error(f"[FAIL] Safety module: {e}")

    # ====================================================================
    # Full AutoGuard model (no graph encoder)
    # ====================================================================
    try:
        from autoguard.models.autoguard_model import AutoGuardModel

        config = ModelConfig()
        model = AutoGuardModel(config, use_graph_encoder=False).to(device)
        x = torch.randint(1, 21, (2, 25))
        output = model(x)
        assert "logits" in output
        assert output["logits"].shape == (2, 25, 21)
        assert output["amp_prediction"].shape == (2, 1)
        assert output["safety"]["safety_score"].shape == (2, 1)
        passed += 1
        logger.info("[PASS] AutoGuard model (forward)")
    except Exception as e:
        failed += 1
        errors.append(f"AutoGuard forward: {e}")
        logger.error(f"[FAIL] AutoGuard forward: {e}")

    # ====================================================================
    # Loss computation
    # ====================================================================
    try:
        from autoguard.training.losses import AutoGuardLoss

        loss_fn = AutoGuardLoss(LossWeights())
        targets = {"tokens": x, "label": torch.ones(2, 1)}
        ld = loss_fn(output, targets, kl_weight=0.01)
        assert "total" in ld
        assert ld["total"].requires_grad
        passed += 1
        logger.info("[PASS] Loss computation")
    except Exception as e:
        failed += 1
        errors.append(f"Loss: {e}")
        logger.error(f"[FAIL] Loss: {e}")

    # ====================================================================
    # Training step (backprop)
    # ====================================================================
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        output = model(x)
        targets = {"tokens": x, "label": torch.ones(2, 1)}
        ld = loss_fn(output, targets, kl_weight=0.01)
        ld["total"].backward()
        optimizer.step()
        passed += 1
        logger.info("[PASS] Training step (backward + optimizer)")
    except Exception as e:
        failed += 1
        errors.append(f"Training step: {e}")
        logger.error(f"[FAIL] Training step: {e}")

    # ====================================================================
    # Generation
    # ====================================================================
    try:
        sequences, _ = model.generate(
            num_samples=5, temperature=0.8, safety_threshold=0.9, max_attempts=100
        )
        assert len(sequences) <= 5
        passed += 1
        logger.info(f"[PASS] Generation ({len(sequences)} seqs)")
    except Exception as e:
        failed += 1
        errors.append(f"Generation: {e}")
        logger.error(f"[FAIL] Generation: {e}")

    # ====================================================================
    # Evaluation metrics
    # ====================================================================
    try:
        from autoguard.evaluation.metrics import AMPMetrics

        metrics = AMPMetrics(training_sequences=set(["KLKLK", "RWRWR"]))
        gen_seqs = ["GIGKFLHSAK", "RLYLRIGRR", "KLKLK"]
        result = metrics.evaluate_batch(gen_seqs, amp_scores=[0.8, 0.9, 0.7])
        assert "novelty" in result
        assert "diversity" in result
        assert result["novelty"] > 0
        passed += 1
        logger.info("[PASS] Evaluation metrics")
    except Exception as e:
        failed += 1
        errors.append(f"Metrics: {e}")
        logger.error(f"[FAIL] Metrics: {e}")

    # ====================================================================
    # GNN baseline
    # ====================================================================
    try:
        from autoguard.models.gnn_baseline import GNNGenerator

        gnn = GNNGenerator(max_len=25, latent_dim=16, hidden_dim=32, num_gnn_layers=2)
        gnn_out = gnn(torch.randint(1, 21, (2, 25)))
        assert gnn_out["logits"].shape == (2, 25, 21)
        seqs, scores = gnn.generate(5, device="cpu")
        assert len(seqs) == 5
        passed += 1
        logger.info("[PASS] GNN baseline")
    except Exception as e:
        failed += 1
        errors.append(f"GNN baseline: {e}")
        logger.error(f"[FAIL] GNN baseline: {e}")

    # ====================================================================
    # HydrAMP adapter
    # ====================================================================
    try:
        from autoguard.evaluation.hydramp_adapter import HydrAMPBaseline

        hamp = HydrAMPBaseline(device="cpu")
        hamp.train(["KLKLKLKLK", "RWRWRWRWR", "GIGKFLHSAK"], epochs=1, batch_size=2)
        seqs, _ = hamp.generate(5)
        assert len(seqs) == 5
        passed += 1
        logger.info("[PASS] HydrAMP adapter")
    except Exception as e:
        failed += 1
        errors.append(f"HydrAMP adapter: {e}")
        logger.error(f"[FAIL] HydrAMP adapter: {e}")

    # ====================================================================
    # Comparison
    # ====================================================================
    try:
        from autoguard.scripts.compare_models import run_comparison

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        results = run_comparison(mode="smoke", device="cpu", output=out_path)
        assert len(results) == 3
        assert any(r.get("name") == "AutoGuard" for r in results)
        os.unlink(out_path)
        passed += 1
        logger.info("[PASS] Model comparison (smoke)")
    except Exception as e:
        failed += 1
        errors.append(f"Comparison: {e}")
        logger.error(f"[FAIL] Comparison: {e}")

    # ====================================================================
    # Sparse Autoencoder
    # ====================================================================
    try:
        from autoguard.models.sparse_autoencoder import SparseAutoencoder

        sae = SparseAutoencoder(input_dim=64, hidden_dim=128, top_k=8)
        rec, feats, loss_d = sae(torch.randn(4, 64))
        assert rec.shape == (4, 64)
        assert feats.shape == (4, 128)
        assert "total" in loss_d
        passed += 1
        logger.info("[PASS] Sparse Autoencoder")
    except Exception as e:
        failed += 1
        errors.append(f"SAE: {e}")
        logger.error(f"[FAIL] SAE: {e}")

    # ====================================================================
    # Summary
    # ====================================================================
    elapsed = time.time() - t0
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"SMOKE TEST RESULTS: {passed}/{total} passed in {elapsed:.1f}s")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e}")
    print(f"{'='*50}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
