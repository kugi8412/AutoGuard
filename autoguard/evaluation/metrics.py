#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation metrics for AMP generation quality.
"""

import torch
import numpy as np
from typing import List, Dict, Set
from collections import Counter


# ===========================================================================
# AMP-challenge-2027 constants (see amp-challenge-2027-main/README.md)
# ===========================================================================
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")  # 20 standard proteinogenic amino acids
CHALLENGE_MIN_LEN = 8                       # sequences must be 8..50 residues
CHALLENGE_MAX_LEN = 50
POTENCY_THRESHOLD_UM = 16.0                 # MIC <= 16 uM counts as a "hit"
MIC_ASSAY_LIMIT_UM = 64.0                   # highest tested MIC (>64 uM = inactive)
HC50_ASSAY_LIMIT_UM = 128.0                 # hemolysis assay limit
TOP_IDENTITY_MAX = 0.80                     # top-100 must be <80% identical to reference



def compute_novelty(generated: List[str], training_set: Set[str]) -> float:
    """Fraction of generated sequences not in training set."""
    if not generated:
        return 0.0
    novel = sum(1 for seq in generated if seq not in training_set)
    return novel / len(generated)


def compute_diversity(sequences: List[str]) -> float:
    """Internal diversity of generated sequences (1 - avg pairwise similarity)."""
    if len(sequences) < 2:
        return 0.0

    def levenshtein_ratio(s1, s2):
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = _levenshtein(s1, s2)
        return 1.0 - distance / max_len

    similarities = []
    n = min(len(sequences), 500)  # Cap for computational efficiency

    for i in range(n):
        for j in range(i + 1, min(i + 50, n)):
            similarities.append(levenshtein_ratio(sequences[i], sequences[j]))

    return 1.0 - np.mean(similarities) if similarities else 0.0


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def amp_score_to_predicted_mic(amp_score: float) -> float:
    """Map a model AMP probability (0..1) to a predicted MIC in uM.

    This is a model-derived simple used only for reporting challenge-style
    statistics (the challenge values come from wet-lab assays). A higher AMP
    probability maps to a lower predicted MIC, capped at the 64 uM assay limit.
    """
    amp_score = float(min(max(amp_score, 0.0), 1.0))
    mic = MIC_ASSAY_LIMIT_UM * (1.0 - amp_score)
    return float(max(0.5, min(MIC_ASSAY_LIMIT_UM, mic)))


def hemolysis_to_predicted_hc50(hemolysis: float) -> float:
    """Map a predicted hemolysis probability (0..1) to a predicted HC50 in uM.
    Lower hemolysis -> higher (safer) HC50, capped at the 128 uM assay limit.
    Reporting proxy only.
    """
    hemolysis = float(min(max(hemolysis, 0.0), 1.0))
    hc50 = HC50_ASSAY_LIMIT_UM * (1.0 - hemolysis)
    return float(max(0.5, min(HC50_ASSAY_LIMIT_UM, hc50)))


def _percentile(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def challenge_sequence_validity(sequences: List[str],
                                reference_set: Set[str] = None) -> Dict:
    """Structural compliance with the AMP-challenge-2027 sequence rules.
    Checks length (8..50), canonical-only alphabet, uniqueness, and overlap
    with the known-antibacterial reference set.
    """
    reference_set = reference_set or set()
    n = len(sequences)
    if n == 0:
        return {
            "num_sequences": 0, "num_unique": 0, "unique_fraction": 0.0,
            "length_valid_fraction": 0.0, "canonical_fraction": 0.0,
            "challenge_valid_fraction": 0.0, "num_overlap_reference": 0,
        }

    def _len_ok(s):
        return CHALLENGE_MIN_LEN <= len(s) <= CHALLENGE_MAX_LEN

    def _canonical(s):
        return len(s) > 0 and all(c in CANONICAL_AA for c in s)

    unique = set(sequences)
    overlap = sum(1 for s in unique if s in reference_set)
    fully_valid = sum(
        1 for s in unique
        if _len_ok(s) and _canonical(s) and s not in reference_set
    )

    return {
        "num_sequences": n,
        "num_unique": len(unique),
        "unique_fraction": len(unique) / n,
        "length_valid_fraction": float(np.mean([_len_ok(s) for s in sequences])),
        "canonical_fraction": float(np.mean([_canonical(s) for s in sequences])),
        "challenge_valid_fraction": fully_valid / len(unique),
        "num_overlap_reference": overlap,
    }


def challenge_activity_metrics(amp_scores: List[float] = None,
                               mic_predictions: List[float] = None,
                               hemolysis_scores: List[float] = None) -> Dict:
    """Challenge-style activity/safety statistics (Success Rate, MIC50, MIC90, SW).
    Definitions follow amp-challenge-2027-main/README.md. When measured MIC /
    HC50 values are unavailable, model-predicted proxies are derived from the
    AMP probability and hemolysis head (clearly prefixed "predicted_").
    """
    out: Dict = {}

    # Resolve predicted MIC values.
    mics = None
    if mic_predictions:
        mics = [float(m) for m in mic_predictions]
    elif amp_scores:
        mics = [amp_score_to_predicted_mic(s) for s in amp_scores]

    if mics:
        out["predicted_mic50"] = _percentile(mics, 50)
        out["predicted_mic90"] = _percentile(mics, 90)
        # Success Rate: fraction meeting the potency threshold (MIC <= 16 uM).
        out["predicted_success_rate"] = float(
            np.mean([m <= POTENCY_THRESHOLD_UM for m in mics])
        )

    # Safety Window = HC50 / MIC50 (challenge "Optimal Selectivity").
    if hemolysis_scores and mics:
        hc50s = [hemolysis_to_predicted_hc50(h) for h in hemolysis_scores]
        out["predicted_hc50_median"] = _percentile(hc50s, 50)
        out["predicted_non_hemolytic_fraction"] = float(
            np.mean([h < 0.2 for h in hemolysis_scores])
        )
        mic50 = out.get("predicted_mic50")
        if mic50 and mic50 > 0:
            out["predicted_safety_window"] = float(_percentile(hc50s, 50) / mic50)

    return out


def top_k_reference_identity(sequences: List[str], reference: List[str],
                             top_k: int = 100) -> Dict:
    """Max Levenshtein-ratio identity of the top-k list against the reference set.
    The challenge requires every top-100 sequence to be <80% identical to any
    known antibacterial peptide. Returns the worst (max) identity and how many
    sequences violate the threshold.
    """
    top = sequences[:top_k]
    if not top or not reference:
        return {"top_k_max_identity": 0.0, "top_k_identity_violations": 0,
                "top_k_identity_ok": True}

    def ratio(s1, s2):
        m = max(len(s1), len(s2))
        return 1.0 if m == 0 else 1.0 - _levenshtein(s1, s2) / m

    max_id = 0.0
    violations = 0
    for s in top:
        best = max(ratio(s, r) for r in reference)
        max_id = max(max_id, best)
        if best > TOP_IDENTITY_MAX:
            violations += 1

    return {
        "top_k_max_identity": float(max_id),
        "top_k_identity_violations": int(violations),
        "top_k_identity_ok": violations == 0,
    }



class AMPMetrics:
    """Comprehensive AMP generation evaluation metrics."""

    def __init__(self, training_sequences: Set[str] = None,
                 reference_sequences: Set[str] = None):
        self.training_sequences = training_sequences or set()
        # Known antibacterial peptides (challenge reference set).
        self.reference_sequences = reference_sequences or set()


    def evaluate_batch(self, generated_sequences: List[str],
                       amp_scores: List[float] = None,
                       safety_scores: List[float] = None,
                       mic_predictions: List[float] = None) -> Dict:
        """Compute all metrics for a batch of generated sequences.

        Returns:
            Dict with comprehensive evaluation metrics
        """
        metrics = {}

        # Basic statistics
        metrics['num_generated'] = len(generated_sequences)
        lengths = [len(s) for s in generated_sequences]
        metrics['mean_length'] = np.mean(lengths) if lengths else 0
        metrics['std_length'] = np.std(lengths) if lengths else 0

        # Novelty
        metrics['novelty'] = compute_novelty(generated_sequences, self.training_sequences)

        # Diversity
        metrics['diversity'] = compute_diversity(generated_sequences)

        # Amino acid composition analysis
        metrics['aa_distribution'] = self._aa_composition(generated_sequences)
        metrics['charge_distribution'] = self._charge_distribution(generated_sequences)
        metrics['hydrophobicity_ratio'] = self._hydrophobicity_ratio(generated_sequences)

        # Activity metrics
        if amp_scores is not None:
            metrics['mean_amp_score'] = np.mean(amp_scores)
            metrics['amp_hit_rate'] = np.mean([s > 0.5 for s in amp_scores])

        # Safety metrics
        if safety_scores is not None:
            metrics['mean_safety_score'] = np.mean(safety_scores)
            metrics['safe_fraction'] = np.mean([s < 0.3 for s in safety_scores])

        # MIC predictions
        if mic_predictions is not None:
            metrics['mean_predicted_mic'] = np.mean(mic_predictions)
            metrics['potent_fraction'] = np.mean([m < 10.0 for m in mic_predictions])

        # Combined quality score
        metrics['quality_score'] = self._compute_quality_score(metrics)

        return metrics

    def evaluate_challenge(self, generated_sequences: List[str],
                           amp_scores: List[float] = None,
                           hemolysis_scores: List[float] = None,
                           mic_predictions: List[float] = None,
                           top_k: int = 100) -> Dict:
        """Compute AMP-challenge-2027 reporting metrics for a generated library.
        Combines structural compliance (length / alphabet / uniqueness /
        reference overlap), challenge activity statistics (Success Rate, MIC50,
        MIC90, Safety Window) and the top-k identity check against the known
        antibacterial reference set.
        """
        result: Dict = {}
        result.update(challenge_sequence_validity(
            generated_sequences, self.reference_sequences))
        result.update(challenge_activity_metrics(
            amp_scores=amp_scores,
            mic_predictions=mic_predictions,
            hemolysis_scores=hemolysis_scores,
        ))
        result.update(top_k_reference_identity(
            generated_sequences, sorted(self.reference_sequences), top_k=top_k))
        return result

    def _aa_composition(self, sequences: List[str]) -> Dict[str, float]:
        """Analyze amino acid composition."""
        all_aa = ''.join(sequences)
        counts = Counter(all_aa)
        total = len(all_aa)
        return {aa: count / total for aa, count in counts.items()} if total > 0 else {}

    def _charge_distribution(self, sequences: List[str]) -> Dict[str, float]:
        """Analyze net charge at pH 7."""
        positive_aa = set('RKH')
        negative_aa = set('DE')
        charges = []
        for seq in sequences:
            pos = sum(1 for aa in seq if aa in positive_aa)
            neg = sum(1 for aa in seq if aa in negative_aa)
            charges.append(pos - neg)
        return {
            'mean_charge': np.mean(charges) if charges else 0,
            'std_charge': np.std(charges) if charges else 0,
            'fraction_cationic': np.mean([c > 0 for c in charges]) if charges else 0,
        }

    def _hydrophobicity_ratio(self, sequences: List[str]) -> float:
        """Mean hydrophobic residue fraction."""
        hydrophobic = set('AILMFVWP')
        ratios = []
        for seq in sequences:
            if len(seq) > 0:
                ratios.append(sum(1 for aa in seq if aa in hydrophobic) / len(seq))
        return np.mean(ratios) if ratios else 0.0

    def _compute_quality_score(self, metrics: Dict) -> float:
        """Composite quality score combining all metrics."""
        score = 0.0
        score += 0.25 * metrics.get('novelty', 0)
        score += 0.15 * metrics.get('diversity', 0)
        score += 0.25 * metrics.get('amp_hit_rate', 0)
        score += 0.20 * metrics.get('safe_fraction', 0)
        score += 0.15 * metrics.get('potent_fraction', 0)
        return score
