#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amino acid properties and physicochemical feature computation.
"""

import numpy as np
from typing import Dict, List


# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Molecular weight (Da)
MOLECULAR_WEIGHT = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
    'E': 147.13, 'Q': 146.15, 'G': 75.03, 'H': 155.16, 'I': 131.17,
    'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
}

# Charge at pH 7
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'E': -1, 'Q': 0, 'G': 0, 'H': 0.5, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

# Side chain volume (A^3)
VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'E': 138.4, 'Q': 143.8, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
}

AA_PROPERTIES = {
    'hydrophobicity': HYDROPHOBICITY,
    'molecular_weight': MOLECULAR_WEIGHT,
    'charge': CHARGE,
    'volume': VOLUME,
}


def compute_peptide_features(sequence: str) -> Dict[str, float]:
    """Compute global physicochemical features for a peptide sequence.

    Returns:
        Dict with computed features
    """
    if not sequence:
        return {}

    n = len(sequence)

    # Basic features
    hydro_values = [HYDROPHOBICITY.get(aa, 0) for aa in sequence]
    charge_values = [CHARGE.get(aa, 0) for aa in sequence]
    weight_values = [MOLECULAR_WEIGHT.get(aa, 0) for aa in sequence]

    features = {
        'length': n,
        'molecular_weight': sum(weight_values) - (n - 1) * 18.02,  # Subtract water
        'net_charge': sum(charge_values),
        'mean_hydrophobicity': np.mean(hydro_values),
        'hydrophobic_moment': _compute_hydrophobic_moment(sequence),
        'fraction_hydrophobic': sum(1 for aa in sequence if aa in 'AILMFVWP') / n,
        'fraction_charged': sum(1 for aa in sequence if aa in 'RKHDE') / n,
        'fraction_positive': sum(1 for aa in sequence if aa in 'RKH') / n,
        'fraction_aromatic': sum(1 for aa in sequence if aa in 'FWY') / n,
        'amphipathicity': _compute_amphipathicity(hydro_values),
        'isoelectric_point': _estimate_pi(sequence),
    }

    return features


def _compute_hydrophobic_moment(sequence: str, angle: float = 100.0) -> float:
    """Compute hydrophobic moment assuming alpha-helix (100° angle)."""
    import math
    sin_sum = 0.0
    cos_sum = 0.0

    for i, aa in enumerate(sequence):
        h = HYDROPHOBICITY.get(aa, 0)
        theta = math.radians(angle * i)
        sin_sum += h * math.sin(theta)
        cos_sum += h * math.cos(theta)

    n = len(sequence)
    return math.sqrt(sin_sum**2 + cos_sum**2) / n if n > 0 else 0.0


def _compute_amphipathicity(hydro_values: List[float]) -> float:
    """
    Compute amphipathicity (variation in hydrophobicity along sequence)
    with Henderson-Hasselbalch.
    """
    if len(hydro_values) < 3:
        return 0.0

    return float(np.std(hydro_values))


def _estimate_pi(sequence: str) -> float:
    """Estimate isoelectric point of peptide."""
    # Simplified Henderson-Hasselbalch
    pka_values = {
        'D': 3.65, 'E': 4.25, 'H': 6.0, 'C': 8.18,
        'Y': 10.07, 'K': 10.53, 'R': 12.48,
    }
    n_term_pka = 9.69
    c_term_pka = 2.34

    def charge_at_ph(ph):
        charge = 1.0 / (1.0 + 10**(ph - n_term_pka))  # N-terminus
        charge -= 1.0 / (1.0 + 10**(c_term_pka - ph))  # C-terminus

        for aa in sequence:
            if aa in 'RKH':
                pka = pka_values.get(aa, 7.0)
                charge += 1.0 / (1.0 + 10**(ph - pka))
            elif aa in 'DECY':
                pka = pka_values.get(aa, 7.0)
                charge -= 1.0 / (1.0 + 10**(pka - ph))

        return charge

    # Binary search for pI
    lo, hi = 0.0, 14.0

    for _ in range(50):
        mid = (lo + hi) / 2
        if charge_at_ph(mid) > 0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2
