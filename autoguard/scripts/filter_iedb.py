#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter IEDB epitope export for disease-specific autoantigens.

Usage:
    python -m autoguard.scripts.filter_iedb --disease "Lupus|SLE" --output data/iedb/sle_epitopes.csv
"""

import argparse
import csv
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_iedb_epitopes(input_path: str,
                         disease_pattern: str,
                         output_path: str,
                         host: str = "Homo sapiens",
                         epitope_type: str = "Linear peptide"):
    """
    Filter IEDB epitope_full_v3.csv for disease-specific linear peptide epitopes.

    The IEDB export has a 2-row header with duplicate column names, so we use
    positional indexing:
        col[1]  = Object Type (e.g. "Linear peptide")
        col[2]  = Name / epitope sequence
        col[9]  = Source Molecule (antigen name)
        col[13] = Source Organism
        col[15] = Species

    Note: The full IEDB export does NOT include a "Disease" column.
    We filter by Source Organism (human autoantigens) and then match
    antigen names against known autoimmune targets.

    Args:
        input_path: Path to epitope_full_v3.csv (unzipped)
        disease_pattern: Regex to match against antigen/molecule names
            e.g., "Lupus|SLE|Ro60|Smith|snRNP|dsDNA|histone|chromatin"
        output_path: Output CSV path
        host: Host organism filter (applied to Source Organism, col[13])
        epitope_type: Object type filter (applied to col[1])
    """
    disease_re = re.compile(disease_pattern, re.IGNORECASE)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_sequences = set()
    kept = 0
    total = 0

    with open(input_path, 'r', encoding='utf-8', errors='replace') as fin, \
         open(output_path, 'w', newline='', encoding='utf-8') as fout:

        # Skip 2-row header (grouped categories + column names)
        next(fin)
        next(fin)

        reader = csv.reader(fin)

        writer = csv.DictWriter(fout, fieldnames=['sequence', 'antigen', 'source_organism',
                                                   'species', 'epitope_iri'])
        writer.writeheader()

        for parts in reader:
            total += 1
            if len(parts) < 16:
                continue

            obj_type = parts[1].strip()
            sequence = parts[2].strip()
            source_mol = parts[9].strip() if len(parts) > 9 else ""
            source_org = parts[13].strip() if len(parts) > 13 else ""
            species = parts[15].strip() if len(parts) > 15 else ""
            epitope_iri = parts[0].strip()

            # Filter by epitope type
            if epitope_type and epitope_type not in obj_type:
                continue

            # Filter by host organism
            if host and host not in source_org and host not in species:
                continue

            # Filter by disease pattern (match against source molecule / antigen name)
            if not disease_re.search(source_mol) and not disease_re.search(source_org):
                continue

            # Validate as peptide
            if not _is_valid_peptide(sequence):
                continue

            seq_upper = sequence.upper()
            if seq_upper in seen_sequences:
                continue
            seen_sequences.add(seq_upper)

            writer.writerow({
                'sequence': seq_upper,
                'antigen': source_mol,
                'source_organism': source_org,
                'species': species,
                'epitope_iri': epitope_iri,
            })
            kept += 1

    logger.info(f"Processed {total} rows → kept {kept} unique epitopes")
    logger.info(f"Output: {output_path}")


def _is_valid_peptide(seq: str) -> bool:
    """Check if string looks like a peptide sequence (only standard AAs, 5-50 residues)."""
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    seq_clean = seq.strip().upper()
    if len(seq_clean) < 5 or len(seq_clean) > 50:
        return False
    return all(c in valid_aa for c in seq_clean)


def main():
    parser = argparse.ArgumentParser(description='Filter IEDB for disease-specific epitopes')
    parser.add_argument('--input', type=str, default='data/iedb/epitope_full_v3.csv',
                        help='Path to IEDB epitope_full_v3.csv')
    parser.add_argument('--disease', type=str, required=True,
                        help='Regex pattern for disease (e.g., "Lupus|SLE")')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV path')
    parser.add_argument('--host', type=str, default='Homo sapiens',
                        help='Host organism filter')
    args = parser.parse_args()

    filter_iedb_epitopes(args.input, args.disease, args.output, host=args.host)


if __name__ == '__main__':
    main()
