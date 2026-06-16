#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full data preparation pipeline for AutoGuard.
Downloads, validates, standardises and splits ALL datasets needed
for the end-to-end experiment:

  1. AMP activity data     -> GRAMPA (CSV), APD6 (FASTA), DRAMP (FASTA), DBAASP (5x CSV)
  2. Safety data           -> ToxinPred (toxic/non-toxic), HemoPI (4x FASTA split)
  3. Mimicry data          -> APD6 human AMPs (positive), IEDB epitopes (negative)
  4. Phylogenetic tree     -> bundled static timetree.nwk (NO SAAP auto-trees)
  5. Challenge reference   -> antibacterial.fasta from amp-challenge-2027-main

After running, the data/ directory will contain:
  data/
  ├── processed/
  │   ├── amp_train.csv          # Unified AMP sequences + labels + MIC
  │   ├── amp_val.csv
  │   ├── amp_test.csv
  │   ├── safety_train.csv       # Toxicity + hemolysis labels
  │   ├── safety_val.csv
  │   ├── safety_test.csv
  │   ├── mimicry_positives.fasta  # Human host defense peptides (attract)
  │   ├── mimicry_negatives.fasta  # Autoantigen epitopes (repel)
  │   └── stats.json             # Dataset statistics
  ├── species_trees/             # Newick files + pre-computed embeddings
  └── raw/                       # Original downloaded files

Usage:
    python -m autoguard.scripts.prepare_data --output_dir data/
    python -m autoguard.scripts.prepare_data --output_dir data/ --skip_download
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 5
MAX_LEN = 25

# Auto-downloadable URLs (direct links, no auth required)
DOWNLOADS = {
    # AMP activity
    "grampa": "https://github.com/zswitten/Antimicrobial-Peptides/raw/master/data/grampa.csv",
    "apd_natural": "https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta",
    "apd_human": "https://aps.unmc.edu/assets/sequences/humanAMPs_APD2024.fasta",
    # Safety (ToxinPred only — HemoPI requires manual download)
    "toxinpred_pos": "https://webs.iiitd.edu.in/raghava/toxinpred/dataset_main_positive.txt",
    "toxinpred_neg": "https://webs.iiitd.edu.in/raghava/toxinpred/dataset_main_negative.txt",
}

# Datasets that need manual download (paths relative to data/raw/)
MANUAL_DOWNLOADS = {
    "dramp": {
        "url": "https://dramp.cpu-bioinfor.org/downloads/",
        "targets": ["dramp/general_amps.fasta"],
        "instructions": (
            "Visit https://dramp.cpu-bioinfor.org/downloads/\n"
            "Download: general_amps.fasta OR general_amps.txt (TSV with MIC)\n"
            "Place in data/raw/dramp/"
        ),
    },
    "dbaasp": {
        "url": "https://dbaasp.org/download",
        "targets": [
            "dbaasp/peptides1.csv",
            "dbaasp/peptides2.csv",
            "dbaasp/peptides3.csv",
            "dbaasp/peptides4.csv",
            "dbaasp/peptides5.csv",
        ],
        "instructions": (
            "Visit https://dbaasp.org/download\n"
            "Download CSV exports: peptides1.csv, peptides2.csv, peptides3.csv, peptides4.csv, peptides5.csv\n"
            "Place them in data/raw/dbaasp/\n"
            "Expected columns: ID, COMPLEXITY, NAME, SEQUENCE, ..."
        ),
    },
    "hemopi": {
        "url": "https://webs.iiitd.edu.in/raghava/hemopi/",
        "targets": [
            "hemopi/pos_train.fa.txt",
            "hemopi/pos_test.fa.txt",
            "hemopi/neg_train.fa.txt",
            "hemopi/neg_test.fa.txt",
        ],
        "instructions": (
            "Visit https://webs.iiitd.edu.in/raghava/hemopi/ → Dataset tab\n"
            "Download FASTA sequence files:\n"
            "  - pos_train.fa.txt, pos_test.fa.txt, neg_train.fa.txt, neg_test.fa.txt\n"
            "Place them in data/raw/hemopi/"
        ),
    },
    "iedb": {
        "url": "https://www.iedb.org/database_export_v3.php",
        "targets": ["iedb/epitope_full_v3.csv"],
        "instructions": (
            "Visit https://www.iedb.org/database_export_v3.php\n"
            "Download: epitope_full_v3.zip (97MB)\n"
            "Unzip → place epitope_full_v3.csv in data/raw/iedb/\n"
            "Format: 2-row header, col[1]=Object Type, col[2]=epitope sequence"
        ),
    },
    "uniprot": {
        "url": "https://www.uniprot.org/uniprot/?query=reviewed:true+length:[5+TO+200]&format=fasta",
        "targets": ["uniprot/uniprot_nonamp.fasta"],
        "instructions": (
            "Option A (recommended): Copy HydrAMP's UniProt file:\n"
            "  cp hydramp-master/data/interim/Uniprot_0_200_no_duplicates.csv data/raw/uniprot/\n"
            "Option B: Download from UniProt:\n"
            "  Visit https://www.uniprot.org/uniprotkb?query=reviewed%3Atrue+AND+length%3A%5B5+TO+200%5D\n"
            "  Download FASTA → place as data/raw/uniprot/uniprot_nonamp.fasta\n"
            "  (~50,000 reviewed short proteins, will be fragmented to 5-25 AA)\n"
            "Option C: Use HydrAMP's prepared negatives via DVC:\n"
            "  cd hydramp-master && dvc pull\n"
            "  File: hydramp-master/data/interim/Uniprot_0_200_no_duplicates.csv"
        ),
    },
}


# ============================================================================
# Download
# ============================================================================

def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a single file. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  Already exists: {dest.name} ({dest.stat().st_size:,} bytes)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading {dest.name}.")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoGuard/0.1"})

        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)

        logger.info(f"  OK: {dest.name} ({dest.stat().st_size:,} bytes)")
        return True
    except Exception as e:
        logger.warning(f"  [WARNING]: Failed to download {url}: {e}")

        if dest.exists():
            dest.unlink()

        return False


def download_all(raw_dir: Path) -> Dict[str, bool]:
    """Download all auto-downloadable datasets."""
    results = {}

    for name, url in DOWNLOADS.items():
        ext = url.rsplit(".", 1)[-1].split("?")[0]

        if ext not in ("csv", "fasta", "txt", "json"):
            ext = "txt"

        parts = name.split("_", 1)

        if len(parts) == 2:
            dest = raw_dir / parts[0] / f"{parts[1]}.{ext}"
        else:
            dest = raw_dir / parts[0] / f"{parts[0]}.{ext}"

        results[name] = download_file(url, dest)

    return results


def check_manual_downloads(raw_dir: Path) -> Dict[str, bool]:
    """Check which manual downloads are present."""
    results = {}

    for name, info in MANUAL_DOWNLOADS.items():
        all_present = True
        for target_rel in info["targets"]:
            target_path = raw_dir / target_rel
            if not (target_path.exists() and target_path.stat().st_size > 0):
                all_present = False
                break

        results[name] = all_present

        if not all_present:
            logger.warning(f"  [WARNING]: Missing or incomplete: {name} — {info['instructions']}")
    return results


# ============================================================================
# Parsing
# ============================================================================

def _valid_seq(seq: str) -> bool:
    s = seq.strip().upper()
    return MIN_LEN <= len(s) <= MAX_LEN and all(c in AA_ALPHABET for c in s)


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    """Parse FASTA -> [(header, sequence)]."""
    seqs = []
    hdr, parts = None, []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith(">"):
            if hdr is not None:
                seqs.append((hdr, "".join(parts).upper()))
            hdr = line[1:]
            parts = []
        elif hdr is not None:
            parts.append(line)

    if hdr is not None:
        seqs.append((hdr, "".join(parts).upper()))

    return seqs


def parse_plain_text(path: Path) -> List[str]:
    """Parse one-sequence-per-line text files (ToxinPred format)."""
    seqs = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(">"):
            continue
        parts = line.split()
        candidate = parts[-1].upper()
        if all(c in AA_ALPHABET for c in candidate) and len(candidate) >= MIN_LEN:
            seqs.append(candidate)

    return seqs


def parse_grampa_csv(path: Path) -> List[Tuple[str, float, str]]:
    """Parse GRAMPA CSV -> [(sequence, log10_MIC, bacterium)].
    Each unique (sequence, bacterium) pair is kept with the best (lowest) MIC.
    """
    results: Dict[Tuple[str, str], float] = {}

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq = row.get("sequence", "").strip().upper()
            try:
                mic = float(row.get("value", ""))
            except (ValueError, TypeError):
                continue
            bacterium = row.get("bacterium", "").strip()
            key = (seq, bacterium)
            if key not in results or mic < results[key]:
                results[key] = mic

    return [(seq, mic, bact) for (seq, bact), mic in results.items()]


# ============================================================================
# Standardisation
# ============================================================================

def _hash_split(seq: str, train=0.8, val=0.1) -> str:
    """Deterministic split based on sequence hash."""
    h = int(hashlib.md5(seq.encode()).hexdigest(), 16) % 1000

    if h < train * 1000:
        return "train"
    elif h < (train + val) * 1000:
        return "val"

    return "test"


def build_amp_dataset(raw_dir: Path) -> Dict[str, List[Dict]]:
    """Build unified AMP dataset from all available sources."""
    all_entries: Dict[str, Dict] = {}

    def _add(seq, label, mic, source, target_species=""):
        seq = seq.strip().upper()
        if not _valid_seq(seq):
            return
        if seq not in all_entries or (mic is not None and
                (all_entries[seq].get("mic") is None or mic < all_entries[seq]["mic"])):
            all_entries[seq] = {"sequence": seq, "label": label, "mic": mic,
                                "source": source, "target_species": target_species}
        elif target_species and not all_entries[seq].get("target_species"):
            all_entries[seq]["target_species"] = target_species

    # GRAMPA
    grampa_path = raw_dir / "grampa" / "grampa.csv"
    if grampa_path.exists():
        for seq, mic, bacterium in parse_grampa_csv(grampa_path):
            # mic is log10(MIC in uM), so threshold 10 uM = log10(10) = 1.0
            label = 1 if mic <= 1.0 else 0
            _add(seq, label, mic, "grampa", target_species=bacterium)

        grampa_entries = [e for e in all_entries.values() if e["source"] == "grampa"]
        logger.info(f"  GRAMPA: loaded {len(grampa_entries)} sequences "
                    f"({sum(1 for e in grampa_entries if e['target_species']):,} with species target)")

    # APD6 natural AMPs
    apd_path = raw_dir / "apd" / "natural.fasta"

    if apd_path.exists():
        seqs = parse_fasta(apd_path)

        for _, seq in seqs:
            _add(seq, 1, None, "apd")

        logger.info(f"  APD6 natural: {len(seqs)} sequences")

    # DRAMP
    dramp_fasta = raw_dir / "dramp" / "general_amps.fasta"
    dramp_tsv = raw_dir / "dramp" / "general_amps.txt"

    if dramp_tsv.exists():
        dramp_count = 0

        with open(dramp_tsv, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seq = row.get("Sequence", "").strip().upper()
                _add(seq, 1, None, "dramp")
                dramp_count += 1

        logger.info(f"  DRAMP (TSV): {dramp_count} sequences")

    elif dramp_fasta.exists():
        seqs = parse_fasta(dramp_fasta)

        for _, seq in seqs:
            _add(seq, 1, None, "dramp")

        logger.info(f"  DRAMP (FASTA): {len(seqs)} sequences")

    # DBAASP
    dbaasp_dir = raw_dir / "dbaasp"
    if dbaasp_dir.exists():
        count = 0
        csv_files = sorted(dbaasp_dir.glob("peptides*.csv"))
        if csv_files:
            for cf in csv_files:
                try:
                    with open(cf, encoding="utf-8", errors="replace") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            seq = row.get("SEQUENCE", "").strip().upper()
                            seq = seq.replace(" ", "").replace("\t", "")
                            complexity = row.get("COMPLEXITY", "").lower()
                            if complexity == "multimer":
                                continue
                            _add(seq, 1, None, "dbaasp")
                            count += 1
                except Exception:
                    pass
            logger.info(f"  DBAASP (CSV export): {count} entries from {len(csv_files)} files")

    # Challenge Reference
    challenge_fasta = raw_dir.parent.parent / "amp-challenge-2027-main" / "data" / "antibacterial.fasta"
    if challenge_fasta.exists():
        seqs = parse_fasta(challenge_fasta)
        for _, seq in seqs:
            _add(seq, 1, None, "challenge")
        logger.info(f"  Challenge reference: {len(seqs)} sequences")

    # Negative Samples from UniProt
    import random
    rng = random.Random(42)

    num_positives = sum(1 for e in all_entries.values() if e["label"] == 1)
    num_existing_neg = sum(1 for e in all_entries.values() if e["label"] == 0)

    # Target: 1:1 balanced
    target_negatives = max(0, num_positives - num_existing_neg)
    neg_added = 0

    # UniProt non-AMP fragments (best negatives — real proteins)
    uniprot_path = raw_dir / "uniprot" / "uniprot_nonamp.fasta"
    if not uniprot_path.exists():
        # Alternative: HydrAMP's UniProt file if available
        alt_paths = [
            raw_dir.parent.parent / "hydramp-master" / "data" / "interim" / "Uniprot_0_200_no_duplicates.csv",
            raw_dir / "uniprot" / "Uniprot_0_200_no_duplicates.csv",
        ]
        for alt in alt_paths:
            if alt.exists():
                uniprot_path = alt
                break

    if uniprot_path.exists() and str(uniprot_path).endswith('.csv'):
        import pandas as pd

        uni_df = pd.read_csv(uniprot_path)
        seq_col = 'Sequence' if 'Sequence' in uni_df.columns else 'sequence'
        uni_seqs = uni_df[seq_col].dropna().tolist()

        # Fragment long sequences into peptide-length windows
        fragments = []
        for seq in uni_seqs:
            seq = seq.strip().upper()
            if len(seq) >= MIN_LEN:
                if len(seq) <= MAX_LEN:
                    fragments.append(seq)
                else:
                    # Sliding window with stride=5
                    for start in range(0, len(seq) - MIN_LEN, 5):
                        frag = seq[start:start + rng.randint(MIN_LEN, MAX_LEN)]
                        if _valid_seq(frag) and len(frag) >= MIN_LEN:
                            fragments.append(frag)

        rng.shuffle(fragments)

        for frag in fragments:
            if neg_added >= target_negatives:
                break

            if frag not in all_entries and _valid_seq(frag):
                all_entries[frag] = {"sequence": frag, "label": 0, "mic": None,
                                     "source": "uniprot_neg", "target_species": ""}
                neg_added += 1

        logger.info(f"  UniProt negatives: {neg_added} (from {uniprot_path.name})")
    elif uniprot_path.exists() and str(uniprot_path).endswith('.fasta'):
        uni_seqs = parse_fasta(uniprot_path)
        fragments = []
        for _, seq in uni_seqs:
            seq = seq.strip().upper()
            if len(seq) <= MAX_LEN and _valid_seq(seq):
                fragments.append(seq)
            elif len(seq) > MAX_LEN:
                for start in range(0, len(seq) - MIN_LEN, 5):
                    frag = seq[start:start + rng.randint(MIN_LEN, MAX_LEN)]
                    if _valid_seq(frag) and len(frag) >= MIN_LEN:
                        fragments.append(frag)

        rng.shuffle(fragments)

        for frag in fragments:
            if neg_added >= target_negatives:
                break

            if frag not in all_entries:
                all_entries[frag] = {"sequence": frag, "label": 0, "mic": None,
                                     "source": "uniprot_neg", "target_species": ""}
                neg_added += 1

        logger.info(f"  UniProt FASTA negatives: {neg_added}")

    # Fragment ToxinPred non-toxic sequences (real peptides, verified non-toxic)
    if neg_added < target_negatives:
        tp_neg_path = raw_dir / "toxinpred" / "neg.txt"

        if tp_neg_path.exists():
            tp_seqs = parse_plain_text(tp_neg_path)
            rng.shuffle(tp_seqs)
            tp_added = 0

            for seq in tp_seqs:
                if neg_added >= target_negatives:
                    break

                if seq not in all_entries and _valid_seq(seq):
                    all_entries[seq] = {"sequence": seq, "label": 0, "mic": None,
                                         "source": "toxinpred_neg", "target_species": ""}
                    neg_added += 1
                    tp_added += 1

            if tp_added:
                logger.info(f"  ToxinPred non-toxic negatives: {tp_added}")

    # Random fragments as last resort
    if neg_added < target_negatives:
        AA_FREQ = list("AAAAAAAALLLLLLLLLLGGGGGGGGVVVVVVVVEEEEEEESSSSSSSIIIIIIIIKKKKKKK"
                       "RRRRRRDDDDDDTTTTTTPPPPPPNNNNFFQQYYHHMCCWW")
        rand_added = 0
        attempts = 0
        while neg_added < target_negatives and attempts < target_negatives * 10:
            attempts += 1
            length = rng.randint(MIN_LEN, MAX_LEN)
            seq = "".join(rng.choices(AA_FREQ, k=length))
            if seq not in all_entries and _valid_seq(seq):
                all_entries[seq] = {"sequence": seq, "label": 0, "mic": None,
                                     "source": "synthetic_neg", "target_species": ""}
                neg_added += 1
                rand_added += 1
        if rand_added:
            logger.info(f"  Synthetic random negatives (fallback): {rand_added}")

    logger.info(f"  Total negatives: {neg_added + num_existing_neg} "
                f"(target was 1:1 = {num_positives})")

    # Split
    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    for entry in all_entries.values():
        s = _hash_split(entry["sequence"])
        splits[s].append(entry)

    return splits


def build_safety_dataset(raw_dir: Path) -> Dict[str, List[Dict]]:
    """Build unified safety dataset from ToxinPred + HemoPI."""
    all_entries: Dict[str, Dict] = {}

    def _add(seq, toxic=None, hemolytic=None):
        seq = seq.strip().upper()
        if not _valid_seq(seq):
            return
        if seq not in all_entries:
            all_entries[seq] = {"sequence": seq, "toxic": toxic, "hemolytic": hemolytic}
        else:
            if toxic is not None and all_entries[seq]["toxic"] is None:
                all_entries[seq]["toxic"] = toxic
            if hemolytic is not None and all_entries[seq]["hemolytic"] is None:
                all_entries[seq]["hemolytic"] = hemolytic

    # ToxinPred
    tp_pos = raw_dir / "toxinpred" / "pos.txt"
    tp_neg = raw_dir / "toxinpred" / "neg.txt"
    if tp_pos.exists():
        for seq in parse_plain_text(tp_pos):
            _add(seq, toxic=1)
        logger.info(f"  ToxinPred positive: {len(parse_plain_text(tp_pos))} sequences")
    if tp_neg.exists():
        for seq in parse_plain_text(tp_neg):
            _add(seq, toxic=0)
        logger.info(f"  ToxinPred negative: {len(parse_plain_text(tp_neg))} sequences")

    # HemoPI
    hp_pos = raw_dir / "hemopi" / "pos.txt"
    hp_neg = raw_dir / "hemopi" / "neg.txt"
    hp_pos_train_fa = raw_dir / "hemopi" / "pos_train.fa.txt"
    hp_pos_test_fa = raw_dir / "hemopi" / "pos_test.fa.txt"
    hp_neg_train_fa = raw_dir / "hemopi" / "neg_train.fa.txt"
    hp_neg_test_fa = raw_dir / "hemopi" / "neg_test.fa.txt"
    hp_train_pos_zip = raw_dir / "hemopi" / "train_pos.zip"

    hemopi_pos_count = 0
    hemopi_neg_count = 0

    fasta_files_exist = any(p.exists() for p in [hp_pos_train_fa, hp_pos_test_fa, hp_neg_train_fa, hp_neg_test_fa])

    if fasta_files_exist:
        for fa_path in [hp_pos_train_fa, hp_pos_test_fa]:
            if fa_path.exists():
                for _, seq in parse_fasta(fa_path):
                    _add(seq, hemolytic=1)
                    hemopi_pos_count += 1
        for fa_path in [hp_neg_train_fa, hp_neg_test_fa]:
            if fa_path.exists():
                for _, seq in parse_fasta(fa_path):
                    _add(seq, hemolytic=0)
                    hemopi_neg_count += 1

        logger.info(f"  HemoPI (FASTA Splits): {hemopi_pos_count} hemolytic, {hemopi_neg_count} non-hemolytic")

    elif hp_pos.exists():
        for seq in parse_plain_text(hp_pos):
            _add(seq, hemolytic=1)
            hemopi_pos_count += 1
        logger.info(f"  HemoPI positive (plain): {hemopi_pos_count} sequences")
        if hp_neg.exists():
            for seq in parse_plain_text(hp_neg):
                _add(seq, hemolytic=0)
                hemopi_neg_count += 1
            logger.info(f"  HemoPI negative (plain): {hemopi_neg_count} sequences")
    elif hp_train_pos_zip.exists():
        logger.warning(
            "  [WARNING]: HemoPI: found ZIP/SDF files but no sequence files.\n"
            "    SDF files contain 3D coordinates, not amino acid sequences.\n"
            "    Please download FASTA sequence files instead:\n"
            "      pos_train.fa.txt, pos_test.fa.txt, neg_train.fa.txt, neg_test.fa.txt\n"
            "    from https://webs.iiitd.edu.in/raghava/hemopi/ → 'Dataset' tab"
        )

    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    for entry in all_entries.values():
        s = _hash_split(entry["sequence"])
        splits[s].append(entry)

    return splits


def build_mimicry_dataset(raw_dir: Path,
                          processed_dir: Path
                          ) -> Dict[str, int]:
    """Build mimicry positive/negative FASTA files."""
    stats = {"positives": 0, "negatives": 0}

    # Positives: human AMPs
    human_path = raw_dir / "apd" / "human.fasta"
    pos_out = processed_dir / "mimicry_positives.fasta"

    if human_path.exists():
        seqs = parse_fasta(human_path)
        valid = [(h, s) for h, s in seqs if _valid_seq(s)]

        with open(pos_out, "w") as f:
            for hdr, seq in valid:
                f.write(f">{hdr}\n{seq}\n")

        stats["positives"] = len(valid)
        logger.info(f"  Mimicry positives (human AMPs): {len(valid)}")
    else:
        defensins = [
            ("HNP-1", "ACYCRIPACIAGERRYGTCIYQGRLWAFCC"),
            ("HBD-1", "DHYNCVSSGGQCLYSACPIFTKIQGTCYRGKAKCCK"),
            ("LL-37", "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"),
            ("HD-5", "ATCYCRTGRCATRESLSGVCEISGRLYRLCCR"),
        ]
        valid = [(h, s[:MAX_LEN]) for h, s in defensins if _valid_seq(s[:MAX_LEN])]

        with open(pos_out, "w") as f:
            for hdr, seq in valid:
                f.write(f">{hdr}\n{seq}\n")

        stats["positives"] = len(valid)
        logger.info(f"  Mimicry positives (fallback defensins): {len(valid)}")

    # Negatives: IEDB epitopes
    iedb_csv = raw_dir / "iedb" / "epitope_full_v3.csv"
    sle_csv = raw_dir / "iedb" / "sle_epitopes.csv"
    neg_out = processed_dir / "mimicry_negatives.fasta"

    neg_seqs: List[Tuple[str, str]] = []

    if sle_csv.exists():
        with open(sle_csv, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq = row.get("sequence", "").strip().upper()
                antigen = row.get("antigen", "unknown")
                if _valid_seq(seq):
                    neg_seqs.append((f"SLE_{antigen[:30]}", seq))

        logger.info(f"  Mimicry negatives (SLE epitopes): {len(neg_seqs)}")

    elif iedb_csv.exists():
        count = 0

        with open(iedb_csv, encoding="utf-8", errors="replace") as f:
            next(f)
            next(f)
            reader = csv.reader(f)
            for parts in reader:
                if count >= 5000:
                    break

                if len(parts) < 16:
                    continue

                obj_type = parts[1].strip()
                seq_name = parts[2].strip().upper()
                source_org = parts[13].strip() if len(parts) > 13 else ""
                species = parts[15].strip() if len(parts) > 15 else ""
                if "Linear" not in obj_type:
                    continue
                if "Homo sapiens" not in source_org and "Homo sapiens" not in species:
                    continue
                if _valid_seq(seq_name):
                    neg_seqs.append((f"IEDB_{count}", seq_name))
                    count += 1
        logger.info(f"  Mimicry negatives (IEDB linear peptides, human): {len(neg_seqs)}")
    else:
        sle_epitopes = [
            ("Ro60_1", "TKYKQRNGWSHKDLLR"),
            ("Ro60_2", "ELYKEKALSVETEKLL"),
            ("SmD1_1", "GRGRGRGRGRGRGRGR"),
            ("RibP_C", "GFGLFD"),
            ("dsDNA_mimic", "DWEYSVWLSN"),
            ("La_SSB_1", "KLEDLERKFREKEQEL"),
            ("Histone_H1", "KKAAGAGAAKK"),
            ("C1q_frag", "GNLGEFWLG"),
        ]
        neg_seqs = [(h, s) for h, s in sle_epitopes if _valid_seq(s)]
        logger.info(f"  Mimicry negatives (fallback SLE epitopes): {len(neg_seqs)}")

    with open(neg_out, "w") as f:
        for hdr, seq in neg_seqs:
            f.write(f">{hdr}\n{seq}\n")

    stats["negatives"] = len(neg_seqs)

    return stats


def copy_phylogenetic_trees(raw_dir: Path,
                            output_dir: Path
                            ) -> int:
    """Make the static, curated phylogenetic tree available under data/species_trees/.
    AutoGuard uses ONLY the bundled, version-controlled "timetree.nwk"
    (a TimeTree-derived ultrametric species tree). The automatically generated
    supertrees / consensus trees from the SnakeAnalysisPhylogenomicsPipeline
    (SAAP) are deliberately NOT used.

    The curated tree ships inside the package at
    "autoguard/data/species_trees/" and is simply copied into the run's
    "<output_dir>/species_trees/" when it is not already present there.
    """
    trees_dest = output_dir / "species_trees"
    trees_dest.mkdir(parents=True, exist_ok=True)

    # Bundled, static curated tree shipped with the package.
    bundled_trees = Path(__file__).resolve().parent.parent / "data" / "species_trees"

    count = 0
    if bundled_trees.exists() and bundled_trees.resolve() != trees_dest.resolve():
        for tree_file in sorted(bundled_trees.glob("*")):
            if tree_file.suffix in (".newick", ".nwk", ".treefile"):
                dest = trees_dest / tree_file.name
                if not dest.exists():
                    shutil.copy2(tree_file, dest)
                count += 1
            elif tree_file.name == "species_metadata.csv":
                dest = trees_dest / tree_file.name
                if not dest.exists():
                    shutil.copy2(tree_file, dest)

    # Count whatever curated trees now live in the destination.
    count = sum(1 for f in trees_dest.glob("*")
                if f.suffix in (".newick", ".nwk", ".treefile"))

    if count:
        logger.info(f"  Using {count} curated static tree file(s) in {trees_dest} "
                    f"(SAAP auto-generated trees are not used)")
    else:
        # Last-resort fallback so the pipeline still runs end-to-end.
        example_tree = (
            "((Escherichia_coli:24.9,Klebsiella_pneumoniae:24.9)'n1':6.2,"
            "(Streptococcus_agalactiae:116.2,Streptococcus_pyogenes:116.2)'n2':854.5);"
        )
        (trees_dest / "timetree.nwk").write_text(example_tree)
        count = 1
        logger.info("  No curated tree found; wrote a minimal fallback timetree.nwk")

    return count


# ============================================================================
# Main pipeline
# ============================================================================

def write_csv(path: Path, entries: List[Dict], fieldnames: List[str]):
    """Write list of dicts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)


def prepare_all(output_dir: Path, skip_download: bool = False):
    """Full data preparation pipeline."""
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    stats = {}

    # Download
    if not skip_download:
        logger.info("\n<== Stage 1: Downloading datasets ==>")
        download_results = download_all(raw_dir)
        stats["downloads"] = download_results
        logger.info(f"\n  Auto-downloads: {sum(download_results.values())}/{len(download_results)} successful")

        logger.info("\n  Checking manual downloads:")
        manual_results = check_manual_downloads(raw_dir)
        stats["manual"] = manual_results
    else:
        logger.info("\n<== Stage 1: Skipping downloads (--skip_download) ==>")

    # Build AMP dataset
    logger.info("\n<== Stage 2: Building AMP activity dataset ==>")
    amp_splits = build_amp_dataset(raw_dir)

    for split, entries in amp_splits.items():
        write_csv(processed_dir / f"amp_{split}.csv", entries,
                  ["sequence", "label", "mic", "source", "target_species"])

    stats["amp"] = {s: len(e) for s, e in amp_splits.items()}
    total_amp = sum(stats["amp"].values())
    logger.info(f"  Total AMP entries: {total_amp} (train={stats['amp']['train']}, "
                f"val={stats['amp']['val']}, test={stats['amp']['test']})")

    # Build safety dataset
    logger.info("\n <== Step 3: Building safety dataset ==>")
    safety_splits = build_safety_dataset(raw_dir)

    for split, entries in safety_splits.items():
        write_csv(processed_dir / f"safety_{split}.csv", entries,
                  ["sequence", "toxic", "hemolytic"])

    stats["safety"] = {s: len(e) for s, e in safety_splits.items()}
    total_safety = sum(stats["safety"].values())
    logger.info(f"  Total safety entries: {total_safety} (train={stats['safety']['train']}, "
                f"val={stats['safety']['val']}, test={stats['safety']['test']})")

    # Build mimicry dataset
    logger.info("\n<== Step 4: Building mimicry contrastive dataset ==>")
    mimicry_stats = build_mimicry_dataset(raw_dir, processed_dir)
    stats["mimicry"] = mimicry_stats

    # Phylogenetic trees
    logger.info("\n<== Step 5: Setting up phylogenetic trees ==>")
    n_trees = copy_phylogenetic_trees(raw_dir, output_dir)
    stats["trees"] = n_trees

    # Copy challenge reference
    logger.info("\n<== Step 6: Challenge reference set ==>")
    challenge_src = raw_dir.parent.parent / "amp-challenge-2027-main" / "data" / "antibacterial.fasta"
    challenge_dest = processed_dir / "antibacterial_reference.fasta"

    if challenge_src.exists() and not challenge_dest.exists():
        shutil.copy2(challenge_src, challenge_dest)
        logger.info(f"  Copied antibacterial reference FASTA")
    elif challenge_dest.exists():
        logger.info(f"  Reference FASTA already present")

    # Save stats
    stats_path = processed_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Summary
    logger.info("\n" + "═" * 60)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("═" * 60)
    logger.info(f"  AMP sequences:     {total_amp:>6}")
    logger.info(f"  Safety sequences:  {total_safety:>6}")
    logger.info(f"  Mimicry positives: {mimicry_stats['positives']:>6}")
    logger.info(f"  Mimicry negatives: {mimicry_stats['negatives']:>6}")
    logger.info(f"  Phylo trees:       {n_trees:>6}")
    logger.info(f"\n  Output: {output_dir.resolve()}")
    logger.info(f"  Stats:  {stats_path}")

    if total_amp == 0:
        logger.warning(
            "\n  [WARNING]: No AMP data found! Make sure downloads completed.\n"
            "    Re-run with: python -m autoguard.scripts.prepare_data --output_dir data/"
        )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download, validate and prepare all datasets for AutoGuard training"
    )
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Root data directory")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download step (use existing raw files)")
    args = parser.parse_args()

    prepare_all(Path(args.output_dir), skip_download=args.skip_download)


if __name__ == "__main__":
    main()
