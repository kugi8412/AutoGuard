#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset loaders for AMP databases.
Supports: GRAMPA, DBAASP, DRAMP, APD6, NCBI datasets.
Also: ToxinPred (toxicity), HemoPI (hemolysis) for safety module.
Handles FASTA, CSV, JSON, and plain-text parsing with MIC thresholds and splits.
"""

import csv
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset


# Standard amino acid alphabet
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}  # 0 = padding
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}
IDX_TO_AA[0] = '<pad>'


def tokenize_sequence(seq: str,
                      max_len: int = 25
                      ) -> torch.Tensor:
    """Convert amino acid sequence to integer tokens.
    Non-canonical residues (B, J, O, U, X, Z) are skipped rather than
    mapped to index 0: index 0 is the padding/stop token, so inserting it
    mid-sequence would silently truncate the peptide at detokenization.
    """
    tokens = [AA_TO_IDX[aa] for aa in seq.upper() if aa in AA_TO_IDX][:max_len]
    # Pad to max_len
    tokens += [0] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


def detokenize_sequence(tokens: torch.Tensor) -> str:
    """Convert integer tokens back to amino acid sequence."""
    seq = ''
    for t in tokens.tolist():

        if t == 0:
            break

        seq += IDX_TO_AA.get(t, 'X')

    return seq


def make_amp_collate(use_graph: bool = False,
                     k_neighbors: int = 5
                     ) -> callable:
    """Build a collate_fn for AMPDataset.
    When "use_graph" is True, peptide molecular graphs are built on the fly
    and batched (requires torch_geometric). The resulting "graph_data" tuple
    "(x, edge_index, edge_attr, batch)" is added to the batch so the GG-FiLM
    graph encoder can be exercised. When False, a plain collate is used.
    """
    from torch.utils.data._utils.collate import default_collate

    graph_builder = None
    pyg_batch = None
    if use_graph:
        try:
            from torch_geometric.data import Data, Batch as _Batch
            from ..models.graph_encoder import PeptideGraphBuilder
            graph_builder = PeptideGraphBuilder(k_neighbors=k_neighbors)
            pyg_batch = (_Batch, Data)
        except Exception:
            graph_builder = None

    def _collate(items):
        keys = items[0].keys()
        tensor_items = [{k: it[k] for k in keys if k != 'sequence'} for it in items]
        batch = default_collate(tensor_items)

        if 'sequence' in keys:
            batch['sequence'] = [it['sequence'] for it in items]

        if graph_builder is not None and 'sequence' in keys:
            _Batch, Data = pyg_batch
            graphs = []

            for it in items:
                nf, ei, ea = graph_builder.sequence_to_graph(it['sequence'])
                graphs.append(Data(x=nf, edge_index=ei, edge_attr=ea))

            g = _Batch.from_data_list(graphs)
            batch['graph_data'] = (g.x, g.edge_index, g.edge_attr, g.batch)

        return batch

    return _collate


def parse_fasta(filepath: str) -> List[Tuple[str, str]]:
    """Parse FASTA file into list of (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):

                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))

                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_header is not None:
        sequences.append((current_header, ''.join(current_seq)))

    return sequences


def parse_grampa_csv(filepath: str,
                     mic_threshold: float = 10.0
                     ) -> Tuple[List[str], List[int], List[float]]:
    """Parse GRAMPA CSV file (columns: sequence, bacterium, strain, value, modifications).

    Returns:
        sequences: peptide sequences
        labels: 1 if MIC <= threshold (active), 0 otherwise
        mic_values: MIC in uM (original unit from GRAMPA)
    """
    sequences, labels, mic_values = [], [], []
    seen = {}  # sequence -> best (lowest) MIC

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get('sequence', '').strip().upper()
            if not seq:
                continue
            try:
                mic = float(row.get('value', '').strip())
            except (ValueError, TypeError):
                continue

            # The best MIC per sequence
            if seq not in seen or mic < seen[seq]:
                seen[seq] = mic

    for seq, mic in seen.items():
        sequences.append(seq)
        mic_values.append(mic)
        labels.append(1 if mic <= mic_threshold else 0)

    return sequences, labels, mic_values


def parse_dbaasp_json(dirpath: str,
                      mic_threshold: float = 10.0
                      ) -> Tuple[List[str], List[int], List[float]]:
    """Parse DBAASP paginated JSON files from REST API.
    Expects directory with page*.json files. Each entry has:
    sequence, targetActivities[{targetOrganism, mic}], hemolytic activity.
    """
    sequences, labels, mic_values = [], [], []
    seen = {}

    json_files = sorted(Path(dirpath).glob('page*.json'))
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        entries = data if isinstance(data, list) else data.get('content', [])
        for entry in entries:
            seq = entry.get('sequence', '').strip().upper()
            if not seq:
                continue
            # Get best MIC from target activities
            activities = entry.get('targetActivities', [])
            best_mic = None
            for act in activities:
                try:
                    mic = float(act.get('mic', act.get('value', '')))
                    if best_mic is None or mic < best_mic:
                        best_mic = mic
                except (ValueError, TypeError):
                    continue

            if best_mic is not None:
                if seq not in seen or best_mic < seen[seq]:
                    seen[seq] = best_mic

    for seq, mic in seen.items():
        sequences.append(seq)
        mic_values.append(mic)
        labels.append(1 if mic <= mic_threshold else 0)

    return sequences, labels, mic_values


def parse_plain_text_sequences(filepath: str) -> List[Tuple[str, str]]:
    """Parse plain-text sequence file (one sequence per line).
    Used for ToxinPred and HemoPI datasets.
    Returns list of (header, sequence) tuples for compatibility with filter_sequences.
    """
    sequences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()

            if not line or line.startswith('#') or line.startswith('>'):
                continue

            # Some files have "index\tsequence" format
            parts = line.split('\t')
            seq = parts[-1].strip().upper() if len(parts) > 1 else line.upper()
            # Validate it looks like a peptide sequence
            if seq and all(c in AA_ALPHABET for c in seq):
                sequences.append((f"seq_{i}", seq))

    return sequences


def filter_sequences(sequences: List[Tuple[str, str]],
                     min_len: int = 5,
                     max_len: int = 25
                     ) -> List[Tuple[str, str]]:
    """Filter sequences by length and valid amino acids."""
    valid_aa = set(AA_ALPHABET)
    filtered = []

    for header, seq in sequences:
        seq_clean = seq.upper().strip()

        if min_len <= len(seq_clean) <= max_len:
            if all(aa in valid_aa for aa in seq_clean):
                filtered.append((header, seq_clean))

    return filtered


def deduplicate_sequences(sequences: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove duplicate sequences (keep first occurrence)."""
    seen = set()
    unique = []
    for header, seq in sequences:
        seq_hash = hashlib.md5(seq.encode()).hexdigest()

        if seq_hash not in seen:
            seen.add(seq_hash)
            unique.append((header, seq))

    return unique


class AMPDataset(Dataset):
    """Antimicrobial peptide dataset with activity labels.
    Loads from FASTA files and provides tokenized sequences with
    binary AMP activity labels and optional MIC values.
    """

    def __init__(self, sequences: List[str], labels: List[int],
                 mic_values: Optional[List[float]] = None,
                 target_species: Optional[List[str]] = None,
                 max_len: int = 25):
        self.sequences = sequences
        self.labels = labels
        self.mic_values = mic_values
        self.target_species = target_species
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = tokenize_sequence(self.sequences[idx], self.max_len)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        item = {'tokens': tokens, 'label': label, 'sequence': self.sequences[idx]}

        if self.mic_values is not None:
            mic = self.mic_values[idx]
            # Use NaN for missing MIC so batches are collatable
            item['mic'] = torch.tensor([mic if mic is not None else float('nan')],
                                       dtype=torch.float32)

        if self.target_species is not None:
            item['target_species'] = self.target_species[idx] or ""

        return item


class MimicryDataset(Dataset):
    """Dataset for molecular mimicry contrastive learning.
    Contains triplets: (peptide, positive_defense_peptide, negative_autoantigen)
    """

    def __init__(self, peptides: List[str],
                 defense_peptides: List[str],
                 autoantigens: List[str],
                 max_len: int = 25):
        self.peptides = peptides
        self.defense_peptides = defense_peptides
        self.autoantigens = autoantigens
        self.max_len = max_len

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        import random
        peptide_tokens = tokenize_sequence(self.peptides[idx], self.max_len)
        # Random positive and negative
        pos_idx = random.randint(0, len(self.defense_peptides) - 1)
        neg_idx = random.randint(0, len(self.autoantigens) - 1)

        return {
            'peptide': peptide_tokens,
            'positive': tokenize_sequence(self.defense_peptides[pos_idx], self.max_len),
            'negative': tokenize_sequence(self.autoantigens[neg_idx], self.max_len),
            'peptide_seq': self.peptides[idx],
            'positive_seq': self.defense_peptides[pos_idx],
            'negative_seq': self.autoantigens[neg_idx],
        }


class CombinedAMPDataset(Dataset):
    """Combined dataset from multiple AMP databases with deduplication.
    Sources: GRAMPA (CSV), DBAASP (JSON), DRAMP (FASTA), APD6 (FASTA), NCBI (FASTA).
    Handles heterogeneous formats and merges into unified tokenized dataset.
    """

    SOURCES = ['grampa', 'dbaasp', 'dramp', 'apd', 'ncbi']

    def __init__(self,
                 data_dir: str,
                 mic_threshold: float = 10.0,
                 min_len: int = 5,
                 max_len: int = 25,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1
                 ):
        self.data_dir = Path(data_dir)
        self.mic_threshold = mic_threshold
        self.max_len = max_len
        self.split = split

        all_sequences = []
        all_labels = []
        all_mic_values = []
        all_sources = []

        for source in self.SOURCES:
            source_dir = self.data_dir / source
            if not source_dir.exists():
                continue

            if source == 'grampa':
                seqs, labels, mics = self._load_grampa(source_dir)
            elif source == 'dbaasp':
                seqs, labels, mics = self._load_dbaasp(source_dir)
            else:
                seqs, labels, mics = self._load_fasta_source(source_dir)

            for seq, label, mic in zip(seqs, labels, mics):
                seq_clean = seq.upper().strip()
                if min_len <= len(seq_clean) <= max_len:
                    if all(aa in set(AA_ALPHABET) for aa in seq_clean):
                        all_sequences.append(seq_clean)
                        all_labels.append(label)
                        all_mic_values.append(mic)
                        all_sources.append(source)

        # Deduplicate (keep entry with lowest MIC)
        best = {}  # seq -> (label, mic, source)
        for seq, label, mic, src in zip(all_sequences, all_labels, all_mic_values, all_sources):
            if seq not in best or (mic is not None and (best[seq][1] is None or mic < best[seq][1])):
                best[seq] = (label, mic, src)

        unique_seqs = list(best.keys())
        unique_labels = [best[s][0] for s in unique_seqs]
        unique_mics = [best[s][1] for s in unique_seqs]
        unique_sources = [best[s][2] for s in unique_seqs]

        # Deterministic split based on sequence hash
        n = len(unique_seqs)
        indices = list(range(n))
        indices.sort(key=lambda i: hashlib.md5(unique_seqs[i].encode()).hexdigest())

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == 'train':
            selected = indices[:train_end]
        elif split == 'val':
            selected = indices[train_end:val_end]
        else:
            selected = indices[val_end:]

        self.sequences = [unique_seqs[i] for i in selected]
        self.labels = [unique_labels[i] for i in selected]
        self.mic_values = [unique_mics[i] for i in selected]
        self.sources = [unique_sources[i] for i in selected]

    def _load_grampa(self, source_dir: Path) -> Tuple[List[str], List[int], List[Optional[float]]]:
        """Load GRAMPA CSV with MIC values."""
        csv_file = source_dir / 'grampa.csv'

        if csv_file.exists():
            return parse_grampa_csv(str(csv_file), self.mic_threshold)

        return [], [], []

    def _load_dbaasp(self, source_dir: Path) -> Tuple[List[str], List[int], List[Optional[float]]]:
        """Load DBAASP JSON pages or processed FASTA."""
        json_files = list(source_dir.glob('page*.json'))

        if json_files:
            return parse_dbaasp_json(str(source_dir), self.mic_threshold)

        # Fallback to FASTA if already processed
        return self._load_fasta_source(source_dir)

    def _load_fasta_source(self, source_dir: Path) -> Tuple[List[str], List[int], List[Optional[float]]]:
        """Load FASTA-based source (DRAMP, APD, NCBI)."""
        sequences, labels, mic_values = [], [], []

        # Load positive AMPs
        pos_files = list(source_dir.glob('*positive*.fasta')) + \
                    list(source_dir.glob('*amp*.fasta')) + \
                    list(source_dir.glob('*active*.fasta')) + \
                    list(source_dir.glob('*natural*.fasta')) + \
                    list(source_dir.glob('*antibacterial*.fasta')) + \
                    list(source_dir.glob('*general*.fasta'))

        for f in pos_files:
            seqs = parse_fasta(str(f))
            for _, seq in seqs:
                sequences.append(seq)
                labels.append(1)
                mic_values.append(None)

        # Load negative (non-AMP) sequences
        neg_files = list(source_dir.glob('*negative*.fasta')) + \
                    list(source_dir.glob('*non_amp*.fasta')) + \
                    list(source_dir.glob('*non-amp*.fasta'))
        for f in neg_files:
            seqs = parse_fasta(str(f))
            for _, seq in seqs:
                sequences.append(seq)
                labels.append(0)
                mic_values.append(None)

        return sequences, labels, mic_values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = tokenize_sequence(self.sequences[idx], self.max_len)
        item = {
            'tokens': tokens,
            'label': torch.tensor([self.labels[idx]], dtype=torch.float32),
            'sequence': self.sequences[idx],
            'source': self.sources[idx],
        }

        if self.mic_values[idx] is not None:
            item['mic'] = torch.tensor([self.mic_values[idx]], dtype=torch.float32)

        return item

    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        from collections import Counter
        source_counts = Counter(self.sources)
        label_counts = Counter(self.labels)
        lengths = [len(s) for s in self.sequences]
        mic_available = sum(1 for m in self.mic_values if m is not None)
        return {
            'total': len(self),
            'by_source': dict(source_counts),
            'positive': label_counts.get(1, 0),
            'negative': label_counts.get(0, 0),
            'mic_available': mic_available,
            'mean_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
        }


class SafetyDataset(Dataset):
    """Dataset for safety module training (toxicity + hemolysis).
    Loads ToxinPred and HemoPI datasets. Supports both FASTA and plain-text formats.
    Provides multi-label output: [toxic, hemolytic].
    """

    def __init__(self, data_dir: str, min_len: int = 5, max_len: int = 25,
                 split: str = 'train', train_ratio: float = 0.8, val_ratio: float = 0.1):
        self.data_dir = Path(data_dir)
        self.max_len = max_len

        sequences = []
        toxic_labels = []
        hemo_labels = []

        # Load ToxinPred
        toxin_dir = self.data_dir / 'toxinpred'
        if toxin_dir.exists():
            for pos_file in self._find_files(toxin_dir, 'positive'):
                seqs = self._load_sequences(pos_file, min_len, max_len)
                for seq in seqs:
                    sequences.append(seq)
                    toxic_labels.append(1)
                    hemo_labels.append(-1)  # -1 = unknown
            for neg_file in self._find_files(toxin_dir, 'negative'):
                seqs = self._load_sequences(neg_file, min_len, max_len)
                for seq in seqs:
                    sequences.append(seq)
                    toxic_labels.append(0)
                    hemo_labels.append(-1)

        # Load HemoPI
        hemo_dir = self.data_dir / 'hemopi'
        if hemo_dir.exists():
            for pos_file in self._find_files(hemo_dir, 'positive'):
                seqs = self._load_sequences(pos_file, min_len, max_len)

                for seq in seqs:
                    sequences.append(seq)
                    toxic_labels.append(-1)  # unknown
                    hemo_labels.append(1)

            for neg_file in self._find_files(hemo_dir, 'negative'):
                seqs = self._load_sequences(neg_file, min_len, max_len)

                for seq in seqs:
                    sequences.append(seq)
                    toxic_labels.append(-1)
                    hemo_labels.append(0)

        # Deduplicate
        seen = {}
        for seq, tox, hem in zip(sequences, toxic_labels, hemo_labels):
            if seq not in seen:
                seen[seq] = [tox, hem]
            else:
                if seen[seq][0] == -1 and tox != -1:
                    seen[seq][0] = tox
                if seen[seq][1] == -1 and hem != -1:
                    seen[seq][1] = hem

        unique_seqs = list(seen.keys())
        unique_labels = [seen[s] for s in unique_seqs]

        # Split
        n = len(unique_seqs)
        indices = list(range(n))
        indices.sort(key=lambda i: hashlib.md5(unique_seqs[i].encode()).hexdigest())

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == 'train':
            selected = indices[:train_end]
        elif split == 'val':
            selected = indices[train_end:val_end]
        else:
            selected = indices[val_end:]

        self.sequences = [unique_seqs[i] for i in selected]
        self.labels = [unique_labels[i] for i in selected]

    def _find_files(self, directory: Path, pattern: str) -> List[Path]:
        """Find data files matching pattern (FASTA or plain text)."""
        files = list(directory.glob(f'*{pattern}*.fasta')) + \
                list(directory.glob(f'*{pattern}*.txt')) + \
                list(directory.glob(f'*{pattern}*main*'))
        return files

    def _load_sequences(self, filepath: Path, min_len: int, max_len: int) -> List[str]:
        """Load sequences from FASTA or plain text file."""
        filepath_str = str(filepath)
        # Try FASTA first
        if filepath.suffix == '.fasta' or filepath.suffix == '.fa':
            seqs = parse_fasta(filepath_str)
        else:
            seqs = parse_plain_text_sequences(filepath_str)

        filtered = filter_sequences(seqs, min_len, max_len)
        return [seq for _, seq in filtered]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = tokenize_sequence(self.sequences[idx], self.max_len)
        tox, hem = self.labels[idx]
        return {
            'tokens': tokens,
            'toxic': torch.tensor([tox], dtype=torch.float32),
            'hemolytic': torch.tensor([hem], dtype=torch.float32),
            'sequence': self.sequences[idx],
            'mask_toxic': torch.tensor([1.0 if tox != -1 else 0.0]),
            'mask_hemolytic': torch.tensor([1.0 if hem != -1 else 0.0]),
        }
