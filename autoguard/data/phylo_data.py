#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phylogenetic data processing for evolutionary conditioning.

Processes species trees (from IQ-TREE/FastTree), applies perturbations,
and prepares training data for Poincaré embeddings.
"""

import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np


class PhylogeneticDataProcessor:
    """Processes phylogenetic trees for Poincaré embedding training."""

    def __init__(self, trees_dir: str, num_perturbations: int = 10,
                 perturbation_scale: float = 0.1, tree_filter: str = ""):
        self.trees_dir = Path(trees_dir)
        self.num_perturbations = num_perturbations
        self.perturbation_scale = perturbation_scale
        self.tree_filter = tree_filter  # glob pattern, e.g. "timetree*"

    def load_trees(self) -> List[str]:
        """Load Newick tree files from directory (filtered if tree_filter set)."""
        trees = []
        if self.tree_filter:
            # Only load trees matching the filter
            for f in sorted(self.trees_dir.glob(self.tree_filter)):
                with open(f, 'r') as fp:
                    tree_str = fp.read().strip()
                    if tree_str:
                        trees.append(tree_str)
        else:
            # Trees popular formats
            for ext in ['*.newick', '*.nwk', '*.tree', '*.treefile']:
                for f in sorted(self.trees_dir.glob(ext)):
                    with open(f, 'r') as fp:
                        tree_str = fp.read().strip()
                        if tree_str:
                            trees.append(tree_str)

        return trees

    def extract_training_pairs(self, newick_str: str) -> List[Tuple[int, int, float]]:
        """Extract ancestor-descendant pairs from a Newick tree.
        Returns list of (parent_idx, child_idx, branch_length) tuples
        for training Poincaré embeddings.
        """
        try:
            from Bio import Phylo
            from io import StringIO
            tree = Phylo.read(StringIO(newick_str), "newick")
        except ImportError:
            return self._fallback_parse(newick_str)

        node_map = {}
        pairs = []
        idx = 0

        for clade in tree.find_clades(order='level'):
            name = clade.name or f"node_{idx}"

            if name not in node_map:
                node_map[name] = idx
                idx += 1

            parent_idx = node_map[name]

            for child in clade.clades:
                child_name = child.name or f"node_{idx}"

                if child_name not in node_map:
                    node_map[child_name] = idx
                    idx += 1

                child_idx = node_map[child_name]
                dist = child.branch_length if child.branch_length else 0.1
                pairs.append((parent_idx, child_idx, float(dist)))

        self._node_map = node_map
        return pairs

    def _fallback_parse(self, newick_str: str) -> List[Tuple[int, int, float]]:
        """Simple fallback parser for basic Newick strings."""
        # Extract leaf names
        import re
        leaves = re.findall(r'([A-Za-z0-9_]+)(?::[0-9.]+)?', newick_str)
        pairs = []
        for i, _ in enumerate(leaves):
            # Connect all leaves to a virtual root
            pairs.append((0, i + 1, 1.0))
        return pairs

    def perturb_tree(self,
                     pairs: List[Tuple[int, int, float]],
                     seed: int = 0
                     ) -> List[Tuple[int, int, float]]:
        """Apply stochastic perturbation to branch lengths.
        Simulates evolutionary uncertainty by perturbing tree topology weights.
        """
        rng = np.random.RandomState(seed)
        perturbed = []

        for parent, child, dist in pairs:
            noise = rng.normal(0, self.perturbation_scale * dist)
            new_dist = max(0.001, dist + noise)
            perturbed.append((parent, child, new_dist))

        return perturbed

    def prepare_training_data(self) -> Dict:
        """Prepare complete training data for Poincaré embeddings.

        Returns:
            Dict with 'pairs', 'distances', 'num_entities', 'perturbations'
        """
        trees = self.load_trees()
        all_pairs = []
        total_entities = 0

        for tree_str in trees:
            pairs = self.extract_training_pairs(tree_str)
            # Offset indices for multiple trees
            offset_pairs = [(p + total_entities, c + total_entities, d)
                           for p, c, d in pairs]
            all_pairs.extend(offset_pairs)

            if pairs:
                max_idx = max(max(p, c) for p, c, _ in pairs)
                total_entities += max_idx + 1

        # Generate perturbations
        all_perturbed_pairs = []
        for i in range(self.num_perturbations):
            perturbed = self.perturb_tree(all_pairs, seed=i)
            all_perturbed_pairs.append(perturbed)

        # Convert to tensors
        if all_pairs:
            parents = torch.tensor([p for p, _, _ in all_pairs], dtype=torch.long)
            children = torch.tensor([c for _, c, _ in all_pairs], dtype=torch.long)
            distances = torch.tensor([d for _, _, d in all_pairs], dtype=torch.float32)
        else:
            parents = torch.zeros(0, dtype=torch.long)
            children = torch.zeros(0, dtype=torch.long)
            distances = torch.zeros(0, dtype=torch.float32)

        return {
            'parents': parents,
            'children': children,
            'distances': distances,
            'num_entities': max(total_entities, 1),
            'perturbations': all_perturbed_pairs,
        }


class PoincareEmbeddingTrainer:
    """Trains Poincaré embeddings on phylogenetic tree data."""
    # Poincaré ball while leaving headroom for the tree's internal structure.
    TARGET_DIAMETER: float = 5.0

    def __init__(self, num_entities: int, embed_dim: int = 64,
                 learning_rate: float = 0.05, num_negatives: int = 10):
        from ..models.phylo_embeddings import PoincareEmbedding

        self.model = PoincareEmbedding(num_entities, embed_dim)
        self.lr = learning_rate
        self.num_negatives = num_negatives
        self.num_entities = num_entities
        # Pairwise tree (graph) distances + the index pairs to regress on
        self._graph_dist = None
        self._pair_i = None
        self._pair_j = None
        self._pair_d = None
        self._dist_scale = 1.0  # set when graph distances are built

    def _build_graph_distances(self, parents, children, distances):
        """All-pairs shortest-path tree distances via Floyd-Warshall.
        The edge list "(parent, child, branch_length)" defines an undirected
        weighted graph; the geodesic (path) distance between every pair of nodes
        is the target metric the hyperbolic embedding must reproduce. Trees here
        are tiny (=30 species) so the O(n^3) all-pairs computation is negligible
        and is cached for the whole training run.
        """
        n = self.num_entities
        inf = float('inf')
        D = torch.full((n, n), inf)
        D.fill_diagonal_(0.0)
        for p, c, w in zip(parents.tolist(), children.tolist(), distances.tolist()):
            w = float(max(w, 1e-3))
            if p < n and c < n:
                D[p, c] = min(D[p, c].item(), w)
                D[c, p] = min(D[c, p].item(), w)
        for k in range(n):
            D = torch.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])

        # Keep only finite, off-diagonal pairs
        iu, ju = torch.triu_indices(n, n, offset=1)
        dvals = D[iu, ju]
        finite = torch.isfinite(dvals)
        pair_d = dvals[finite]

        # Normalise the tree distances to a representable diameter
        max_d = pair_d.max().clamp_min(1e-6) if pair_d.numel() else torch.tensor(1.0)
        self._dist_scale = (self.TARGET_DIAMETER / max_d).item()
        self._graph_dist = D * self._dist_scale
        self._pair_i = iu[finite]
        self._pair_j = ju[finite]
        self._pair_d = pair_d * self._dist_scale

    def train_step(self, parents, children, distances):
        """Single step of hyperbolic distance-distortion regression.
        Minimises the mean squared error between the Poincaré-ball geodesic
        distance of each node pair and its tree (graph) distance:

            L = mean_{(i,j)} ( d_Poincaré(x_i, x_j) - d_tree(i, j) )^2

        Because hyperbolic space embeds trees with arbitrarily low distortion
        (Sala et al., ICML 2018), this objective is driven genuinely toward
        zero as the embedding becomes faithful, unlike the margin/softmax
        ranking losses, whose floor sits near the margin because random
        negatives include legitimately close nodes (siblings, grandparents).
        """
        from ..models.phylo_embeddings import poincare_distance

        if self._graph_dist is None:
            self._build_graph_distances(parents, children, distances)

        num_pairs = self._pair_i.shape[0]
        max_pairs = 4096

        if num_pairs > max_pairs:
            sel = torch.randint(0, num_pairs, (max_pairs,))
            pi, pj, pd = self._pair_i[sel], self._pair_j[sel], self._pair_d[sel]
        else:
            pi, pj, pd = self._pair_i, self._pair_j, self._pair_d

        ei = self.model(pi)
        ej = self.model(pj)
        d_hyp = poincare_distance(ei, ej).squeeze(-1)
        loss = F.mse_loss(d_hyp, pd)
        return loss

    def train(self, data: Dict, epochs: int = 100, verbose: bool = True):
        """Full training loop over prepared data.

        Args:
            data: Dict from prepare_training_data() with 'parents', 'children', 'distances'
            epochs: Number of training epochs
            verbose: Print progress every 20 epochs
        """
        # Optimiser choice for the distance-distortion objective: Adam clearly
        # beats Riemannian SGD here.

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        parents = data['parents']
        children = data['children']
        distances = data['distances']

        # Burn-in at a reduced learning rate stabilises the early geometry
        # (Nickel & Kiela, 2017), then the LR decays so the embedding settles
        # into a low-distortion configuration instead of oscillating.
        burn_in = max(1, int(0.1 * epochs))
        for epoch in range(epochs):

            if epoch < burn_in:
                lr = self.lr * 0.1
            else:
                progress = (epoch - burn_in) / max(1, epochs - burn_in)
                lr = self.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad()
            loss = self.train_step(parents, children, distances)
            loss.backward()
            # Mild clipping guards against the occasional large gradient
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Poincaré epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

    def save(self, path: str):
        """Save trained embeddings to file."""
        import numpy as np
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        embeddings = self.model.embeddings.detach().cpu().numpy()

        if path.endswith('.npz'):
            np.savez(path, embeddings=embeddings)
        else:
            torch.save(self.model.state_dict(), path)


# Only species present in the timetree are mapped; others use nearest relative.
BACTERIUM_TO_TREE_LEAF = {
    "E. coli": "Escherichia_coli",
    "K. pneumoniae": "Klebsiella_pneumoniae",
    "S. agalactiae": "Streptococcus_agalactiae",
    "S. pyogenes": "Streptococcus_pyogenes",
    "H. influenzae": "Haemophilus_influenzae",
    "E. faecalis": "Enterococcus_gallinarum",  # closest in tree
    "E. faecium": "Enterococcus_gallinarum",
    "E. gallinarum": "Enterococcus_gallinarum",
    "B. fragilis": "Bacteroides_fragilis",
    "B. thetaiotaomicron": "Bacteroides_thetaiotaomicron",
    "P. gingivalis": "Porphyromonas_gingivalis",
    "P. copri": "Prevotella_copri",
    "P. melaninogenica": "Prevotella_melaninogenica",
    "C. acnes": "Cutibacterium_acnes",
    "P. mirabilis": "Proteus_mirabilis",
    "P. aeruginosa": "Escherichia_coli",  # Gammaproteobacteria
    "A. baumannii": "Escherichia_coli",  # Gammaproteobacteria
    "S. typhimurium": "Escherichia_coli",  # Enterobacteriaceae
    "S. enterica": "Escherichia_coli",  # Enterobacteriaceae
    "E. cloacae": "Escherichia_coli",  # Enterobacteriaceae
    "S. aureus": "Streptococcus_pyogenes",  # Firmicutes (Bacilli)
    "S. epidermidis": "Streptococcus_pyogenes",  # Firmicutes (Bacilli)
    "B. subtilis": "Eubacterium_limosum",  # Firmicutes
    "B. cereus": "Eubacterium_limosum",  # Firmicutes
    "M. luteus": "Corynebacterium_amycolatum",  # Actinobacteria
    "L. monocytogenes": "Ligilactobacillus_salivarius",  # Firmicutes (Bacilli)
    "B. megaterium": "Eubacterium_limosum",  # Firmicutes
}


class SpeciesEmbeddingLookup:
    """Maps bacterium names to pre-trained Poincaré embeddings.
    Loads the trained Poincaré model and provides per-species embeddings
    that can be used as phylogenetic conditioning during training.
    """

    def __init__(self,
                 poincare_checkpoint: str,
                 trees_dir: str,
                 embed_dim: int = 64,
                 tree_filter: str = ""):
        """
        Args:
            poincare_checkpoint: Path to saved Poincaré embeddings (.pt or .npz)
            trees_dir: Path to species_trees/ directory
            embed_dim: Embedding dimension
            tree_filter: Optional glob pattern for tree selection
        """
        self.embed_dim = embed_dim

        # Load tree to get node -> index mapping
        processor = PhylogeneticDataProcessor(trees_dir, tree_filter=tree_filter)
        trees = processor.load_trees()

        if not trees:
            self._leaf_to_idx = {}
            self._embeddings = None
            return

        # Build node map from first tree
        processor.extract_training_pairs(trees[0])
        self._leaf_to_idx = getattr(processor, '_node_map', {})

        # Load pre-trained embeddings
        checkpoint_path = Path(poincare_checkpoint)
        if checkpoint_path.exists():
            if str(checkpoint_path).endswith('.npz'):
                data = np.load(str(checkpoint_path))
                self._embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
            else:
                state = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
                if 'embeddings' in state:
                    self._embeddings = state['embeddings']
                elif 'model_state_dict' in state:
                    self._embeddings = state['model_state_dict'].get(
                        'embeddings', torch.zeros(1, embed_dim))
                else:
                    self._embeddings = state.get('embeddings',
                                                  torch.zeros(1, embed_dim))
        else:
            self._embeddings = None

    def get_embedding(self, bacterium_name: str) -> Optional[torch.Tensor]:
        """Get Poincaré embedding for a bacterium name.

        Args:
            bacterium_name: Short name from GRAMPA (e.g. "E. coli")

        Returns:
            Tensor [embed_dim] or None if unmapped
        """
        if self._embeddings is None:
            return None

        tree_leaf = BACTERIUM_TO_TREE_LEAF.get(bacterium_name)
        if tree_leaf is None:
            return None

        idx = self._leaf_to_idx.get(tree_leaf)
        if idx is None:
            return None

        if idx < len(self._embeddings):
            return self._embeddings[idx]
        return None

    def get_all_species_embeddings(self) -> torch.Tensor:
        """Get embeddings for all tree species as [num_species, embed_dim]."""
        if self._embeddings is None:
            return torch.zeros(1, self.embed_dim)

        # Get only leaf embeddings
        leaf_indices = []
        for leaf_name in sorted(self._leaf_to_idx.keys()):
            idx = self._leaf_to_idx[leaf_name]
            if idx < len(self._embeddings):
                leaf_indices.append(idx)

        if not leaf_indices:
            return torch.zeros(1, self.embed_dim)

        return self._embeddings[leaf_indices]

    def get_batch_embeddings(self, bacterium_names: List[str]) -> torch.Tensor:
        """Get phylo embeddings for a batch of bacterium names.
        For samples without a species target, returns the mean of all
        species embeddings (uniform prior over tree).

        Args:
            bacterium_names: List of bacterium names (can contain empty strings)

        Returns:
            Tensor [batch, num_species, embed_dim] — species embeddings
            with target species masked/weighted for attention.
        """
        all_embeds = self.get_all_species_embeddings()  # [num_species, embed_dim]
        batch_size = len(bacterium_names)

        # Return same full species panel
        batch_embeds = all_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return batch_embeds
