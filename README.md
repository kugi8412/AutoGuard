# AutoGuard
<img src="./Logo_AutoGuard.png" alt="Logo" width="250"/></td>

**Evolutionary-Conditioned Graph-Based AMP Generation with Molecular Mimicry Detection**

AutoGuard is an end-to-end deep learning framework for generating antimicrobial peptides (AMPs) that are potent, evolutionary-robust, immunologically safe, and interpretable. It integrates:

- **HydrAMP-style Bi-GRU encoder/decoder** (Szymczak et al., 2023)
- **GG-FiLM graph encoder** (Brockschmidt, 2020) for physicochemical features
- **VQ-VAE discrete codebook** (van den Oord et al., 2017) replacing continuous latent
- **Poincaré phylogenetic embeddings** (Nickel & Kiela, 2017) for evolutionary conditioning
- **Contrastive molecular mimicry detection** (ESM-2 based) for autoantigen avoidance
- **Multi-task safety module** (toxicity, hemolysis, immunogenicity)
- **Sparse Autoencoder** for mechanistic interpretability (Cunningham et al., 2024)

---

## Quick Start (End-to-End)

### 0. Get data

```bash
bash get_data.sh
```

### 1. Install

```bash
# Clone and enter the workspace
cd AutoGuard

# Create virtual environment (Python 3.9+)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install AutoGuard (minimal)
pip install -e autoguard/

# For full features (graph encoder + ESM-2):
pip install -e "autoguard/[full]"
```

### 2. Smoke Test

```bash
python -m autoguard.scripts.smoke_test
```

This runs 14 checks covering:
- Config & tokenization
- Encoder/Decoder forward pass
- VQ-VAE quantization
- Phylogenetic conditioning
- Safety module
- Full model forward + backward
- AMP generation
- Evaluation metrics
- GNN baseline
- HydrAMP baseline adapter
- Model comparison harness
- Sparse Autoencoder

### 3. Compare Models

```bash
# Quick smoke comparison (~1 min, CPU)
python -m autoguard.scripts.compare_models --mode smoke

# Full comparison (longer, use --device cuda if available)
python -m autoguard.scripts.compare_models --mode full --device cuda
```

Output: `comparison_report.json` with metrics table:

| Model     | Novelty | Diversity | AMP Hit Rate | Quality Score |
|-----------|---------|-----------|--------------|---------------|
| AutoGuard | 1.00    | 0.89      | 1.00         | 0.63          |
| HydrAMP   | 1.00    | 0.52      | 1.00         | 0.58          |
| GNN-VAE   | 1.00    | 0.91      | 0.22         | 0.44          |

*(Results from smoke test with 50 generated sequences each; full mode with 1000 sequences and 2000 training sequences produces similar ranking)*

---

## Reproducible Snakemake Workflow

```
autoguard/workflow/Snakefile                 # combined AutoGuard-vs-HydrAMP pipeline
autoguard/workflow/Snakefile_autoguard       # AutoGuard-only, one rule per stage
autoguard/workflow/Snakefile_hydramp         # HydrAMP-only, two-phase training
autoguard/workflow/config/config.yaml        # Windows / local venv defaults (CPU smoke)
autoguard/workflow/config/config_linux.yaml  # Linux / cluster defaults (full-data GPU)
autoguard/workflow/envs/autoguard.yaml       # conda environment
autoguard/data/                              # prepared data + static timetree.nwk
autoguard/checkpoints, autoguard/results, autoguard/logs   # outputs
```

DAG (combined `Snakefile`): `prepare_data -> poincare -> train_autoguard ->
{train_sae, eval_autoguard}`, `train_hydramp -> eval_hydramp`, then `compare`.

Run from the **repository root**:

```bash
# Windows (PowerShell / cmd.exe)
snakemake -s autoguard/workflow/Snakefile --cores 1

# Linux / macOS (bash), conda env recocomended
conda env create -f autoguard/workflow/envs/autoguard.yaml
conda activate autoguard
snakemake -s autoguard/workflow/Snakefile \
    --configfile autoguard/workflow/config/config_linux.yaml --cores 4

# Train only one model (per-stage workflows, epoch budgets fully config-driven)
snakemake -s autoguard/workflow/Snakefile_autoguard --cores 1
snakemake -s autoguard/workflow/Snakefile_hydramp   --cores 1

# Preview the DAG (either OS)
snakemake -s autoguard/workflow/Snakefile -n

# Override defaults on the command line
snakemake -s autoguard/workflow/Snakefile --cores 1 \
    --config seed=7 autoguard_epochs=20 max_train=0 device=cuda use_esm=true
```

Defaults differ per config: `config.yaml` (Windows / local venv) is a fast
**CPU smoke** profile (10 epochs, `max_train: 1024`); `config_linux.yaml`
(cluster) is a **full-data GPU** profile (`max_train: 0`, converged epoch
budgets, `device: cuda`). Both use seed 42 and the static `timetree.nwk`
phylogeny, with ESM-2 cross-attention **optional** (`use_esm: true`, needs
`fair-esm`). The evaluation step reports AMP-challenge-2027 metrics (see below)
alongside the generation-quality metrics.

### AMP-Challenge-2027 reporting metrics

`evaluate_model` and the comparison report now include the metrics required by
[`amp-challenge-2027-main/README.md`](./amp-challenge-2027-main/README.md):

| Metric | Meaning |
|--------|---------|
| `unique_fraction` | Fraction of unique sequences (no duplicates) |
| `length_valid_fraction` | Fraction within the 8–50 residue rule |
| `canonical_fraction` | Fraction using only the 20 standard amino acids |
| `challenge_valid_fraction` | Unique sequences that are valid **and** absent from `antibacterial.fasta` |
| `num_overlap_reference` | Count identical to a known antibacterial peptide |
| `predicted_success_rate` | Fraction meeting the potency threshold (MIC ≤ 16 µM) |
| `predicted_mic50` / `predicted_mic90` | Median / 90th-percentile predicted MIC (µM) |
| `predicted_safety_window` | Predicted HC50 / MIC50 (Optimal Selectivity) |
| `predicted_non_hemolytic_fraction` | Fraction with low predicted hemolysis |
| `top_k_max_identity` / `top_k_identity_violations` | Top-100 Levenshtein identity vs reference (must be <80%) |

> MIC / HC50 values are **model-predicted proxies** (prefixed `predicted_`),
> since wet-lab assay values are not available locally. Structural metrics
> (length, alphabet, uniqueness, reference overlap, top-k identity) are exact.

---

## Full Control of Model Parameters (config.py via the workflow)

Snakemake controls the **training schedule** (epochs / batch size / lr / device /
`max_train`) through `config.yaml`. The **architecture, loss weights and data
thresholds** in [`config.py`](./autoguard/config.py) are controlled by a separate
override file passed with `--model_config`:

```
autoguard/workflow/config/model_params.yaml   # maps onto ModelConfig / LossWeights / DataConfig
```

It has three sections that mirror the dataclasses in `config.py`:

```yaml
model:                 # -> ModelConfig (architecture, codebook, decoder, SAE, thresholds)
  latent_dim: 48
  num_codebook_vectors: 128
  toxicity_threshold: 0.3
loss_weights:          # -> LossWeights (multi-task loss term weights)
  mimicry_penalty: 0.2
  safety_penalty: 0.15
data:                  # -> DataConfig (paths, thresholds, splits)
  tree_filter: "timetree*"
```

The Snakefiles read the `model_config:` key in `config.yaml` / `config_linux.yaml`
and pass it to every training stage, so one file gives you full control of all
parameters. Omit any key to keep its `config.py` default; unknown keys are
ignored with a warning. You can also pass it directly:

```bash
python -m autoguard.scripts.train --stage full --data_dir data/ \
    --model_config autoguard/workflow/config/model_params.yaml
```

> Precedence: `config.py` defaults < `model_params.yaml` < CLI/workflow schedule
> (epochs/batch/lr always win). Changing architecture fields (e.g. `latent_dim`,
> `num_codebook_vectors`) makes existing checkpoints incompatible — retrain from
> scratch. If you change `latent_dim`, set `codebook_dim` to the same value.

## Hyperparameter Search (all stages)

[`hyperparameter_search.py`](./autoguard/scripts/hyperparameter_search.py) runs a grid or
random search across **any** parameter — `model.*`, `loss_weights.*`, `data.*`,
and per-stage `epochs` / `batch_size` / `lr` — for the staged training protocol.
Each trial writes a per-trial config, runs the listed stages in order, evaluates
the checkpoint, and records the chosen metric; the best config is saved as
`best_model_params.yaml` (ready to feed back into `--model_config`).

```bash
python -m autoguard.scripts.hyperparameter_search \
    --search_space autoguard/workflow/config/search_space.yaml
```

Edit [`search_space.yaml`](./autoguard/workflow/config/search_space.yaml) to set `method`
(`grid`/`random`), the optimized `metric` (any key in `metrics.json`), the
`stages` run per trial, and the candidate value lists, e.g.:

```yaml
method: grid
metric: challenge_valid_fraction
stages: ["poincare", "full", "sae"]
search_space:
  full.lr: [0.0003, 0.0005]
  full.batch_size: [64, 128]
  model.num_codebook_vectors: [128, 256]
  loss_weights.mimicry_penalty: [0.1, 0.2]
```

Outputs land in `autoguard/results/search/`: `search_results.csv`,
`search_summary.json`, and `best_model_params.yaml`.

## Google Colab (download from GitHub + inference + SAE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kugi8412//AutoGuard/blob/main/autoguard_colab.ipynb)

Self-contained Colab notebook that clones the repo from a **user-set GitHub
URL**, downloads a trained checkpoint, generates peptides, renders all figures
(length / amino-acid composition / AMP-score / safety distributions), visualises
the **VQ-VAE codebook** (usage histogram, code-vector heatmap, PCA), runs the
Sparse Autoencoder to surface interpretable features, and produces **XAI
per-residue saliency** maps for the AMP and safety heads. Set `GITHUB_REPO`,
`GITHUB_BRANCH` and `CHECKPOINT_PATH` in the first config cell.

---

## Full Training Pipeline


```bash
# Downloads GRAMPA, APD6 and standardizes into data/processed/
python -m autoguard.scripts.prepare_data --output_dir data/
```

This creates:
- `data/processed/amp_train.csv` (51,756 sequences)
- `data/processed/amp_val.csv` (6,347 sequences)  
- `data/processed/amp_test.csv` (6,517 sequences)
- `data/processed/mimicry_positives.fasta` / `mimicry_negatives.fasta`
- `data/species_trees/` (bundled **static** `timetree.nwk` — SAAP auto-trees are not used)

### Step 2: Download Additional Datasets

```bash
# Create directory structure + print download instructions
python -m autoguard.scripts.download_data --setup_only --output_dir data/

# Auto-download where possible (GRAMPA, APD6)
python -m autoguard.scripts.download_data --dataset all --auto --output_dir data/
```

Optional manual downloads for richer training:
- DBAASP: https://dbaasp.org/api?page=rest
- DRAMP: https://dramp.cpu-bioinfor.org/downloads/
- IEDB: https://www.iedb.org/database_export_v3.php
- ToxinPred: https://webs.iiitd.edu.in/raghava/toxinpred/dataset.php
- HemoPI: https://webs.iiitd.edu.in/raghava/hemopi/datasets.php

### Step 3: Phylogenetic Tree (static, bundled)

Phylogenetic conditioning uses **only** the curated static tree bundled with the
package at `autoguard/data/species_trees/timetree.nwk` (a TimeTree-derived
ultrametric species tree). It is selected via `tree_filter: "timetree*"`.

> The automatically generated supertrees / consensus trees from the
> SnakeAnalysisPhylogenomicsPipeline (SAAP) are **deliberately not used**.

No extra step is required — `prepare_data` copies the bundled tree into the
run's `data/species_trees/` automatically.

### Step 4: Train AutoGuard

```bash
python -m autoguard.scripts.train \
    --data_dir data/ \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-4 \
    --device cuda \
    --save_dir checkpoints/
```

Training stages (automatic):
1. Encoder warm-up (reconstruction only)
2. VQ codebook learning
3. Full multi-task training with KL/Gumbel annealing

### Step 5: Generate AMPs

```bash
python -m autoguard.scripts.generate \
    --checkpoint checkpoints/best_model.pt \
    --num_samples 1000 \
    --temperature 0.5 \
    --safety_threshold 0.3 \
    --output generated_amps.json \
    --output_fasta generated_amps.fasta
```

### Step 6: Evaluate

```bash
python -m autoguard.scripts.evaluate \
    --generated generated_amps.json \
    --training_data data/ \
    --output_report evaluation_report.md
```

### Step 7: SLE-Safe Generation

```bash
# Filter IEDB for SLE epitopes
python -m autoguard.scripts.filter_iedb --disease "Lupus|SLE" \
    --input data/iedb/epitope_full_v3.csv \
    --output data/iedb/sle_epitopes.csv

# Train with SLE-specific mimicry penalty
python -m autoguard.scripts.train \
    --data_dir data/ \
    --epochs 200 \
    --device cuda
```

---

## Baseline Comparison Details

### HydrAMP Baseline

The HydrAMP PyTorch implementation lives in `amp-challenge-2027-main/`. It implements the complete cVAE architecture from Szymczak et al. (2023):
- Bidirectional GRU encoder
- Autoregressive GRU→LSTM decoder with Gumbel-Softmax
- Conv1D+LSTM AMP classifier + LSTM MIC classifier
- Sleep-phase training with unconstrained generation

To train standalone:
```bash
cd amp-challenge-2027-main
pip install -e .
uv run train_hydramp --data-dir ./data --epochs 50
```

### GNN-VAE Baseline

A lightweight graph neural network VAE (`autoguard/models/gnn_baseline.py`) that:
- Uses per-residue physicochemical node features
- Implements dense message-passing (no torch-geometric needed)
- Has a GRU decoder for sequence generation
- Includes an AMP scoring head

Designed to demonstrate the benefit of AutoGuard's discrete codebook and multimodal conditioning by ablation.

---

## Model Mechanism & Mathematics

AutoGuard is a **conditional, discrete-latent sequence model**. A peptide
$x = (x_1,\dots,x_L)$ over the 20-letter amino-acid alphabet is encoded into a
discrete latent, conditioned on evolutionary / structural / safety signals via
cross-attention, and decoded **autoregressively**. This section describes each
mechanism with its governing equations and references.

### 1. Token & positional embeddings

Each residue index $x_t \in \{0,\dots,20\}$ (0 = `<pad>`) is mapped to a dense
vector by a learned lookup table $E \in \mathbb{R}^{V \times d}$ ($V=21$,
$d=$ `embedding_dim`), and combined with a sinusoidal positional code so the
model is order-aware:

$$
e_t = E[x_t] + p_t,\qquad
p_{t,2i} = \sin\!\Big(\tfrac{t}{10000^{2i/d}}\Big),\quad
p_{t,2i+1} = \cos\!\Big(\tfrac{t}{10000^{2i/d}}\Big).
$$

Embeddings turn discrete symbols into a differentiable geometric space where
semantically similar residues lie close together — the standard input
representation for all modern sequence models (Mikolov et al., 2013; Vaswani et
al., 2017). The graph encoder additionally builds per-residue **node features**
$h_v \in \mathbb{R}^{36}$ (8 physicochemical + 20 one-hot + 8 positional).

### 2. Sequence encoder (Bidirectional GRU)

The embedded sequence is read by a bidirectional GRU (Cho et al., 2014) that
produces forward/backward hidden states whose concatenation is pooled into the
approximate posterior parameters of a VAE (Kingma & Welling, 2014):

$$
\overrightarrow{h_t} = \mathrm{GRU}(e_t,\overrightarrow{h_{t-1}}),\quad
\overleftarrow{h_t} = \mathrm{GRU}(e_t,\overleftarrow{h_{t+1}}),\quad
[\mu,\;\log\sigma^2] = W\,[\overrightarrow{h_L};\overleftarrow{h_1}] + b .
$$

### 3. Graph encoder with FiLM (GG-FiLM)

Each peptide is also a $k$-NN graph over residues. Messages are modulated by
**Feature-wise Linear Modulation** (Perez et al., 2018; Brockschmidt, 2020):
a conditioning vector produces per-channel scale $\gamma$ and shift $\beta$ that
gate the neighbour messages,

$$
h_v^{(l+1)} = \phi\!\Big(\textstyle\sum_{u\in\mathcal{N}(v)}
\gamma(c)\odot \big(W^{(l)} h_u^{(l)}\big) + \beta(c)\Big),
$$

where $\odot$ is element-wise product and $\phi$ a nonlinearity. FiLM lets the
conditioning signal *reshape* the graph computation rather than merely being
concatenated to it.

### 4. Poincaré phylogenetic embedding

Species relationships are embedded in **hyperbolic space**, which represents
tree-like (exponentially branching) hierarchies with far lower distortion than
Euclidean space (Nickel & Kiela, 2017). For points $u,v$ in the Poincaré ball
$\mathbb{B}^n=\{x:\lVert x\rVert<1\}$,

The embedding is pre-trained (Stage 1, `poincare`) by regressing hyperbolic
distance onto patristic tree distance $d_T$:
$\mathcal{L}_{\text{phylo}}=\sum_{i,j}\big(d_{\mathbb{B}}(z_i,z_j)-d_T(i,j)\big)^2$.
The resulting species vector $z_{\text{phylo}}$ becomes a conditioning token.

### 5. Cross-attention multimodal fusion

The continuous latent (queries $Q$) attends over a small stack of conditioning
tokens — phylogenetic, mimicry (ESM-2), safety, MIC, ESM context (keys/values
$K,V$) — using multi-head **scaled dot-product attention** (Vaswani et al.,
2017), implemented in [`models/fusion.py`](./autoguard/models/fusion.py):

Cross-attention (as opposed to self-attention) lets generation be *steered* by
external biology: each residue position decides how much to listen to each
conditioning modality, and the attention weights double as an **XAI** signal.

### 6. Discrete latent — VQ-VAE codebook (per-position grid)

The encoder produces **one latent vector per residue position** — a latent grid
$Z_e = (z_e^{1},\dots,z_e^{L})$ rather than a single whole-peptide vector — and
**each** position is independently **quantized** to its nearest entry in a
learned codebook $\{c_k\}_{k=1}^{K}$ ($K=512$, $c_k\in\mathbb{R}^{48}$), yielding
a *sequence of discrete codes* (van den Oord et al., 2017):

Per-position quantization is essential: compressing a whole peptide into one code
is far too lossy (a 512-entry codebook cannot represent every peptide), so the
reconstruction loss plateaus high; a grid of $L$ codes gives the decoder enough
capacity to reconstruct faithfully.

Because $\arg\min$ has no gradient, the **straight-through estimator** copies the
decoder gradient back to the encoder, and the codebook is trained with a
commitment loss (here the codebook itself is updated by **EMA**, decay 0.99):

### 7. Autoregressive decoder

The decoder is **autoregressive**: it factorizes the joint sequence probability
into a product of per-position conditionals (Sutskever et al., 2014; Bengio et
al., 2003), each conditioned on the quantized latent **at that position**
$q^{t}$, the fused conditioning $c$, and all previously emitted residues:

A GRU/LSTM stack emits logits $\ell_t$; training uses **teacher forcing** with
cross-entropy, and sampling uses temperature $\tau$ (and optionally
**Gumbel-Softmax**, Jang et al., 2017, for differentiable discrete sampling):

### 8. Multi-task objective

Stage 4 (`full`) optimizes a weighted sum (weights in `LossWeights`):

$\mathcal{L}_{\text{mimicry}}$ is a **contrastive** loss (InfoNCE-style; Oord et
al., 2018) on ESM-2 embeddings (Lin et al., 2023) that pushes generated peptides
away from human self-antigens, while $\mathcal{L}_{\text{safety}}$ supervises the
toxicity / hemolysis / immunogenicity heads.

### 9. Interpretability (post-hoc)

A **sparse autoencoder** (Cunningham et al., 2024) is trained on the quantized
activations to recover monosemantic features, which give three complementary
levels of explanation (see [`evaluation/xai.py`](./autoguard/evaluation/xai.py)).

### Learning-mechanism references

- **Embeddings:** Mikolov et al., "Distributed Representations of Words." *NeurIPS* 2013; Bengio et al., "A Neural Probabilistic Language Model." *JMLR* 2003.
- **Recurrent encoder/decoder & autoregression:** Cho et al., "GRU / RNN Encoder–Decoder." *EMNLP* 2014; Sutskever et al., "Sequence to Sequence Learning." *NeurIPS* 2014.
- **Cross-attention:** Vaswani et al., "Attention Is All You Need." *NeurIPS* 2017.
- **FiLM conditioning:** Perez et al., "FiLM." *AAAI* 2018; Brockschmidt, "GNN-FiLM." *ICML* 2020.
- **VQ-VAE discrete latent:** van den Oord et al., "Neural Discrete Representation Learning." *NeurIPS* 2017.
- **Gumbel-Softmax sampling:** Jang et al., "Categorical Reparameterization with Gumbel-Softmax." *ICLR* 2017.
- **Hyperbolic embeddings:** Nickel & Kiela, "Poincaré Embeddings." *NeurIPS* 2017.
- **Contrastive learning / ESM-2:** van den Oord et al., "Representation Learning with CPC." 2018; Lin et al., "ESM-2." *Science* 2023.
- **VAE objective:** Kingma & Welling, "Auto-Encoding Variational Bayes." *ICLR* 2014.
- **Sparse autoencoders (XAI):** Cunningham et al., "Sparse Autoencoders Find Interpretable Features." *ICLR* 2024.

---

## Project Structure

```
autoguard/
├── config.py              # Model & training hyperparameters
├── pyproject.toml         # Package definition (pip install -e .)
├── models/
│   ├── autoguard_model.py # Full AutoGuard architecture
│   ├── hydramp_base.py    # HydrAMP encoder/decoder (PyTorch port)
│   ├── graph_encoder.py   # GG-FiLM graph neural network (optional torch_geometric)
│   ├── vqvae.py           # VQ-VAE discrete codebook
│   ├── phylo_embeddings.py# Poincaré ball phylogenetic conditioner
│   ├── mimicry_module.py  # ESM-2 contrastive mimicry detection
│   ├── safety_module.py   # Multi-task toxicity/hemolysis/immunogenicity
│   ├── fusion.py          # Cross-attention multimodal fusion
│   ├── sparse_autoencoder.py # XAI sparse dictionary learning
│   └── gnn_baseline.py    # Lightweight GNN-VAE baseline
├── data/
│   ├── datasets.py        # AMP/Safety/Mimicry dataset loaders
│   ├── peptide_graph.py   # PyG graph dataset (optional)
│   └── phylo_data.py      # Phylogenetic tree processing
├── training/
│   ├── trainer.py         # AutoGuard training loop
│   ├── losses.py          # Multi-component loss function
│   └── contrastive.py     # Contrastive mimicry pre-training
├── evaluation/
│   ├── metrics.py         # AMP generation quality metrics
│   ├── comparison.py      # Baseline comparison framework
│   ├── hydramp_adapter.py # HydrAMP baseline adapter
│   └── xai.py             # Interpretability toolkit
├── scripts/
│   ├── train.py           # Main training entry point (5-stage protocol)
│   ├── train_hydramp.py   # HydrAMP baseline two-phase training
│   ├── train_baselines.py # GNN-VAE / baseline training
│   ├── prepare_data.py    # Download + standardize all datasets
│   ├── generate.py        # AMP generation
│   ├── generate_challenge.py # AMP-Challenge-2027 five-category libraries
│   ├── generate_and_analyze.py # Generate + physicochemical analysis
│   ├── evaluate.py        # Evaluation (report)
│   ├── evaluate_model.py  # Per-model eval incl. challenge metrics
│   ├── compare_models.py  # Head-to-head model comparison
│   ├── compare_report.py  # Side-by-side comparison report (md + json)
│   ├── hyperparameter_search.py # Grid/random search across all stages
│   ├── make_report_plots.py  # Figures for the report (run manually)
│   ├── full_experiment.py # Physicochemical comparison plots
│   ├── esm_comparison.py / analyze_esm.py  # ESM-2 embedding analysis
│   ├── explain.py         # SAE interpretability / XAI
│   ├── download_data.py   # Dataset download helper
│   ├── filter_iedb.py     # IEDB SLE epitope filtering
│   └── smoke_test.py      # End-to-end validation (14 checks)
└── utils/
    ├── amino_acids.py     # Physicochemical features
    └── hyperbolic.py      # Poincaré ball geometry
```

---

## Hardware Requirements

| Mode | GPU | RAM | Time |
|------|-----|-----|------|
| Smoke test | None (CPU) | 4 GB | ~30s |
| Model comparison | None (CPU) | 8 GB | ~5 min |
| Full training | NVIDIA 5090 32GB | 64 GB | ~4h |

---

## Citation

If you use AutoGuard, please cite:

```bibtex
@software{autoguard2026,
  title={AutoGuard: Evolutionary-Conditioned Graph-Based AMP Generation},
  author={Jakub Giezgała},
  year={2026},
  url={https://github.com/kugi8412/AutoGuard}
}
```

Key references:
1. Szymczak et al. "Discovering highly potent AMPs with HydrAMP." *Nat Commun* 14, 1453 (2023)
2. Brockschmidt. "GNN-FiLM." *ICML* 2020
3. van den Oord et al. "Neural Discrete Representation Learning." *NeurIPS* 2017
4. Nickel & Kiela. "Poincaré Embeddings." *NeurIPS* 2017
5. Lin et al. "ESM-2." *Science* 379, 1123–1130 (2023)
6. Cunningham et al. "Sparse Autoencoders Find Interpretable Features." *ICLR* 2024
