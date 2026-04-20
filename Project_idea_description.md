# Engineering of Antimicrobial Peptides via Evolutionary-Conditioned VQ-VAE and Molecular Mimicry Detection

## 1. Background

The global antimicrobial resistance (AMR) crisis necessitates the rapid discovery of novel therapeutics, such as antimicrobial peptides (AMPs). Generative deep learning models, like HydrAMP[1], have significantly advanced *in silico* AMP design. However, the computational biology field is currently hindered by a reproducibility crisis driven by flawed validation metrics, data leakage, and a heavy reliance on arbitrary post-hoc manual curation.

Furthermore, current generative models face critical architectural and biological limitations. First, they project discrete amino acid sequences into continuous latent spaces, which poorly represent the highly multimodal biophysical forces governing protein folding. Second, they ignore the evolutionary trajectory of pathogens, leading to generated drugs that are highly susceptible to rapid mutational escape. Finally, they overlook the phenomenon of __ensemble molecular mimicry__. By blindly optimizing for target disruption, AI models can inadvertently generate peptides that structurally and thermodynamically mimic human autoantigens, risking catastrophic autoimmune cross-reactivity[2].

## 2. Research Question

Could an Evolutionary-Conditioned Vector Quantized Variational Autoencoder (ecVQ-VAE) resolve these systemic issues by integrating perturbed phylogenomic embeddings and contrastive learning-based molecular mimicry detection?

The solution, I believe, lies in replacing continuous sequence embeddings with a discrete tokenised latent space. This space should be conditioned on biological and evolutionary vectors. It should autonomously generate highly potent, mutation-resistant and immunologically safe peptides. This would eliminate the need for extensive computational post-filtering.

## 3. Proposed Analysis

### Data Sources

I will curate peptide sequences from databases like:

- Antimicrobial Peptide Database - https://aps.unmc.edu/
- Database of Antimicrobial Activity and Structure of Peptides - https://dbaasp.org/home
- DRAMP Database - https://dramp.cpu-bioinfor.org/

establishing strict clinical utility thresholds (e.g., $MIC \le 10 \mu g/mL$).

For phylogenetic inputs, we will extract whole bacterial proteomes from:

- NCBI Datasets — https://www.ncbi.nlm.nih.gov/datasets/

For immunological constraints, we will compile structural datasets of known human autoantigens and natural host defense peptides from:

- Protein Data Bank (PDB) — https://www.rcsb.org/
- Immune Epitope Database (IEDB) — https://www.iedb.org/

### Methods & Algorithms

My framework integrates three distinct modalities.

**Phylogenomics Module:**  
The Analysis Phylogenomics Pipeline will process full proteomes using:

- MMseqs2 — https://github.com/soedinglab/MMseqs2
- IQ-TREE — http://www.iqtree.org/

to construct robust species trees. These trees will undergo statistical perturbations and be embedded into a hyperbolic space ($Z_{phylo}$) to map immutable evolutionary targets.

**Molecular Mimicry Detection:**  
A contrastive learning module utilizing a pre-trained protein language model:

- ESM-2 — https://github.com/facebookresearch/esm

will generate a mimicry vector ($C_{mimic}$). This vector will penalize similarities to human autoantigens while rewarding thermodynamic mimicry of natural immunomodulators.

**Generative Model:**  
An ecVQ-VAE will manage discrete peptide generation. A cross-attention fusion module in the decoder will integrate discrete sequence tokens with continuous $Z_{phylo}$ and $C_{mimic}$ vectors, guiding the generation process.

### End-to-End Project Workflow

The project will follow a fully end-to-end pipeline beginning with curated peptide, proteome, and immunological datasets, followed by preprocessing, quality control, and feature extraction for each modality. Phylogenomic trees will be inferred and perturbed to produce evolutionary conditioning vectors, while the contrastive mimicry module will encode structural and functional similarity constraints against host proteins. These signals will be fused inside the ecVQ-VAE decoder, which will generate candidate peptides in a discrete latent space and then pass them through a unified validation stage that checks potency, novelty, evolutionary robustness, and safety. The final output of the system will therefore be a ranked list of AMP candidates with associated scores, traceable generation paths, and interpretable conditioning factors.

--

## References

[1] Szymczak, P., Możejko, M., Grzegorzek, T. et al. *Discovering highly potent antimicrobial peptides with deep generative model HydrAMP*. Nat Commun 14, 1453 (2023). https://doi.org/10.1038/s41467-023-36994-z

[2] Maoz-Segal R, Andrade P. *Molecular Mimicry and Autoimmunity. Infection and Autoim-
munity*, 2015 Feb 6:27–44. https://pmc.ncbi.nlm.nih.gov/articles/PMC7151819/
