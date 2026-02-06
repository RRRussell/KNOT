# KNOT: Knowledge Graph and Omics Integration Framework for Target Identification

![Flowchart](./KNOT_Flowchart.png)

A Graph Neural Network (GNN) framework that combines positive-unlabeled learning with heterogeneous graph networks for gene druggability prediction.

## Installation

```bash
# PyTorch (CUDA 12.1)
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
pip install torch-geometric==2.6.1
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Other dependencies
pip install pandas==2.0.3 numpy==1.24.1 scikit-learn==1.3.2 scipy==1.10.1 tqdm networkx==3.0
```

## Data Structure

The framework now uses separate files for features and labels:

```
data/
├── gene_features.tsv                  # Gene features (503 features across 16 categories)
├── gene_labels.tsv                    # Task labels and annotations
├── gene_feature_summary.tsv           # Feature category breakdown
└── edge/                              # Network edge files
    ├── human.source                   # RegNetwork regulatory edges
    ├── trrust_rawdata.human.tsv       # TRRUST transcription factor edges
    ├── coexpression_edges_99p.tsv     # Co-expression edges
    └── ppi_symbol_links.tsv           # STRING PPI edges
```

## Quick Start

### Evaluation Mode (Train/Val/Test)

```bash
# Default task with all features
python main.py --task triage_tier12_vs_others --mode evaluation

# Use only DepMap features
python main.py --task triage_tier12_vs_others --mode evaluation --feature-config depmap_only

# Specific task with custom settings
python main.py --task triage_tier12_vs_others --mode evaluation --epochs 100 --edge-config ppi_only
```

### Inference Mode (Rank All Genes)

```bash
# Generate gene rankings
python main.py --task triage_tier12_vs_others --mode inference

# Custom configuration
python main.py --task cancer_relevant_targets --mode inference --feature-config depmap_plus_pharos --epochs 200
```

## Tasks

KNOT evaluates **15 biologically grounded target identification tasks** spanning  
(i) evidence strength,  
(ii) disease scope, and  
(iii) drug modality.

All tasks are formulated as **binary positive–unlabeled (PU) learning problems** at the gene level.

<table>
  <tr>
    <th>Category</th>
    <th>Task</th>
    <th>Task ID</th>
    <th>Description</th>
  </tr>

  <!-- PHAROS -->
  <tr>
    <td rowspan="2" align="center"><b>PHAROS<br/>(Disease-agnostic)</b></td>
    <td>Clinical Targets (Tclin)</td>
    <td><code>pharos_tclin_vs_others</code></td>
    <td>FDA-approved clinically validated drug targets</td>
  </tr>
  <tr>
    <td>Clinical & Chemical Targets</td>
    <td><code>pharos_tclin_tchem_vs_others</code></td>
    <td>Tclin or Tchem targets versus weak/no evidence</td>
  </tr>

  <!-- TRIAGE -->
  <tr>
    <td rowspan="2" align="center"><b>Triage Assessment<br/>(Disease-agnostic)</b></td>
    <td>Top-Tier Targets</td>
    <td><code>triage_tier1_vs_others</code></td>
    <td>Highest-confidence druggable targets (Tier 1)</td>
  </tr>
  <tr>
    <td>High-Confidence Targets</td>
    <td><code>triage_tier12_vs_others</code></td>
    <td>Known druggable targets (Tier 1–2)</td>
  </tr>

  <!-- CANCER -->
  <tr>
    <td rowspan="6" align="center"><b>Cancer Druggability<br/>(Domain-specific)</b></td>
    <td>Cancer-Relevant Targets</td>
    <td><code>cancer_relevant_targets</code></td>
    <td>Curated targets relevant to cancer biology</td>
  </tr>
  <tr>
    <td>Cancer-Type-Specific Targets</td>
    <td><code>cancer_type_specific_targets</code></td>
    <td>Targets specific to individual cancer types</td>
  </tr>
  <tr>
    <td>Pan-Cancer Targets</td>
    <td><code>pan_cancer_targets</code></td>
    <td>Targets recurrent across multiple cancer types</td>
  </tr>
  <tr>
    <td>Tier 1 Cancer Targets</td>
    <td><code>pan_cancer_T1_targets</code></td>
    <td>Approved cancer drug targets</td>
  </tr>
  <tr>
    <td>Tier 1–2 Cancer Targets</td>
    <td><code>pan_cancer_T12_targets</code></td>
    <td>Approved or repurposed cancer targets</td>
  </tr>
  <tr>
    <td>Tier 1–3 Cancer Targets</td>
    <td><code>pan_cancer_T123_targets</code></td>
    <td>Approved or investigational cancer targets</td>
  </tr>

  <!-- MODALITY -->
  <tr>
    <td rowspan="5" align="center"><b>Drug Modality<br/>(Domain-specific)</b></td>
    <td>Small Molecule Targets (Approved)</td>
    <td><code>sm_bucket1_vs_others</code></td>
    <td>Targets of approved small-molecule drugs</td>
  </tr>
  <tr>
    <td>Small Molecule Targets (Clinical+)</td>
    <td><code>sm_bucket123_vs_others</code></td>
    <td>Approved or clinical-stage small-molecule targets</td>
  </tr>
  <tr>
    <td>Antibody Targets (Approved)</td>
    <td><code>ab_bucket1_vs_others</code></td>
    <td>Targets of approved antibody therapeutics</td>
  </tr>
  <tr>
    <td>Antibody Targets (Clinical+)</td>
    <td><code>ab_bucket123_vs_others</code></td>
    <td>Approved or clinical-stage antibody targets</td>
  </tr>
  <tr>
    <td>PROTAC Targets</td>
    <td><code>protac_bucket1234_vs_others</code></td>
    <td>Targets supported by literature-curated PROTAC evidence</td>
  </tr>
</table>

## Feature Configurations

### Available Options
- `all_features`: All 503 features (21 DepMap + 482 non-DepMap)
- `depmap_only`: Only 21 DepMap features (CRISPR, CNV, expression, mutation)
- `non_depmap_only`: Only 482 non-DepMap features
- `depmap_plus_pharos`: DepMap + Pharos druggability features
- `depmap_plus_gnomad`: DepMap + GnomAD gene intolerance features

### Feature Categories (503 total)

| Category | Features | Description |
|----------|----------|-------------|
| **DepMap** | 21 | CRISPR dependency, CNV, expression, mutation |
| **ExAC** | 19 | CNV counts, intolerance scores, gene size |
| **GnomAD** | 15 | Gene intolerance metrics |
| **InterPro** | 97 | Protein family classification |
| **Pharos** | 8 | Druggability and genome features |
| **GWAS** | 6 | Disease/trait associations |
| **Genic Intolerance** | 6 | RVIS and MTR scores |
| **Mouse Genes** | 4 | Essential gene orthologs |
| **InWeb** | 4 | PPI network features |
| **Reactome** | 2 | Pathway annotations |
| **STRING_db** | 10 | Protein interaction data |
| **CTDbase** | 307 | Chemical-gene interactions |
| **DGIdb** | 1 | Drug-gene interaction types |
| **MGI** | 1 | Mouse phenotype data |
| **OMIM** | 1 | Disease associations |
| **GTEx** | 1 | Tissue expression specificity |

## Edge Configurations

- `all`: All networks (regulatory + co-expression + PPI)
- `regulatory`: Gene regulatory networks (RegNetwork + TRRUST)
- `functional`: Co-expression + PPI  
- `ppi_only`: Protein-protein interactions only
- `coexp_only`: Co-expression only
- `reg_ppi`: Regulatory + PPI
- `reg_coexp`: Regulatory + co-expression

## Key Parameters

- `--task`: Druggability task (default: triage_tier12_vs_others)
- `--mode`: evaluation or inference (default: evaluation)
- `--feature-config`: Feature configuration (default: all_features)
- `--edge-config`: Edge type configuration (default: all)
- `--epochs`: Training epochs (default: 200)
- `--seed`: Random seed (default: 42)
- `--save-dir`: Output directory (default: ./results)

## Output Files

**Evaluation Mode:**
- `eval_results_[task]_[features]_[timestamp].json`: Performance metrics
- `model_[task]_[features]_[timestamp].pt`: Model checkpoint

**Inference Mode:**
- `gene_ranking_[task]_[features]_[timestamp].csv`: Ranked gene list
- `inference_model_[task]_[features]_[timestamp].pt`: Model checkpoint

## Examples

```bash
# Compare DepMap-only vs all features for known druggable prediction
python main.py --task triage_tier12_vs_others --feature-config depmap_only --mode evaluation
python main.py --task triage_tier12_vs_others --feature-config all_features --mode evaluation

# Test different network configurations  
python main.py --task pharos_tclin_vs_others --edge-config regulatory --mode evaluation
python main.py --task pharos_tclin_vs_others --edge-config functional --mode evaluation

# Generate rankings for cancer targets
python main.py --task cancer_relevant_targets --mode inference --feature-config depmap_plus_pharos

```

## File Structure

```
KNOT_v0/
├── main.py              # Main training script
├── config.py            # Configuration settings
├── data_loader.py       # Data loading and preprocessing  
├── models.py            # GNN model architectures
├── trainer.py           # Training pipeline
├── utils.py             # Utility functions
├── results/             # Output directory
├── checkpoints/         # Model checkpoints
└── data/                # Dataset files
    ├── gene_features.tsv
    ├── gene_labels.tsv
    └── edge/
```
