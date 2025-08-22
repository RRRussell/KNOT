# KNOT: Knowledge Graph and Omics Integration Framework for Target Identification

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

## Quick Start

### Evaluation Mode (Train/Val/Test)

```bash
# Default task
python main.py --task known_druggable --mode evaluation

# Specific task with custom settings
python main.py --task approved_targets --mode evaluation --epochs 100 --edge-config ppi_only
```

### Inference Mode (Rank All Genes)

```bash
# Generate gene rankings
python main.py --task known_druggable --mode inference

# Custom configuration
python main.py --task cancer_druggability --mode inference --edge-config all --epochs 200
```

## Tasks

Ten biologically motivated tasks across evidence levels, clinical relevance, and drug modalities.

<table>
<thead>
<tr>
<th>Category</th>
<th>Task</th>
<th>Task ID</th>
<th>Positive Class (n)</th>
<th>Unknown Class (n)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2"><b>Pharos Label (Disease-agnostic)</b></td>
<td>Strong vs Weak Evidence</td>
<td><code>strong_vs_weak_evidence</code></td>
<td>Tclin + Tchem (2,562)</td>
<td>Tbio + Tdark (16,470)</td>
</tr>
<tr>
<td>Approved Drug Targets vs Others</td>
<td><code>approved_targets</code></td>
<td>Tclin (700)</td>
<td>Tchem + Tbio + Tdark (18,332)</td>
</tr>
<tr>
<td rowspan="2"><b>Triage Label (Disease-agnostic)</b></td>
<td>Known Druggable vs Others</td>
<td><code>known_druggable</code></td>
<td>Tier1–3B (4,311)</td>
<td>All other genes (14,721)</td>
</tr>
<tr>
<td>High-confidence vs Low-confidence Targets</td>
<td><code>high_confidence</code></td>
<td>Tier1 &amp; Tier2 (2,048)</td>
<td>Tier3A &amp; Tier3B (2,263)</td>
</tr>
<tr>
<td><b>Oncology (Domain-specific)</b></td>
<td>Cancer-related Druggability</td>
<td><code>cancer_druggability</code></td>
<td>Cancer-sm (319)</td>
<td>All other genes (18,731)</td>
</tr>
<tr>
<td rowspan="5"><b>Drug Modality (Domain-specific)</b></td>
<td>SM Drug Target vs Non-SM</td>
<td><code>sm_target</code></td>
<td>SM-labeled (879)</td>
<td>All other genes (18,153)</td>
</tr>
<tr>
<td>SM Targets: Cancer vs Non-cancer</td>
<td><code>sm_cancer</code></td>
<td>SM + CPD (696)</td>
<td>SM + Non-CPD (183)</td>
</tr>
<tr>
<td>AB Drug Target vs Non-AB</td>
<td><code>ab_target</code></td>
<td>AB-labeled (246)</td>
<td>All other genes (18,786)</td>
</tr>
<tr>
<td>AB Targets: Cancer vs Non-cancer</td>
<td><code>ab_cancer</code></td>
<td>AB + CPD (173)</td>
<td>AB + Non-CPD (73)</td>
</tr>
<tr>
<td>PROTAC Target vs Non-PROTAC</td>
<td><code>protac</code></td>
<td>PROTAC-labeled (269)</td>
<td>All other genes (18,763)</td>
</tr>
</tbody>
</table>

## Edge Configurations

- `all`: All networks (regulatory + co-expression + PPI)
- `regulatory`: Gene regulatory networks
- `functional`: Co-expression + PPI
- `ppi_only`: Protein-protein interactions only

## Key Parameters

- `--task`: Druggability task (default: known_druggable)
- `--mode`: evaluation or inference (default: evaluation)
- `--edge-config`: Edge type configuration (default: all)
- `--epochs`: Training epochs (default: 200)
- `--seed`: Random seed (default: 42)
- `--save-dir`: Output directory (default: ./results)

## Output Files

**Evaluation Mode:**
- `eval_results_[task]_[timestamp].json`: Performance metrics
- `model_[task]_[timestamp].pt`: Model checkpoint

**Inference Mode:**
- `gene_ranking_[task]_[timestamp].csv`: Ranked gene list
- `inference_model_[task]_[timestamp].pt`: Model checkpoint