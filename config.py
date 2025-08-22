#!/usr/bin/env python3
"""
Configuration file for KNOT GNN druggability prediction
"""

import torch
from pathlib import Path

# ===============================================================================
# PATHS
# ===============================================================================

# Data paths
DATA_DIR = Path('/home/zihend1/Genesis/KNOT/data/processed')
INPUT_FILE = DATA_DIR / 'gene_druggability_v2.tsv'
OUTPUT_DIR = Path('./results')
CHECKPOINT_DIR = Path('./checkpoints')

# Edge data paths
EDGE_PATHS = {
    'regnet': "/extra/zhanglab0/INDV/zihend1/Genesis/KNOT/Edge/GRN/RegNetwork/Human/human.source",
    'trrust': "/extra/zhanglab0/INDV/zihend1/Genesis/KNOT/Edge/GRN/TRRUST/trrust_rawdata.human.tsv",
    'coexp': "/extra/zhanglab0/INDV/zihend1/Genesis/KNOT/Edge/GCN/GENEFRIENDS/homo_sapiens_sapiens/GTEx/coexpression_edges_99p.tsv",
    'ppi': "/extra/zhanglab0/INDV/zihend1/Genesis/KNOT/Edge/PPI/STRING/ppi_symbol_links.tsv"
}

# ===============================================================================
# FEATURES
# ===============================================================================

# DepMap features
DEPMAP_FEATURES = [
    'crispr_dependency_mean', 'crispr_dependency_std', 'crispr_dependency_min',
    'crispr_dependency_max', 'crispr_dependency_median', 'crispr_dependency_strong_count',
    'cnv_wgs_mean', 'cnv_wgs_std', 'cnv_wgs_min', 'cnv_wgs_max', 'cnv_wgs_median',
    'expression_mean', 'expression_std', 'expression_min', 'expression_max',
    'expression_median', 'expression_high_prop',
    'damaging_mutation_sum', 'damaging_high_af_count', 'hotspot_mutation_sum',
    'hotspot_high_af_count'
]

# Columns to exclude from features
EXCLUDE_COLUMNS = [
    "Gene_Symbol", "known_gene", "druggability_tier", "idgTDL",
    "druggability_tier_numeric", "Target Development Level",
    # All task columns
    "strong_vs_weak_evidence", "approved_targets_vs_others",
    "known_druggable_vs_others", "high_vs_low_confidence",
    "cancer_druggability", "sm_target_vs_non_sm",
    "sm_cancer_vs_non_cancer", "ab_target_vs_non_ab",
    "ab_cancer_vs_non_cancer", "protac_vs_non_protac"
]

# ===============================================================================
# DRUGGABILITY TASKS
# ===============================================================================

DRUGGABILITY_TASKS = {
    'strong_vs_weak_evidence': {
        'label_col': 'strong_vs_weak_evidence',
        'display_name': 'Strong vs Weak Evidence',
        'description': 'Distinguish between strong and weak evidence for druggability',
        'task_type': 'binary'
    },
    'approved_targets': {
        'label_col': 'approved_targets_vs_others',
        'display_name': 'Approved Drug Targets vs Others',
        'description': 'Identify FDA-approved drug targets',
        'task_type': 'binary'
    },
    'known_druggable': {
        'label_col': 'known_druggable_vs_others',
        'display_name': 'Known Druggable vs Others',
        'description': 'Predict known druggable targets',
        'task_type': 'binary'
    },
    'high_confidence': {
        'label_col': 'high_vs_low_confidence',
        'display_name': 'High-confidence vs Low-confidence Targets',
        'description': 'Distinguish high-confidence from low-confidence targets',
        'task_type': 'binary_subset'
    },
    'cancer_druggability': {
        'label_col': 'cancer_druggability',
        'display_name': 'Cancer-related Druggability',
        'description': 'Predict cancer-specific druggability',
        'task_type': 'binary'
    },
    'sm_target': {
        'label_col': 'sm_target_vs_non_sm',
        'display_name': 'SM Drug Target vs Non-SM',
        'description': 'Predict small molecule druggability',
        'task_type': 'binary'
    },
    'sm_cancer': {
        'label_col': 'sm_cancer_vs_non_cancer',
        'display_name': 'SM Targets: Cancer vs Non-cancer',
        'description': 'Distinguish cancer vs non-cancer small molecule targets',
        'task_type': 'binary_subset'
    },
    'ab_target': {
        'label_col': 'ab_target_vs_non_ab',
        'display_name': 'AB Drug Target vs Non-AB',
        'description': 'Predict antibody druggability',
        'task_type': 'binary'
    },
    'ab_cancer': {
        'label_col': 'ab_cancer_vs_non_cancer',
        'display_name': 'AB Targets: Cancer vs Non-cancer',
        'description': 'Distinguish cancer vs non-cancer antibody targets',
        'task_type': 'binary_subset'
    },
    'protac': {
        'label_col': 'protac_vs_non_protac',
        'display_name': 'PROTAC Target vs Non-PROTAC',
        'description': 'Predict PROTAC druggability',
        'task_type': 'binary'
    }
}

# Task order for display
TASK_ORDER = [
    'strong_vs_weak_evidence',
    'approved_targets',
    'known_druggable',
    'high_confidence',
    'cancer_druggability',
    'sm_target',
    'sm_cancer',
    'ab_target',
    'ab_cancer',
    'protac'
]

# ===============================================================================
# MODEL HYPERPARAMETERS
# ===============================================================================

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model architecture
HIDDEN_CHANNELS = 512
NUM_LAYERS = 3
NUM_HEADS = 8
DROPOUT = 0.2

# Training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 256

# Data splits
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Graph construction
MAX_NEIGHBORS_PER_NODE = 10
NEIGHBOR_SAMPLING = [25, 20, 15]  # For each layer

# Edge type configurations
EDGE_CONFIGS = {
    'all': ['regnet', 'trrust', 'coexp', 'ppi'],
    'regulatory': ['regnet', 'trrust'],
    'functional': ['coexp', 'ppi'],
    'ppi_only': ['ppi'],
    'coexp_only': ['coexp'],
    'reg_ppi': ['regnet', 'ppi'],
    'reg_coexp': ['regnet', 'coexp']
}

# Default configuration
DEFAULT_EDGE_CONFIG = 'all'
DEFAULT_TASK = 'known_druggable'
DEFAULT_SEED = 42