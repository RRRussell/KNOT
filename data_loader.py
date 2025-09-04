#!/usr/bin/env python3
"""
Data loading and preprocessing for KNOT GNN druggability prediction
Updated for new data structure
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, Data
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from config import *


class DrugabilityDataLoader:
    """
    Data loader for druggability prediction
    Handles graph construction and feature preprocessing
    """
    
    def __init__(self, features_file=FEATURES_FILE, labels_file=LABELS_FILE, 
                 edge_config='all', feature_config='all_features'):
        """
        Initialize data loader
        
        Args:
            features_file: Path to gene features dataset
            labels_file: Path to gene labels dataset
            edge_config: Edge configuration name from EDGE_CONFIGS
            feature_config: Feature configuration name from FEATURE_CONFIGS
        """
        self.features_file = features_file
        self.labels_file = labels_file
        self.edge_config = edge_config if edge_config in EDGE_CONFIGS else 'all'
        self.feature_config = feature_config if feature_config in FEATURE_CONFIGS else 'all_features'
        self.features_df = None
        self.labels_df = None
        self.merged_df = None
        self.edge_data = None
        self.scaler = None
        self.imputer = None
        
    def load_data(self):
        """Load the features and labels datasets"""
        print(f"Loading features from: {self.features_file}")
        self.features_df = pd.read_csv(self.features_file, sep='\t')
        print(f"Loaded features: {len(self.features_df)} genes with {len(self.features_df.columns)-1} features")
        
        print(f"Loading labels from: {self.labels_file}")
        self.labels_df = pd.read_csv(self.labels_file, sep='\t')
        print(f"Loaded labels: {len(self.labels_df)} genes with {len(self.labels_df.columns)-1} label columns")
        
        # Merge features and labels
        self.merged_df = self.features_df.merge(self.labels_df, on='Gene_Symbol', how='inner')
        print(f"Merged dataset: {len(self.merged_df)} genes")
        
        # Check feature availability
        selected_features = FEATURE_CONFIGS[self.feature_config]
        available_features = [f for f in selected_features if f in self.features_df.columns]
        missing_features = [f for f in selected_features if f not in self.features_df.columns]
        
        print(f"Feature config '{self.feature_config}': {len(available_features)}/{len(selected_features)} available")
        if missing_features:
            print(f"Missing features ({len(missing_features)}): {missing_features[:10]}...")
        
        return self.merged_df
    
    def load_edges(self):
        """Load edge data based on configuration"""
        edge_types = EDGE_CONFIGS[self.edge_config]
        print(f"Loading edges for configuration '{self.edge_config}': {edge_types}")
        
        self.edge_data = {}
        
        for edge_type in edge_types:
            if edge_type not in EDGE_PATHS:
                continue
                
            print(f"  Loading {edge_type} edges...")
            
            if edge_type == 'regnet':
                edges = self._load_regnet()
            elif edge_type == 'trrust':
                edges = self._load_trrust()
            elif edge_type == 'coexp':
                edges = self._load_coexp()
            elif edge_type == 'ppi':
                edges = self._load_ppi()
            else:
                edges = []
            
            if edges:
                edges = self._limit_edges_per_node(edges, MAX_NEIGHBORS_PER_NODE)
                self.edge_data[edge_type] = edges
                print(f"    Loaded {len(edges):,} {edge_type} edges")
        
        return self.edge_data
    
    def _load_regnet(self):
        """Load RegNetwork regulatory edges"""
        edges = []
        try:
            with open(EDGE_PATHS['regnet']) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        src, _, tgt, _ = parts[:4]
                        edges.append((src, tgt))
        except Exception as e:
            print(f"    Error loading RegNetwork: {e}")
        return edges
    
    def _load_trrust(self):
        """Load TRRUST transcription factor edges"""
        try:
            df = pd.read_csv(EDGE_PATHS['trrust'], sep="\t", header=None,
                           names=["TF", "Target", "Effect", "PubMed"])
            return list(zip(df['TF'], df['Target']))
        except Exception as e:
            print(f"    Error loading TRRUST: {e}")
            return []
    
    def _load_coexp(self):
        """Load co-expression edges"""
        try:
            df = pd.read_csv(EDGE_PATHS['coexp'], sep="\t")
            # Sample if too large
            if len(df) > 1000000:
                df = df.sample(n=1000000, random_state=42)
            # Assuming columns are Gene1, Gene2
            gene1_col = df.columns[0]
            gene2_col = df.columns[1]
            return list(zip(df[gene1_col], df[gene2_col]))
        except Exception as e:
            print(f"    Error loading Coexpression: {e}")
            return []
    
    def _load_ppi(self):
        """Load protein-protein interaction edges"""
        try:
            df = pd.read_csv(EDGE_PATHS['ppi'], sep="\t")
            # Filter by confidence score if available
            if 'combined_score' in df.columns:
                df = df[df['combined_score'] > 700]
            # Assuming columns are gene1, gene2
            gene1_col = df.columns[0]
            gene2_col = df.columns[1]
            return list(zip(df[gene1_col], df[gene2_col]))
        except Exception as e:
            print(f"    Error loading PPI: {e}")
            return []
    
    def _limit_edges_per_node(self, edge_list, max_neighbors):
        """
        Limit the number of edges per node to prevent memory issues
        
        Args:
            edge_list: List of (source, target) tuples
            max_neighbors: Maximum neighbors per node
            
        Returns:
            Limited edge list
        """
        if not edge_list:
            return edge_list
        
        df = pd.DataFrame(edge_list, columns=['src', 'tgt'])
        limited_df = df.groupby('src').apply(
            lambda x: x.sample(n=min(len(x), max_neighbors), random_state=42)
        ).reset_index(drop=True)
        
        return list(zip(limited_df['src'], limited_df['tgt']))
    
    def prepare_task_data(self, task_name, mode='evaluation'):
        """
        Prepare data for a specific task
        
        Args:
            task_name: Name of the task from DRUGGABILITY_TASKS
            mode: 'evaluation' for train/val/test split, 'inference' for all labeled data
            
        Returns:
            Dictionary containing prepared data and metadata
        """
        if self.merged_df is None:
            self.load_data()
        if self.edge_data is None:
            self.load_edges()
        
        if task_name not in DRUGGABILITY_TASKS:
            raise ValueError(f"Unknown task: {task_name}")
        
        task_info = DRUGGABILITY_TASKS[task_name]
        label_col = task_info['label_col']
        
        print(f"\nPreparing data for task: {task_name}")
        print(f"  Display Name: {task_info['display_name']}")
        print(f"  Label Column: {label_col}")
        print(f"  Mode: {mode}")
        
        # Check if label column exists
        if label_col not in self.merged_df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset")
        
        # Filter data: only use samples with valid labels (not NaN/null)
        valid_mask = self.merged_df[label_col].notna()
        task_data = self.merged_df[valid_mask].copy()
        
        print(f"  Using {len(task_data)} samples (filtered from {len(self.merged_df)} total)")
        
        # Get labels
        y = task_data[label_col].values.astype(int)
        
        # Check class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"  Class distribution: {dict(zip(unique_classes, counts))}")
        
        # Prepare features
        feature_cols = self._get_feature_columns(task_data)
        X = task_data[feature_cols].values
        
        # Impute missing values and standardize
        self.imputer = SimpleImputer(strategy='median')
        X = self.imputer.fit_transform(X)
        
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Create graph structure
        genes = task_data['Gene_Symbol'].tolist()
        gene2idx = {gene: i for i, gene in enumerate(genes)}
        
        # Build graph data
        graph_data = self._build_graph_data(X, y, genes, gene2idx)
        
        # Create splits for evaluation mode
        if mode == 'evaluation':
            train_mask, val_mask, test_mask = self._create_splits(len(genes), y)
            graph_data.train_mask = train_mask
            graph_data.val_mask = val_mask
            graph_data.test_mask = test_mask
        else:  # inference mode - all data is training data
            graph_data.train_mask = torch.ones(len(genes), dtype=torch.bool)
            graph_data.val_mask = torch.zeros(len(genes), dtype=torch.bool)
            graph_data.test_mask = torch.zeros(len(genes), dtype=torch.bool)
        
        return {
            'data': graph_data,
            'feature_names': feature_cols,
            'gene_symbols': genes,
            'task_info': task_info,
            'mode': mode,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_config': self.feature_config
        }
    
    def _get_feature_columns(self, df):
        """Get feature columns based on feature configuration"""
        selected_features = FEATURE_CONFIGS[self.feature_config]
        feature_cols = [f for f in selected_features if f in df.columns]
        
        print(f"  Selected {len(feature_cols)} features from config '{self.feature_config}'")
        
        # Print feature breakdown if using all features
        if self.feature_config == 'all_features':
            depmap_count = len([f for f in DEPMAP_FEATURES if f in feature_cols])
            non_depmap_count = len(feature_cols) - depmap_count
            print(f"    DepMap: {depmap_count}, Non-DepMap: {non_depmap_count}")
        
        return feature_cols
    
    def _build_graph_data(self, X, y, genes, gene2idx):
        """
        Build PyTorch Geometric graph data
        
        Args:
            X: Feature matrix
            y: Labels
            genes: List of gene symbols
            gene2idx: Mapping from gene symbol to index
            
        Returns:
            PyTorch Geometric Data object
        """
        X_tensor = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Build edges
        edge_list = []
        edge_types = []
        edge_type_map = {etype: i for i, etype in enumerate(self.edge_data.keys())}
        
        for edge_type, edges in self.edge_data.items():
            type_id = edge_type_map[edge_type]
            for src_gene, tgt_gene in edges:
                if src_gene in gene2idx and tgt_gene in gene2idx:
                    src_idx = gene2idx[src_gene]
                    tgt_idx = gene2idx[tgt_gene]
                    if src_idx != tgt_idx:  # Skip self-loops
                        edge_list.append([src_idx, tgt_idx])
                        edge_types.append(type_id)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).T
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty(0, dtype=torch.long)
        
        # Create Data object
        data = Data(
            x=X_tensor,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y_tensor,
            num_nodes=len(genes)
        )
        
        print(f"  Graph: {data.num_nodes} nodes, {data.num_edges} edges")
        
        return data
    
    def _create_splits(self, n_samples, labels, random_state=42):
        """
        Create train/val/test splits
        
        Args:
            n_samples: Number of samples
            labels: Labels for stratification
            random_state: Random seed
            
        Returns:
            Tuple of (train_mask, val_mask, test_mask)
        """
        indices = np.arange(n_samples)
        
        try:
            # Stratified split
            train_val_idx, test_idx = train_test_split(
                indices, test_size=TEST_SIZE, random_state=random_state, stratify=labels
            )
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=VAL_SIZE/(1-TEST_SIZE),
                random_state=random_state, stratify=labels[train_val_idx]
            )
        except:
            # If stratification fails, split without it
            train_val_idx, test_idx = train_test_split(
                indices, test_size=TEST_SIZE, random_state=random_state
            )
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=random_state
            )
        
        # Create masks
        train_mask = torch.zeros(n_samples, dtype=torch.bool)
        val_mask = torch.zeros(n_samples, dtype=torch.bool)
        test_mask = torch.zeros(n_samples, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        print(f"  Splits - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        
        return train_mask, val_mask, test_mask
    
    def convert_to_hetero(self, data):
        """
        Convert homogeneous data to heterogeneous format for HGT
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            HeteroData object
        """
        hetero_data = HeteroData()
        
        # Add node features and labels
        hetero_data['gene'].x = data.x
        hetero_data['gene'].y = data.y
        
        # Add masks if they exist
        if hasattr(data, 'train_mask'):
            hetero_data['gene'].train_mask = data.train_mask
            hetero_data['gene'].val_mask = data.val_mask
            hetero_data['gene'].test_mask = data.test_mask
        
        # Add edges by type
        edge_type_map = {i: etype for i, etype in enumerate(self.edge_data.keys())}
        
        if data.num_edges > 0:
            for type_id, edge_type in edge_type_map.items():
                edge_mask = (data.edge_type == type_id)
                if edge_mask.sum() > 0:
                    edge_index = data.edge_index[:, edge_mask]
                    hetero_data['gene', edge_type, 'gene'].edge_index = edge_index
        
        # Add self-loops if no edges (fallback)
        if len(hetero_data.edge_types) == 0:
            num_nodes = data.x.shape[0]
            self_loops = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
            hetero_data['gene', 'self_loop', 'gene'].edge_index = self_loops
        
        return hetero_data
    
    def get_feature_breakdown(self):
        """Get breakdown of features by category"""
        if self.features_df is None:
            self.load_data()
        
        breakdown = {}
        
        # DepMap features
        depmap_available = [f for f in DEPMAP_FEATURES if f in self.features_df.columns]
        breakdown['DepMap'] = {
            'available': len(depmap_available),
            'total': len(DEPMAP_FEATURES),
            'features': depmap_available
        }
        
        # Non-DepMap features by category
        for category, features in NON_DEPMAP_FEATURES.items():
            available = [f for f in features if f in self.features_df.columns]
            breakdown[category] = {
                'available': len(available),
                'total': len(features),
                'features': available
            }
        
        return breakdown