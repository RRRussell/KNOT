#!/usr/bin/env python3
"""
GNN model architectures for KNOT druggability prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear, LayerNorm


class MultiTaskHGT(torch.nn.Module):
    """
    Heterogeneous Graph Transformer for druggability prediction
    Uses relation-specific attention to capture different types of gene-gene interactions
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, metadata,
                 heads=8, num_layers=3, dropout=0.2):
        """
        Initialize HGT model
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (2 for binary classification)
            metadata: Tuple of (node_types, edge_types) for heterogeneous graph
            heads: Number of attention heads
            num_layers: Number of HGT layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers
        
        # Input projection for each node type
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)
        
        # HGT convolution layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, heads)
            self.convs.append(conv)
            self.norms.append(LayerNorm(hidden_channels))
            self.dropouts.append(torch.nn.Dropout(dropout))
        
        # Output projection
        self.output = torch.nn.Sequential(
            LayerNorm(hidden_channels),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, x_dict, edge_index_dict, node_type='gene'):
        """
        Forward pass
        
        Args:
            x_dict: Dictionary of node features by node type
            edge_index_dict: Dictionary of edge indices by edge type
            node_type: Target node type for output
            
        Returns:
            Output logits for target node type
        """
        # Input projection
        for ntype, x in x_dict.items():
            x_dict[ntype] = self.lin_dict[ntype](x).relu()
        
        # Apply HGT layers with residual connections
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            x_dict_prev = {k: v.clone() for k, v in x_dict.items()}
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply normalization, dropout, and residual connection
            x_dict[node_type] = norm(x_dict[node_type])
            x_dict[node_type] = dropout(x_dict[node_type])
            x_dict[node_type] = x_dict[node_type] + x_dict_prev[node_type]
        
        # Output projection
        out = self.output(x_dict[node_type])
        return out


class PULoss(nn.Module):
    """
    Positive-Unlabeled (PU) learning loss based on unbiased risk estimator
    Handles label uncertainty by treating unlabeled samples appropriately
    """
    
    def __init__(self, prior=0.5, nnpu=True):
        """
        Initialize PU loss
        
        Args:
            prior: Estimated proportion of positive samples in unlabeled data
            nnpu: Whether to use non-negative PU learning
        """
        super().__init__()
        self.prior = prior
        self.nnpu = nnpu  # Non-negative PU learning
    
    def forward(self, outputs, targets):
        """
        Calculate PU loss
        
        Args:
            outputs: Model predictions (logits)
            targets: Ground truth labels (1 for positive, 0 for unlabeled)
            
        Returns:
            PU loss value
        """
        outputs = outputs.float()
        targets = targets.float()
        
        # Get predicted probabilities for positive class
        probs = F.softmax(outputs, dim=1)[:, 1]
        
        # Separate positive and unlabeled samples
        positive = (targets == 1).float()
        unlabeled = (targets == 0).float()
        
        n_positive = positive.sum() + 1e-7
        n_unlabeled = unlabeled.sum() + 1e-7
        
        # Positive risk
        positive_risk = -(positive * torch.log(probs + 1e-7)).sum() / n_positive
        
        # Negative risk (from unlabeled)
        negative_risk = -(unlabeled * torch.log(1 - probs + 1e-7)).sum() / n_unlabeled
        
        # PU risk estimator
        objective = self.prior * positive_risk + negative_risk - self.prior * negative_risk
        
        if self.nnpu:
            # Non-negative correction to prevent overfitting
            objective = torch.max(objective, torch.tensor(0.0, device=outputs.device))
        
        # Add small regularization term
        reg_loss = F.cross_entropy(outputs, targets.long())
        return 0.8 * objective + 0.2 * reg_loss