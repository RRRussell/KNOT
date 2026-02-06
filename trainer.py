#!/usr/bin/env python3
"""
Training pipeline for GNN druggability prediction
Updated to support parameter sensitivity analysis
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import numpy as np

from config import *
from models import MultiTaskHGT, PULoss
from utils import calculate_metrics, print_metrics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def _get_train_node_features(hetero_data, device):
    # Use gene node features for prior estimation (no graph needed)
    x = hetero_data['gene'].x
    train_mask = hetero_data['gene'].train_mask
    y = hetero_data['gene'].y
    return x[train_mask].to(device), y[train_mask].to(device)

class _PUPropensityMLP(nn.Module):
    """Aux classifier for P vs U: outputs P(labeled=1 | x)."""
    def __init__(self, in_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits

def estimate_pu_prior_elkan_noto(hetero_data, device, epochs=20, lr=1e-3, batch_size=2048, seed=42):
    """
    Estimate class prior pi = P(y=1) from PU data using Elkan-Noto.

    Assumptions:
      - train y=1 are labeled positives (P)
      - train y=0 are unlabeled (U)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train = _get_train_node_features(hetero_data, device)
    # labeled indicator s: P nodes are 1, U nodes are 0
    s_train = (y_train == 1).float()

    in_dim = x_train.size(-1)
    clf = _PUPropensityMLP(in_dim=in_dim).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)

    # simple class imbalance handling for P vs U
    n_pos = int((s_train == 1).sum().item())
    n_unl = int((s_train == 0).sum().item())
    # avoid div by zero
    pos_weight = torch.tensor([max(n_unl / max(n_pos, 1), 1.0)], device=device)

    clf.train()
    idx = torch.arange(x_train.size(0), device=device)

    for _ in range(epochs):
        perm = idx[torch.randperm(idx.numel())]
        for start in range(0, perm.numel(), batch_size):
            b = perm[start:start+batch_size]
            logits = clf(x_train[b])
            loss = F.binary_cross_entropy_with_logits(logits, s_train[b], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()

    clf.eval()
    # s_hat(x) = P(labeled=1 | x)
    s_hat = torch.sigmoid(clf(x_train))

    # c ≈ E_{x~P} s_hat(x)
    pos_mask = (s_train == 1)
    if pos_mask.sum().item() == 0:
        # fallback: cannot estimate; return small prior
        return 0.01
    c = s_hat[pos_mask].mean().clamp(min=1e-6, max=1.0)

    # pi ≈ E_{x~train} s_hat(x)/c
    pi = (s_hat / c).mean().clamp(min=1e-6, max=1.0)

    return float(pi.item())

class GNNTrainer:
    """Trainer for GNN models"""
    
    def __init__(self, model, device=DEVICE):
        self.model = model.to(device)
        self.device = device
        self.best_model_state = None
        
    def train_epoch(self, loader, optimizer, criterion):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            out = self.model(batch.x_dict, batch.edge_index_dict)
            batch_size = batch['gene'].batch_size
            
            batch_out = out[:batch_size]
            batch_labels = batch['gene'].y[:batch_size]
            
            loss = criterion(batch_out, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        return total_loss / total_samples
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x_dict, batch.edge_index_dict)
            batch_size = batch['gene'].batch_size
            
            batch_out = out[:batch_size]
            preds = batch_out.argmax(dim=-1)
            probs = F.softmax(batch_out, dim=-1)[:, 1]
            labels = batch['gene'].y[:batch_size]
            
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(labels)
        
        # Handle empty loader case (e.g., in inference mode with no val/test data)
        if not all_labels:
            return {
                'adjusted_f1': 0.0,
                'roc_auc': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        all_labels = torch.cat(all_labels).cpu().numpy()
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_probs = torch.cat(all_probs).cpu().numpy()
        
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        return metrics
    
    def train(self, hetero_data, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
              weight_decay=WEIGHT_DECAY, patience=PATIENCE, batch_size=BATCH_SIZE,
              use_pu_loss=True, verbose=True, return_training_curves=False):
        """Full training pipeline"""
        
        # Initialize training curves tracking
        training_curves = {
            'train_loss': [],
            'train_adj_f1': [],
            'val_adj_f1': [],
            'val_auc': [],
            'learning_rate': [],
            'epoch': []
        } if return_training_curves else None
        
        # Create data loaders
        num_neighbors = NEIGHBOR_SAMPLING[:self.model.num_layers]
        
        train_loader = NeighborLoader(
            hetero_data, num_neighbors=num_neighbors, batch_size=batch_size,
            input_nodes=('gene', hetero_data['gene'].train_mask), shuffle=True
        )
        
        # Check if we have validation and test data
        has_val = hetero_data['gene'].val_mask.sum().item() > 0
        has_test = hetero_data['gene'].test_mask.sum().item() > 0
        
        if has_val:
            val_loader = NeighborLoader(
                hetero_data, num_neighbors=num_neighbors, batch_size=batch_size,
                input_nodes=('gene', hetero_data['gene'].val_mask)
            )
        else:
            val_loader = None
            
        if has_test:
            test_loader = NeighborLoader(
                hetero_data, num_neighbors=num_neighbors, batch_size=batch_size,
                input_nodes=('gene', hetero_data['gene'].test_mask)
            )
        else:
            test_loader = None
        
        # Setup loss function
        train_labels = hetero_data['gene'].y[hetero_data['gene'].train_mask].cpu().numpy()

        if use_pu_loss and len(np.unique(train_labels)) > 1:
            # Estimate PU prior pi instead of naive labeled fraction
            prior = estimate_pu_prior_elkan_noto(
                hetero_data,
                device=self.device,
                epochs=20,
                lr=1e-3,
                batch_size=2048,
                seed=42
            )
            criterion = PULoss(prior=prior, nnpu=True).to(self.device)
            if verbose:
                print(f"Using PU Loss with estimated prior(pi)={prior:.4f} (Elkan-Noto)")
        else:
            criterion = nn.CrossEntropyLoss().to(self.device)
            if verbose:
                print("Using CrossEntropy Loss")
                
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
        
        # Training loop
        best_val_metric = -1
        best_epoch = 0
        patience_counter = 0
        
        if verbose:
            print(f"\nStarting training for {num_epochs} epochs...")
            if has_val:
                print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Adj F1':<11} | {'Val AUC':<9} | {'LR':<10}")
                print("-" * 60)
            else:
                print(f"{'Epoch':<6} | {'Train Loss':<11} | {'LR':<10}")
                print("-" * 40)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Evaluate training performance (for tracking curves)
            train_metrics = None
            if return_training_curves:
                train_metrics = self.evaluate(train_loader)
                train_adj_f1 = train_metrics.get('adjusted_f1', 0)
                if np.isnan(train_adj_f1):
                    train_adj_f1 = 0
            
            # Validate if we have validation data
            val_metrics = None
            val_adjusted_f1 = 0
            val_auc = 0
            
            if has_val:
                val_metrics = self.evaluate(val_loader)
                val_adjusted_f1 = val_metrics.get('adjusted_f1', 0)
                val_auc = val_metrics.get('roc_auc', 0)
                
                if np.isnan(val_adjusted_f1):
                    val_adjusted_f1 = 0
                if np.isnan(val_auc):
                    val_auc = 0
                
                # Update learning rate
                scheduler.step(val_adjusted_f1)
                
                # Check for improvement
                if val_adjusted_f1 > best_val_metric:
                    best_val_metric = val_adjusted_f1
                    best_epoch = epoch
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Print progress
                if verbose and (epoch % 10 == 0 or epoch == 1 or patience_counter == 0):
                    print(f"{epoch:<6} | {train_loss:<11.4f} | {val_adjusted_f1:<11.4f} | "
                          f"{val_auc:<9.4f} | {current_lr:<10.2e}")
            else:
                # No validation data (inference mode)
                # Save best model based on training loss
                if epoch == 1 or train_loss < best_val_metric or best_val_metric < 0:
                    best_val_metric = -train_loss  # Use negative loss as metric
                    best_epoch = epoch
                    self.best_model_state = self.model.state_dict().copy()
                
                # Print progress
                if verbose and (epoch % 10 == 0 or epoch == 1):
                    print(f"{epoch:<6} | {train_loss:<11.4f} | {current_lr:<10.2e}")
            
            # Record training curves
            if return_training_curves:
                training_curves['epoch'].append(epoch)
                training_curves['train_loss'].append(train_loss)
                training_curves['train_adj_f1'].append(train_adj_f1 if train_metrics else 0)
                training_curves['val_adj_f1'].append(val_adjusted_f1)
                training_curves['val_auc'].append(val_auc)
                training_curves['learning_rate'].append(current_lr)
            
            # Early stopping (only if we have validation data)
            if has_val and patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation
        results = {
            'train': self.evaluate(train_loader),
            'best_epoch': best_epoch,
            'best_val_metric': best_val_metric,
            'total_epochs': epoch
        }
        
        if has_val:
            results['val'] = self.evaluate(val_loader)
        
        if has_test:
            results['test'] = self.evaluate(test_loader)
            
        # Add training curves to results if requested
        if return_training_curves:
            results['training_curves'] = training_curves
        
        if verbose:
            print(f"\nFinal evaluation (best model from epoch {best_epoch}):")
            print_metrics(results['train'], "Train")
            if has_val:
                print_metrics(results['val'], "Validation")
            if has_test:
                print_metrics(results['test'], "Test")
        
        return results
    
    @torch.no_grad()
    def predict_all(self, hetero_data, batch_size=BATCH_SIZE):
        """Predict scores for all nodes (inference mode)"""
        self.model.eval()
        
        # Create loader for all nodes
        all_mask = torch.ones(hetero_data['gene'].num_nodes, dtype=torch.bool)
        loader = NeighborLoader(
            hetero_data,
            num_neighbors=NEIGHBOR_SAMPLING[:self.model.num_layers],
            batch_size=batch_size,
            input_nodes=('gene', all_mask)
        )
        
        all_scores = []
        all_indices = []
        
        for batch in tqdm(loader, desc="Predicting"):
            batch = batch.to(self.device)
            out = self.model(batch.x_dict, batch.edge_index_dict)
            batch_size = batch['gene'].batch_size
            
            batch_out = out[:batch_size]
            scores = F.softmax(batch_out, dim=-1)[:, 1]  # Probability of positive class
            
            # Get original node indices
            batch_indices = batch['gene'].n_id[:batch_size]
            
            all_scores.append(scores)
            all_indices.append(batch_indices)
        
        # Concatenate and reorder
        all_scores = torch.cat(all_scores)
        all_indices = torch.cat(all_indices)
        
        # Sort by original indices to maintain order
        sorted_indices = torch.argsort(all_indices)
        final_scores = all_scores[sorted_indices].cpu().numpy()
        
        return final_scores
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_model_state': self.best_model_state
        }, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model_state = checkpoint.get('best_model_state')
        print(f"Checkpoint loaded from {filepath}")
        
    def get_parameter_count(self):
        """Get detailed parameter count information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Parameter breakdown by layer type
        param_breakdown = {}
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    param_breakdown[name] = module_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_breakdown': param_breakdown
        }
    
    def get_model_info(self):
        """Get comprehensive model information for analysis"""
        param_info = self.get_parameter_count()
        
        model_info = {
            'model_class': self.model.__class__.__name__,
            'device': str(self.device),
            'parameter_info': param_info,
            'model_config': {}
        }
        
        # Try to extract model configuration
        if hasattr(self.model, 'hidden_channels'):
            model_info['model_config']['hidden_channels'] = self.model.hidden_channels
        if hasattr(self.model, 'num_layers'):
            model_info['model_config']['num_layers'] = self.model.num_layers
        if hasattr(self.model, 'heads'):
            model_info['model_config']['heads'] = self.model.heads
        if hasattr(self.model, 'dropout'):
            model_info['model_config']['dropout'] = self.model.dropout
        
        return model_info