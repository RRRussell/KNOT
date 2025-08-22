#!/usr/bin/env python3
"""
Utility functions for KNOT GNN druggability prediction
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    accuracy_score, balanced_accuracy_score, matthews_corrcoef
)


def calculate_adjusted_f1(y_true, y_pred_proba):
    """
    Calculate adjusted F1 score for PU learning
    Based on KNOT paper: F1_adj = R_soft^2 / p_bar
    
    Args:
        y_true: True labels (1 for positive, 0 for unlabeled)
        y_pred_proba: Predicted probabilities
    
    Returns:
        Adjusted F1 score
    """
    try:
        true_pos_mask = (y_true == 1)
        if not np.any(true_pos_mask):
            return 0.0
        
        probs_clipped = np.clip(y_pred_proba, 0.0, 1.0)
        
        # Soft recall: average probability on positive examples
        soft_recall = np.mean(probs_clipped[true_pos_mask])
        
        # Average predicted probability across all examples
        pred_pos_prob = np.mean(probs_clipped)
        
        if pred_pos_prob > 0:
            # Adjusted F1 = (soft_recall^2) / pred_pos_prob
            return (soft_recall ** 2) / pred_pos_prob
        else:
            return 0.0
    except:
        return 0.0


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive metrics for evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Handle tensor inputs
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if y_pred_proba is not None and torch.is_tensor(y_pred_proba):
        y_pred_proba = y_pred_proba.cpu().numpy()
    
    try:
        # Basic confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_positives': tp, 'true_negatives': tn,
            'false_positives': fp, 'false_negatives': fn
        })
        
        # Classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['matthews_corr'] = matthews_corrcoef(y_true, y_pred)
        
        # Probability-based metrics
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            metrics['adjusted_f1'] = calculate_adjusted_f1(y_true, y_pred_proba)
        else:
            metrics.update({'roc_auc': np.nan, 'pr_auc': np.nan, 'adjusted_f1': np.nan})
            
    except Exception as e:
        print(f"Metric calculation error: {e}")
        return {}
    
    return metrics


def print_metrics(metrics, prefix=""):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for the printout
    """
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")
    
    print(f"  Accuracy:          {metrics.get('accuracy', 0):.4f}")
    print(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"  Precision:         {metrics.get('precision', 0):.4f}")
    print(f"  Recall:            {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:          {metrics.get('f1_score', 0):.4f}")
    print(f"  Adjusted F1 (PU):  {metrics.get('adjusted_f1', 0):.4f}")
    print(f"  ROC AUC:           {metrics.get('roc_auc', 0):.4f}")
    print(f"  PR AUC:            {metrics.get('pr_auc', 0):.4f}")
    print(f"  Matthews Corr:     {metrics.get('matthews_corr', 0):.4f}")


def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False