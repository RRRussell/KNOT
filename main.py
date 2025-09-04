#!/usr/bin/env python3
"""
Enhanced main script for KNOT GNN druggability prediction
Added features for parameter sensitivity analysis
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import time
import torch

from config import *
from data_loader import DrugabilityDataLoader
from models import MultiTaskHGT
from trainer import GNNTrainer
from utils import set_seed, print_metrics


def main():
    parser = argparse.ArgumentParser(description='KNOT GNN for Gene Druggability Prediction')
    
    # Task and mode
    parser.add_argument('--task', type=str, default=DEFAULT_TASK,
                       choices=list(DRUGGABILITY_TASKS.keys()),
                       help='Druggability prediction task')
    parser.add_argument('--mode', type=str, default='evaluation',
                       choices=['evaluation', 'inference'],
                       help='Mode: evaluation (train/val/test) or inference (rank all genes)')
    
    # Edge and feature configurations
    parser.add_argument('--edge-config', type=str, default=DEFAULT_EDGE_CONFIG,
                       choices=list(EDGE_CONFIGS.keys()),
                       help='Edge type configuration')
    parser.add_argument('--feature-config', type=str, default=DEFAULT_FEATURE_CONFIG,
                       choices=list(FEATURE_CONFIGS.keys()),
                       help='Feature configuration')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=HIDDEN_CHANNELS,
                       help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=NUM_LAYERS,
                       help='Number of GNN layers')
    parser.add_argument('--num-heads', type=int, default=NUM_HEADS,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                       help='Early stopping patience')
    parser.add_argument('--use_pu_loss', type=bool, default=True,
                       help='Use pu loss')
    
    # Other options
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help='Random seed')
    parser.add_argument('--save-dir', type=str, default=str(OUTPUT_DIR),
                       help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    parser.add_argument('--show-feature-breakdown', action='store_true',
                       help='Show detailed feature breakdown')
    
    # Enhanced options for parameter sensitivity analysis
    parser.add_argument('--save-training-curves', action='store_true',
                       help='Save detailed training curves for analysis')
    parser.add_argument('--save-model-info', action='store_true', default=True,
                       help='Save detailed model information')
    parser.add_argument('--track-gpu-usage', action='store_true',
                       help='Track GPU memory usage during training')
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='Unique experiment identifier for tracking')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Record experiment start time
    experiment_start_time = time.time()
    
    if not args.quiet:
        print("="*80)
        print("KNOT GNN DRUGGABILITY PREDICTION")
        print("="*80)
        print(f"Task: {args.task}")
        print(f"Task Name: {DRUGGABILITY_TASKS[args.task]['display_name']}")
        print(f"Mode: {args.mode}")
        print(f"Edge configuration: {args.edge_config}")
        print(f"Feature configuration: {args.feature_config}")
        print(f"Device: {DEVICE}")
        
        # Enhanced parameter reporting for sensitivity analysis
        print(f"\nModel Parameters:")
        print(f"  Hidden dimensions: {args.hidden_dim}")
        print(f"  Number of layers: {args.num_layers}")
        print(f"  Attention heads: {args.num_heads}")
        print(f"  Dropout: {args.dropout}")
        
        print(f"\nTraining Parameters:")
        print(f"  Learning rate: {args.lr}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Max epochs: {args.epochs}")
        print(f"  Patience: {args.patience}")
        print(f"  Use PU Loss: {args.use_pu_loss}")
        print(f"  Random seed: {args.seed}")
        
        if args.experiment_id:
            print(f"  Experiment ID: {args.experiment_id}")
        
        print("="*80)
    
    # Initialize data loader
    data_loader = DrugabilityDataLoader(
        edge_config=args.edge_config,
        feature_config=args.feature_config
    )
    
    # Show feature breakdown if requested
    if args.show_feature_breakdown and not args.quiet:
        print("\nFeature Breakdown:")
        breakdown = data_loader.get_feature_breakdown()
        for category, info in breakdown.items():
            print(f"  {category}: {info['available']}/{info['total']}")
        print()
    
    # Load and prepare data
    data_loading_start = time.time()
    data_info = data_loader.prepare_task_data(args.task, mode=args.mode)
    data_loading_time = time.time() - data_loading_start
    
    # Convert to heterogeneous format
    hetero_data = data_loader.convert_to_hetero(data_info['data'])
    hetero_data = hetero_data.to(DEVICE)
    
    # Initialize model
    metadata = (list(hetero_data.node_types), list(hetero_data.edge_types))
    model = MultiTaskHGT(
        in_channels=data_info['data'].x.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=2,  # Binary classification
        metadata=metadata,
        heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if not args.quiet:
        print(f"\nModel initialized with {total_params:,} parameters")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Model architecture details for sensitivity analysis
        if args.save_model_info:
            print(f"\nDetailed Model Architecture:")
            print(f"  Input features: {data_info['data'].x.shape[1]}")
            print(f"  Hidden dimensions: {args.hidden_dim}")
            print(f"  Number of layers: {args.num_layers}")
            print(f"  Attention heads: {args.num_heads}")
            print(f"  Node types: {len(hetero_data.node_types)}")
            print(f"  Edge types: {len(hetero_data.edge_types)}")
    
    # Initialize trainer
    trainer = GNNTrainer(model, device=DEVICE)
    
    if args.mode == 'evaluation':
        # Evaluation mode: train/val/test split
        if not args.quiet:
            print("\nRunning in EVALUATION mode...")
        
        # Track GPU usage if requested
        if args.track_gpu_usage and torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
        
        training_start_time = time.time()
        results = trainer.train(
            hetero_data,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            batch_size=args.batch_size,
            verbose=not args.quiet,
            use_pu_loss=args.use_pu_loss,
            return_training_curves=args.save_training_curves
        )
        training_time = time.time() - training_start_time
        
        # Record GPU usage
        gpu_stats = {}
        if args.track_gpu_usage and torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            max_memory_used = torch.cuda.max_memory_allocated()
            gpu_stats = {
                'initial_memory_mb': initial_memory / 1024 / 1024,
                'final_memory_mb': final_memory / 1024 / 1024,
                'max_memory_mb': max_memory_used / 1024 / 1024,
                'memory_increase_mb': (final_memory - initial_memory) / 1024 / 1024
            }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced results with timing and system information
        enhanced_results = {}
        
        # Copy original results
        for split, metrics in results.items():
            if isinstance(metrics, dict):
                enhanced_results[split] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items()
                }
        
        # Add comprehensive experiment metadata
        enhanced_results['experiment_metadata'] = {
            'experiment_id': args.experiment_id,
            'timestamp': timestamp,
            'task': args.task,
            'task_display_name': DRUGGABILITY_TASKS[args.task]['display_name'],
            'mode': args.mode,
            'edge_config': args.edge_config,
            'feature_config': args.feature_config,
            'seed': args.seed,
        }
        
        # Model configuration
        enhanced_results['model_config'] = {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_features': data_info['data'].x.shape[1],
        }
        
        # Training configuration
        enhanced_results['training_config'] = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'use_pu_loss': args.use_pu_loss,
        }
        
        # Timing information
        enhanced_results['timing'] = {
            'data_loading_time_seconds': data_loading_time,
            'training_time_seconds': training_time,
            'total_experiment_time_seconds': time.time() - experiment_start_time,
        }
        
        # System information
        enhanced_results['system_info'] = {
            'device': str(DEVICE),
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        # GPU usage statistics
        if gpu_stats:
            enhanced_results['gpu_stats'] = gpu_stats
        
        # Data statistics
        enhanced_results['data_stats'] = {
            'num_nodes': hetero_data['gene'].x.shape[0],
            'num_features': hetero_data['gene'].x.shape[1],
            'num_edges': sum(hetero_data[edge_type].edge_index.shape[1] 
                           for edge_type in hetero_data.edge_types),
            'num_edge_types': len(hetero_data.edge_types),
            'pos_samples': int((data_info['data'].y == 1).sum()),
            'neg_samples': int((data_info['data'].y == 0).sum()),
            'pos_ratio': float((data_info['data'].y == 1).float().mean()),
        }
        
        # Training curves (if requested)
        if args.save_training_curves and 'training_curves' in results:
            enhanced_results['training_curves'] = results['training_curves']
        
        # Save results
        results_file = os.path.join(args.save_dir, f"eval_results_{args.task}_{args.feature_config}_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        if not args.quiet:
            print(f"\nEnhanced results saved to {results_file}")
            print(f"\nTiming Summary:")
            print(f"  Data loading: {data_loading_time:.2f}s")
            print(f"  Training: {training_time:.2f}s")
            print(f"  Total experiment: {time.time() - experiment_start_time:.2f}s")
            
            if gpu_stats:
                print(f"\nGPU Memory Usage:")
                print(f"  Peak memory: {gpu_stats['max_memory_mb']:.1f} MB")
                print(f"  Memory increase: {gpu_stats['memory_increase_mb']:.1f} MB")
        
        # Save checkpoint
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"model_{args.task}_{args.feature_config}_{timestamp}.pt")
        trainer.save_checkpoint(checkpoint_file)
        
    else:  # inference mode
        # Inference mode: train on all labeled data, rank all genes
        if not args.quiet:
            print("\nRunning in INFERENCE mode...")
            print("Training on all labeled data...")
        
        # Train on all data
        training_start_time = time.time()
        results = trainer.train(
            hetero_data,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )
        training_time = time.time() - training_start_time
        
        # Get predictions for all genes
        if not args.quiet:
            print("\nGenerating predictions for all genes...")
        all_scores = trainer.predict_all(hetero_data, batch_size=args.batch_size)
        
        # Create ranking dataframe
        gene_symbols = data_info['gene_symbols']
        labels = data_info['data'].y.cpu().numpy()
        
        ranking_df = pd.DataFrame({
            'Gene_Symbol': gene_symbols,
            'Druggability_Score': all_scores,
            'Original_Label': labels,
            'Rank': pd.Series(all_scores).rank(ascending=False, method='min').astype(int)
        })
        
        # Sort by score
        ranking_df = ranking_df.sort_values('Druggability_Score', ascending=False)
        
        # Save ranking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ranking_file = os.path.join(args.save_dir, f"gene_ranking_{args.task}_{args.feature_config}_{timestamp}.csv")
        ranking_df.to_csv(ranking_file, index=False)
        
        if not args.quiet:
            print(f"\nGene ranking saved to {ranking_file}")
            
            # Print top genes
            print("\nTop 20 genes by druggability score:")
            print(ranking_df[['Rank', 'Gene_Symbol', 'Druggability_Score', 'Original_Label']].head(20).to_string(index=False))
        
        # Save model
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"inference_model_{args.task}_{args.feature_config}_{timestamp}.pt")
        trainer.save_checkpoint(checkpoint_file)
        
        # Enhanced inference results
        inference_results = {
            'experiment_metadata': {
                'experiment_id': args.experiment_id,
                'timestamp': timestamp,
                'task': args.task,
                'mode': args.mode,
                'total_genes_ranked': len(ranking_df),
                'training_time_seconds': training_time,
                'total_experiment_time_seconds': time.time() - experiment_start_time,
            },
            'ranking_stats': {
                'score_mean': float(all_scores.mean()),
                'score_std': float(all_scores.std()),
                'score_min': float(all_scores.min()),
                'score_max': float(all_scores.max()),
                'positive_genes': int((labels == 1).sum()),
                'negative_genes': int((labels == 0).sum()),
            }
        }
        
        # Check how well we rank known positives
        positive_ranks = ranking_df[ranking_df['Original_Label'] == 1]['Rank'].values
        if len(positive_ranks) > 0:
            inference_results['positive_ranking_stats'] = {
                'median_rank': float(np.median(positive_ranks)),
                'mean_rank': float(np.mean(positive_ranks)),
                'top_100': int((positive_ranks <= 100).sum()),
                'top_500': int((positive_ranks <= 500).sum()),
                'top_1000': int((positive_ranks <= 1000).sum()),
            }
            
            if not args.quiet:
                print(f"\nKnown positives ranking:")
                print(f"  Median rank: {np.median(positive_ranks):.0f}/{len(ranking_df)}")
                print(f"  Mean rank: {np.mean(positive_ranks):.1f}/{len(ranking_df)}")
                print(f"  Top 100: {(positive_ranks <= 100).sum()}/{len(positive_ranks)}")
                print(f"  Top 500: {(positive_ranks <= 500).sum()}/{len(positive_ranks)}")
                print(f"  Top 1000: {(positive_ranks <= 1000).sum()}/{len(positive_ranks)}")
        
        # Save inference results
        inference_results_file = os.path.join(args.save_dir, f"inference_results_{args.task}_{args.feature_config}_{timestamp}.json")
        with open(inference_results_file, 'w') as f:
            json.dump(inference_results, f, indent=2)
    
    if not args.quiet:
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)


if __name__ == "__main__":
    main()
