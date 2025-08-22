#!/usr/bin/env python3
"""
Main script for KNOT GNN druggability prediction
Supports both evaluation mode (train/val/test) and inference mode (rank all genes)
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

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
    
    # Edge configuration
    parser.add_argument('--edge-config', type=str, default=DEFAULT_EDGE_CONFIG,
                       choices=list(EDGE_CONFIGS.keys()),
                       help='Edge type configuration')
    
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
    
    # Other options
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help='Random seed')
    parser.add_argument('--save-dir', type=str, default=str(OUTPUT_DIR),
                       help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("="*80)
    print("KNOT GNN DRUGGABILITY PREDICTION")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Task Name: {DRUGGABILITY_TASKS[args.task]['display_name']}")
    print(f"Mode: {args.mode}")
    print(f"Edge configuration: {args.edge_config}")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    # Initialize data loader
    data_loader = DrugabilityDataLoader(
        edge_config=args.edge_config
    )
    
    # Load and prepare data
    data_info = data_loader.prepare_task_data(args.task, mode=args.mode)
    
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
    print(f"\nModel initialized with {total_params:,} parameters")
    
    # Initialize trainer
    trainer = GNNTrainer(model, device=DEVICE)
    
    if args.mode == 'evaluation':
        # Evaluation mode: train/val/test split
        print("\nRunning in EVALUATION mode...")
        
        results = trainer.train(
            hetero_data,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        results_file = os.path.join(args.save_dir, f"eval_results_{args.task}_{timestamp}.json")
        with open(results_file, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            json_results = {
                split: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in metrics.items()}
                for split, metrics in results.items()
                if isinstance(metrics, dict)
            }
            # Add configuration
            json_results['config'] = {
                'task': args.task,
                'mode': args.mode,
                'edge_config': args.edge_config,
                'seed': args.seed,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr
            }
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {results_file}")
        
        # Save checkpoint
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"model_{args.task}_{timestamp}.pt")
        trainer.save_checkpoint(checkpoint_file)
        
    else:  # inference mode
        # Inference mode: train on all labeled data, rank all genes
        print("\nRunning in INFERENCE mode...")
        print("Training on all labeled data...")
        
        # Train on all data
        results = trainer.train(
            hetero_data,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )
        
        # Get predictions for all genes
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
        ranking_file = os.path.join(args.save_dir, f"gene_ranking_{args.task}_{timestamp}.csv")
        ranking_df.to_csv(ranking_file, index=False)
        print(f"\nGene ranking saved to {ranking_file}")
        
        # Print top genes
        print("\nTop 20 genes by druggability score:")
        print(ranking_df[['Rank', 'Gene_Symbol', 'Druggability_Score', 'Original_Label']].head(20).to_string(index=False))
        
        # Save model
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"inference_model_{args.task}_{timestamp}.pt")
        trainer.save_checkpoint(checkpoint_file)
        
        # Summary statistics
        print("\n" + "="*60)
        print("INFERENCE SUMMARY")
        print("="*60)
        print(f"Total genes ranked: {len(ranking_df)}")
        print(f"Genes with positive labels: {(ranking_df['Original_Label'] == 1).sum()}")
        print(f"Genes with negative/unlabeled: {(ranking_df['Original_Label'] == 0).sum()}")
        print(f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
        print(f"Mean score: {all_scores.mean():.4f} ± {all_scores.std():.4f}")
        
        # Check how well we rank known positives
        positive_ranks = ranking_df[ranking_df['Original_Label'] == 1]['Rank'].values
        if len(positive_ranks) > 0:
            print(f"\nKnown positives ranking:")
            print(f"  Median rank: {np.median(positive_ranks):.0f}/{len(ranking_df)}")
            print(f"  Mean rank: {np.mean(positive_ranks):.1f}/{len(ranking_df)}")
            print(f"  Top 100: {(positive_ranks <= 100).sum()}/{len(positive_ranks)}")
            print(f"  Top 500: {(positive_ranks <= 500).sum()}/{len(positive_ranks)}")
            print(f"  Top 1000: {(positive_ranks <= 1000).sum()}/{len(positive_ranks)}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()