#!/usr/bin/env python3
"""
Hierarchical Hyperparameter Search for SimplifiedSpectralCF
Systematically searches through hyperparameters in order of importance
"""

import subprocess
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical hyperparameter search")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset to search on: ml-100k, gowalla, yelp2018, amazon-book, lastfm')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for each experiment')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout for each experiment in seconds')
    parser.add_argument('--save_results', type=str, default='hp_search_results.json',
                       help='File to save results')
    parser.add_argument('--skip_to_stage', type=int, default=1,
                       help='Skip to specific stage (1=lr, 2=decay, 3=eigen, 4=filter)')
    return parser.parse_args()


def get_dataset_specific_ranges(dataset):
    """Get dataset-specific hyperparameter ranges"""
    
    # Common ranges
    lr_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
    decay_values = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    filter_orders = [3, 4, 5, 6, 7, 8]
    
    # Dataset-specific eigenvalue ranges
    if dataset == 'ml-100k':
        u_n_eigen = list(range(15, 50, 5))  # [15, 20, 25, 30, 35, 40, 45]
        i_n_eigen = list(range(25, 65, 5))  # [25, 30, 35, 40, 45, 50, 55, 60]
    elif dataset == 'lastfm':
        u_n_eigen = list(range(10, 40, 5))  # [10, 15, 20, 25, 30, 35]
        i_n_eigen = list(range(20, 55, 5))  # [20, 25, 30, 35, 40, 45, 50]
    elif dataset == 'gowalla':
        u_n_eigen = list(range(20, 65, 5))  # [20, 25, 30, 35, 40, 45, 50, 55, 60]
        i_n_eigen = list(range(30, 80, 5))  # [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    elif dataset == 'yelp2018':
        u_n_eigen = list(range(25, 75, 5))  # [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        i_n_eigen = list(range(40, 90, 5))  # [40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    elif dataset == 'amazon-book':
        u_n_eigen = list(range(30, 80, 5))  # [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
        i_n_eigen = list(range(50, 105, 5)) # [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    else:
        # Default ranges
        u_n_eigen = list(range(15, 50, 5))
        i_n_eigen = list(range(25, 65, 5))
    
    return {
        'lr_values': lr_values,
        'decay_values': decay_values,
        'u_n_eigen': u_n_eigen,
        'i_n_eigen': i_n_eigen,
        'filter_orders': filter_orders
    }


def run_experiment(dataset, lr=0.01, decay=1e-2, n_eigen_user=20, n_eigen_item=40, 
                  filter_order=6, use_user_gram=True, use_item_gram=True, 
                  epochs=50, patience=5, timeout=300):
    """Run a single experiment and return results"""
    
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset,
        "--lr", str(lr),
        "--decay", str(decay),
        "--n_eigen_user", str(n_eigen_user),
        "--n_eigen_item", str(n_eigen_item),
        "--filter_order", str(filter_order),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--n_epoch_eval", "5"
    ]
    
    if not use_user_gram:
        cmd.append("--no_user_gram")
    if not use_item_gram:
        cmd.append("--no_item_gram")
    
    print(f"Running: lr={lr}, decay={decay}, u_eigen={n_eigen_user}, i_eigen={n_eigen_item}, filter={filter_order}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        # Parse results
        lines = result.stdout.split('\n')
        final_line = None
        for line in lines:
            if "Final Results:" in line:
                final_line = line
                break
        
        if final_line and result.returncode == 0:
            # Parse metrics
            parts = final_line.split(", ")
            recall = float(parts[0].split("=")[1])
            precision = float(parts[1].split("=")[1])
            ndcg = float(parts[2].split("=")[1])
            
            return {
                'success': True,
                'recall': recall,
                'precision': precision,
                'ndcg': ndcg,
                'lr': lr,
                'decay': decay,
                'n_eigen_user': n_eigen_user,
                'n_eigen_item': n_eigen_item,
                'filter_order': filter_order,
                'use_user_gram': use_user_gram,
                'use_item_gram': use_item_gram
            }
        else:
            return {'success': False, 'error': 'Could not parse results or non-zero exit'}
            
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def stage1_search_lr(dataset, ranges, epochs, patience, timeout):
    """Stage 1: Search for best learning rate"""
    print("\n" + "="*80)
    print("STAGE 1: LEARNING RATE SEARCH")
    print("="*80)
    
    # Use reasonable defaults for other parameters
    decay = 1e-2
    n_eigen_user = ranges['u_n_eigen'][len(ranges['u_n_eigen'])//2]  # Middle value
    n_eigen_item = ranges['i_n_eigen'][len(ranges['i_n_eigen'])//2]  # Middle value
    filter_order = 6
    
    best_lr = None
    best_ndcg = 0.0
    lr_results = []
    
    for lr in ranges['lr_values']:
        result = run_experiment(
            dataset, lr=lr, decay=decay, n_eigen_user=n_eigen_user, 
            n_eigen_item=n_eigen_item, filter_order=filter_order,
            epochs=epochs, patience=patience, timeout=timeout
        )
        
        if result['success']:
            lr_results.append(result)
            print(f"  lr={lr}: NDCG={result['ndcg']:.6f}, Recall={result['recall']:.6f}")
            
            if result['ndcg'] > best_ndcg:
                best_ndcg = result['ndcg']
                best_lr = lr
                print(f"    → New best NDCG: {best_ndcg:.6f}")
        else:
            print(f"  lr={lr}: FAILED - {result['error']}")
    
    print(f"\nStage 1 Result: Best lr = {best_lr} (NDCG = {best_ndcg:.6f})")
    return best_lr, lr_results


def stage2_search_decay(dataset, best_lr, ranges, epochs, patience, timeout):
    """Stage 2: Search for best weight decay"""
    print("\n" + "="*80)
    print("STAGE 2: WEIGHT DECAY SEARCH")
    print("="*80)
    
    # Use best lr from stage 1 and reasonable defaults for others
    n_eigen_user = ranges['u_n_eigen'][len(ranges['u_n_eigen'])//2]
    n_eigen_item = ranges['i_n_eigen'][len(ranges['i_n_eigen'])//2]
    filter_order = 6
    
    best_decay = None
    best_ndcg = 0.0
    decay_results = []
    
    for decay in ranges['decay_values']:
        result = run_experiment(
            dataset, lr=best_lr, decay=decay, n_eigen_user=n_eigen_user,
            n_eigen_item=n_eigen_item, filter_order=filter_order,
            epochs=epochs, patience=patience, timeout=timeout
        )
        
        if result['success']:
            decay_results.append(result)
            print(f"  decay={decay}: NDCG={result['ndcg']:.6f}, Recall={result['recall']:.6f}")
            
            if result['ndcg'] > best_ndcg:
                best_ndcg = result['ndcg']
                best_decay = decay
                print(f"    → New best NDCG: {best_ndcg:.6f}")
        else:
            print(f"  decay={decay}: FAILED - {result['error']}")
    
    print(f"\nStage 2 Result: Best decay = {best_decay} (NDCG = {best_ndcg:.6f})")
    return best_decay, decay_results


def stage3_search_eigenvalues(dataset, best_lr, best_decay, ranges, epochs, patience, timeout):
    """Stage 3: Search for best eigenvalue settings"""
    print("\n" + "="*80)
    print("STAGE 3: EIGENVALUE SEARCH")
    print("="*80)
    
    filter_order = 6
    best_eigen_combo = None
    best_ndcg = 0.0
    eigen_results = []
    
    # Grid search over eigenvalue combinations
    for n_eigen_user in ranges['u_n_eigen']:
        for n_eigen_item in ranges['i_n_eigen']:
            result = run_experiment(
                dataset, lr=best_lr, decay=best_decay, 
                n_eigen_user=n_eigen_user, n_eigen_item=n_eigen_item,
                filter_order=filter_order, epochs=epochs, 
                patience=patience, timeout=timeout
            )
            
            if result['success']:
                eigen_results.append(result)
                print(f"  u_eigen={n_eigen_user}, i_eigen={n_eigen_item}: NDCG={result['ndcg']:.6f}")
                
                if result['ndcg'] > best_ndcg:
                    best_ndcg = result['ndcg']
                    best_eigen_combo = (n_eigen_user, n_eigen_item)
                    print(f"    → New best NDCG: {best_ndcg:.6f}")
            else:
                print(f"  u_eigen={n_eigen_user}, i_eigen={n_eigen_item}: FAILED - {result['error']}")
    
    print(f"\nStage 3 Result: Best eigenvalues = {best_eigen_combo} (NDCG = {best_ndcg:.6f})")
    return best_eigen_combo, eigen_results


def stage4_search_filters(dataset, best_lr, best_decay, best_eigen_combo, ranges, epochs, patience, timeout):
    """Stage 4: Search for best filter configurations"""
    print("\n" + "="*80)
    print("STAGE 4: FILTER CONFIGURATION SEARCH")
    print("="*80)
    
    best_n_eigen_user, best_n_eigen_item = best_eigen_combo
    best_config = None
    best_ndcg = 0.0
    filter_results = []
    
    # Test different filter orders
    for filter_order in ranges['filter_orders']:
        # Test with both user and item grams
        result = run_experiment(
            dataset, lr=best_lr, decay=best_decay,
            n_eigen_user=best_n_eigen_user, n_eigen_item=best_n_eigen_item,
            filter_order=filter_order, use_user_gram=True, use_item_gram=True,
            epochs=epochs, patience=patience, timeout=timeout
        )
        
        if result['success']:
            filter_results.append(result)
            print(f"  filter_order={filter_order}, both_grams: NDCG={result['ndcg']:.6f}")
            
            if result['ndcg'] > best_ndcg:
                best_ndcg = result['ndcg']
                best_config = (filter_order, True, True)
                print(f"    → New best NDCG: {best_ndcg:.6f}")
        
        # Test with only item grams
        result = run_experiment(
            dataset, lr=best_lr, decay=best_decay,
            n_eigen_user=best_n_eigen_user, n_eigen_item=best_n_eigen_item,
            filter_order=filter_order, use_user_gram=False, use_item_gram=True,
            epochs=epochs, patience=patience, timeout=timeout
        )
        
        if result['success']:
            filter_results.append(result)
            print(f"  filter_order={filter_order}, item_only: NDCG={result['ndcg']:.6f}")
            
            if result['ndcg'] > best_ndcg:
                best_ndcg = result['ndcg']
                best_config = (filter_order, False, True)
                print(f"    → New best NDCG: {best_ndcg:.6f}")
        
        # Test with only user grams
        result = run_experiment(
            dataset, lr=best_lr, decay=best_decay,
            n_eigen_user=best_n_eigen_user, n_eigen_item=best_n_eigen_item,
            filter_order=filter_order, use_user_gram=True, use_item_gram=False,
            epochs=epochs, patience=patience, timeout=timeout
        )
        
        if result['success']:
            filter_results.append(result)
            print(f"  filter_order={filter_order}, user_only: NDCG={result['ndcg']:.6f}")
            
            if result['ndcg'] > best_ndcg:
                best_ndcg = result['ndcg']
                best_config = (filter_order, True, False)
                print(f"    → New best NDCG: {best_ndcg:.6f}")
    
    print(f"\nStage 4 Result: Best filter config = {best_config} (NDCG = {best_ndcg:.6f})")
    return best_config, filter_results


def main():
    args = parse_args()
    
    print(f"🔍 Hierarchical Hyperparameter Search for {args.dataset.upper()}")
    print(f"Configuration: epochs={args.epochs}, patience={args.patience}, timeout={args.timeout}s")
    
    # Get dataset-specific ranges
    ranges = get_dataset_specific_ranges(args.dataset)
    
    print(f"\nSearch Ranges:")
    print(f"  Learning rates: {ranges['lr_values']}")
    print(f"  Weight decays: {ranges['decay_values']}")
    print(f"  User eigenvalues: {ranges['u_n_eigen']}")
    print(f"  Item eigenvalues: {ranges['i_n_eigen']}")
    print(f"  Filter orders: {ranges['filter_orders']}")
    
    # Store all results
    all_results = {
        'dataset': args.dataset,
        'ranges': ranges,
        'stages': {}
    }
    
    start_time = time.time()
    
    # Stage 1: Learning Rate Search
    if args.skip_to_stage <= 1:
        best_lr, lr_results = stage1_search_lr(
            args.dataset, ranges, args.epochs, args.patience, args.timeout
        )
        all_results['stages']['lr'] = {
            'best_lr': best_lr,
            'results': lr_results
        }
    else:
        # Use default if skipping
        best_lr = 0.1
        print(f"Skipping to stage {args.skip_to_stage}, using default lr={best_lr}")
    
    # Stage 2: Weight Decay Search  
    if args.skip_to_stage <= 2:
        best_decay, decay_results = stage2_search_decay(
            args.dataset, best_lr, ranges, args.epochs, args.patience, args.timeout
        )
        all_results['stages']['decay'] = {
            'best_decay': best_decay,
            'results': decay_results
        }
    else:
        best_decay = 1e-2
        print(f"Skipping to stage {args.skip_to_stage}, using default decay={best_decay}")
    
    # Stage 3: Eigenvalue Search
    if args.skip_to_stage <= 3:
        best_eigen_combo, eigen_results = stage3_search_eigenvalues(
            args.dataset, best_lr, best_decay, ranges, args.epochs, args.patience, args.timeout
        )
        all_results['stages']['eigenvalues'] = {
            'best_eigen_combo': best_eigen_combo,
            'results': eigen_results
        }
    else:
        # Use middle values if skipping
        best_eigen_combo = (
            ranges['u_n_eigen'][len(ranges['u_n_eigen'])//2],
            ranges['i_n_eigen'][len(ranges['i_n_eigen'])//2]
        )
        print(f"Skipping to stage {args.skip_to_stage}, using default eigenvalues={best_eigen_combo}")
    
    # Stage 4: Filter Configuration Search
    if args.skip_to_stage <= 4:
        best_config, filter_results = stage4_search_filters(
            args.dataset, best_lr, best_decay, best_eigen_combo, ranges, 
            args.epochs, args.patience, args.timeout
        )
        all_results['stages']['filters'] = {
            'best_config': best_config,
            'results': filter_results
        }
    
    total_time = time.time() - start_time
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Total search time: {total_time:.1f}s")
    print(f"Best learning rate: {best_lr}")
    print(f"Best weight decay: {best_decay}")
    print(f"Best eigenvalues: {best_eigen_combo}")
    if args.skip_to_stage <= 4:
        filter_order, use_user, use_item = best_config
        print(f"Best filter order: {filter_order}")
        print(f"Best gram config: user={use_user}, item={use_item}")
    
    # Find overall best result
    all_experiments = []
    for stage_results in all_results['stages'].values():
        if 'results' in stage_results:
            all_experiments.extend([r for r in stage_results['results'] if r['success']])
    
    if all_experiments:
        best_overall = max(all_experiments, key=lambda x: x['ndcg'])
        print(f"\nBest overall performance:")
        print(f"  NDCG@20: {best_overall['ndcg']:.6f}")
        print(f"  Recall@20: {best_overall['recall']:.6f}")
        print(f"  Precision@20: {best_overall['precision']:.6f}")
        print(f"  Configuration: lr={best_overall['lr']}, decay={best_overall['decay']}, " +
              f"u_eigen={best_overall['n_eigen_user']}, i_eigen={best_overall['n_eigen_item']}, " +
              f"filter={best_overall['filter_order']}")
        
        # Generate optimal command
        cmd_parts = [
            f"python main.py --dataset {args.dataset}",
            f"--lr {best_overall['lr']}",
            f"--decay {best_overall['decay']}",
            f"--n_eigen_user {best_overall['n_eigen_user']}",
            f"--n_eigen_item {best_overall['n_eigen_item']}",
            f"--filter_order {best_overall['filter_order']}",
            f"--epochs {args.epochs}"
        ]
        
        if not best_overall['use_user_gram']:
            cmd_parts.append("--no_user_gram")
        if not best_overall['use_item_gram']:
            cmd_parts.append("--no_item_gram")
            
        print(f"\nOptimal command:")
        print(" ".join(cmd_parts))
    
    # Save results
    all_results['total_time'] = total_time
    all_results['search_completed'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(args.save_results, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    main()