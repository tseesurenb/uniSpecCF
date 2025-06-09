#!/usr/bin/env python3
"""
Ablation Study for SimplifiedSpectralCF
Systematically evaluates the contribution of different model components
"""

import subprocess
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
from itertools import product


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study for SimplifiedSpectralCF")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset: ml-100k, gowalla, yelp2018, amazon-book, lastfm')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for each experiment')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout for each experiment in seconds')
    parser.add_argument('--save_results', type=str, default='ablation_results.json',
                       help='File to save results')
    parser.add_argument('--runs_per_config', type=int, default=3,
                       help='Number of runs per configuration for statistical significance')
    parser.add_argument('--baseline_config', type=str, 
                       help='JSON file with baseline hyperparameters (from hp_search)')
    parser.add_argument('--studies', type=str, nargs='+', 
                       default=['gram_matrices', 'eigenvalues', 'filter_orders', 'spectral_vs_mf', 'normalization'],
                       help='Which ablation studies to run')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed error messages and commands')
    parser.add_argument('--test_baseline', action='store_true',
                       help='Test baseline configuration first to check setup')
    return parser.parse_args()


def load_baseline_config(config_file):
    """Load baseline hyperparameters from hp_search results"""
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            hp_results = json.load(f)
        
        # Extract best configuration from hp search results
        all_experiments = []
        for stage_results in hp_results['stages'].values():
            if 'results' in stage_results:
                all_experiments.extend([r for r in stage_results['results'] if r['success']])
        
        if all_experiments:
            best = max(all_experiments, key=lambda x: x['ndcg'])
            return {
                'lr': best['lr'],
                'decay': best['decay'],
                'n_eigen_user': best['n_eigen_user'],
                'n_eigen_item': best['n_eigen_item'],
                'filter_order': best['filter_order'],
                'use_user_gram': best.get('use_user_gram', True),
                'use_item_gram': best.get('use_item_gram', True),
                'user_norm': best.get('user_norm', 'symmetric'),
                'item_norm': best.get('item_norm', 'symmetric')
            }
    
    # Default configuration if no baseline file
    return {
        'lr': 0.1,
        'decay': 1e-2,
        'n_eigen_user': 25,
        'n_eigen_item': 40,
        'filter_order': 6,
        'use_user_gram': True,
        'use_item_gram': True,
        'user_norm': 'symmetric',
        'item_norm': 'symmetric'
    }


def run_single_experiment(dataset, config, epochs=50, patience=5, timeout=300, seed=None, verbose=False):
    """Run a single experiment with given configuration"""
    
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset,
        "--lr", str(config['lr']),
        "--decay", str(config['decay']),
        "--n_eigen_user", str(config['n_eigen_user']),
        "--n_eigen_item", str(config['n_eigen_item']),
        "--filter_order", str(config['filter_order']),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--n_epoch_eval", "5"
    ]
    
    if not config.get('use_user_gram', True):
        cmd.append("--no_user_gram")
    if not config.get('use_item_gram', True):
        cmd.append("--no_item_gram")
    
    # Add normalization parameters if specified and supported
    if 'user_norm' in config and config['user_norm'] != 'symmetric':
        cmd.extend(["--user_norm", config['user_norm']])
    if 'item_norm' in config and config['item_norm'] != 'symmetric':
        cmd.extend(["--item_norm", config['item_norm']])
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    if verbose:
        print(f"    Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if verbose and result.returncode != 0:
            print(f"    STDERR: {result.stderr[:500]}")
            print(f"    STDOUT: {result.stdout[:500]}")
        
        # Parse results
        lines = result.stdout.split('\n')
        final_line = None
        for line in lines:
            if "Final Results:" in line:
                final_line = line
                break
        
        if final_line and result.returncode == 0:
            parts = final_line.split(", ")
            recall = float(parts[0].split("=")[1])
            precision = float(parts[1].split("=")[1])
            ndcg = float(parts[2].split("=")[1])
            
            return {
                'success': True,
                'recall': recall,
                'precision': precision,
                'ndcg': ndcg,
                'config': config.copy()
            }
        else:
            error_msg = f'Non-zero exit ({result.returncode})'
            if result.stderr:
                error_msg += f': {result.stderr[:200]}'
            return {'success': False, 'error': error_msg}
            
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def run_multiple_runs(dataset, config, runs=3, epochs=50, patience=5, timeout=300, verbose=False):
    """Run multiple experiments with same config for statistical significance"""
    results = []
    
    for run in range(runs):
        seed = 42 + run  # Different seed for each run
        result = run_single_experiment(dataset, config, epochs, patience, timeout, seed, verbose)
        results.append(result)
        
        if result['success']:
            print(f"    Run {run+1}/{runs}: NDCG={result['ndcg']:.4f}, Recall={result['recall']:.4f}")
        else:
            print(f"    Run {run+1}/{runs}: FAILED - {result['error']}")
            if verbose and run == 0:  # Show details for first failure
                print(f"      Config: {config}")
    
    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    if successful_results:
        ndcgs = [r['ndcg'] for r in successful_results]
        recalls = [r['recall'] for r in successful_results]
        precisions = [r['precision'] for r in successful_results]
        
        stats = {
            'ndcg_mean': np.mean(ndcgs),
            'ndcg_std': np.std(ndcgs),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls),
            'precision_mean': np.mean(precisions),
            'precision_std': np.std(precisions),
            'success_rate': len(successful_results) / len(results),
            'num_runs': len(results)
        }
        
        print(f"    Summary: NDCG={stats['ndcg_mean']:.4f}±{stats['ndcg_std']:.4f}, " +
              f"Recall={stats['recall_mean']:.4f}±{stats['recall_std']:.4f}")
        
        return stats, results
    else:
        return None, results


def ablation_gram_matrices(dataset, baseline_config, runs, epochs, patience, timeout):
    """Study 1: Ablation of Gram matrices"""
    print("\n" + "="*80)
    print("ABLATION STUDY 1: GRAM MATRICES")
    print("="*80)
    
    configurations = [
        {'name': 'Both Gram Matrices', 'use_user_gram': True, 'use_item_gram': True},
        {'name': 'User Gram Only', 'use_user_gram': True, 'use_item_gram': False},
        {'name': 'Item Gram Only', 'use_user_gram': False, 'use_item_gram': True},
        {'name': 'No Gram Matrices', 'use_user_gram': False, 'use_item_gram': False},
    ]
    
    results = {}
    
    for config_info in configurations:
        print(f"\nTesting: {config_info['name']}")
        
        test_config = baseline_config.copy()
        test_config.update({k: v for k, v in config_info.items() if k != 'name'})
        
        stats, raw_results = run_multiple_runs(
            dataset, test_config, runs, epochs, patience, timeout, verbose=True
        )
        
        results[config_info['name']] = {
            'config': test_config,
            'stats': stats,
            'raw_results': raw_results
        }
    
    return results


def ablation_eigenvalues(dataset, baseline_config, runs, epochs, patience, timeout):
    """Study 2: Ablation of eigenvalue dimensions"""
    print("\n" + "="*80)
    print("ABLATION STUDY 2: EIGENVALUE DIMENSIONS")
    print("="*80)
    
    base_user = baseline_config['n_eigen_user']
    base_item = baseline_config['n_eigen_item']
    
    configurations = [
        {'name': 'Baseline', 'n_eigen_user': base_user, 'n_eigen_item': base_item},
        {'name': 'Half User Eigen', 'n_eigen_user': base_user // 2, 'n_eigen_item': base_item},
        {'name': 'Half Item Eigen', 'n_eigen_user': base_user, 'n_eigen_item': base_item // 2},
        {'name': 'Double User Eigen', 'n_eigen_user': base_user * 2, 'n_eigen_item': base_item},
        {'name': 'Double Item Eigen', 'n_eigen_user': base_user, 'n_eigen_item': base_item * 2},
        {'name': 'Minimal Eigen (10,15)', 'n_eigen_user': 10, 'n_eigen_item': 15},
        {'name': 'Large Eigen', 'n_eigen_user': min(base_user * 3, 100), 'n_eigen_item': min(base_item * 3, 150)},
    ]
    
    results = {}
    
    for config_info in configurations:
        print(f"\nTesting: {config_info['name']} " +
              f"(u_eigen={config_info['n_eigen_user']}, i_eigen={config_info['n_eigen_item']})")
        
        test_config = baseline_config.copy()
        test_config.update({k: v for k, v in config_info.items() if k != 'name'})
        
        stats, raw_results = run_multiple_runs(
            dataset, test_config, runs, epochs, patience, timeout
        )
        
        results[config_info['name']] = {
            'config': test_config,
            'stats': stats,
            'raw_results': raw_results
        }
    
    return results


def ablation_filter_orders(dataset, baseline_config, runs, epochs, patience, timeout):
    """Study 3: Ablation of filter orders"""
    print("\n" + "="*80)
    print("ABLATION STUDY 3: FILTER ORDERS")
    print("="*80)
    
    filter_orders = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    results = {}
    
    for filter_order in filter_orders:
        print(f"\nTesting: Filter Order = {filter_order}")
        
        test_config = baseline_config.copy()
        test_config['filter_order'] = filter_order
        
        stats, raw_results = run_multiple_runs(
            dataset, test_config, runs, epochs, patience, timeout
        )
        
        results[f'Filter Order {filter_order}'] = {
            'config': test_config,
            'stats': stats,
            'raw_results': raw_results
        }
    
    return results


def ablation_normalization(dataset, baseline_config, runs, epochs, patience, timeout):
    """Study 5: Ablation of normalization strategies"""
    print("\n" + "="*80)
    print("ABLATION STUDY 5: NORMALIZATION STRATEGIES")
    print("="*80)
    
    # Define normalization types and their mathematical descriptions
    norm_descriptions = {
        'none': 'Raw adjacency matrix A',
        'symmetric': 'D^(-1/2) A D^(-1/2) (symmetric normalization)',
        'left': 'D^(-1) A (left/row normalization)', 
        'right': 'A D^(-1) (right/column normalization)',
        'row': 'Row-wise L1 normalization',
        'col': 'Column-wise L1 normalization'
    }
    
    configurations = [
        # === STANDARD COMBINATIONS ===
        {'name': 'Symmetric (Both)', 'user_norm': 'symmetric', 'item_norm': 'symmetric', 
         'category': 'Standard'},
        {'name': 'Left (Both)', 'user_norm': 'left', 'item_norm': 'left',
         'category': 'Standard'},
        {'name': 'Right (Both)', 'user_norm': 'right', 'item_norm': 'right',
         'category': 'Standard'},
        {'name': 'Row (Both)', 'user_norm': 'row', 'item_norm': 'row',
         'category': 'Standard'},
        {'name': 'Col (Both)', 'user_norm': 'col', 'item_norm': 'col',
         'category': 'Standard'},
        {'name': 'No Normalization', 'user_norm': 'none', 'item_norm': 'none',
         'category': 'Standard'},
        
        # === ASYMMETRIC COMBINATIONS ===
        {'name': 'Sym User + Left Item', 'user_norm': 'symmetric', 'item_norm': 'left',
         'category': 'Asymmetric'},
        {'name': 'Left User + Sym Item', 'user_norm': 'left', 'item_norm': 'symmetric',
         'category': 'Asymmetric'},
        {'name': 'Sym User + Right Item', 'user_norm': 'symmetric', 'item_norm': 'right',
         'category': 'Asymmetric'},
        {'name': 'Right User + Sym Item', 'user_norm': 'right', 'item_norm': 'symmetric',
         'category': 'Asymmetric'},
        {'name': 'Row User + Col Item', 'user_norm': 'row', 'item_norm': 'col',
         'category': 'Asymmetric'},
        {'name': 'Col User + Row Item', 'user_norm': 'col', 'item_norm': 'row',
         'category': 'Asymmetric'},
        
        # === MIXED WITH NO NORMALIZATION ===
        {'name': 'Sym User + None Item', 'user_norm': 'symmetric', 'item_norm': 'none',
         'category': 'Mixed'},
        {'name': 'None User + Sym Item', 'user_norm': 'none', 'item_norm': 'symmetric',
         'category': 'Mixed'},
        {'name': 'Left User + None Item', 'user_norm': 'left', 'item_norm': 'none',
         'category': 'Mixed'},
        {'name': 'None User + Left Item', 'user_norm': 'none', 'item_norm': 'left',
         'category': 'Mixed'},
        
        # === EXTREME CASES ===
        {'name': 'Row User + None Item', 'user_norm': 'row', 'item_norm': 'none',
         'category': 'Extreme'},
        {'name': 'None User + Col Item', 'user_norm': 'none', 'item_norm': 'col',
         'category': 'Extreme'},
    ]
    
    results = {}
    category_results = {}
    
    for config_info in configurations:
        category = config_info['category']
        if category not in category_results:
            category_results[category] = []
            
        print(f"\n[{category}] Testing: {config_info['name']}")
        print(f"  User: {norm_descriptions[config_info['user_norm']]}")
        print(f"  Item: {norm_descriptions[config_info['item_norm']]}")
        
        test_config = baseline_config.copy()
        test_config.update({k: v for k, v in config_info.items() 
                           if k not in ['name', 'category']})
        
        stats, raw_results = run_multiple_runs(
            dataset, test_config, runs, epochs, patience, timeout
        )
        
        result_entry = {
            'config': test_config,
            'stats': stats,
            'raw_results': raw_results,
            'category': category,
            'user_norm_desc': norm_descriptions[config_info['user_norm']],
            'item_norm_desc': norm_descriptions[config_info['item_norm']]
        }
        
        results[config_info['name']] = result_entry
        category_results[category].append((config_info['name'], result_entry))
    
    # Print category-wise analysis
    print(f"\n" + "="*60)
    print("NORMALIZATION ANALYSIS BY CATEGORY")
    print("="*60)
    
    for category, cat_results in category_results.items():
        print(f"\n{category.upper()} NORMALIZATION:")
        print("-" * 40)
        
        valid_cat_results = [(name, data) for name, data in cat_results 
                            if data['stats'] is not None]
        
        if valid_cat_results:
            # Sort by NDCG within category
            valid_cat_results.sort(key=lambda x: x[1]['stats']['ndcg_mean'], reverse=True)
            
            print(f"{'Configuration':<25} {'NDCG@20':<12}")
            print("-" * 40)
            
            for name, data in valid_cat_results:
                stats = data['stats']
                print(f"{name:<25} {stats['ndcg_mean']:.4f}±{stats['ndcg_std']:.3f}")
                
            # Highlight best in category
            best_name, best_data = valid_cat_results[0]
            print(f"  → Best {category}: {best_name} (NDCG: {best_data['stats']['ndcg_mean']:.4f})")
        else:
            print("  No successful results in this category.")
    
    return results


def ablation_spectral_vs_mf(dataset, baseline_config, runs, epochs, patience, timeout):
    """Study 4: Spectral Components vs Matrix Factorization baseline"""
    print("\n" + "="*80)
    print("ABLATION STUDY 4: SPECTRAL COMPONENTS vs MATRIX FACTORIZATION")
    print("="*80)
    
    configurations = [
        {
            'name': 'Full SpectralCF',
            'use_spectral': True,
            'description': 'Complete spectral filtering with gram matrices'
        },
        {
            'name': 'No Filter (eigen only)',
            'filter_order': 0,
            'description': 'Only eigendecomposition, no polynomial filtering'
        },
        {
            'name': 'Linear Filter',
            'filter_order': 1,
            'description': 'Linear spectral filtering (K=1)'
        },
        {
            'name': 'No User Spectral',
            'n_eigen_user': 0,
            'description': 'Remove user spectral component entirely'
        },
        {
            'name': 'No Item Spectral',
            'n_eigen_item': 0,
            'description': 'Remove item spectral component entirely'
        },
        {
            'name': 'Minimal Spectral',
            'n_eigen_user': 5,
            'n_eigen_item': 5,
            'filter_order': 1,
            'description': 'Minimal spectral components'
        }
    ]
    
    results = {}
    
    for config_info in configurations:
        print(f"\nTesting: {config_info['name']}")
        print(f"  Description: {config_info['description']}")
        
        test_config = baseline_config.copy()
        test_config.update({k: v for k, v in config_info.items() 
                           if k not in ['name', 'description', 'use_spectral']})
        
        stats, raw_results = run_multiple_runs(
            dataset, test_config, runs, epochs, patience, timeout
        )
        
        results[config_info['name']] = {
            'config': test_config,
            'stats': stats,
            'raw_results': raw_results,
            'description': config_info['description']
        }
    
    return results


def print_study_summary(study_name, results):
    """Print summary of a single ablation study"""
    print(f"\n{study_name} - SUMMARY:")
    print("-" * 60)
    
    # Sort by NDCG performance
    valid_results = [(name, data) for name, data in results.items() 
                     if data['stats'] is not None]
    
    if not valid_results:
        print("No successful results in this study.")
        return
    
    valid_results.sort(key=lambda x: x[1]['stats']['ndcg_mean'], reverse=True)
    
    print(f"{'Configuration':<25} {'NDCG@20':<12} {'Recall@20':<12} {'Precision@20':<12}")
    print("-" * 65)
    
    for name, data in valid_results:
        stats = data['stats']
        print(f"{name:<25} "
              f"{stats['ndcg_mean']:.4f}±{stats['ndcg_std']:.3f}  "
              f"{stats['recall_mean']:.4f}±{stats['recall_std']:.3f}   "
              f"{stats['precision_mean']:.4f}±{stats['precision_std']:.3f}")


def print_final_summary(all_results):
    """Print comprehensive summary across all studies"""
    print("\n" + "="*100)
    print("COMPREHENSIVE ABLATION STUDY SUMMARY")
    print("="*100)
    
    # Collect all configurations and their best NDCG
    all_configs = []
    
    for study_name, study_results in all_results.items():
        if study_name == 'metadata':
            continue
            
        for config_name, config_data in study_results.items():
            if config_data['stats'] is not None:
                all_configs.append({
                    'study': study_name,
                    'config': config_name,
                    'ndcg': config_data['stats']['ndcg_mean'],
                    'ndcg_std': config_data['stats']['ndcg_std'],
                    'recall': config_data['stats']['recall_mean'],
                    'precision': config_data['stats']['precision_mean'],
                    'full_config': config_data['config']
                })
    
    # Sort by performance
    all_configs.sort(key=lambda x: x['ndcg'], reverse=True)
    
    print(f"\nTOP 10 CONFIGURATIONS ACROSS ALL STUDIES:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Study':<20} {'Configuration':<25} {'NDCG@20':<12} {'Recall@20':<12}")
    print("-" * 100)
    
    for i, config in enumerate(all_configs[:10]):
        print(f"{i+1:<4} {config['study']:<20} {config['config']:<25} "
              f"{config['ndcg']:.4f}±{config['ndcg_std']:.3f}  "
              f"{config['recall']:.4f}")
    
    if all_configs:
        best_config = all_configs[0]
        print(f"\n🏆 BEST OVERALL CONFIGURATION:")
        print(f"   Study: {best_config['study']}")
        print(f"   Configuration: {best_config['config']}")
        print(f"   NDCG@20: {best_config['ndcg']:.6f} ± {best_config['ndcg_std']:.6f}")
        print(f"   Recall@20: {best_config['recall']:.6f}")
        print(f"   Precision@20: {best_config['precision']:.6f}")
        
        # Print the exact hyperparameters
        print(f"\n   Hyperparameters:")
        for key, value in best_config['full_config'].items():
            print(f"     {key}: {value}")


def main():
    args = parse_args()
    
    print(f"🔬 Ablation Study for SimplifiedSpectralCF on {args.dataset.upper()}")
    print(f"Configuration: epochs={args.epochs}, patience={args.patience}, runs_per_config={args.runs_per_config}")
    
    # Load baseline configuration
    baseline_config = load_baseline_config(args.baseline_config)
    print(f"\nBaseline Configuration:")
    for key, value in baseline_config.items():
        print(f"  {key}: {value}")
    
    start_time = time.time()
    
    # Test baseline configuration first if requested
    if args.test_baseline:
        print("\n" + "="*80)
        print("TESTING BASELINE CONFIGURATION")
        print("="*80)
        
        test_config = baseline_config.copy()
        # Remove normalization args if they might not be supported
        if not args.verbose:
            test_config.pop('user_norm', None)
            test_config.pop('item_norm', None)
        
        print("Testing baseline configuration...")
        result = run_single_experiment(
            args.dataset, test_config, args.epochs, args.patience, args.timeout, 42, args.verbose
        )
        
        if result['success']:
            print(f"✅ Baseline test successful: NDCG={result['ndcg']:.4f}")
        else:
            print(f"❌ Baseline test failed: {result['error']}")
            print("Please fix the baseline configuration before running ablation studies.")
            return
    
    all_results = {
        'metadata': {
            'dataset': args.dataset,
            'baseline_config': baseline_config,
            'runs_per_config': args.runs_per_config,
            'studies_conducted': args.studies,
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Run selected studies
    if 'gram_matrices' in args.studies:
        gram_results = ablation_gram_matrices(
            args.dataset, baseline_config, args.runs_per_config, 
            args.epochs, args.patience, args.timeout
        )
        all_results['Gram Matrices'] = gram_results
        print_study_summary("GRAM MATRICES", gram_results)
    
    if 'eigenvalues' in args.studies:
        eigen_results = ablation_eigenvalues(
            args.dataset, baseline_config, args.runs_per_config,
            args.epochs, args.patience, args.timeout
        )
        all_results['Eigenvalue Dimensions'] = eigen_results
        print_study_summary("EIGENVALUE DIMENSIONS", eigen_results)
    
    if 'filter_orders' in args.studies:
        filter_results = ablation_filter_orders(
            args.dataset, baseline_config, args.runs_per_config,
            args.epochs, args.patience, args.timeout
        )
        all_results['Filter Orders'] = filter_results
        print_study_summary("FILTER ORDERS", filter_results)
    
    if 'normalization' in args.studies:
        norm_results = ablation_normalization(
            args.dataset, baseline_config, args.runs_per_config,
            args.epochs, args.patience, args.timeout
        )
        all_results['Normalization'] = norm_results
        print_study_summary("NORMALIZATION STRATEGIES", norm_results)
    
    if 'spectral_vs_mf' in args.studies:
        spectral_results = ablation_spectral_vs_mf(
            args.dataset, baseline_config, args.runs_per_config,
            args.epochs, args.patience, args.timeout
        )
        all_results['Spectral vs MF'] = spectral_results
        print_study_summary("SPECTRAL vs MATRIX FACTORIZATION", spectral_results)
    
    total_time = time.time() - start_time
    all_results['metadata']['total_time'] = total_time
    all_results['metadata']['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Print comprehensive summary
    print_final_summary(all_results)
    
    print(f"\nTotal ablation study time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Save results
    with open(args.save_results, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()