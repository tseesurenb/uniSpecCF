#!/usr/bin/env python3
"""
Normalization Ablation Study for SimplifiedSpectralCF
Tests impact of: no normalization, symmetric degree normalization, softmax normalization
"""

import subprocess
import sys
import time
import json
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Normalization ablation study")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset: ml-100k, gowalla, yelp2018, amazon-book, lastfm')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs for each experiment')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout for each experiment in seconds')
    parser.add_argument('--runs_per_config', type=int, default=3,
                       help='Number of runs per configuration')
    parser.add_argument('--save_results', type=str, default='normalization_ablation_results.json',
                       help='File to save results')
    
    # Use best hyperparameters from hp_search
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--n_eigen_user', type=int, default=15, help='User eigenvalues')
    parser.add_argument('--n_eigen_item', type=int, default=30, help='Item eigenvalues')
    parser.add_argument('--filter_order', type=int, default=6, help='Filter order')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def run_single_experiment(args, norm_config, seed=None):
    """Run a single experiment with given normalization configuration"""
    
    cmd = [
        sys.executable, "main.py",
        "--dataset", args.dataset,
        "--lr", str(args.lr),
        "--decay", str(args.decay),
        "--n_eigen_user", str(args.n_eigen_user),
        "--n_eigen_item", str(args.n_eigen_item),
        "--filter_order", str(args.filter_order),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--n_epoch_eval", "5"
    ]
    
    # Add normalization configuration
    if norm_config['type'] == 'single':
        cmd.extend(["--single_norm", norm_config['norm_type']])
    elif norm_config['type'] == 'combination':
        cmd.extend(["--norm_types"] + norm_config['norm_types'])
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    if args.verbose:
        print(f"    Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        
        if args.verbose and result.returncode != 0:
            print(f"    STDERR: {result.stderr[:300]}")
            print(f"    STDOUT: {result.stdout[:300]}")
        
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
                'norm_config': norm_config
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


def run_multiple_runs(args, norm_config):
    """Run multiple experiments with same normalization config"""
    results = []
    
    print(f"\nTesting: {norm_config['name']}")
    print(f"  Description: {norm_config['description']}")
    
    for run in range(args.runs_per_config):
        seed = 42 + run
        result = run_single_experiment(args, norm_config, seed)
        results.append(result)
        
        if result['success']:
            print(f"    Run {run+1}/{args.runs_per_config}: "
                  f"NDCG={result['ndcg']:.4f}, Recall={result['recall']:.4f}")
        else:
            print(f"    Run {run+1}/{args.runs_per_config}: FAILED - {result['error']}")
    
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
        
        print(f"    Summary: NDCG={stats['ndcg_mean']:.4f}±{stats['ndcg_std']:.4f}, "
              f"Recall={stats['recall_mean']:.4f}±{stats['recall_std']:.4f}")
        
        return stats, results
    else:
        return None, results


def analyze_results(all_results):
    """Analyze and compare normalization impact"""
    print("\n" + "="*80)
    print("NORMALIZATION ABLATION ANALYSIS")
    print("="*80)
    
    # Get successful configurations
    successful_configs = []
    for config_name, result_data in all_results['results'].items():
        if result_data['stats'] is not None:
            successful_configs.append({
                'name': config_name,
                'ndcg': result_data['stats']['ndcg_mean'],
                'ndcg_std': result_data['stats']['ndcg_std'],
                'recall': result_data['stats']['recall_mean'],
                'precision': result_data['stats']['precision_mean'],
                'config': result_data['norm_config']
            })
    
    if not successful_configs:
        print("❌ No successful configurations to analyze")
        return
    
    # Sort by performance
    successful_configs.sort(key=lambda x: x['ndcg'], reverse=True)
    
    print(f"\n📊 PERFORMANCE RANKING:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Normalization':<25} {'NDCG@20':<12} {'Recall@20':<12} {'Precision@20':<12}")
    print("-" * 80)
    
    for i, config in enumerate(successful_configs):
        print(f"{i+1:<4} {config['name']:<25} "
              f"{config['ndcg']:.4f}±{config['ndcg_std']:.3f}  "
              f"{config['recall']:.4f}  {config['precision']:.4f}")
    
    # Find single normalization results for comparison
    single_norms = [c for c in successful_configs if c['config'].get('type') == 'single']
    combination_norms = [c for c in successful_configs if c['config'].get('type') == 'combination']
    
    if len(single_norms) >= 2:
        best_single = single_norms[0]
        worst_single = single_norms[-1]
        improvement = ((best_single['ndcg'] - worst_single['ndcg']) / worst_single['ndcg']) * 100
        
        print(f"\n💡 SINGLE NORMALIZATION INSIGHTS:")
        print(f"   🥇 Best single: {best_single['name']} (NDCG: {best_single['ndcg']:.4f})")
        print(f"   🥉 Worst single: {worst_single['name']} (NDCG: {worst_single['ndcg']:.4f})")
        print(f"   📈 Performance gap: {improvement:.1f}% improvement")
        
        # Identify the best normalization type
        if len(single_norms) == 3:
            norm_ranking = []
            for config in single_norms:
                norm_type = config['config']['norm_type']
                norm_ranking.append((norm_type, config['ndcg']))
            
            norm_ranking.sort(key=lambda x: x[1], reverse=True)
            print(f"\n🏆 NORMALIZATION TYPE RANKING:")
            for i, (norm_type, ndcg) in enumerate(norm_ranking):
                if norm_type == 'none':
                    name = "No Normalization"
                elif norm_type == 'symmetric':
                    name = "Symmetric Degree"
                elif norm_type == 'softmax':
                    name = "Softmax"
                else:
                    name = norm_type
                print(f"   {i+1}. {name}: {ndcg:.4f}")
    
    # Compare single vs combination
    if single_norms and combination_norms:
        best_single = max(single_norms, key=lambda x: x['ndcg'])
        best_combination = max(combination_norms, key=lambda x: x['ndcg'])
        
        print(f"\n🔄 SINGLE vs COMBINATION:")
        print(f"   Best Single: {best_single['name']} (NDCG: {best_single['ndcg']:.4f})")
        print(f"   Best Combination: {best_combination['name']} (NDCG: {best_combination['ndcg']:.4f})")
        
        if best_combination['ndcg'] > best_single['ndcg']:
            improvement = ((best_combination['ndcg'] - best_single['ndcg']) / best_single['ndcg']) * 100
            print(f"   ✅ Combination improves performance by {improvement:.1f}%")
        else:
            decline = ((best_single['ndcg'] - best_combination['ndcg']) / best_combination['ndcg']) * 100
            print(f"   📝 Single normalization works {decline:.1f}% better than combination")


def main():
    args = parse_args()
    
    print(f"🔬 Normalization Ablation Study on {args.dataset.upper()}")
    print(f"Testing impact of different normalization strategies")
    print(f"Configuration: epochs={args.epochs}, runs={args.runs_per_config}")
    
    # Define normalization configurations to test
    normalization_configs = [
        # Single normalization types
        {
            'name': 'No Normalization Only',
            'type': 'single',
            'norm_type': 'none',
            'description': 'Only raw adjacency matrices (no normalization)'
        },
        {
            'name': 'Symmetric Degree Only',
            'type': 'single', 
            'norm_type': 'symmetric',
            'description': 'Only D^(-1/2) A D^(-1/2) symmetric degree normalization'
        },
        {
            'name': 'Softmax Only',
            'type': 'single',
            'norm_type': 'softmax',
            'description': 'Only softmax normalization for numerical stability'
        },
        
        # Combinations for comparison
        {
            'name': 'No Norm + Softmax',
            'type': 'combination',
            'norm_types': ['none', 'softmax'],
            'description': 'Learned combination of no normalization and softmax (top 2 performers)'
        },
        {
            'name': 'All Three Combined',
            'type': 'combination',
            'norm_types': ['none', 'symmetric', 'softmax'],
            'description': 'Learned combination of all three normalization types (default)'
        },
        {
            'name': 'Symmetric + Softmax',
            'type': 'combination',
            'norm_types': ['symmetric', 'softmax'],
            'description': 'Learned combination of symmetric and softmax normalization'
        }
    ]
    
    start_time = time.time()
    all_results = {
        'metadata': {
            'dataset': args.dataset,
            'baseline_config': {
                'lr': args.lr,
                'decay': args.decay,
                'n_eigen_user': args.n_eigen_user,
                'n_eigen_item': args.n_eigen_item,
                'filter_order': args.filter_order
            },
            'runs_per_config': args.runs_per_config,
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'results': {}
    }
    
    print("\n" + "="*80)
    print("RUNNING NORMALIZATION ABLATION EXPERIMENTS")
    print("="*80)
    
    # Test each normalization configuration
    for norm_config in normalization_configs:
        stats, raw_results = run_multiple_runs(args, norm_config)
        
        all_results['results'][norm_config['name']] = {
            'norm_config': norm_config,
            'stats': stats,
            'raw_results': raw_results
        }
    
    total_time = time.time() - start_time
    all_results['metadata']['total_time'] = total_time
    all_results['metadata']['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Analyze results
    analyze_results(all_results)
    
    print(f"\nTotal study time: {total_time:.1f}s")
    
    # Save results
    with open(args.save_results, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()