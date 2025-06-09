#!/usr/bin/env python3
"""
PolyCF Gamma Values Ablation Study for SimplifiedSpectralCF
Tests impact of different gamma values and their combinations
Following PolyCF approach: create multiple adjacency views first, then gram matrices
"""

import subprocess
import sys
import time
import json
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="PolyCF gamma values ablation study")
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
    parser.add_argument('--save_results', type=str, default='polycf_gamma_results.json',
                       help='File to save results')
    
    # Use best hyperparameters from hp_search
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--n_eigen_user', type=int, default=15, help='User eigenvalues')
    parser.add_argument('--n_eigen_item', type=int, default=30, help='Item eigenvalues')
    parser.add_argument('--filter_order', type=int, default=6, help='Filter order')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def run_single_experiment(args, gamma_config, seed=None):
    """Run a single experiment with given gamma configuration"""
    
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
    
    # Add gamma configuration
    if gamma_config['type'] == 'single':
        cmd.extend(["--single_gamma", str(gamma_config['gamma'])])
    elif gamma_config['type'] == 'combination':
        cmd.extend(["--gamma_values"] + [str(g) for g in gamma_config['gammas']])
    
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
                'gamma_config': gamma_config
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


def run_multiple_runs(args, gamma_config):
    """Run multiple experiments with same gamma config"""
    results = []
    
    print(f"\nTesting: {gamma_config['name']}")
    print(f"  Description: {gamma_config['description']}")
    
    for run in range(args.runs_per_config):
        seed = 42 + run
        result = run_single_experiment(args, gamma_config, seed)
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
    """Analyze and compare gamma value impact"""
    print("\n" + "="*80)
    print("POLYCF GAMMA VALUES ABLATION ANALYSIS")
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
                'config': result_data['gamma_config']
            })
    
    if not successful_configs:
        print("❌ No successful configurations to analyze")
        return
    
    # Sort by performance
    successful_configs.sort(key=lambda x: x['ndcg'], reverse=True)
    
    print(f"\n📊 PERFORMANCE RANKING:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Configuration':<35} {'NDCG@20':<12} {'Recall@20':<12} {'Precision@20':<12}")
    print("-" * 100)
    
    for i, config in enumerate(successful_configs):
        print(f"{i+1:<4} {config['name']:<35} "
              f"{config['ndcg']:.4f}±{config['ndcg_std']:.3f}  "
              f"{config['recall']:.4f}  {config['precision']:.4f}")
    
    # Find single gamma results for comparison
    single_gammas = [c for c in successful_configs if c['config'].get('type') == 'single']
    combination_gammas = [c for c in successful_configs if c['config'].get('type') == 'combination']
    
    if len(single_gammas) >= 2:
        best_single = single_gammas[0]
        worst_single = single_gammas[-1]
        improvement = ((best_single['ndcg'] - worst_single['ndcg']) / worst_single['ndcg']) * 100
        
        print(f"\n💡 SINGLE GAMMA INSIGHTS:")
        print(f"   🥇 Best single: {best_single['name']} (NDCG: {best_single['ndcg']:.4f})")
        print(f"   🥉 Worst single: {worst_single['name']} (NDCG: {worst_single['ndcg']:.4f})")
        print(f"   📈 Performance gap: {improvement:.1f}% improvement")
        
        # Identify the best gamma value
        if len(single_gammas) >= 3:
            gamma_ranking = []
            for config in single_gammas:
                gamma_value = config['config']['gamma']
                gamma_ranking.append((gamma_value, config['ndcg']))
            
            gamma_ranking.sort(key=lambda x: x[1], reverse=True)
            print(f"\n🏆 GAMMA VALUE RANKING:")
            for i, (gamma, ndcg) in enumerate(gamma_ranking):
                if gamma == 0.0:
                    name = "Raw/Column-norm (γ=0.0)"
                elif gamma == 0.5:
                    name = "Symmetric (γ=0.5)"
                elif gamma == 1.0:
                    name = "Row-norm (γ=1.0)"
                else:
                    name = f"γ={gamma:.1f}"
                print(f"   {i+1}. {name}: {ndcg:.4f}")
    
    # Compare single vs combination
    if single_gammas and combination_gammas:
        best_single = max(single_gammas, key=lambda x: x['ndcg'])
        best_combination = max(combination_gammas, key=lambda x: x['ndcg'])
        
        print(f"\n🔄 SINGLE vs COMBINATION:")
        print(f"   Best Single: {best_single['name']} (NDCG: {best_single['ndcg']:.4f})")
        print(f"   Best Combination: {best_combination['name']} (NDCG: {best_combination['ndcg']:.4f})")
        
        if best_combination['ndcg'] > best_single['ndcg']:
            improvement = ((best_combination['ndcg'] - best_single['ndcg']) / best_single['ndcg']) * 100
            print(f"   ✅ PolyCF multi-view improves performance by {improvement:.1f}%")
        else:
            decline = ((best_single['ndcg'] - best_combination['ndcg']) / best_combination['ndcg']) * 100
            print(f"   📝 Single gamma works {decline:.1f}% better than combinations")
    
    # Theoretical insights
    print(f"\n🔬 THEORETICAL INSIGHTS:")
    print(f"   • γ=0.0: Column normalization D_c^(-1) - emphasizes popular items")
    print(f"   • γ=0.5: Symmetric normalization D_r^(-0.5) D_c^(-0.5) - balanced view")
    print(f"   • γ=1.0: Row normalization D_r^(-1) - user-centric, equal weight per user")
    print(f"   • Combinations: Multiple views capture different collaborative patterns")


def main():
    args = parse_args()
    
    print(f"🔬 PolyCF Gamma Values Ablation Study on {args.dataset.upper()}")
    print(f"Testing impact of different gamma values following PolyCF approach")
    print(f"Configuration: epochs={args.epochs}, runs={args.runs_per_config}")
    
    # Define gamma configurations to test
    gamma_configs = [
        # Single gamma values (PolyCF style)
        {
            'name': 'Raw View Only (γ=0.0)',
            'type': 'single',
            'gamma': 0.0,
            'description': 'Column normalization D_c^(-1) - emphasizes popular items'
        },
        {
            'name': 'Symmetric Only (γ=0.5)',
            'type': 'single', 
            'gamma': 0.5,
            'description': 'Symmetric normalization D_r^(-0.5) D_c^(-0.5) - balanced view'
        },
        {
            'name': 'Row-norm Only (γ=1.0)',
            'type': 'single',
            'gamma': 1.0,
            'description': 'Row normalization D_r^(-1) - user-centric view'
        },
        
        # Additional single gamma values
        {
            'name': 'Light Symmetric (γ=0.25)',
            'type': 'single',
            'gamma': 0.25,
            'description': 'Between raw and symmetric normalization'
        },
        {
            'name': 'Heavy Symmetric (γ=0.75)',
            'type': 'single',
            'gamma': 0.75,
            'description': 'Between symmetric and row normalization'
        },
        
        # PolyCF combinations
        {
            'name': 'PolyCF Classic (0.0, 0.5, 1.0)',
            'type': 'combination',
            'gammas': [0.0, 0.5, 1.0],
            'description': 'Classic PolyCF: Raw + Symmetric + Row views'
        },
        {
            'name': 'Symmetric Focus (0.25, 0.5, 0.75)',
            'type': 'combination',
            'gammas': [0.25, 0.5, 0.75],
            'description': 'Fine-grained around symmetric normalization'
        },
        {
            'name': 'Raw + Symmetric (0.0, 0.5)',
            'type': 'combination',
            'gammas': [0.0, 0.5],
            'description': 'Popular items + balanced view combination'
        },
        {
            'name': 'Symmetric + Row (0.5, 1.0)',
            'type': 'combination',
            'gammas': [0.5, 1.0],
            'description': 'Balanced + user-centric view combination'
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
    print("RUNNING POLYCF GAMMA VALUES EXPERIMENTS")
    print("="*80)
    
    # Test each gamma configuration
    for gamma_config in gamma_configs:
        stats, raw_results = run_multiple_runs(args, gamma_config)
        
        all_results['results'][gamma_config['name']] = {
            'gamma_config': gamma_config,
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