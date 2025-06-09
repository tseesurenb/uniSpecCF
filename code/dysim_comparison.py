#!/usr/bin/env python3
"""
DySimSpectralCF vs SimplifiedSpectralCF Comparison Study
Compare similarity-based spectral filtering with original PolyCF approach
"""

import subprocess
import sys
import time
import json
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="DySimSpectralCF vs SimplifiedSpectralCF comparison study")
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
    parser.add_argument('--save_results', type=str, default='dysim_comparison_results.json',
                       help='File to save results')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--n_eigen_user', type=int, default=15, help='User eigenvalues')
    parser.add_argument('--n_eigen_item', type=int, default=30, help='Item eigenvalues')
    parser.add_argument('--filter_order', type=int, default=6, help='Filter order')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def run_single_experiment(args, config, seed=None):
    """Run a single experiment with given configuration"""
    
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
    
    # Add configuration-specific parameters
    if config['type'] == 'original_polycf':
        # Original PolyCF with gamma values
        cmd.extend(["--model", "uspec"])
        cmd.extend(["--gamma_values"] + [str(g) for g in config.get('gamma_values', [0.0, 0.5, 1.0])])
    elif config['type'] == 'dysim_spectral':
        # DySimSpectralCF parameters
        cmd.extend([
            "--model", "dysim_spectral",
            "--k_u", str(config.get('k_u', 50)),
            "--k_i", str(config.get('k_i', 20)),
            "--similarity_type", config.get('similarity_type', 'cosine')
        ])
        if config.get('use_attention', True):
            cmd.append("--use_attention")
        else:
            cmd.append("--no_attention")
        
        # Graph construction options
        if not config.get('use_user_similarity', True):
            cmd.append("--no_user_similarity")
        if not config.get('use_item_similarity', True):
            cmd.append("--no_item_similarity")
        if not config.get('use_original_interactions', True):
            cmd.append("--no_original_interactions")
    
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
                'config': config
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


def run_multiple_runs(args, config):
    """Run multiple experiments with same configuration"""
    results = []
    
    print(f"\nTesting: {config['name']}")
    print(f"  Description: {config['description']}")
    
    for run in range(args.runs_per_config):
        seed = 42 + run
        result = run_single_experiment(args, config, seed)
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
    """Analyze and compare DySimSpectralCF vs SimplifiedSpectralCF"""
    print("\n" + "="*80)
    print("DYSIMSPECTRALCF vs SIMPLIFIEDSPECTRALCF COMPARISON")
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
                'config': result_data['config']
            })
    
    if not successful_configs:
        print("❌ No successful configurations to analyze")
        return
    
    # Sort by performance
    successful_configs.sort(key=lambda x: x['ndcg'], reverse=True)
    
    print(f"\n📊 PERFORMANCE RANKING:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Method':<40} {'NDCG@20':<12} {'Recall@20':<12} {'Precision@20':<12}")
    print("-" * 100)
    
    for i, config in enumerate(successful_configs):
        print(f"{i+1:<4} {config['name']:<40} "
              f"{config['ndcg']:.4f}±{config['ndcg_std']:.3f}  "
              f"{config['recall']:.4f}  {config['precision']:.4f}")
    
    # Separate analysis by method type
    dysim_configs = [c for c in successful_configs if c['config']['type'] == 'dysim_spectral']
    polycf_configs = [c for c in successful_configs if c['config']['type'] == 'original_polycf']
    
    print(f"\n🔬 METHODOLOGY COMPARISON:")
    
    if dysim_configs and polycf_configs:
        best_dysim = max(dysim_configs, key=lambda x: x['ndcg'])
        best_polycf = max(polycf_configs, key=lambda x: x['ndcg'])
        
        print(f"   🏆 Best DySimSpectralCF: {best_dysim['name']} (NDCG: {best_dysim['ndcg']:.4f})")
        print(f"   🥇 Best PolyCF: {best_polycf['name']} (NDCG: {best_polycf['ndcg']:.4f})")
        
        if best_dysim['ndcg'] > best_polycf['ndcg']:
            improvement = ((best_dysim['ndcg'] - best_polycf['ndcg']) / best_polycf['ndcg']) * 100
            print(f"   ✅ DySimSpectralCF improves over PolyCF by {improvement:.1f}%")
        else:
            decline = ((best_polycf['ndcg'] - best_dysim['ndcg']) / best_dysim['ndcg']) * 100
            print(f"   📝 PolyCF outperforms DySimSpectralCF by {decline:.1f}%")
    
    # Analyze DySimSpectralCF variations
    if len(dysim_configs) > 1:
        print(f"\n🎯 DYSIMSPECTRALCF VARIATIONS:")
        for config in dysim_configs:
            similarity_type = config['config'].get('similarity_type', 'cosine')
            k_u = config['config'].get('k_u', 50)
            k_i = config['config'].get('k_i', 20)
            attention = "with attention" if config['config'].get('use_attention', True) else "no attention"
            print(f"   • {config['name']}: {similarity_type}, k_u={k_u}, k_i={k_i}, {attention} → NDCG: {config['ndcg']:.4f}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Similarity-based approach vs. PolyCF normalization approach")
    print(f"   • Top-K neighbor selection vs. Multi-view adjacency construction")
    print(f"   • Attention mechanisms vs. Spectral normalization")
    print(f"   • User/item similarity graphs vs. Gram matrix eigenspaces")
    
    # Best configuration recommendation
    if successful_configs:
        best_overall = successful_configs[0]
        print(f"\n🏆 BEST OVERALL CONFIGURATION:")
        print(f"   Method: {best_overall['name']}")
        print(f"   Performance: NDCG@20={best_overall['ndcg']:.4f}, Recall@20={best_overall['recall']:.4f}")
        
        if best_overall['config']['type'] == 'dysim_spectral':
            config = best_overall['config']
            print(f"   Parameters: similarity={config.get('similarity_type', 'cosine')}, "
                  f"k_u={config.get('k_u', 50)}, k_i={config.get('k_i', 20)}")
        else:
            config = best_overall['config']
            print(f"   Parameters: gamma_values={config.get('gamma_values', [0.0, 0.5, 1.0])}")


def main():
    args = parse_args()
    
    print(f"🔬 DySimSpectralCF vs SimplifiedSpectralCF Comparison on {args.dataset.upper()}")
    print(f"Testing similarity-based vs PolyCF normalization approaches")
    print(f"Configuration: epochs={args.epochs}, runs={args.runs_per_config}")
    
    # Define configurations to test
    configurations = [
        # Original PolyCF approaches
        {
            'name': 'PolyCF Classic (0.0, 0.5, 1.0)',
            'type': 'original_polycf',
            'gamma_values': [0.0, 0.5, 1.0],
            'description': 'Original PolyCF with three normalization views'
        },
        {
            'name': 'PolyCF Symmetric Only (0.5)',
            'type': 'original_polycf',
            'gamma_values': [0.5],
            'description': 'Single symmetric normalization view'
        },
        {
            'name': 'PolyCF Fine-grained (0.25, 0.5, 0.75)',
            'type': 'original_polycf',
            'gamma_values': [0.25, 0.5, 0.75],
            'description': 'Fine-grained normalization around symmetric'
        },
        
        # DySimSpectralCF approaches
        {
            'name': 'DySimSpectral Cosine + Attention',
            'type': 'dysim_spectral',
            'k_u': 50,
            'k_i': 20,
            'similarity_type': 'cosine',
            'use_attention': True,
            'use_user_similarity': True,
            'use_item_similarity': True,
            'use_original_interactions': True,
            'description': 'Full DySimSpectralCF with cosine similarity and attention'
        },
        {
            'name': 'DySimSpectral Jaccard + Attention',
            'type': 'dysim_spectral',
            'k_u': 50,
            'k_i': 20,
            'similarity_type': 'jaccard',
            'use_attention': True,
            'use_user_similarity': True,
            'use_item_similarity': True,
            'use_original_interactions': True,
            'description': 'DySimSpectralCF with Jaccard similarity and attention'
        },
        {
            'name': 'DySimSpectral No Attention',
            'type': 'dysim_spectral',
            'k_u': 50,
            'k_i': 20,
            'similarity_type': 'cosine',
            'use_attention': False,
            'use_user_similarity': True,
            'use_item_similarity': True,
            'use_original_interactions': True,
            'description': 'DySimSpectralCF without symmetric softmax attention'
        },
        {
            'name': 'DySimSpectral Item-Only',
            'type': 'dysim_spectral',
            'k_u': 50,
            'k_i': 20,
            'similarity_type': 'cosine',
            'use_attention': True,
            'use_user_similarity': False,
            'use_item_similarity': True,
            'use_original_interactions': True,
            'description': 'Only item-item similarity matrices'
        },
        {
            'name': 'DySimSpectral User-Only',
            'type': 'dysim_spectral',
            'k_u': 50,
            'k_i': 20,
            'similarity_type': 'cosine',
            'use_attention': True,
            'use_user_similarity': True,
            'use_item_similarity': False,
            'use_original_interactions': True,
            'description': 'Only user-user similarity matrices'
        },
        {
            'name': 'DySimSpectral Pure Similarity',
            'type': 'dysim_spectral',
            'k_u': 80,
            'k_i': 40,
            'similarity_type': 'cosine',
            'use_attention': True,
            'use_user_similarity': True,
            'use_item_similarity': True,
            'use_original_interactions': False,
            'description': 'Pure similarity-based without original interactions'
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
    print("RUNNING COMPARISON EXPERIMENTS")
    print("="*80)
    
    # Test each configuration
    for config in configurations:
        stats, raw_results = run_multiple_runs(args, config)
        
        all_results['results'][config['name']] = {
            'config': config,
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