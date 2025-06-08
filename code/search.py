#!/usr/bin/env python3
"""
Flexible Experiment Runner for Multi-hop Analysis
Run comprehensive experiments on any dataset
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive multi-hop experiments")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset to run experiments on: ml-100k, gowalla, yelp2018, amazon-book, lastfm')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick experiments (fewer configurations)')
    parser.add_argument('--no_svd', action='store_true',
                       help='Skip SVD experiments (faster)')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='Maximum number of experiments to run')
    return parser.parse_args()

def get_dataset_configs(dataset):
    """Get dataset-specific optimal configurations"""
    configs = {
        'ml-100k': {
            'eigenvals': [25, 35, 65],
            'best_eigenval': 65,
            'svd_ranks': [128, 256, 512],
            'best_svd_rank': 256
        },
        'gowalla': {
            'eigenvals': [64, 128, 256],
            'best_eigenval': 128,
            'svd_ranks': [256, 512, 1024],
            'best_svd_rank': 512
        },
        'yelp2018': {
            'eigenvals': [128, 256, 512],
            'best_eigenval': 256,
            'svd_ranks': [512, 1024, 2048],
            'best_svd_rank': 1024
        },
        'amazon-book': {
            'eigenvals': [256, 512, 1024],
            'best_eigenval': 512,
            'svd_ranks': [1024, 2048, 4096],
            'best_svd_rank': 2048
        },
        'lastfm': {
            'eigenvals': [32, 64, 128],
            'best_eigenval': 64,
            'svd_ranks': [128, 256, 512],
            'best_svd_rank': 256
        }
    }
    
    # Default config for unknown datasets
    default_config = {
        'eigenvals': [64, 128, 256],
        'best_eigenval': 128,
        'svd_ranks': [256, 512, 1024],
        'best_svd_rank': 512
    }
    
    return configs.get(dataset, default_config)

def run_experiment(dataset, n_eigen, n_hops, combine_hops=False, additional_args=""):
    """Run a single experiment and capture results"""
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset,
        "--n_eigen", str(n_eigen),
        "--n_hops", str(n_hops)
    ]
    
    if combine_hops:
        cmd.append("--combine_hops")
    
    if additional_args:
        cmd.extend(additional_args.split())
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # Longer timeout for larger datasets
        
        # Extract key metrics from output
        lines = result.stdout.split('\n')
        final_line = None
        for line in lines:
            if "Final Results:" in line:
                final_line = line
                break
        
        if final_line:
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
                'output': result.stdout
            }
        else:
            return {'success': False, 'error': 'Could not parse results', 'output': result.stdout}
            
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def generate_experiments(dataset, quick=False, no_svd=False):
    """Generate experiment configurations based on dataset and options"""
    
    config = get_dataset_configs(dataset)
    eigenvals = config['eigenvals']
    best_eigenval = config['best_eigenval']
    svd_ranks = config['svd_ranks']
    best_svd_rank = config['best_svd_rank']
    
    experiments = []
    
    # === BASELINE MULTI-HOP EXPERIMENTS ===
    experiments.extend([
        (dataset, best_eigenval, 1, False, "1-hop baseline", ""),
        (dataset, best_eigenval, 2, False, "2-hop baseline", ""),
        (dataset, best_eigenval, 3, False, "3-hop baseline", ""),
        (dataset, best_eigenval, 4, False, "4-hop baseline", ""),
    ])
    
    # === MULTI-HOP COMBINATION EXPERIMENTS ===
    experiments.extend([
        (dataset, best_eigenval, 2, True, "2-hop combined", ""),
        (dataset, best_eigenval, 3, True, "3-hop combined", ""),
        (dataset, best_eigenval, 4, True, "4-hop combined", ""),
    ])
    
    if not quick:
        # === EIGENVALUE TUNING ===
        for eigen in eigenvals:
            if eigen != best_eigenval:
                experiments.extend([
                    (dataset, eigen, 2, False, f"2-hop baseline ({eigen} eigen)", ""),
                    (dataset, eigen, 3, True, f"3-hop combined ({eigen} eigen)", ""),
                ])
    
    if not no_svd:
        # === SVD ENHANCEMENT EXPERIMENTS ===
        experiments.extend([
            (dataset, best_eigenval, 2, False, "2-hop baseline + SVD", "--use_svd"),
            (dataset, best_eigenval, 2, True, "2-hop combined + SVD", "--use_svd"),
            (dataset, eigenvals[1], 3, True, f"3-hop combined + SVD ({eigenvals[1]} eigen)", "--use_svd"),
            (dataset, best_eigenval, 4, True, "4-hop combined + SVD", "--use_svd"),
        ])
        
        if not quick:
            # Test different SVD ranks
            for rank in svd_ranks:
                experiments.append((dataset, best_eigenval, 2, False, f"2-hop + SVD-{rank}", f"--use_svd --n_svd {rank}"))
            
            # Test different SVD weights
            for weight in [0.1, 0.3, 0.5]:
                experiments.append((dataset, best_eigenval, 2, False, f"2-hop + SVD (w={weight})", f"--use_svd --svd_weight {weight}"))
            
            # === HYBRID EXPERIMENTS ===
            experiments.extend([
                (dataset, eigenvals[1], 3, True, "3-hop combined + SVD (w=0.2)", "--use_svd --svd_weight 0.2"),
                (dataset, best_eigenval, 4, True, "4-hop combined + SVD (w=0.4)", "--use_svd --svd_weight 0.4"),
                (dataset, best_eigenval, 1, False, "1-hop + SVD", "--use_svd"),
                (dataset, best_eigenval, 3, False, "3-hop + SVD", "--use_svd"),
            ])
    
    return experiments

def run_comprehensive_experiments():
    """Run comprehensive experiments across different configurations"""
    
    args = parse_args()
    dataset = args.dataset
    quick = args.quick
    no_svd = args.no_svd
    max_experiments = args.max_experiments
    
    print(f"🚀 Starting comprehensive experiments on {dataset.upper()}")
    print(f"Options: {'Quick' if quick else 'Full'}, {'No SVD' if no_svd else 'Include SVD'}")
    
    # Generate experiments
    experiments = generate_experiments(dataset, quick, no_svd)
    
    if max_experiments:
        experiments = experiments[:max_experiments]
    
    print(f"Total experiments: {len(experiments)}")
    print("Categories: Baseline Multi-hop, SVD Enhancement, Fine-tuning, Hybrid")
    
    results = []
    
    for i, (ds, n_eigen, n_hops, combine_hops, description, additional_args) in enumerate(experiments):
        
        print(f"\n[{i+1}/{len(experiments)}] {description}")
        start_time = time.time()
        
        result = run_experiment(ds, n_eigen, n_hops, combine_hops, additional_args)
        
        if result['success']:
            elapsed = time.time() - start_time
            results.append({
                'description': description,
                'dataset': ds,
                'n_eigen': n_eigen,
                'n_hops': n_hops,
                'combine_hops': combine_hops,
                'recall': result['recall'],
                'precision': result['precision'],
                'ndcg': result['ndcg'],
                'time': elapsed
            })
            
            print(f"✓ Success: NDCG={result['ndcg']:.6f}, Recall={result['recall']:.6f} ({elapsed:.1f}s)")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Print summary with categories
    print(f"\n{'='*120}")
    print(f"EXPERIMENT SUMMARY FOR {dataset.upper()} (sorted by NDCG@20)")
    print(f"{'='*120}")
    print(f"{'Description':<50} {'NDCG@20':<10} {'Recall@20':<12} {'Precision@20':<14} {'Time':<8} {'Category':<15}")
    print("-" * 120)
    
    # Sort by NDCG
    results.sort(key=lambda x: x['ndcg'], reverse=True)
    
    for result in results:
        # Categorize results
        desc = result['description']
        if 'SVD' in desc:
            category = "SVD-Enhanced"
        elif result['combine_hops']:
            category = "Multi-hop Comb"
        else:
            category = "Baseline"
            
        print(f"{result['description']:<50} {result['ndcg']:<10.6f} {result['recall']:<12.6f} {result['precision']:<14.6f} {result['time']:<8.1f}s {category:<15}")
    
    print(f"\n{'='*120}")
    print(f"TOP 5 CONFIGURATIONS FOR {dataset.upper()}:")
    print(f"{'='*120}")
    
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['description']}")
        print(f"   NDCG@20: {result['ndcg']:.6f}, Recall@20: {result['recall']:.6f}")
        cmd_parts = [f"--dataset {result['dataset']}", f"--n_eigen {result['n_eigen']}", f"--n_hops {result['n_hops']}"]
        if result['combine_hops']:
            cmd_parts.append("--combine_hops")
        # Parse additional args from description
        if 'SVD' in result['description']:
            cmd_parts.append("--use_svd")
            if 'w=' in result['description']:
                weight = result['description'].split('w=')[1].split(')')[0]
                cmd_parts.append(f"--svd_weight {weight}")
            if 'SVD-' in result['description'] and 'SVD-Enhanced' not in result['description']:
                svd_dim = result['description'].split('SVD-')[1].split(' ')[0]
                if svd_dim.isdigit():
                    cmd_parts.append(f"--n_svd {svd_dim}")
        print(f"   Command: python main.py {' '.join(cmd_parts)}")
        print()
    
    # Performance analysis
    print(f"{'='*120}")
    print(f"PERFORMANCE ANALYSIS FOR {dataset.upper()}:")
    print(f"{'='*120}")
    
    # Best in each category
    best_baseline = max([r for r in results if 'SVD' not in r['description'] and not r['combine_hops']], 
                       key=lambda x: x['ndcg'], default=None)
    best_combined = max([r for r in results if 'SVD' not in r['description'] and r['combine_hops']], 
                       key=lambda x: x['ndcg'], default=None)
    best_svd = max([r for r in results if 'SVD' in r['description']], 
                   key=lambda x: x['ndcg'], default=None)
    
    if best_baseline:
        print(f"Best Baseline: {best_baseline['description']} (NDCG: {best_baseline['ndcg']:.6f})")
    if best_combined:
        print(f"Best Combined: {best_combined['description']} (NDCG: {best_combined['ndcg']:.6f})")
        if best_baseline:
            improvement = (best_combined['ndcg'] - best_baseline['ndcg']) / best_baseline['ndcg'] * 100
            print(f"Hop Combination Improvement: {improvement:+.2f}%")
    if best_svd:
        print(f"Best SVD-Enhanced: {best_svd['description']} (NDCG: {best_svd['ndcg']:.6f})")
        if best_baseline:
            improvement = (best_svd['ndcg'] - best_baseline['ndcg']) / best_baseline['ndcg'] * 100
            print(f"SVD Improvement over Baseline: {improvement:+.2f}%")
    
    # Dataset-specific insights
    print(f"\n📊 DATASET INSIGHTS:")
    if len(results) > 0:
        avg_time = sum(r['time'] for r in results) / len(results)
        print(f"Average experiment time: {avg_time:.1f}s")
        
        hop_performance = {}
        for r in results:
            if 'SVD' not in r['description'] and not r['combine_hops']:
                hop_performance[r['n_hops']] = r['ndcg']
        
        if hop_performance:
            best_hop = max(hop_performance, key=hop_performance.get)
            print(f"Best performing hop count: {best_hop}-hop (NDCG: {hop_performance[best_hop]:.6f})")

if __name__ == "__main__":
    run_comprehensive_experiments()