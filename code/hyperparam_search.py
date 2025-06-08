#!/usr/bin/env python3
"""
Hierarchical Hyperparameter Search for Multi-hop Spectral CF
Single file that works with existing main.py - no changes needed to other files
"""

import subprocess
import sys
import time
import argparse
import json
import numpy as np
import itertools
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical Hyperparameter Search")
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset: ml-100k, gowalla, yelp2018, amazon-book, lastfm')
    
    # Search strategy
    parser.add_argument('--search_type', type=str, choices=['hierarchical', 'grid', 'random'], 
                       default='hierarchical', help='Search strategy')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Max trials for random search or per stage for hierarchical')
    parser.add_argument('--quick', action='store_true',
                       help='Quick search with fewer parameters')
    
    # Output
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results (auto-generated if not provided)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()

class HierarchicalSearch:
    """Hierarchical hyperparameter search with multiple stages"""
    
    def __init__(self, dataset: str, quick: bool = False, verbose: bool = False):
        self.dataset = dataset
        self.quick = quick
        self.verbose = verbose
        self.results = []
        self.best_configs = {}
        
        # Define search spaces
        self._define_search_spaces()
    
    def _define_search_spaces(self):
        """Define hierarchical search spaces"""
        
        # Stage 1: Architecture search (hop patterns)
        self.stage1_space = {
            'n_hops': [1, 2, 3, 4],
            'combine_hops': [False, True],
            'n_eigen': self._get_eigen_range(),
            'epochs': [20] if self.quick else [30]
        }
        
        # Stage 2: Spectral parameter tuning  
        self.stage2_space = {
            'n_eigen': self._get_eigen_range(extended=True),
            'filter_order': [4, 5, 6, 7, 8] if not self.quick else [5, 6, 7],
            'lr': [0.0005, 0.001, 0.002, 0.005] if not self.quick else [0.001, 0.002],
            'decay': [1e-4, 1e-3, 1e-2] if not self.quick else [1e-3, 1e-2],
            'epochs': [30] if self.quick else [50]
        }
        
        # Stage 3: Training optimization
        self.stage3_space = {
            'lr': [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.003, 0.005],
            'decay': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            'train_u_batch_size': [256, 512, 1000, 1500, 2000],
            'patience': [3, 5, 7, 10],
            'epochs': [50] if self.quick else [100]
        }
        
        # Stage 4: SVD enhancement (optional)
        self.stage4_space = {
            'use_svd': [True],
            'n_svd': self._get_svd_range(),
            'svd_weight': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            'epochs': [30] if self.quick else [50]
        }
    
    def _get_eigen_range(self, extended=False):
        """Get eigenvalue range based on dataset"""
        ranges = {
            'ml-100k': [16, 25, 35, 45, 65, 85, 128] if extended else [25, 35, 65],
            'gowalla': [64, 128, 256, 512, 1024] if extended else [128, 256, 512],
            'yelp2018': [128, 256, 512, 1024, 2048] if extended else [256, 512, 1024],
            'amazon-book': [256, 512, 1024, 2048] if extended else [512, 1024],
            'lastfm': [16, 32, 64, 128, 256] if extended else [32, 64, 128]
        }
        return ranges.get(self.dataset, [64, 128, 256])
    
    def _get_svd_range(self):
        """Get SVD rank range based on dataset"""
        ranges = {
            'ml-100k': [64, 128, 256, 512],
            'gowalla': [256, 512, 1024],
            'yelp2018': [512, 1024, 2048],
            'amazon-book': [1024, 2048, 4096],
            'lastfm': [64, 128, 256, 512]
        }
        return ranges.get(self.dataset, [128, 256, 512])
    
    def run_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with given config"""
        
        # Build command
        cmd = [sys.executable, "main.py", "--dataset", self.dataset]
        
        # Add parameters
        for key, value in config.items():
            if key == 'combine_hops' and value:
                cmd.append('--combine_hops')
            elif key == 'use_svd' and value:
                cmd.append('--use_svd')
            elif key != 'combine_hops' and key != 'use_svd':
                cmd.extend([f"--{key}", str(value)])
        
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse results
            lines = result.stdout.split('\n')
            for line in lines:
                if "Final Results:" in line:
                    parts = line.split(", ")
                    recall = float(parts[0].split("=")[1])
                    precision = float(parts[1].split("=")[1])
                    ndcg = float(parts[2].split("=")[1])
                    
                    return {
                        'config': config,
                        'recall': recall,
                        'precision': precision,
                        'ndcg': ndcg,
                        'success': True
                    }
            
            return {'config': config, 'success': False, 'error': 'Could not parse results'}
            
        except subprocess.TimeoutExpired:
            return {'config': config, 'success': False, 'error': 'Timeout'}
        except Exception as e:
            return {'config': config, 'success': False, 'error': str(e)}
    
    def stage1_architecture_search(self, max_trials=None):
        """Stage 1: Find best architecture (hop patterns)"""
        print(f"\n🔍 STAGE 1: Architecture Search (Hop Patterns)")
        print(f"Testing hop configurations for {self.dataset}")
        
        # Generate all combinations for architecture search
        configs = []
        for n_hops in self.stage1_space['n_hops']:
            for combine_hops in self.stage1_space['combine_hops']:
                for n_eigen in self.stage1_space['n_eigen']:
                    config = {
                        'n_hops': n_hops,
                        'combine_hops': combine_hops,
                        'n_eigen': n_eigen,
                        'epochs': self.stage1_space['epochs'][0]
                    }
                    configs.append(config)
        
        # Limit trials if specified
        if max_trials and len(configs) > max_trials:
            configs = random.sample(configs, max_trials)
        
        print(f"Running {len(configs)} architecture configurations...")
        
        results = []
        for i, config in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] Testing {config['n_hops']}-hop" + 
                  (" combined" if config['combine_hops'] else "") + 
                  f" (eigen={config['n_eigen']})")
            
            result = self.run_experiment(config)
            if result['success']:
                results.append(result)
                print(f"  ✓ NDCG: {result['ndcg']:.6f}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
        
        # Find best architecture
        if results:
            best = max(results, key=lambda x: x['ndcg'])
            self.best_configs['stage1'] = best['config']
            print(f"\n🏆 Best Architecture: {best['config']['n_hops']}-hop" +
                  (" combined" if best['config']['combine_hops'] else "") +
                  f" (NDCG: {best['ndcg']:.6f})")
            return results
        else:
            print("❌ No successful results in Stage 1")
            return []
    
    def stage2_spectral_tuning(self, max_trials=None):
        """Stage 2: Tune spectral parameters"""
        print(f"\n🔍 STAGE 2: Spectral Parameter Tuning")
        
        if 'stage1' not in self.best_configs:
            print("❌ No best config from Stage 1, skipping Stage 2")
            return []
        
        base_config = self.best_configs['stage1'].copy()
        print(f"Base config: {base_config}")
        
        # Generate parameter combinations
        configs = []
        param_combinations = list(itertools.product(
            self.stage2_space['n_eigen'],
            self.stage2_space['filter_order'],
            self.stage2_space['lr'],
            self.stage2_space['decay']
        ))
        
        if max_trials and len(param_combinations) > max_trials:
            param_combinations = random.sample(param_combinations, max_trials)
        
        for n_eigen, filter_order, lr, decay in param_combinations:
            config = base_config.copy()
            config.update({
                'n_eigen': n_eigen,
                'filter_order': filter_order,
                'lr': lr,
                'decay': decay,
                'epochs': self.stage2_space['epochs'][0]
            })
            configs.append(config)
        
        print(f"Running {len(configs)} spectral parameter combinations...")
        
        results = []
        for i, config in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] eigen={config['n_eigen']}, " +
                  f"filter={config['filter_order']}, lr={config['lr']}, decay={config['decay']}")
            
            result = self.run_experiment(config)
            if result['success']:
                results.append(result)
                print(f"  ✓ NDCG: {result['ndcg']:.6f}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
        
        # Find best spectral config
        if results:
            best = max(results, key=lambda x: x['ndcg'])
            self.best_configs['stage2'] = best['config']
            print(f"\n🏆 Best Spectral Config: NDCG={best['ndcg']:.6f}")
            print(f"  Parameters: {best['config']}")
            return results
        else:
            print("❌ No successful results in Stage 2")
            return []
    
    def stage3_training_optimization(self, max_trials=None):
        """Stage 3: Optimize training parameters"""
        print(f"\n🔍 STAGE 3: Training Optimization")
        
        if 'stage2' not in self.best_configs:
            print("❌ No best config from Stage 2, skipping Stage 3")
            return []
        
        base_config = self.best_configs['stage2'].copy()
        print(f"Base config from Stage 2")
        
        # Generate training parameter combinations
        configs = []
        param_combinations = list(itertools.product(
            self.stage3_space['lr'],
            self.stage3_space['decay'],
            self.stage3_space['train_u_batch_size'],
            self.stage3_space['patience']
        ))
        
        if max_trials and len(param_combinations) > max_trials:
            param_combinations = random.sample(param_combinations, max_trials)
        
        for lr, decay, batch_size, patience in param_combinations:
            config = base_config.copy()
            config.update({
                'lr': lr,
                'decay': decay,
                'train_u_batch_size': batch_size,
                'patience': patience,
                'epochs': self.stage3_space['epochs'][0]
            })
            configs.append(config)
        
        print(f"Running {len(configs)} training optimization combinations...")
        
        results = []
        for i, config in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] lr={config['lr']}, decay={config['decay']}, " +
                  f"batch={config['train_u_batch_size']}, patience={config['patience']}")
            
            result = self.run_experiment(config)
            if result['success']:
                results.append(result)
                print(f"  ✓ NDCG: {result['ndcg']:.6f}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
        
        # Find best training config
        if results:
            best = max(results, key=lambda x: x['ndcg'])
            self.best_configs['stage3'] = best['config']
            print(f"\n🏆 Best Training Config: NDCG={best['ndcg']:.6f}")
            return results
        else:
            print("❌ No successful results in Stage 3")
            return []
    
    def stage4_svd_enhancement(self, max_trials=None):
        """Stage 4: SVD enhancement optimization"""
        print(f"\n🔍 STAGE 4: SVD Enhancement")
        
        if 'stage3' not in self.best_configs:
            print("❌ No best config from Stage 3, skipping Stage 4")
            return []
        
        base_config = self.best_configs['stage3'].copy()
        print(f"Testing SVD enhancement on best config from Stage 3")
        
        # Generate SVD parameter combinations
        configs = []
        for n_svd in self.stage4_space['n_svd']:
            for svd_weight in self.stage4_space['svd_weight']:
                config = base_config.copy()
                config.update({
                    'use_svd': True,
                    'n_svd': n_svd,
                    'svd_weight': svd_weight,
                    'epochs': self.stage4_space['epochs'][0]
                })
                configs.append(config)
        
        if max_trials and len(configs) > max_trials:
            configs = random.sample(configs, max_trials)
        
        print(f"Running {len(configs)} SVD enhancement combinations...")
        
        results = []
        for i, config in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] SVD rank={config['n_svd']}, weight={config['svd_weight']}")
            
            result = self.run_experiment(config)
            if result['success']:
                results.append(result)
                print(f"  ✓ NDCG: {result['ndcg']:.6f}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
        
        # Find best SVD config
        if results:
            best = max(results, key=lambda x: x['ndcg'])
            self.best_configs['stage4'] = best['config']
            
            # Compare with non-SVD baseline
            baseline_ndcg = max(self.results, key=lambda x: x['ndcg'])['ndcg'] if self.results else 0
            improvement = (best['ndcg'] - baseline_ndcg) / baseline_ndcg * 100
            
            print(f"\n🏆 Best SVD Config: NDCG={best['ndcg']:.6f}")
            print(f"SVD Improvement: {improvement:+.2f}%")
            return results
        else:
            print("❌ No successful results in Stage 4")
            return []
    
    def run_hierarchical_search(self, max_trials_per_stage=None):
        """Run complete hierarchical search"""
        print(f"🚀 Starting Hierarchical Hyperparameter Search for {self.dataset.upper()}")
        print(f"Mode: {'Quick' if self.quick else 'Full'}")
        
        all_results = []
        
        # Stage 1: Architecture
        stage1_results = self.stage1_architecture_search(max_trials_per_stage)
        all_results.extend(stage1_results)
        self.results.extend(stage1_results)
        
        # Stage 2: Spectral tuning
        if stage1_results:
            stage2_results = self.stage2_spectral_tuning(max_trials_per_stage)
            all_results.extend(stage2_results)
            self.results.extend(stage2_results)
        
        # Stage 3: Training optimization
        if 'stage2' in self.best_configs:
            stage3_results = self.stage3_training_optimization(max_trials_per_stage)
            all_results.extend(stage3_results)
            self.results.extend(stage3_results)
        
        # Stage 4: SVD enhancement
        if 'stage3' in self.best_configs:
            stage4_results = self.stage4_svd_enhancement(max_trials_per_stage)
            all_results.extend(stage4_results)
            self.results.extend(stage4_results)
        
        return all_results
    
    def print_final_summary(self):
        """Print final summary of hierarchical search"""
        print(f"\n{'='*80}")
        print(f"HIERARCHICAL SEARCH SUMMARY FOR {self.dataset.upper()}")
        print(f"{'='*80}")
        
        if not self.results:
            print("❌ No successful results found")
            return
        
        # Overall best
        best_overall = max(self.results, key=lambda x: x['ndcg'])
        print(f"🏆 BEST OVERALL CONFIGURATION:")
        print(f"   NDCG@20: {best_overall['ndcg']:.6f}")
        print(f"   Recall@20: {best_overall['recall']:.6f}")
        print(f"   Config: {best_overall['config']}")
        
        # Best per stage
        print(f"\n📊 BEST PER STAGE:")
        for stage, config in self.best_configs.items():
            stage_results = [r for r in self.results if all(
                r['config'].get(k) == v for k, v in config.items()
                if k in r['config']
            )]
            if stage_results:
                best_stage = max(stage_results, key=lambda x: x['ndcg'])
                print(f"   {stage}: NDCG={best_stage['ndcg']:.6f}")
        
        # Performance analysis
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        ndcgs = [r['ndcg'] for r in self.results if r['success']]
        if ndcgs:
            print(f"   Total trials: {len(self.results)}")
            print(f"   Best NDCG: {max(ndcgs):.6f}")
            print(f"   Average NDCG: {np.mean(ndcgs):.6f}")
            print(f"   Std NDCG: {np.std(ndcgs):.6f}")
        
        # Command for best config
        print(f"\n💻 COMMAND FOR BEST CONFIG:")
        cmd_parts = [f"python main.py --dataset {self.dataset}"]
        for key, value in best_overall['config'].items():
            if key == 'combine_hops' and value:
                cmd_parts.append('--combine_hops')
            elif key == 'use_svd' and value:
                cmd_parts.append('--use_svd')
            elif key not in ['combine_hops', 'use_svd']:
                cmd_parts.append(f"--{key} {value}")
        print(f"   {' '.join(cmd_parts)}")

def main():
    args = parse_args()
    
    # Initialize search
    search = HierarchicalSearch(
        dataset=args.dataset,
        quick=args.quick,
        verbose=args.verbose
    )
    
    # Run search
    if args.search_type == 'hierarchical':
        search.run_hierarchical_search(max_trials_per_stage=args.n_trials // 4)
    
    # Print summary
    search.print_final_summary()
    
    # Save results
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = f"hyperparam_results_{args.dataset}_{args.search_type}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'search_type': args.search_type,
            'quick': args.quick,
            'results': search.results,
            'best_configs': search.best_configs
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()