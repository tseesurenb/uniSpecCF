"""
Minimal Main Script for DySimSpectralCF
"""

import numpy as np
import torch
import time
import argparse

# Set random seed
def set_seed(seed=2025):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='lastfm')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--train_u_batch_size', type=int, default=1000)
parser.add_argument('--eval_u_batch_size', type=int, default=500)
parser.add_argument('--filter_order', type=int, default=6)
parser.add_argument('--k_u', type=int, default=30)
parser.add_argument('--k_i', type=int, default=15)
parser.add_argument('--n_eigen_user', type=int, default=16)
parser.add_argument('--n_eigen_item', type=int, default=32)
parser.add_argument('--use_attention', action='store_true', default=True)
parser.add_argument('--no_attention', dest='use_attention', action='store_false')
parser.add_argument('--cache_dir', type=str, default='cache')
parser.add_argument('--chunk_size', type=int, default=1000)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--n_epoch_eval', type=int, default=5)
args = parser.parse_args()

config = vars(args)

# Load dataset
if args.dataset == 'lastfm':
    from dataloader import LastFM
    dataset = LastFM()
if args.dataset == 'ml-100k':
    from dataloader import ML100K
    dataset = ML100K()
else:
    from dataloader import Loader
    dataset = Loader(path=f"../data/{args.dataset}")

# Create model
from model import DySimSpectralCF
print(f"Creating DySimSpectralCF model...")
adj_mat = dataset.UserItemNet.tolil()
model = DySimSpectralCF(adj_mat, config)

# Train model
print("Training model...")
start_time = time.time()
model.train()
print(f"Model initialized in {time.time() - start_time:.2f}s")

# Train and evaluate
import procedure
trained_model, final_results = procedure.train_and_evaluate(dataset, model, config)

# Results
print(f"\nFinal Results:")
print(f"Recall@20: {final_results['recall'][0]:.6f}")
print(f"Precision@20: {final_results['precision'][0]:.6f}")
print(f"NDCG@20: {final_results['ndcg'][0]:.6f}")

# Print learned weights
if hasattr(trained_model, 'combination_weights'):
    weights = torch.softmax(trained_model.combination_weights, dim=0)
    print(f"\nLearned combination weights: {weights.detach().cpu().numpy()}")