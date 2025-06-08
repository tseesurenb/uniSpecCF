'''
Minimal Main Script for Simple Universal Spectral CF
'''

import world
import utils
import procedure
import time
from register import dataset, MODELS

# Set random seed
utils.set_seed(world.seed)

print(f"Simple Universal Spectral CF")
print(f"Dataset: {world.config['dataset']}")
print(f"Device: {world.device}")

# Create model
print(f"Creating model...")
model_start = time.time()
adj_mat = dataset.UserItemNet.tolil()

UniversalSpectralCF = MODELS['uspec']
Recmodel = UniversalSpectralCF(adj_mat, world.config)
Recmodel.train()
print(f"Model created and trained in {time.time() - model_start:.2f}s")

# Dataset info
print(f"Users: {dataset.n_users:,}, Items: {dataset.m_items:,}")
print(f"Training: {dataset.trainDataSize:,}, Test: {len(dataset.testDict):,}")

# Training
print(f"Starting training...")
training_start = time.time()
trained_model, final_results = procedure.train_and_evaluate(dataset, Recmodel, world.config)
total_time = time.time() - training_start

# Final results
print(f"Training time: {total_time:.2f}s")
print(f"Final Results: Recall@20={final_results['recall'][0]:.6f}, "
      f"Precision@20={final_results['precision'][0]:.6f}, "
      f"NDCG@20={final_results['ndcg'][0]:.6f}")