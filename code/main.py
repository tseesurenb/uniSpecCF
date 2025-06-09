'''
Main Script for SimplifiedSpectralCF and DySimSpectralCF
'''

import world
import utils
import procedure
import time
import torch
from register import dataset, MODELS

# Set random seed
utils.set_seed(world.seed)

print(f"Dataset: {world.config['dataset']}")
print(f"Device: {world.device}")
print(f"Model: {world.config['model']}")

# Create model based on configuration
print(f"Creating {world.config['model']} model...")
model_start = time.time()
adj_mat = dataset.UserItemNet.tolil()

# Select the appropriate model class
if world.config['model'] in MODELS:
    ModelClass = MODELS[world.config['model']]
    print(f"Using model class: {ModelClass.__name__}")
else:
    print(f"Warning: Model '{world.config['model']}' not found, using SimplifiedSpectralCF")
    ModelClass = MODELS['uspec']

# Create and train model
Recmodel = ModelClass(adj_mat, world.config)
Recmodel.train()
print(f"Model created and trained in {time.time() - model_start:.2f}s")

# Dataset info
print(f"Users: {dataset.n_users:,}, Items: {dataset.m_items:,}")
print(f"Training: {dataset.trainDataSize:,}, Test: {len(dataset.testDict):,}")
if hasattr(dataset, 'valDict') and len(dataset.valDict) > 0:
    print(f"Validation: {len(dataset.valDict):,}")

# Model-specific info
print(f"\nModel Configuration:")
if hasattr(Recmodel, 'gamma_values'):
    # SimplifiedSpectralCF
    print(f"User gram: {world.config['use_user_gram']}")
    print(f"Item gram: {world.config['use_item_gram']}")
    print(f"Gamma values: {Recmodel.gamma_values}")
elif hasattr(Recmodel, 'k_u'):
    # DySimSpectralCF
    print(f"Similarity type: {Recmodel.similarity_type}")
    print(f"Top-K: users={Recmodel.k_u}, items={Recmodel.k_i}")
    print(f"Attention: {Recmodel.use_attention}")
    print(f"User similarity: {Recmodel.use_user_similarity}")
    print(f"Item similarity: {Recmodel.use_item_similarity}")
    print(f"Original interactions: {Recmodel.use_original_interactions}")

print(f"Filter order: {world.config['filter_order']}")
if world.config['use_full_eigen']:
    print(f"Eigenspace: Full spectrum")
else:
    print(f"Eigenspace: Item={world.config['n_eigen_item']}, User={world.config['n_eigen_user']}")

if hasattr(Recmodel, 'similarity_matrices'):
    print(f"Number of similarity matrices: {len(Recmodel.similarity_matrices)}")
elif hasattr(Recmodel, 'gram_matrices'):
    print(f"Number of gram matrices: {len(Recmodel.gram_matrices)}")

# Training
print(f"\nStarting training...")
training_start = time.time()
trained_model, final_results = procedure.train_and_evaluate(dataset, Recmodel, world.config)
total_time = time.time() - training_start

# Final results
print(f"\nTraining time: {total_time:.2f}s")
print(f"Final Results: Recall@20={final_results['recall'][0]:.6f}, "
      f"Precision@20={final_results['precision'][0]:.6f}, "
      f"NDCG@20={final_results['ndcg'][0]:.6f}")

# Print final learned weights
if hasattr(trained_model, 'combination_weights'):
    final_weights = torch.softmax(trained_model.combination_weights, dim=0)
    print(f"\nFinal Learned Weights:")
    
    if hasattr(trained_model, 'similarity_matrices'):
        # DySimSpectralCF
        matrix_names = list(trained_model.similarity_matrices.keys())
    elif hasattr(trained_model, 'gram_matrices'):
        # SimplifiedSpectralCF
        matrix_names = list(trained_model.gram_matrices.keys())
    else:
        matrix_names = [f"Matrix_{i}" for i in range(len(final_weights))]
    
    for i, (name, weight) in enumerate(zip(matrix_names, final_weights)):
        print(f"  {name}: {weight:.4f}")
else:
    print("\nNo combination weights found (single matrix model)")