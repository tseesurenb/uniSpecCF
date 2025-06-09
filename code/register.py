'''
Smart Model Registration - Automatically chooses efficient versions for large datasets
'''

import world
import dataloader
import model

# Dataset loading
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

# Check dataset size for automatic model selection
adj_mat = dataset.UserItemNet.tolil()
n_users, n_items = adj_mat.shape
dataset_size = n_users * n_items

# Thresholds for model selection (in millions of potential interactions)
LARGE_DATASET_THRESHOLD = 100_000_000  # 100M interactions
MEMORY_EFFICIENT_THRESHOLD = 500_000_000  # 500M interactions

print(f"Dataset size: {n_users:,} users × {n_items:,} items = {dataset_size:,} potential interactions")

# Import efficient version if available
try:
    from memory_efficient_dysim import DySimSpectralCF_Efficient
    EFFICIENT_AVAILABLE = True
    print("Memory-efficient DySimSpectralCF available")
except ImportError:
    print("Memory-efficient DySimSpectralCF not available, using standard version")
    EFFICIENT_AVAILABLE = False

# Model registration with automatic selection
MODELS = {
    'uspec': model.SimplifiedSpectralCF,
    'simplified': model.SimplifiedSpectralCF,
    'universal': model.UniversalSpectralCF
}

# Add DySimSpectralCF variants
if world.config['model'] in ['dysim_spectral', 'dysim', 'similarity_spectral']:
    if dataset_size > MEMORY_EFFICIENT_THRESHOLD and EFFICIENT_AVAILABLE:
        print(f"Large dataset detected ({dataset_size:,} interactions), using memory-efficient DySimSpectralCF")
        DySimSpectralCF_Selected = DySimSpectralCF_Efficient
        # Update config for memory efficiency
        world.config['use_full_eigen'] = False
        world.config['n_eigen_item'] = min(world.config.get('n_eigen_item', 30), 32)
        world.config['n_eigen_user'] = min(world.config.get('n_eigen_user', 15), 16)
        world.config['k_u'] = min(world.config.get('k_u', 50), 30)
        world.config['k_i'] = min(world.config.get('k_i', 20), 15)
    elif dataset_size > LARGE_DATASET_THRESHOLD:
        print(f"Medium-large dataset detected ({dataset_size:,} interactions), using standard DySimSpectralCF with reduced parameters")
        DySimSpectralCF_Selected = model.DySimSpectralCF
        # Reduce parameters for better performance
        world.config['use_full_eigen'] = False
        world.config['n_eigen_item'] = min(world.config.get('n_eigen_item', 30), 64)
        world.config['n_eigen_user'] = min(world.config.get('n_eigen_user', 15), 32)
    else:
        print(f"Small-medium dataset ({dataset_size:,} interactions), using standard DySimSpectralCF")
        DySimSpectralCF_Selected = model.DySimSpectralCF

    MODELS.update({
        'dysim_spectral': DySimSpectralCF_Selected,
        'dysim': DySimSpectralCF_Selected,
        'similarity_spectral': DySimSpectralCF_Selected
    })
else:
    # Add standard DySimSpectralCF for other model types
    MODELS.update({
        'dysim_spectral': model.DySimSpectralCF,
        'dysim': model.DySimSpectralCF,
        'similarity_spectral': model.DySimSpectralCF
    })

print(f"Available models: {list(MODELS.keys())}")
if world.config['model'] not in MODELS:
    print(f"Warning: Model '{world.config['model']}' not found, using 'uspec' instead")
    world.config['model'] = 'uspec'

# Print selected configuration
if world.config['model'] in ['dysim_spectral', 'dysim', 'similarity_spectral']:
    print(f"\nDySimSpectralCF Configuration for {world.dataset}:")
    print(f"  Selected model: {MODELS[world.config['model']].__name__}")
    print(f"  k_u: {world.config.get('k_u', 50)}, k_i: {world.config.get('k_i', 20)}")
    print(f"  n_eigen_user: {world.config.get('n_eigen_user', 15)}, n_eigen_item: {world.config.get('n_eigen_item', 30)}")
    print(f"  use_full_eigen: {world.config.get('use_full_eigen', False)}")