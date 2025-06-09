'''
Safe Model Registration - Handles missing classes gracefully
'''

import world
import dataloader

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

print(f"Dataset size: {n_users:,} users × {n_items:,} items = {dataset_size:,} potential interactions")

# Import models safely
try:
    import model
    print(f"Available classes in model module: {[attr for attr in dir(model) if not attr.startswith('_')]}")
except Exception as e:
    print(f"Error importing model module: {e}")
    raise

# Safe model registration
MODELS = {}

# Try to register SimplifiedSpectralCF variants
for class_name in ['SimplifiedSpectralCF', 'UniversalSpectralCF']:
    if hasattr(model, class_name):
        ModelClass = getattr(model, class_name)
        MODELS['uspec'] = ModelClass
        MODELS['simplified'] = ModelClass
        MODELS['universal'] = ModelClass
        print(f"Registered SimplifiedSpectralCF as: {class_name}")
        break
else:
    print("Warning: No SimplifiedSpectralCF class found")

# Try to register DySimSpectralCF variants
dysim_class = None
for class_name in ['DySimSpectralCF', 'DynamicSimilaritySpectralCF']:
    if hasattr(model, class_name):
        dysim_class = getattr(model, class_name)
        print(f"Found DySimSpectralCF as: {class_name}")
        break

if dysim_class:
    # Adjust parameters for large datasets
    if dataset_size > 100_000_000:  # 100M interactions
        print(f"Large dataset detected, adjusting parameters for memory efficiency")
        world.config['use_full_eigen'] = False
        world.config['n_eigen_item'] = min(world.config.get('n_eigen_item', 30), 32)
        world.config['n_eigen_user'] = min(world.config.get('n_eigen_user', 15), 16)
        world.config['k_u'] = min(world.config.get('k_u', 50), 25)  # Reduced further
        world.config['k_i'] = min(world.config.get('k_i', 20), 10)  # Reduced further
        print(f"Adjusted parameters: k_u={world.config['k_u']}, k_i={world.config['k_i']}")
        print(f"Adjusted eigenspace: n_eigen_user={world.config['n_eigen_user']}, n_eigen_item={world.config['n_eigen_item']}")
    
    MODELS.update({
        'dysim_spectral': dysim_class,
        'dysim': dysim_class,
        'similarity_spectral': dysim_class
    })
else:
    print("Warning: No DySimSpectralCF class found")

# Ensure we have at least one model
if not MODELS:
    raise RuntimeError("No valid models found in model module")

print(f"Available models: {list(MODELS.keys())}")

# Validate selected model
if world.config['model'] not in MODELS:
    available_models = list(MODELS.keys())
    print(f"Warning: Model '{world.config['model']}' not found")
    print(f"Available models: {available_models}")
    if available_models:
        fallback_model = available_models[0]
        print(f"Using fallback model: {fallback_model}")
        world.config['model'] = fallback_model
    else:
        raise RuntimeError("No models available")

# Print final configuration
print(f"\nFinal Configuration:")
print(f"  Selected model: {world.config['model']} -> {MODELS[world.config['model']].__name__}")
if world.config['model'] in ['dysim_spectral', 'dysim', 'similarity_spectral']:
    print(f"  k_u: {world.config.get('k_u', 50)}, k_i: {world.config.get('k_i', 20)}")
    print(f"  n_eigen_user: {world.config.get('n_eigen_user', 15)}, n_eigen_item: {world.config.get('n_eigen_item', 30)}")
    print(f"  use_full_eigen: {world.config.get('use_full_eigen', False)}")