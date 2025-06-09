'''
Model Registration for SimplifiedSpectralCF and DySimSpectralCF
'''

import world
import dataloader
import model

# Import DySimSpectralCF if available
try:
    from model import DySimSpectralCF
    DYSIM_AVAILABLE = True
except ImportError:
    print("Warning: DySimSpectralCF not available, falling back to SimplifiedSpectralCF")
    DYSIM_AVAILABLE = False

# Dataset loading
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

# Model registration
MODELS = {
    'uspec': model.SimplifiedSpectralCF,
    'simplified': model.SimplifiedSpectralCF,  # Alternative name
    'universal': model.UniversalSpectralCF    # Backward compatibility
}

# Add DySimSpectralCF if available
if DYSIM_AVAILABLE:
    MODELS.update({
        'dysim_spectral': model.DySimSpectralCF,
        'dysim': model.DySimSpectralCF,  # Short name
        'similarity_spectral': model.DySimSpectralCF  # Descriptive name
    })

print(f"Available models: {list(MODELS.keys())}")
if world.config['model'] not in MODELS:
    print(f"Warning: Model '{world.config['model']}' not found, using 'uspec' instead")
    world.config['model'] = 'uspec'