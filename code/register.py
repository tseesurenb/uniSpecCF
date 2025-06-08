'''
Fixed Model Registration for SVD-Enhanced Model
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

# Model registration - updated for SVD-enhanced model
MODELS = {
    'uspec': model.SVDEnhancedSpectralCF,
    'svd_enhanced': model.SVDEnhancedSpectralCF,  # Alternative name
}

# Backward compatibility - check if old model exists
if hasattr(model, 'UniversalSpectralCF'):
    MODELS['universal'] = model.UniversalSpectralCF