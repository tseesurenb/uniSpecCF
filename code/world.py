'''
World Configuration for both SimplifiedSpectralCF and DySimSpectralCF
'''

import torch
from parse import parse_args

args = parse_args()

config = {
    'train_u_batch_size': args.train_u_batch_size,
    'eval_u_batch_size': args.eval_u_batch_size,
    'dataset': args.dataset,
    'lr': args.lr,
    'decay': args.decay,
    'epochs': args.epochs,
    'filter_order': args.filter_order,
    'verbose': args.verbose,
    'val_ratio': args.val_ratio,
    'patience': args.patience,
    'n_epoch_eval': args.n_epoch_eval,
    'min_delta': args.min_delta,
    
    # SimplifiedSpectralCF parameters - manual control
    'use_full_eigen': args.use_full_eigen,  # Manual: --use_full_eigen for PolyCF style
    'n_eigen_item': args.n_eigen_item,     # Manual: separate for item grams
    'n_eigen_user': args.n_eigen_user,     # Manual: separate for user grams
    'use_user_gram': args.use_user_gram,
    'use_item_gram': args.use_item_gram,
    
    # PolyCF-style normalization control
    'gamma_values': None,  # Will be set below
    'single_gamma': args.single_gamma,
    
    # Legacy normalization control (for backward compatibility)
    'norm_types': args.norm_types,
    'single_norm': args.single_norm,
    
    # DySimSpectralCF parameters (new)
    'k_u': getattr(args, 'k_u', 50),  # Default values if not present
    'k_i': getattr(args, 'k_i', 20),
    'similarity_type': getattr(args, 'similarity_type', 'cosine'),
    'use_attention': getattr(args, 'use_attention', True),
    'use_user_similarity': getattr(args, 'use_user_similarity', True),
    'use_item_similarity': getattr(args, 'use_item_similarity', True),
    'use_original_interactions': getattr(args, 'use_original_interactions', True),
    
    # Model selection
    'model': args.model,
}

# Handle gamma values configuration
if args.single_gamma is not None:
    # Use only single gamma value for ablation studies
    config['gamma_values'] = [args.single_gamma]
    print(f"Using single gamma value: {args.single_gamma}")
else:
    # Use multiple gamma values for PolyCF approach
    config['gamma_values'] = args.gamma_values
    print(f"Using gamma values: {args.gamma_values}")

# Backward compatibility: convert old normalization types to gamma values
if args.single_norm is not None:
    print("WARNING: --single_norm is deprecated, converting to gamma values")
    if args.single_norm == 'none':
        config['gamma_values'] = [1.0]  # Row normalization
    elif args.single_norm == 'symmetric':
        config['gamma_values'] = [0.5]  # Symmetric normalization
    else:
        config['gamma_values'] = [0.5]  # Default to symmetric

if args.norm_types is not None:
    print("WARNING: --norm_types is deprecated, converting to gamma values")
    gamma_mapping = {
        'none': 1.0,
        'symmetric': 0.5,
        'softmax': 0.0  # Map to raw view
    }
    config['gamma_values'] = [gamma_mapping.get(norm, 0.5) for norm in args.norm_types]

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config['device'] = device

seed = args.seed
dataset = args.dataset
topks = eval(args.topks)

# Print configuration summary
print(f"\n=== Configuration Summary ===")
print(f"Model: {config['model']}")
print(f"Dataset: {config['dataset']}")
print(f"Device: {device}")

if config['model'] == 'dysim_spectral':
    print(f"\nDySimSpectralCF Parameters:")
    print(f"  Top-K: users={config['k_u']}, items={config['k_i']}")
    print(f"  Similarity: {config['similarity_type']}")
    print(f"  Attention: {config['use_attention']}")
    print(f"  User similarity: {config['use_user_similarity']}")
    print(f"  Item similarity: {config['use_item_similarity']}")
    print(f"  Original interactions: {config['use_original_interactions']}")
else:
    print(f"\nSimplifiedSpectralCF Parameters:")
    print(f"  User gram: {config['use_user_gram']}")
    print(f"  Item gram: {config['use_item_gram']}")
    print(f"  Gamma values: {config['gamma_values']}")

print(f"  Full eigenspace: {config['use_full_eigen']}")
if not config['use_full_eigen']:
    print(f"  Eigenspace: Item={config['n_eigen_item']}, User={config['n_eigen_user']}")
print(f"  Filter order: {config['filter_order']}")
print("="*30)