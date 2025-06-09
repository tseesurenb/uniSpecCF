'''
Argument Parser for DySimSpectralCF - Similarity-Centric Spectral Collaborative Filtering
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DySimSpectralCF: Similarity-Centric Spectral Collaborative Filtering")

    # Basic parameters
    parser.add_argument('--lr', type=float, default=0.05, help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--train_u_batch_size', type=int, default=1000, help='training batch size')
    parser.add_argument('--eval_u_batch_size', type=int, default=500, help="evaluation batch size")
    parser.add_argument('--epochs', type=int, default=50)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='gowalla', 
                       help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, ml-100k]")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation ratio')
    
    # Model parameters
    parser.add_argument('--filter_order', type=int, default=6, help='Chebyshev filter order')
    
    # Eigenspace parameters
    parser.add_argument('--use_full_eigen', action='store_true', default=False,
                       help='use full eigenspace for all similarity matrices')
    parser.add_argument('--n_eigen_item', type=int, default=30,
                       help='number of eigenvalues for item similarity matrices')
    parser.add_argument('--n_eigen_user', type=int, default=15,
                       help='number of eigenvalues for user similarity matrices')
    
    # DySimGCF-inspired parameters
    parser.add_argument('--k_u', type=int, default=50,
                       help='top-k users for user similarity graph construction')
    parser.add_argument('--k_i', type=int, default=20,
                       help='top-k items for item similarity graph construction')
    parser.add_argument('--similarity_type', type=str, default='cosine', choices=['cosine', 'jaccard'],
                       help='similarity computation method')
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='use symmetric softmax attention on similarity matrices')
    parser.add_argument('--no_attention', dest='use_attention', action='store_false',
                       help='disable symmetric softmax attention')
    
    # Graph construction control
    parser.add_argument('--use_user_similarity', action='store_true', default=True,
                       help='use user-user similarity matrices')
    parser.add_argument('--use_item_similarity', action='store_true', default=True,
                       help='use item-item similarity matrices')
    parser.add_argument('--use_original_interactions', action='store_true', default=True,
                       help='include original interaction-based gram matrices')
    parser.add_argument('--no_user_similarity', dest='use_user_similarity', action='store_false',
                       help='disable user-user similarity matrices')
    parser.add_argument('--no_item_similarity', dest='use_item_similarity', action='store_false',
                       help='disable item-item similarity matrices')
    parser.add_argument('--no_original_interactions', dest='use_original_interactions', action='store_false',
                       help='disable original interaction-based matrices')
    
    # Training control
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='evaluate every N epochs')
    
    # Experiment
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level')
    
    # Backward compatibility with existing SimplifiedSpectralCF parameters
    parser.add_argument('--use_user_gram', action='store_true', default=True,
                       help='use user gram matrices (R R^T projected to item space)')
    parser.add_argument('--use_item_gram', action='store_true', default=True,
                       help='use item gram matrices (R^T R)')
    parser.add_argument('--no_user_gram', dest='use_user_gram', action='store_false',
                       help='disable user gram matrices')
    parser.add_argument('--no_item_gram', dest='use_item_gram', action='store_false',
                       help='disable item gram matrices')
    
    # PolyCF-style gamma values (for backward compatibility)
    parser.add_argument('--gamma_values', type=float, nargs='+', 
                       default=[0.0, 0.5, 1.0],
                       help='Gamma values for PolyCF adjacency views (default: [0.0, 0.5, 1.0])')
    parser.add_argument('--single_gamma', type=float, default=None,
                       help='Use only single gamma value (for ablation studies)')
    
    # Legacy normalization control (for backward compatibility)
    parser.add_argument('--norm_types', type=str, nargs='+', 
                       default=None,
                       help='DEPRECATED: use --gamma_values instead')
    parser.add_argument('--single_norm', type=str, default=None,
                       help='DEPRECATED: use --single_gamma instead')
    
    # Legacy compatibility
    parser.add_argument('--model', type=str, default='uspec', help='model name')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='minimum improvement')
    
    return parser.parse_args()