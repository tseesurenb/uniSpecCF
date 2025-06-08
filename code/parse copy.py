'''
Enhanced Argument Parser with Hop Combination Options
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Simple Universal Spectral CF")

    # Basic parameters
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-2, help="weight decay")
    parser.add_argument('--train_u_batch_size', type=int, default=1000, help='training batch size')
    parser.add_argument('--eval_u_batch_size', type=int, default=500, help="evaluation batch size")
    parser.add_argument('--epochs', type=int, default=50)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='gowalla', 
                       help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, ml-100k]")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation ratio')
    
    # Model parameters
    parser.add_argument('--n_eigen', type=int, default=128, help='number of eigenvalues')
    parser.add_argument('--filter_order', type=int, default=6, help='spectral filter order')
    parser.add_argument('--n_hops', type=int, default=2, choices=[1, 2, 3, 4], 
                       help='number of hops: 1 (User→Item), 2 (User→Item→User→Item), 3 (3-hop), 4 (4-hop)')
    parser.add_argument('--combine_hops', action='store_true', 
                       help='combine multiple hop patterns (1 through n_hops) instead of using only n_hops')
    
    # Training control
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='evaluate every N epochs')
    
    # Experiment
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level')
    
    # Legacy compatibility
    parser.add_argument('--model', type=str, default='uspec', help='model name')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='minimum improvement')
    
    return parser.parse_args()