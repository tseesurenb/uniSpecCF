'''
Fixed World Configuration with all SVD parameters
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
    'n_eigen': args.n_eigen,
    'n_svd': args.n_svd,  # Add SVD parameters
    'filter_order': args.filter_order,
    'n_hops': args.n_hops,
    'combine_hops': args.combine_hops,
    'use_svd': args.use_svd,  # Add SVD flag
    'svd_weight': args.svd_weight,  # Add SVD weight
    'verbose': args.verbose,
    'val_ratio': args.val_ratio,
    'patience': args.patience,
    'n_epoch_eval': args.n_epoch_eval,
    'min_delta': args.min_delta,
}

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config['device'] = device

seed = args.seed
dataset = args.dataset
topks = eval(args.topks)