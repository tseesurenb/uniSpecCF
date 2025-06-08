'''
Fixed World Configuration with combine_hops support
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
    'filter_order': args.filter_order,
    'n_hops': args.n_hops,
    'combine_hops': args.combine_hops,  # Add this line
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