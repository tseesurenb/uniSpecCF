'''
Training Procedure for SimplifiedSpectralCF and DySimSpectralCF
'''

import world
import numpy as np
import torch
import torch.nn as nn
import utils
from time import time


class MSELoss:
    def __init__(self, model, config):
        self.model = model
        base_lr = config['lr']
        weight_decay = config['decay']
        
        # Collect all learnable parameters from both model types
        filter_params = []
        for filter_module in model.filters.values():
            filter_params.extend(list(filter_module.parameters()))
        filter_params.append(model.combination_weights)
        
        self.opt = torch.optim.Adam(filter_params, lr=base_lr, weight_decay=weight_decay)
    
    def train_step(self, users, target_ratings):
        """Training step for both SimplifiedSpectralCF and DySimSpectralCF"""
        self.opt.zero_grad()
        
        # Get base scores - different for each model type
        if hasattr(self.model, 'norm_adj'):
            # SimplifiedSpectralCF
            base_scores = self.model.norm_adj[users]
        elif hasattr(self.model, 'base_adj'):
            # DySimSpectralCF
            base_scores = self.model.base_adj[users]
        else:
            raise AttributeError("Model must have either norm_adj or base_adj")
        
        all_filtered_scores = []
        
        # Apply each matrix with its filter - works for both model types
        for name, eigen_data in self.model.eigendata.items():
            eigenvals = eigen_data['eigenvals']
            eigenvecs = eigen_data['eigenvecs']
            
            # Apply Chebyshev filter
            filter_response = self.model.filters[name](eigenvals)
            filter_matrix = eigenvecs @ torch.diag(filter_response) @ eigenvecs.t()
            
            # Apply filtering - different logic for DySimSpectralCF
            if hasattr(self.model, 'similarity_matrices') and 'user' in name and 'original' not in name:
                # DySimSpectralCF user similarity: filter users then project to items
                user_embeddings = torch.eye(self.model.n_users, device=self.model.device)[users]
                filtered_users = user_embeddings @ filter_matrix
                filtered_scores = filtered_users @ self.model.base_adj
            else:
                # SimplifiedSpectralCF or DySimSpectralCF item similarity/original matrices
                filtered_scores = base_scores @ filter_matrix
            
            all_filtered_scores.append(filtered_scores)
        
        # Combine with learned weights
        if len(all_filtered_scores) > 1:
            weights = torch.softmax(self.model.combination_weights, dim=0)
            predicted_ratings = sum(w * s for w, s in zip(weights, all_filtered_scores))
        else:
            predicted_ratings = all_filtered_scores[0]
        
        loss = torch.mean((predicted_ratings - target_ratings) ** 2)
        loss.backward()
        
        # Gradient clipping
        all_params = []
        for filter_module in self.model.filters.values():
            all_params.extend(list(filter_module.parameters()))
        all_params.append(self.model.combination_weights)
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        
        self.opt.step()
        return loss.cpu().item()


def create_target_ratings(dataset, users, device=None):
    """Create target ratings"""
    if device is None:
        device = world.device
        
    batch_size = len(users)
    n_items = dataset.m_items
    target_ratings = torch.zeros(batch_size, n_items, device=device)
    
    for i, user in enumerate(users):
        pos_items = dataset.allPos[user]
        if len(pos_items) > 0:
            target_ratings[i, pos_items] = 1.0
    
    return target_ratings


def train_epoch(dataset, model, loss_class, epoch, config):
    """Train one epoch"""
    n_users = dataset.n_users
    train_batch_size = config['train_u_batch_size']
    
    if train_batch_size == -1:
        train_batch_size = n_users
        users_per_epoch = n_users
    else:
        users_per_epoch = min(n_users, max(1000, n_users // 4))
    
    # Sample users
    np.random.seed(epoch * 42)
    sampled_users = np.random.choice(n_users, users_per_epoch, replace=False)
    sampled_users = [int(u) for u in sampled_users]
    
    total_loss = 0.0
    n_batches = max(1, users_per_epoch // train_batch_size)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, users_per_epoch)
        user_indices = sampled_users[start_idx:end_idx]
        
        users = torch.LongTensor(user_indices).to(world.device)
        target_ratings = create_target_ratings(dataset, user_indices, device=world.device)
        
        batch_loss = loss_class.train_step(users, target_ratings)
        total_loss += batch_loss
        
        if train_batch_size == n_users:
            break
    
    return total_loss / n_batches


def evaluate(dataset, model, data_dict, config):
    """Evaluate model"""
    if len(data_dict) == 0:
        return {'recall': np.zeros(len(world.topks)),
                'precision': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks))}
    
    eval_batch_size = config['eval_u_batch_size']
    max_K = max(world.topks)
    
    results = {'recall': np.zeros(len(world.topks)),
               'precision': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    users = list(data_dict.keys())
    all_results = []
    
    for batch_users in utils.minibatch(users, batch_size=eval_batch_size):
        batch_users = [int(u) for u in batch_users]
        
        training_items = dataset.getUserPosItems(batch_users)
        ground_truth = [data_dict[u] for u in batch_users]
        
        ratings = model.getUsersRating(batch_users)
        ratings = torch.from_numpy(ratings)
        
        # Exclude training items
        for i, items in enumerate(training_items):
            if len(items) > 0:
                ratings[i, items] = -float('inf')
        
        _, top_items = torch.topk(ratings, k=max_K)
        
        batch_result = compute_metrics(ground_truth, top_items.cpu().numpy())
        all_results.append(batch_result)
    
    # Aggregate results
    for result in all_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    
    n_users = len(users)
    results['recall'] /= n_users
    results['precision'] /= n_users
    results['ndcg'] /= n_users
    
    return results


def compute_metrics(ground_truth, predictions):
    """Compute metrics"""
    relevance = utils.getLabel(ground_truth, predictions)
    
    recall, precision, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(ground_truth, relevance, k)
        recall.append(ret['recall'])
        precision.append(ret['precision'])
        ndcg.append(utils.NDCGatK_r(ground_truth, relevance, k))
    
    return {'recall': np.array(recall),
            'precision': np.array(precision),
            'ndcg': np.array(ndcg)}


def train_and_evaluate(dataset, model, config):
    """Complete training pipeline for both SimplifiedSpectralCF and DySimSpectralCF"""
    
    # Determine model type for appropriate messaging
    if hasattr(model, 'similarity_matrices'):
        model_name = "DySimSpectralCF"
        matrix_dict = model.similarity_matrices
    elif hasattr(model, 'gram_matrices'):
        model_name = "SimplifiedSpectralCF"
        matrix_dict = model.gram_matrices
    else:
        model_name = "Unknown Model"
        matrix_dict = {}
    
    print(f"Starting {model_name} training...")
    
    # Print model configuration
    if matrix_dict:
        print(f"Matrices: {list(matrix_dict.keys())}")
    if hasattr(model, 'combination_weights'):
        print(f"Initial combination weights: {model.combination_weights.data}")
    
    # Check validation
    has_validation = hasattr(dataset, 'valDict') and len(dataset.valDict) > 0
    
    loss_class = MSELoss(model, config)
    best_ndcg = 0.0
    best_epoch = 0
    no_improvement = 0
    
    total_epochs = config['epochs']
    patience = config['patience']
    eval_every = config['n_epoch_eval']
    
    training_start = time()
    
    for epoch in range(total_epochs):
        avg_loss = train_epoch(dataset, model, loss_class, epoch, config)
        
        if (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1:
            eval_data = dataset.valDict if has_validation else dataset.testDict
            results = evaluate(dataset, model, eval_data, config)
            current_ndcg = results['ndcg'][0]
            
            # Print learned weights occasionally
            if (epoch + 1) % (eval_every * 2) == 0:
                weights = torch.softmax(model.combination_weights, dim=0)
                print(f"Learned weights at epoch {epoch+1}: {weights.data}")
            
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                no_improvement = 0
                print(f"Epoch {epoch+1}: NDCG = {current_ndcg:.6f} (best)")
            else:
                no_improvement += 1
                print(f"Epoch {epoch+1}: NDCG = {current_ndcg:.6f}")
            
            if no_improvement >= patience // eval_every:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    final_results = evaluate(dataset, model, dataset.testDict, config)
    
    # Print final learned weights
    final_weights = torch.softmax(model.combination_weights, dim=0)
    print(f"Final learned weights: {final_weights.data}")
    
    return model, final_results