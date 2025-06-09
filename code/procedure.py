"""
Minimal Training Procedure
"""

import numpy as np
import torch
import torch.nn as nn
import utils
from time import time


class MSELoss:
    def __init__(self, model, config):
        self.model = model
        
        # Collect parameters
        params = []
        for filter_module in model.filters.values():
            params.extend(list(filter_module.parameters()))
        params.append(model.combination_weights)
        
        self.opt = torch.optim.Adam(params, lr=config['lr'], weight_decay=config['decay'])
    
    def train_step(self, users, target_ratings):
        self.opt.zero_grad()
        
        base_scores = self.model.base_adj[users]
        all_scores = []
        
        for name, eigen_data in self.model.eigendata.items():
            eigenvals = eigen_data['eigenvals']
            eigenvecs = eigen_data['eigenvecs']
            
            filter_response = self.model.filters[name](eigenvals)
            filter_matrix = eigenvecs @ torch.diag(filter_response) @ eigenvecs.t()
            
            if 'user_sim' in name:
                user_embeddings = torch.eye(self.model.n_users, device=self.model.device)[users]
                filtered_users = user_embeddings @ filter_matrix
                filtered_scores = filtered_users @ self.model.base_adj
            else:
                filtered_scores = base_scores @ filter_matrix
            
            all_scores.append(filtered_scores)
        
        if len(all_scores) > 1:
            weights = torch.softmax(self.model.combination_weights, dim=0)
            predicted = sum(w * s for w, s in zip(weights, all_scores))
        else:
            predicted = all_scores[0]
        
        loss = torch.mean((predicted - target_ratings) ** 2)
        loss.backward()
        
        # Gradient clipping
        all_params = []
        for filter_module in self.model.filters.values():
            all_params.extend(list(filter_module.parameters()))
        all_params.append(self.model.combination_weights)
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        
        self.opt.step()
        return loss.cpu().item()


def create_target_ratings(dataset, users, device):
    batch_size = len(users)
    n_items = dataset.m_items
    target_ratings = torch.zeros(batch_size, n_items, device=device)
    
    for i, user in enumerate(users):
        pos_items = dataset.allPos[user]
        if len(pos_items) > 0:
            target_ratings[i, pos_items] = 1.0
    
    return target_ratings


def train_epoch(dataset, model, loss_class, epoch, config):
    n_users = dataset.n_users
    batch_size = config['train_u_batch_size']
    users_per_epoch = min(n_users, max(1000, n_users // 4))
    
    np.random.seed(epoch * 42)
    sampled_users = np.random.choice(n_users, users_per_epoch, replace=False)
    
    total_loss = 0.0
    n_batches = max(1, users_per_epoch // batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, users_per_epoch)
        user_indices = sampled_users[start_idx:end_idx]
        
        users = torch.LongTensor(user_indices).to(device)
        target_ratings = create_target_ratings(dataset, user_indices, device)
        
        batch_loss = loss_class.train_step(users, target_ratings)
        total_loss += batch_loss
    
    return total_loss / n_batches


def evaluate(dataset, model, data_dict, config):
    if len(data_dict) == 0:
        return {'recall': np.zeros(1), 'precision': np.zeros(1), 'ndcg': np.zeros(1)}
    
    batch_size = config['eval_u_batch_size']
    results = {'recall': 0.0, 'precision': 0.0, 'ndcg': 0.0}
    
    users = list(data_dict.keys())
    total_users = 0
    
    for batch_users in utils.minibatch(users, batch_size=batch_size):
        batch_users = [int(u) for u in batch_users]
        
        training_items = dataset.getUserPosItems(batch_users)
        ground_truth = [data_dict[u] for u in batch_users]
        
        ratings = model.getUsersRating(batch_users)
        ratings = torch.from_numpy(ratings)
        
        # Exclude training items
        for i, items in enumerate(training_items):
            if len(items) > 0:
                ratings[i, items] = -float('inf')
        
        _, top_items = torch.topk(ratings, k=20)
        
        # Compute metrics for this batch
        relevance = utils.getLabel(ground_truth, top_items.cpu().numpy())
        
        for i, gt in enumerate(ground_truth):
            if len(gt) > 0:
                # Recall@20
                hit = relevance[i][:20].sum()
                results['recall'] += hit / len(gt)
                results['precision'] += hit / 20
                
                # NDCG@20
                dcg = sum(hit / np.log2(j + 2) for j, hit in enumerate(relevance[i][:20]))
                idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(gt), 20)))
                results['ndcg'] += dcg / idcg if idcg > 0 else 0
                
                total_users += 1
    
    if total_users > 0:
        results['recall'] /= total_users
        results['precision'] /= total_users
        results['ndcg'] /= total_users
    
    return {
        'recall': np.array([results['recall']]),
        'precision': np.array([results['precision']]),
        'ndcg': np.array([results['ndcg']])
    }


def train_and_evaluate(dataset, model, config):
    loss_class = MSELoss(model, config)
    best_ndcg = 0.0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        avg_loss = train_epoch(dataset, model, loss_class, epoch, config)
        
        if (epoch + 1) % config['n_epoch_eval'] == 0:
            eval_data = dataset.valDict if len(dataset.valDict) > 0 else dataset.testDict
            results = evaluate(dataset, model, eval_data, config)
            current_ndcg = results['ndcg'][0]
            
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience'] // config['n_epoch_eval']:
                break
    
    final_results = evaluate(dataset, model, dataset.testDict, config)
    return model, final_results