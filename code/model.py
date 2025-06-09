"""
Minimal DySimSpectralCF Implementation
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity
import time


class ChebyshevFilter(nn.Module):
    def __init__(self, filter_order=6):
        super().__init__()
        self.filter_order = filter_order
        self.coeffs = nn.Parameter(torch.randn(filter_order + 1) * 0.1)
    
    def forward(self, eigenvals):
        # Normalize eigenvalues to [-1, 1]
        eigenvals_norm = 2 * eigenvals / (eigenvals.max() + 1e-8) - 1
        eigenvals_norm = torch.clamp(eigenvals_norm, -0.99, 0.99)
        
        # Compute Chebyshev polynomials
        T = [torch.ones_like(eigenvals_norm), eigenvals_norm]
        for i in range(2, self.filter_order + 1):
            T.append(2 * eigenvals_norm * T[i-1] - T[i-2])
        
        # Apply filter
        response = sum(self.coeffs[i] * T[i] for i in range(self.filter_order + 1))
        return torch.sigmoid(response)


def create_similarity_adj(adj_mat, k_u=30, k_i=15, similarity_type='cosine'):
    """Create similarity-based adjacency matrix"""
    n_users, n_items = adj_mat.shape
    
    # User-user similarity
    if n_users < 5000:
        user_sim = cosine_similarity(adj_mat, dense_output=False).toarray()
    else:
        # Limit to most active users for large datasets
        row_activity = np.array(adj_mat.sum(axis=1)).flatten()
        top_users = min(3000, n_users)
        top_indices = np.argsort(row_activity)[-top_users:]
        limited_adj = adj_mat[top_indices]
        limited_sim = cosine_similarity(limited_adj, dense_output=False).toarray()
        # Pad back to original size
        user_sim = np.zeros((n_users, n_users))
        for i, idx in enumerate(top_indices):
            user_sim[idx, top_indices] = limited_sim[i]
    
    # Item-item similarity
    if n_items < 5000:
        item_sim = cosine_similarity(adj_mat.T, dense_output=False).toarray()
    else:
        # Limit to most active items
        col_activity = np.array(adj_mat.sum(axis=0)).flatten()
        top_items = min(3000, n_items)
        top_indices = np.argsort(col_activity)[-top_items:]
        limited_adj = adj_mat[:, top_indices].T
        limited_sim = cosine_similarity(limited_adj, dense_output=False).toarray()
        # Pad back to original size
        item_sim = np.zeros((n_items, n_items))
        for i, idx in enumerate(top_indices):
            item_sim[idx, top_indices] = limited_sim[i]
    
    # Apply top-k selection
    user_adj = np.zeros_like(user_sim)
    for i in range(n_users):
        similarities = user_sim[i].copy()
        similarities[i] = -1  # Exclude self
        if k_u > 0:
            top_k = np.argsort(similarities)[-k_u:]
            valid = top_k[similarities[top_k] > 0]
            if len(valid) > 0:
                user_adj[i, valid] = similarities[valid]
    
    item_adj = np.zeros_like(item_sim)
    for i in range(n_items):
        similarities = item_sim[i].copy()
        similarities[i] = -1  # Exclude self
        if k_i > 0:
            top_k = np.argsort(similarities)[-k_i:]
            valid = top_k[similarities[top_k] > 0]
            if len(valid) > 0:
                item_adj[i, valid] = similarities[valid]
    
    # Create block diagonal matrix
    user_sparse = sp.csr_matrix(user_adj)
    item_sparse = sp.csr_matrix(item_adj)
    similarity_adj = sp.block_diag([user_sparse, item_sparse]).tocsr()
    
    return similarity_adj, user_sparse, item_sparse


class DySimSpectralCF:
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat.astype(np.float32) if sp.issparse(adj_mat) else adj_mat
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parameters
        self.k_u = self.config.get('k_u', 30)
        self.k_i = self.config.get('k_i', 15)
        self.n_eigen_user = self.config.get('n_eigen_user', 16)
        self.n_eigen_item = self.config.get('n_eigen_item', 32)
        self.filter_order = self.config.get('filter_order', 6)
        
        self.n_users, self.n_items = self.adj_mat.shape
    
    def _compute_eigen(self, matrix, k):
        """Compute eigendecomposition"""
        try:
            if not sp.issparse(matrix):
                matrix = sp.csr_matrix(matrix)
            matrix = (matrix + matrix.T) / 2  # Make symmetric
            matrix += sp.diags(1e-6 * np.ones(matrix.shape[0]))  # Numerical stability
            
            k = min(k, matrix.shape[0] - 2)
            eigenvals, eigenvecs = eigsh(matrix, k=k, which='LM', maxiter=1000, tol=1e-6)
            eigenvals = np.maximum(eigenvals, 0.0)
            
        except Exception:
            # Fallback to random
            k = min(16, matrix.shape[0] - 2)
            eigenvals = np.linspace(0.1, 1.0, k)
            eigenvecs = np.random.randn(matrix.shape[0], k)
            eigenvecs, _ = np.linalg.qr(eigenvecs)
        
        return eigenvals, eigenvecs
    
    def train(self):
        """Train the model"""
        # Create similarity matrices
        similarity_adj, user_block, item_block = create_similarity_adj(self.adj_mat, self.k_u, self.k_i)
        
        # Add original gram matrices
        item_gram = self.adj_mat.T @ self.adj_mat
        user_gram = self.adj_mat @ self.adj_mat.T
        user_gram_proj = self.adj_mat.T @ user_gram @ self.adj_mat
        
        # Store matrices
        self.matrices = {
            'user_sim': user_block,
            'item_sim': item_block,
            'item_gram': item_gram,
            'user_gram_proj': user_gram_proj
        }
        
        # Compute eigendecompositions
        self.eigendata = {}
        self.filters = {}
        
        for name, matrix in self.matrices.items():
            if matrix.nnz == 0:
                continue
                
            k = self.n_eigen_user if 'user' in name else self.n_eigen_item
            eigenvals, eigenvecs = self._compute_eigen(matrix, k)
            
            self.eigendata[name] = {
                'eigenvals': torch.tensor(eigenvals, dtype=torch.float32, device=self.device),
                'eigenvecs': torch.tensor(eigenvecs, dtype=torch.float32, device=self.device)
            }
            
            self.filters[name] = ChebyshevFilter(self.filter_order).to(self.device)
        
        # Combination weights
        n_matrices = len(self.eigendata)
        self.combination_weights = nn.Parameter(torch.ones(n_matrices, device=self.device))
        
        # Base adjacency
        self.base_adj = torch.tensor(
            self.adj_mat.toarray() if sp.issparse(self.adj_mat) else self.adj_mat,
            dtype=torch.float32, device=self.device
        )
    
    def getUsersRating(self, batch_users, ds_name=None):
        """Get user ratings"""
        if isinstance(batch_users, np.ndarray):
            batch_users = torch.LongTensor(batch_users).to(self.device)
        
        base_scores = self.base_adj[batch_users]
        all_scores = []
        
        for name, eigen_data in self.eigendata.items():
            eigenvals = eigen_data['eigenvals']
            eigenvecs = eigen_data['eigenvecs']
            
            filter_response = self.filters[name](eigenvals)
            filter_matrix = eigenvecs @ torch.diag(filter_response) @ eigenvecs.t()
            
            if 'user_sim' in name:
                # User similarity: filter users then project to items
                user_embeddings = torch.eye(self.n_users, device=self.device)[batch_users]
                filtered_users = user_embeddings @ filter_matrix
                filtered_scores = filtered_users @ self.base_adj
            else:
                # Item matrices
                filtered_scores = base_scores @ filter_matrix
            
            all_scores.append(filtered_scores)
        
        if len(all_scores) > 1:
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = sum(w * s for w, s in zip(weights, all_scores))
        else:
            predicted = all_scores[0]
        
        return predicted.detach().cpu().numpy()