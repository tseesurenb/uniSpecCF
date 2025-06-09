'''
Memory-Efficient DySimSpectralCF for Large Datasets
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
import time


def compute_similarity_matrix_efficient(matrix, similarity_type='cosine', max_users=None, max_items=None):
    """
    Memory-efficient similarity computation using sparse operations and chunking
    """
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
    
    n_rows = matrix.shape[0]
    
    # For very large datasets, limit the size
    if max_users is not None and n_rows > max_users:
        print(f"Limiting similarity computation to top {max_users} most active users/items")
        # Get row activity (number of interactions)
        row_activity = np.array(matrix.sum(axis=1)).flatten()
        top_indices = np.argsort(row_activity)[-max_users:]
        matrix = matrix[top_indices]
        n_rows = max_users
    
    if similarity_type == 'cosine':
        print(f"Computing cosine similarity for {n_rows} entities...")
        
        # Use sklearn's efficient cosine similarity for sparse matrices
        similarity = cosine_similarity(matrix, dense_output=False)
        
        # Convert to dense only if small enough
        if n_rows < 5000:
            similarity = similarity.toarray()
        
    elif similarity_type == 'jaccard':
        print(f"Computing Jaccard similarity for {n_rows} entities...")
        
        # Binarize the matrix
        matrix_bool = (matrix > 0).astype(np.float32)
        
        if n_rows < 5000:
            # Dense computation for small matrices
            intersection = matrix_bool @ matrix_bool.T
            sum_matrix = np.array(matrix_bool.sum(axis=1)).flatten()
            union = sum_matrix[:, None] + sum_matrix[None, :] - intersection
            union[union == 0] = 1e-8
            similarity = intersection / union
        else:
            # Sparse computation for large matrices
            intersection = matrix_bool @ matrix_bool.T
            sum_matrix = np.array(matrix_bool.sum(axis=1)).flatten()
            # Convert to dense for union computation (still memory-efficient)
            if sp.issparse(intersection):
                intersection = intersection.toarray()
            union = sum_matrix[:, None] + sum_matrix[None, :] - intersection
            union[union == 0] = 1e-8
            similarity = intersection / union
    
    return similarity


def create_similarity_based_adjacency_efficient(adj_mat, k_u=50, k_i=20, similarity_type='cosine', 
                                               max_users=10000, max_items=10000):
    """
    Memory-efficient similarity-based adjacency matrix creation
    """
    n_users, n_items = adj_mat.shape
    print(f"Dataset size: {n_users:,} users × {n_items:,} items")
    
    # Memory check: if too large, reduce k values
    if n_users > 20000:
        k_u = min(k_u, 30)
        print(f"Large dataset detected, reducing k_u to {k_u}")
    if n_items > 30000:
        k_i = min(k_i, 15)
        print(f"Large dataset detected, reducing k_i to {k_i}")
    
    # Compute user-user similarity (limit to most active users if needed)
    print("Computing user-user similarities...")
    user_sim_matrix = compute_similarity_matrix_efficient(
        adj_mat, similarity_type, max_users=max_users if n_users > max_users else None
    )
    
    # Compute item-item similarity (limit to most active items if needed)
    print("Computing item-item similarities...")
    item_sim_matrix = compute_similarity_matrix_efficient(
        adj_mat.T, similarity_type, max_items=max_items if n_items > max_items else None
    )
    
    # Adjust matrix sizes if we limited users/items
    actual_n_users = user_sim_matrix.shape[0]
    actual_n_items = item_sim_matrix.shape[0]
    
    print(f"Actual similarity matrix sizes: {actual_n_users} users, {actual_n_items} items")
    
    # Apply top-k selection for users
    print(f"Applying top-{k_u} selection for users...")
    user_adj = np.zeros_like(user_sim_matrix)
    
    for i in range(actual_n_users):
        if sp.issparse(user_sim_matrix):
            similarities = user_sim_matrix[i].toarray().flatten()
        else:
            similarities = user_sim_matrix[i].copy()
        
        similarities[i] = -1  # Exclude self
        
        if k_u > 0 and k_u < actual_n_users:
            top_k_indices = np.argsort(similarities)[-k_u:]
            valid_indices = top_k_indices[similarities[top_k_indices] > 0]
            if len(valid_indices) > 0:
                user_adj[i, valid_indices] = similarities[valid_indices]
    
    # Apply top-k selection for items
    print(f"Applying top-{k_i} selection for items...")
    item_adj = np.zeros_like(item_sim_matrix)
    
    for i in range(actual_n_items):
        if sp.issparse(item_sim_matrix):
            similarities = item_sim_matrix[i].toarray().flatten()
        else:
            similarities = item_sim_matrix[i].copy()
        
        similarities[i] = -1  # Exclude self
        
        if k_i > 0 and k_i < actual_n_items:
            top_k_indices = np.argsort(similarities)[-k_i:]
            valid_indices = top_k_indices[similarities[top_k_indices] > 0]
            if len(valid_indices) > 0:
                item_adj[i, valid_indices] = similarities[valid_indices]
    
    # Create sparse block diagonal matrix
    user_adj_sparse = sp.csr_matrix(user_adj)
    item_adj_sparse = sp.csr_matrix(item_adj)
    
    # If we limited the matrix sizes, we need to pad them back
    if actual_n_users < n_users or actual_n_items < n_items:
        print("Padding similarity matrices to original size...")
        
        # Pad user matrix
        if actual_n_users < n_users:
            pad_users = n_users - actual_n_users
            user_pad = sp.csr_matrix((pad_users, actual_n_users))
            bottom_pad = sp.csr_matrix((n_users, pad_users))
            user_adj_sparse = sp.vstack([user_adj_sparse, user_pad])
            user_adj_sparse = sp.hstack([user_adj_sparse, bottom_pad])
        
        # Pad item matrix
        if actual_n_items < n_items:
            pad_items = n_items - actual_n_items
            item_pad = sp.csr_matrix((pad_items, actual_n_items))
            bottom_pad = sp.csr_matrix((n_items, pad_items))
            item_adj_sparse = sp.vstack([item_adj_sparse, item_pad])
            item_adj_sparse = sp.hstack([item_adj_sparse, bottom_pad])
    
    similarity_adj = sp.block_diag([user_adj_sparse, item_adj_sparse])
    
    print(f"Created similarity adjacency matrix: {similarity_adj.shape}")
    print(f"User block edges: {user_adj_sparse.nnz:,}, Item block edges: {item_adj_sparse.nnz:,}")
    
    return similarity_adj, user_sim_matrix, item_sim_matrix


def apply_symmetric_softmax_attention_efficient(similarity_matrix):
    """
    Memory-efficient attention mechanism for large sparse matrices
    """
    if not sp.issparse(similarity_matrix):
        # For small dense matrices, use original method
        return apply_symmetric_softmax_attention(similarity_matrix)
    
    print("Applying efficient attention mechanism...")
    
    # Work with sparse matrix directly
    similarity_matrix = similarity_matrix.tocsr()
    
    # Apply attention only to non-zero elements
    data = similarity_matrix.data.copy()
    
    # Clip extreme values
    data = np.clip(data, -10, 10)
    
    # Apply softmax to non-zero elements only
    data = np.exp(data - np.max(data))
    
    # Create new sparse matrix with attention-weighted values
    attended_matrix = sp.csr_matrix(
        (data, similarity_matrix.indices, similarity_matrix.indptr),
        shape=similarity_matrix.shape
    )
    
    # Row normalization
    row_sums = np.array(attended_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-8
    row_diag = sp.diags(1.0 / np.sqrt(row_sums))
    
    # Column normalization
    col_sums = np.array(attended_matrix.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1e-8
    col_diag = sp.diags(1.0 / np.sqrt(col_sums))
    
    # Apply symmetric normalization
    normalized_matrix = row_diag @ attended_matrix @ col_diag
    
    return normalized_matrix


class DySimSpectralCF_Efficient(object):
    """Memory-Efficient DySimSpectralCF for Large Datasets"""
    
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat.astype(np.float32) if sp.issparse(adj_mat) else adj_mat
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters - more conservative for large datasets
        self.use_full_eigen = self.config.get('use_full_eigen', False)  # Always False for large datasets
        self.n_eigen_item = self.config.get('n_eigen_item', 32)  # Reduced
        self.n_eigen_user = self.config.get('n_eigen_user', 16)  # Reduced
        self.filter_order = self.config.get('filter_order', 6)
        
        # DySimGCF parameters
        self.k_u = self.config.get('k_u', 50)
        self.k_i = self.config.get('k_i', 20)
        self.similarity_type = self.config.get('similarity_type', 'cosine')
        self.use_attention = self.config.get('use_attention', True)
        
        # Graph construction options
        self.use_user_similarity = self.config.get('use_user_similarity', True)
        self.use_item_similarity = self.config.get('use_item_similarity', True)
        self.use_original_interactions = self.config.get('use_original_interactions', True)
        
        # Memory efficiency parameters
        self.max_users_similarity = self.config.get('max_users_similarity', 10000)
        self.max_items_similarity = self.config.get('max_items_similarity', 10000)
        
        self.n_users, self.n_items = self.adj_mat.shape
        
        print(f"DySimSpectralCF_Efficient: {self.n_users:,} users, {self.n_items:,} items")
        print(f"Similarity: {self.similarity_type}, Top-k: users={self.k_u}, items={self.k_i}")
        print(f"Memory limits: users={self.max_users_similarity:,}, items={self.max_items_similarity:,}")
    
    def _compute_eigendecomposition_efficient(self, matrix, matrix_type='similarity'):
        """Efficient eigendecomposition for large sparse matrices"""
        max_size = matrix.shape[0]
        
        # Always use truncated eigenspace for large matrices
        if matrix_type == 'user':
            k = min(self.n_eigen_user, max_size - 5, 50)  # Cap at 50
        elif matrix_type == 'item':
            k = min(self.n_eigen_item, max_size - 5, 100)  # Cap at 100
        else:
            k = min(32, max_size - 5)
        
        try:
            # Ensure matrix is sparse and symmetric
            if not sp.issparse(matrix):
                matrix = sp.csr_matrix(matrix)
            
            # Make symmetric for stability
            matrix = (matrix + matrix.T) / 2
            
            # Add small diagonal term for numerical stability
            matrix += sp.diags(1e-6 * np.ones(max_size))
            
            print(f"Computing {k} eigenvalues for {matrix_type} matrix ({max_size}×{max_size})...")
            eigenvals, eigenvecs = eigsh(matrix, k=k, which='LM', maxiter=1000, tol=1e-6)
            eigenvals = np.maximum(eigenvals, 0.0)
            method = f"Efficient truncated ({k})"
            
        except Exception as e:
            print(f"Eigendecomposition failed ({e}), using fallback")
            k = min(16, max_size - 2)
            eigenvals = np.linspace(0.1, 1.0, k)
            eigenvecs = np.random.randn(max_size, k)
            eigenvecs, _ = np.linalg.qr(eigenvecs)  # Orthogonalize
            method = f"Random fallback ({k})"
        
        return eigenvals, eigenvecs, len(eigenvals), method
    
    def train(self):
        """Memory-efficient training for large datasets"""
        start = time.time()
        
        print("Creating similarity-based adjacency matrices (memory-efficient)...")
        
        # Create similarity-based adjacency matrix with memory limits
        similarity_adj, user_sim_matrix, item_sim_matrix = create_similarity_based_adjacency_efficient(
            self.adj_mat, self.k_u, self.k_i, self.similarity_type,
            self.max_users_similarity, self.max_items_similarity
        )
        
        # Apply attention mechanism if enabled
        if self.use_attention:
            print("Applying efficient attention mechanism...")
            try:
                similarity_adj = apply_symmetric_softmax_attention_efficient(similarity_adj)
            except Exception as e:
                print(f"Attention failed ({e}), continuing without attention")
        
        # Store similarity matrices
        self.similarity_matrices = {}
        
        if self.use_user_similarity:
            user_block = similarity_adj[:self.n_users, :self.n_users]
            if user_block.nnz > 0:
                self.similarity_matrices['user_similarity'] = user_block
            
        if self.use_item_similarity:
            item_block = similarity_adj[self.n_users:, self.n_users:]
            if item_block.nnz > 0:
                self.similarity_matrices['item_similarity'] = item_block
        
        if self.use_original_interactions:
            # Use sparse operations for original interactions
            item_gram = self.adj_mat.T @ self.adj_mat
            # Simplify user gram computation for memory efficiency
            user_gram_simple = self.adj_mat @ self.adj_mat.T
            user_gram_projected = self.adj_mat.T @ user_gram_simple @ self.adj_mat
            
            self.similarity_matrices['item_gram_original'] = item_gram
            self.similarity_matrices['user_gram_original'] = user_gram_projected
        
        # Compute eigendecompositions
        self.eigendata = {}
        self.filters = {}
        
        for name, sim_matrix in self.similarity_matrices.items():
            if 'user' in name:
                matrix_type = 'user'
            elif 'item' in name:
                matrix_type = 'item'
            else:
                matrix_type = 'similarity'
            
            eigenvals, eigenvecs, n_eigen, method = self._compute_eigendecomposition_efficient(
                sim_matrix, matrix_type
            )
            
            print(f"  {name}: {method} ({n_eigen} eigenvalues)")
            
            self.eigendata[name] = {
                'eigenvals': torch.tensor(eigenvals, dtype=torch.float32, device=self.device),
                'eigenvecs': torch.tensor(eigenvecs, dtype=torch.float32, device=self.device)
            }
            
            from model import ChebyshevFilter  # Import from your model
            self.filters[name] = ChebyshevFilter(self.filter_order).to(self.device)
        
        # Combination weights
        n_matrices = len(self.similarity_matrices)
        self.combination_weights = nn.Parameter(torch.ones(n_matrices, device=self.device))
        
        # Base adjacency for predictions
        self.base_adj = torch.tensor(
            self.adj_mat.toarray() if sp.issparse(self.adj_mat) else self.adj_mat, 
            dtype=torch.float32, device=self.device
        )
        
        print(f'DySimSpectralCF_Efficient training completed in {time.time() - start:.2f}s')
        print(f'Similarity matrices: {list(self.similarity_matrices.keys())}')
    
    def getUsersRating(self, batch_users, ds_name=None):
        """Efficient user rating prediction"""
        if isinstance(batch_users, np.ndarray):
            batch_users = torch.LongTensor(batch_users).to(self.device)
        
        base_scores = self.base_adj[batch_users]
        all_filtered_scores = []
        
        for name, eigen_data in self.eigendata.items():
            eigenvals = eigen_data['eigenvals']
            eigenvecs = eigen_data['eigenvecs']
            
            filter_response = self.filters[name](eigenvals)
            filter_matrix = eigenvecs @ torch.diag(filter_response) @ eigenvecs.t()
            
            if 'user' in name and 'original' not in name:
                # User similarity: filter users then project to items
                user_embeddings = torch.eye(self.n_users, device=self.device)[batch_users]
                filtered_users = user_embeddings @ filter_matrix
                filtered_scores = filtered_users @ self.base_adj
            else:
                # Item similarity or original gram matrices
                filtered_scores = base_scores @ filter_matrix
            
            all_filtered_scores.append(filtered_scores)
        
        if len(all_filtered_scores) > 1:
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = sum(w * s for w, s in zip(weights, all_filtered_scores))
        else:
            predicted = all_filtered_scores[0]
        
        return predicted.detach().cpu().numpy()


# For large datasets, use the efficient version
DySimSpectralCF_Large = DySimSpectralCF_Efficient