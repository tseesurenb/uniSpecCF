"""
Memory-Efficient DySimSpectralCF with Caching
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import os
import pickle
import hashlib


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


def get_cache_filename(prefix, adj_shape, k_u, k_i, use_attention):
    """Generate unique cache filename based on parameters"""
    param_str = f"{adj_shape[0]}x{adj_shape[1]}_ku{k_u}_ki{k_i}_att{use_attention}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return f"cache_{prefix}_{param_hash}.pkl"


def apply_symmetric_softmax_attention_sparse(similarity_matrix):
    """Memory-efficient attention for sparse matrices"""
    if not sp.issparse(similarity_matrix):
        similarity_matrix = sp.csr_matrix(similarity_matrix)
    
    # Work with sparse matrix data directly
    data = similarity_matrix.data.copy()
    data = np.clip(data, -10, 10)
    
    # Apply softmax to non-zero elements only
    data = np.exp(data - np.max(data))
    
    # Row normalization for sparse matrix
    attended_matrix = sp.csr_matrix(
        (data, similarity_matrix.indices, similarity_matrix.indptr),
        shape=similarity_matrix.shape
    )
    
    # Efficient row normalization
    row_sums = np.array(attended_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-8
    row_inv_sqrt = 1.0 / np.sqrt(row_sums)
    
    # Column normalization
    col_sums = np.array(attended_matrix.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1e-8
    col_inv_sqrt = 1.0 / np.sqrt(col_sums)
    
    # Apply symmetric normalization
    row_diag = sp.diags(row_inv_sqrt)
    col_diag = sp.diags(col_inv_sqrt)
    normalized_matrix = row_diag @ attended_matrix @ col_diag
    
    return normalized_matrix


def create_similarity_adj_chunked(adj_mat, k_u=30, k_i=15, use_attention=True, 
                                 cache_dir="cache", chunk_size=1000):
    """Memory-efficient similarity computation with chunking and caching"""
    os.makedirs(cache_dir, exist_ok=True)
    
    n_users, n_items = adj_mat.shape
    cache_file = os.path.join(cache_dir, get_cache_filename("similarity", adj_mat.shape, k_u, k_i, use_attention))
    
    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"Loading similarity matrices from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['similarity_adj'], cached_data['user_sparse'], cached_data['item_sparse']
        except Exception as e:
            print(f"Cache loading failed: {e}, recomputing...")
    
    print(f"Computing similarity matrices for {n_users}×{n_items} matrix...")
    
    # Convert to sparse if needed
    if not sp.issparse(adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
    
    # Memory-efficient user-user similarity with chunking
    print("Computing user-user similarities (chunked)...")
    user_adj = sp.lil_matrix((n_users, n_users))
    
    for start_idx in range(0, n_users, chunk_size):
        end_idx = min(start_idx + chunk_size, n_users)
        user_chunk = adj_mat[start_idx:end_idx]
        
        # Compute similarities for this chunk
        if user_chunk.nnz > 0:
            user_sim_chunk = (user_chunk @ adj_mat.T).toarray()
            
            # Apply top-k selection for this chunk
            for i, global_i in enumerate(range(start_idx, end_idx)):
                similarities = user_sim_chunk[i].copy()
                similarities[global_i] = -1  # Exclude self
                
                if k_u > 0 and similarities.max() > 0:
                    top_k = np.argsort(similarities)[-k_u:]
                    valid = top_k[similarities[top_k] > 0]
                    if len(valid) > 0:
                        user_adj[global_i, valid] = similarities[valid]
    
    # Convert to CSR for efficiency
    user_adj = user_adj.tocsr()
    print(f"User similarity matrix: {user_adj.nnz:,} non-zeros")
    
    # Memory-efficient item-item similarity with chunking
    print("Computing item-item similarities (chunked)...")
    item_adj = sp.lil_matrix((n_items, n_items))
    
    adj_mat_T = adj_mat.T.tocsr()  # Transpose once
    
    for start_idx in range(0, n_items, chunk_size):
        end_idx = min(start_idx + chunk_size, n_items)
        item_chunk = adj_mat_T[start_idx:end_idx]
        
        # Compute similarities for this chunk
        if item_chunk.nnz > 0:
            item_sim_chunk = (item_chunk @ adj_mat_T.T).toarray()
            
            # Apply top-k selection for this chunk
            for i, global_i in enumerate(range(start_idx, end_idx)):
                similarities = item_sim_chunk[i].copy()
                similarities[global_i] = -1  # Exclude self
                
                if k_i > 0 and similarities.max() > 0:
                    top_k = np.argsort(similarities)[-k_i:]
                    valid = top_k[similarities[top_k] > 0]
                    if len(valid) > 0:
                        item_adj[global_i, valid] = similarities[valid]
    
    # Convert to CSR for efficiency
    item_adj = item_adj.tocsr()
    print(f"Item similarity matrix: {item_adj.nnz:,} non-zeros")
    
    # Apply attention mechanism if enabled
    if use_attention:
        print("Applying attention mechanism...")
        if user_adj.nnz > 0:
            user_adj = apply_symmetric_softmax_attention_sparse(user_adj)
        if item_adj.nnz > 0:
            item_adj = apply_symmetric_softmax_attention_sparse(item_adj)
    
    # Create block diagonal matrix
    similarity_adj = sp.block_diag([user_adj, item_adj]).tocsr()
    print(f"Combined similarity matrix: {similarity_adj.nnz:,} non-zeros")
    
    # Cache the results
    print(f"Caching similarity matrices to: {cache_file}")
    try:
        cache_data = {
            'similarity_adj': similarity_adj,
            'user_sparse': user_adj,
            'item_sparse': item_adj
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Cache saving failed: {e}")
    
    return similarity_adj, user_adj, item_adj


def compute_gram_matrices_cached(adj_mat, cache_dir="cache"):
    """Compute and cache gram matrices"""
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, get_cache_filename("gram", adj_mat.shape, 0, 0, False))
    
    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"Loading gram matrices from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['item_gram'], cached_data['user_gram_proj']
        except Exception as e:
            print(f"Gram cache loading failed: {e}, recomputing...")
    
    print("Computing gram matrices...")
    
    # Convert to sparse if needed
    if not sp.issparse(adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
    
    # Item gram matrix: R.T @ R
    print("Computing item gram matrix (R.T @ R)...")
    item_gram = adj_mat.T @ adj_mat
    
    # User gram projected to item space: R.T @ (R @ R.T) @ R
    print("Computing user gram projection...")
    user_gram = adj_mat @ adj_mat.T
    user_gram_proj = adj_mat.T @ user_gram @ adj_mat
    
    # Cache the results
    print(f"Caching gram matrices to: {cache_file}")
    try:
        cache_data = {
            'item_gram': item_gram,
            'user_gram_proj': user_gram_proj
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Gram cache saving failed: {e}")
    
    return item_gram, user_gram_proj


class DySimSpectralCF:
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat.astype(np.float32) if sp.issparse(adj_mat) else adj_mat
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parameters
        self.k_u = self.config.get('k_u', 30)
        self.k_i = self.config.get('k_i', 15)
        self.use_attention = self.config.get('use_attention', True)
        self.n_eigen_user = self.config.get('n_eigen_user', 16)
        self.n_eigen_item = self.config.get('n_eigen_item', 32)
        self.filter_order = self.config.get('filter_order', 6)
        self.cache_dir = self.config.get('cache_dir', 'cache')
        self.chunk_size = self.config.get('chunk_size', 1000)
        
        self.n_users, self.n_items = self.adj_mat.shape
        
        # Adjust parameters for large datasets
        if self.n_users > 10000 or self.n_items > 10000:
            self.k_u = min(self.k_u, 20)
            self.k_i = min(self.k_i, 10)
            self.n_eigen_user = min(self.n_eigen_user, 12)
            self.n_eigen_item = min(self.n_eigen_item, 24)
            self.chunk_size = min(self.chunk_size, 500)
            print(f"Large dataset detected, adjusted parameters: k_u={self.k_u}, k_i={self.k_i}")
    
    def _compute_eigen_cached(self, matrix, k, matrix_name, cache_dir="cache"):
        """Compute eigendecomposition with caching"""
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache filename based on matrix properties
        matrix_hash = hashlib.md5(str(matrix.shape).encode()).hexdigest()[:8]
        cache_file = os.path.join(cache_dir, f"eigen_{matrix_name}_{matrix_hash}_k{k}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Loaded eigendecomposition for {matrix_name} from cache")
                return cached_data['eigenvals'], cached_data['eigenvecs']
            except Exception as e:
                print(f"Eigen cache loading failed for {matrix_name}: {e}")
        
        print(f"Computing eigendecomposition for {matrix_name} ({matrix.shape})...")
        
        try:
            if not sp.issparse(matrix):
                matrix = sp.csr_matrix(matrix)
            
            # Make symmetric and add stability
            matrix = (matrix + matrix.T) / 2
            matrix += sp.diags(1e-6 * np.ones(matrix.shape[0]))
            
            k = min(k, matrix.shape[0] - 5)
            if k <= 0:
                raise ValueError("k too small")
            
            eigenvals, eigenvecs = eigsh(matrix, k=k, which='LM', maxiter=1000, tol=1e-6)
            eigenvals = np.maximum(eigenvals, 0.0)
            
        except Exception as e:
            print(f"Eigendecomposition failed for {matrix_name}: {e}, using fallback")
            k = min(8, matrix.shape[0] - 2)
            eigenvals = np.linspace(0.1, 1.0, k)
            eigenvecs = np.random.randn(matrix.shape[0], k)
            eigenvecs, _ = np.linalg.qr(eigenvecs)
        
        # Cache the results
        try:
            cache_data = {
                'eigenvals': eigenvals,
                'eigenvecs': eigenvecs
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached eigendecomposition for {matrix_name}")
        except Exception as e:
            print(f"Eigen cache saving failed for {matrix_name}: {e}")
        
        return eigenvals, eigenvecs
    
    def train(self):
        """Train the model with memory-efficient operations and caching"""
        print(f"Training DySimSpectralCF for {self.n_users}×{self.n_items} matrix...")
        
        # Create similarity matrices with caching
        similarity_adj, user_block, item_block = create_similarity_adj_chunked(
            self.adj_mat, self.k_u, self.k_i, self.use_attention, 
            self.cache_dir, self.chunk_size
        )
        
        # Compute gram matrices with caching
        item_gram, user_gram_proj = compute_gram_matrices_cached(self.adj_mat, self.cache_dir)
        
        # Store matrices (only keep non-empty ones)
        self.matrices = {}
        
        if user_block.nnz > 0:
            self.matrices['user_sim'] = user_block
        if item_block.nnz > 0:
            self.matrices['item_sim'] = item_block
        if item_gram.nnz > 0:
            self.matrices['item_gram'] = item_gram
        if user_gram_proj.nnz > 0:
            self.matrices['user_gram_proj'] = user_gram_proj
        
        print(f"Active matrices: {list(self.matrices.keys())}")
        
        # Compute eigendecompositions with caching
        self.eigendata = {}
        self.filters = {}
        
        for name, matrix in self.matrices.items():
            k = self.n_eigen_user if 'user' in name else self.n_eigen_item
            eigenvals, eigenvecs = self._compute_eigen_cached(matrix, k, name, self.cache_dir)
            
            self.eigendata[name] = {
                'eigenvals': torch.tensor(eigenvals, dtype=torch.float32, device=self.device),
                'eigenvecs': torch.tensor(eigenvecs, dtype=torch.float32, device=self.device)
            }
            
            self.filters[name] = ChebyshevFilter(self.filter_order).to(self.device)
            print(f"  {name}: {len(eigenvals)} eigenvalues")
        
        # Combination weights
        n_matrices = len(self.eigendata)
        if n_matrices > 1:
            self.combination_weights = nn.Parameter(torch.ones(n_matrices, device=self.device))
        else:
            self.combination_weights = torch.ones(1, device=self.device)
        
        # Base adjacency (convert to dense only for small matrices)
        if self.n_users * self.n_items < 1000000:  # 1M threshold
            self.base_adj = torch.tensor(
                self.adj_mat.toarray() if sp.issparse(self.adj_mat) else self.adj_mat,
                dtype=torch.float32, device=self.device
            )
        else:
            # Keep as sparse for large matrices
            self.base_adj_sparse = self.adj_mat.tocsr()
            self.base_adj = None
        
        print(f"Model training completed. Matrices: {n_matrices}")
    
    def getUsersRating(self, batch_users, ds_name=None):
        """Memory-efficient user rating prediction"""
        # Handle both list and tensor inputs
        if isinstance(batch_users, list):
            batch_users = torch.LongTensor(batch_users).to(self.device)
        elif isinstance(batch_users, np.ndarray):
            batch_users = torch.LongTensor(batch_users).to(self.device)
        elif not isinstance(batch_users, torch.Tensor):
            batch_users = torch.LongTensor([batch_users]).to(self.device)
        
        # Handle sparse base adjacency for large matrices
        if self.base_adj is not None:
            base_scores = self.base_adj[batch_users]
        else:
            # Convert sparse to dense for this batch only
            user_indices = batch_users.cpu().numpy()
            base_scores_np = self.base_adj_sparse[user_indices].toarray()
            base_scores = torch.tensor(base_scores_np, dtype=torch.float32, device=self.device)
        
        all_scores = []
        
        for name, eigen_data in self.eigendata.items():
            eigenvals = eigen_data['eigenvals']
            eigenvecs = eigen_data['eigenvecs']
            
            filter_response = self.filters[name](eigenvals)
            filter_matrix = eigenvecs @ torch.diag(filter_response) @ eigenvecs.t()
            
            if 'user_sim' in name:
                # User similarity: filter users then project to items
                batch_size = len(batch_users)
                user_embeddings = torch.zeros(batch_size, self.n_users, device=self.device)
                user_embeddings[range(batch_size), batch_users] = 1.0
                
                filtered_users = user_embeddings @ filter_matrix
                
                if self.base_adj is not None:
                    filtered_scores = filtered_users @ self.base_adj
                else:
                    # Use sparse matrix multiplication
                    filtered_users_np = filtered_users.detach().cpu().numpy()
                    filtered_scores_np = filtered_users_np @ self.base_adj_sparse.toarray()
                    filtered_scores = torch.tensor(filtered_scores_np, dtype=torch.float32, device=self.device)
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