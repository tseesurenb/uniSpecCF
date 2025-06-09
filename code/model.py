'''
Simplified Spectral CF and DySimSpectralCF Models
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time


class ChebyshevFilter(nn.Module):
    """Learnable Chebyshev spectral filter"""
    
    def __init__(self, filter_order=6, init_type='practical'):
        super().__init__()
        
        # Initialize with practical values from theory
        if init_type == 'practical':
            # Good starting coefficients for collaborative filtering
            init_coeffs = [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015]
        else:
            # Zero initialization except first coefficient
            init_coeffs = [1.0] + [0.0] * filter_order
        
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(init_coeffs[:filter_order + 1]):
            coeffs_data[i] = val
            
        self.coeffs = nn.Parameter(coeffs_data)
    
    def forward(self, eigenvalues):
        """Apply Chebyshev polynomial filter"""
        # Map eigenvalues to [-1, 1] for Chebyshev polynomials
        lambda_max = torch.max(eigenvalues)
        lambda_min = torch.min(eigenvalues)
        
        if lambda_max > lambda_min:
            scaled_eigen = 2.0 * (eigenvalues - lambda_min) / (lambda_max - lambda_min) - 1.0
        else:
            scaled_eigen = torch.zeros_like(eigenvalues)
        
        # Compute Chebyshev polynomials T_k(x)
        result = self.coeffs[0] * torch.ones_like(scaled_eigen)
        
        if len(self.coeffs) > 1:
            T_prev = torch.ones_like(scaled_eigen)  # T_0(x) = 1
            T_curr = scaled_eigen.clone()           # T_1(x) = x
            result += self.coeffs[1] * T_curr
            
            for k in range(2, len(self.coeffs)):
                T_next = 2.0 * scaled_eigen * T_curr - T_prev  # T_{k+1}(x) = 2xT_k(x) - T_{k-1}(x)
                result += self.coeffs[k] * T_next
                T_prev, T_curr = T_curr, T_next
        
        return torch.sigmoid(result) + 1e-6


def create_polycf_adjacency_views(adj_mat, gamma_values=None):
    """
    Create multiple adjacency views with different normalization parameters γ
    Following PolyCF approach: R_γ = D_r^(-γ) R D_c^(-(1-γ))
    """
    if gamma_values is None:
        gamma_values = [0.0, 0.5, 1.0]  # Raw, Symmetric, Row-normalized
    
    adjacency_views = {}
    
    # Compute degree matrices
    row_degrees = np.array(adj_mat.sum(axis=1)).flatten()
    col_degrees = np.array(adj_mat.sum(axis=0)).flatten()
    
    # Add small epsilon to avoid division by zero
    row_degrees = row_degrees + 1e-8
    col_degrees = col_degrees + 1e-8
    
    for gamma in gamma_values:
        # Compute D_r^(-γ) and D_c^(-(1-γ))
        d_row_power = np.power(row_degrees, -gamma)
        d_col_power = np.power(col_degrees, -(1-gamma))
        
        # Handle infinite values
        d_row_power[np.isinf(d_row_power)] = 0.
        d_col_power[np.isinf(d_col_power)] = 0.
        
        # Create diagonal matrices
        d_row_diag = sp.diags(d_row_power)
        d_col_diag = sp.diags(d_col_power)
        
        # Create normalized adjacency: R_γ = D_r^(-γ) R D_c^(-(1-γ))
        norm_adj = d_row_diag @ adj_mat @ d_col_diag
        
        # Name the view based on gamma value
        if gamma == 0.0:
            name = "raw"  # Column normalized only
        elif gamma == 0.5:
            name = "symmetric"  # Symmetric normalization
        elif gamma == 1.0:
            name = "row_norm"  # Row normalized only
        else:
            name = f"gamma_{gamma:.1f}"
        
        adjacency_views[name] = norm_adj
    
    return adjacency_views


def compute_similarity_matrix(matrix, similarity_type='cosine'):
    """
    Compute similarity matrix using cosine or jaccard similarity
    Following DySimGCF approach
    """
    if similarity_type == 'cosine':
        # Cosine similarity: dot product of normalized vectors
        norm_matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        similarity = norm_matrix @ norm_matrix.T
    elif similarity_type == 'jaccard':
        # Jaccard similarity: intersection over union for binary data
        matrix_bool = (matrix > 0).astype(float)
        intersection = matrix_bool @ matrix_bool.T
        sum_matrix = np.sum(matrix_bool, axis=1, keepdims=True)
        union = sum_matrix + sum_matrix.T - intersection
        similarity = intersection / (union + 1e-8)
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")
    
    return similarity


def create_similarity_based_adjacency(adj_mat, k_u=50, k_i=20, similarity_type='cosine'):
    """
    Create similarity-based adjacency matrix following DySimGCF approach
    """
    n_users, n_items = adj_mat.shape
    
    # Convert to dense if sparse for similarity computation
    if sp.issparse(adj_mat):
        adj_dense = adj_mat.toarray()
    else:
        adj_dense = adj_mat
    
    # Compute user-user similarity from interaction patterns
    user_sim_matrix = compute_similarity_matrix(adj_dense, similarity_type)
    
    # Compute item-item similarity from interaction patterns  
    item_sim_matrix = compute_similarity_matrix(adj_dense.T, similarity_type)
    
    # Apply top-k selection for users
    user_adj = np.zeros_like(user_sim_matrix)
    for i in range(n_users):
        # Get top-k similar users (excluding self)
        similarities = user_sim_matrix[i].copy()
        similarities[i] = -1  # Exclude self
        if k_u > 0:
            top_k_indices = np.argsort(similarities)[-min(k_u, len(similarities)-1):]
            user_adj[i, top_k_indices] = similarities[top_k_indices]
    
    # Apply top-k selection for items
    item_adj = np.zeros_like(item_sim_matrix)
    for i in range(n_items):
        # Get top-k similar items (excluding self)
        similarities = item_sim_matrix[i].copy()
        similarities[i] = -1  # Exclude self
        if k_i > 0:
            top_k_indices = np.argsort(similarities)[-min(k_i, len(similarities)-1):]
            item_adj[i, top_k_indices] = similarities[top_k_indices]
    
    # Create block adjacency matrix: [user_adj, 0; 0, item_adj]
    similarity_adj = sp.block_diag([sp.csr_matrix(user_adj), sp.csr_matrix(item_adj)])
    
    return similarity_adj, user_sim_matrix, item_sim_matrix


def apply_symmetric_softmax_attention(similarity_matrix):
    """
    Apply symmetric softmax normalization as in DySimGCF
    """
    # Convert to dense if sparse
    if sp.issparse(similarity_matrix):
        sim_dense = similarity_matrix.toarray()
    else:
        sim_dense = similarity_matrix.copy()
    
    # Avoid numerical issues
    sim_dense = np.clip(sim_dense, -10, 10)
    
    # Apply softmax normalization
    exp_sim = np.exp(sim_dense - np.max(sim_dense, axis=1, keepdims=True))
    
    # Symmetric normalization: geometric mean of row and column normalization
    row_sums = np.sum(exp_sim, axis=1, keepdims=True) + 1e-8
    col_sums = np.sum(exp_sim, axis=0, keepdims=True) + 1e-8
    
    # Geometric mean normalization
    normalized = exp_sim / np.sqrt(row_sums * col_sums)
    
    return sp.csr_matrix(normalized)


class SimplifiedSpectralCF(object):
    """Simplified Spectral CF with PolyCF-style Multi-view Normalization"""
    
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat.astype(np.float32) if sp.issparse(adj_mat) else adj_mat
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.use_full_eigen = self.config.get('use_full_eigen', True)
        self.n_eigen_item = self.config.get('n_eigen_item', 128)
        self.n_eigen_user = self.config.get('n_eigen_user', 128)
        self.filter_order = self.config.get('filter_order', 6)
        self.use_user_gram = self.config.get('use_user_gram', True)
        self.use_item_gram = self.config.get('use_item_gram', True)
        
        # PolyCF-style gamma values for different adjacency views
        self.gamma_values = self.config.get('gamma_values', [0.0, 0.5, 1.0])
        
        self.n_users, self.n_items = self.adj_mat.shape
        
        print(f"SimplifiedSpectralCF: {self.n_users:,} users, {self.n_items:,} items")
        print(f"User gram: {self.use_user_gram}, Item gram: {self.use_item_gram}")
        print(f"Gamma values: {self.gamma_values}")
    
    def _compute_eigendecomposition(self, gram_matrix, matrix_type='item'):
        """Compute eigendecomposition for a gram matrix"""
        if self.use_full_eigen:
            try:
                eigenvals, eigenvecs = eigsh(gram_matrix, k=self.n_items-2, which='LM')
                eigenvals = np.maximum(eigenvals, 0.0)
                n_eigen = len(eigenvals)
                method = "Full eigenspace (sparse)"
            except:
                try:
                    gram_dense = gram_matrix.toarray() if sp.issparse(gram_matrix) else gram_matrix
                    eigenvals, eigenvecs = np.linalg.eigh(gram_dense)
                    idx = np.argsort(eigenvals)[::-1]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
                    eigenvals = np.maximum(eigenvals, 0.0)
                    n_eigen = len(eigenvals)
                    method = "Full eigenspace (dense)"
                except:
                    k = min(128, self.n_items - 2)
                    eigenvals = np.linspace(0.1, 1.0, k)
                    eigenvecs = np.eye(self.n_items, k)
                    n_eigen = k
                    method = "Fallback"
        else:
            if matrix_type == 'item':
                k = min(self.n_eigen_item, self.n_items - 2)
            else:
                k = min(self.n_eigen_user, self.n_items - 2)
            
            try:
                eigenvals, eigenvecs = eigsh(gram_matrix, k=k, which='LM')
                eigenvals = np.maximum(eigenvals, 0.0)
                n_eigen = k
                method = f"Truncated ({k})"
            except:
                eigenvals = np.linspace(0.1, 1.0, k)
                eigenvecs = np.eye(self.n_items, k)
                n_eigen = k
                method = f"Fallback ({k})"
        
        return eigenvals, eigenvecs, n_eigen, method
    
    def train(self):
        """Training: create multiple adjacency views, compute gram matrices"""
        start = time.time()
        
        print(f"Creating {len(self.gamma_values)} adjacency views...")
        adjacency_views = create_polycf_adjacency_views(self.adj_mat, self.gamma_values)
        
        self.gram_matrices = {}
        
        # For each adjacency view, create gram matrices
        for view_name, norm_adj in adjacency_views.items():
            if self.use_item_gram:
                item_gram = norm_adj.T @ norm_adj
                key = f'item_gram_{view_name}'
                self.gram_matrices[key] = item_gram
            
            if self.use_user_gram:
                user_gram = norm_adj @ norm_adj.T
                projected_user_gram = norm_adj.T @ user_gram @ norm_adj
                key = f'user_gram_{view_name}'
                self.gram_matrices[key] = projected_user_gram
        
        print(f"Created {len(self.gram_matrices)} gram matrices")
        
        # Compute eigendecompositions
        self.eigendata = {}
        self.filters = {}
        
        for name, gram_matrix in self.gram_matrices.items():
            matrix_type = 'item' if 'item_gram' in name else 'user'
            eigenvals, eigenvecs, n_eigen, method = self._compute_eigendecomposition(gram_matrix, matrix_type)
            
            print(f"  {name}: {method} ({n_eigen} eigenvalues)")
            
            self.eigendata[name] = {
                'eigenvals': torch.tensor(eigenvals, dtype=torch.float32, device=self.device),
                'eigenvecs': torch.tensor(eigenvecs, dtype=torch.float32, device=self.device)
            }
            
            self.filters[name] = ChebyshevFilter(self.filter_order).to(self.device)
        
        # Combination weights
        n_matrices = len(self.gram_matrices)
        self.combination_weights = nn.Parameter(torch.ones(n_matrices, device=self.device))
        
        # Use symmetric view for predictions
        if 'symmetric' in adjacency_views:
            prediction_adj = adjacency_views['symmetric']
        else:
            prediction_adj = list(adjacency_views.values())[0]
        
        self.norm_adj = torch.tensor(prediction_adj.toarray(), dtype=torch.float32, device=self.device)
        
        print(f'SimplifiedSpectralCF training completed in {time.time() - start:.2f}s')
    
    def getUsersRating(self, batch_users, ds_name=None):
        """Get user ratings using multi-view spectral filtering"""
        if isinstance(batch_users, np.ndarray):
            batch_users = torch.LongTensor(batch_users).to(self.device)
        
        base_scores = self.norm_adj[batch_users]
        all_filtered_scores = []
        
        for name, eigen_data in self.eigendata.items():
            eigenvals = eigen_data['eigenvals']
            eigenvecs = eigen_data['eigenvecs']
            
            filter_response = self.filters[name](eigenvals)
            filter_matrix = eigenvecs @ torch.diag(filter_response) @ eigenvecs.t()
            
            filtered_scores = base_scores @ filter_matrix
            all_filtered_scores.append(filtered_scores)
        
        if len(all_filtered_scores) > 1:
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = sum(w * s for w, s in zip(weights, all_filtered_scores))
        else:
            predicted = all_filtered_scores[0]
        
        return predicted.detach().cpu().numpy()


class DySimSpectralCF(object):
    """DySimSpectralCF: Similarity-Centric Spectral Collaborative Filtering"""
    
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat.astype(np.float32) if sp.issparse(adj_mat) else adj_mat
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.use_full_eigen = self.config.get('use_full_eigen', True)
        self.n_eigen_item = self.config.get('n_eigen_item', 128)
        self.n_eigen_user = self.config.get('n_eigen_user', 128)
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
        
        self.n_users, self.n_items = self.adj_mat.shape
        
        print(f"DySimSpectralCF: {self.n_users:,} users, {self.n_items:,} items")
        print(f"Similarity: {self.similarity_type}, Top-k: users={self.k_u}, items={self.k_i}")
        print(f"Attention: {self.use_attention}")
    
    def _compute_eigendecomposition(self, similarity_matrix, matrix_type='similarity'):
        """Compute eigendecomposition for similarity-based matrix"""
        max_size = similarity_matrix.shape[0]
        
        if self.use_full_eigen:
            try:
                eigenvals, eigenvecs = eigsh(similarity_matrix, k=max_size-2, which='LM')
                eigenvals = np.maximum(eigenvals, 0.0)
                method = "Full eigenspace"
            except:
                try:
                    sim_dense = similarity_matrix.toarray() if sp.issparse(similarity_matrix) else similarity_matrix
                    eigenvals, eigenvecs = np.linalg.eigh(sim_dense)
                    idx = np.argsort(eigenvals)[::-1]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
                    eigenvals = np.maximum(eigenvals, 0.0)
                    method = "Full eigenspace (dense)"
                except:
                    k = min(64, max_size - 2)
                    eigenvals = np.linspace(0.1, 1.0, k)
                    eigenvecs = np.eye(max_size, k)
                    method = "Fallback"
        else:
            if matrix_type == 'user':
                k = min(self.n_eigen_user, max_size - 2)
            elif matrix_type == 'item':
                k = min(self.n_eigen_item, max_size - 2)
            else:
                k = min(64, max_size - 2)
            
            try:
                eigenvals, eigenvecs = eigsh(similarity_matrix, k=k, which='LM')
                eigenvals = np.maximum(eigenvals, 0.0)
                method = f"Truncated ({k})"
            except:
                eigenvals = np.linspace(0.1, 1.0, k)
                eigenvecs = np.eye(max_size, k)
                method = f"Fallback ({k})"
        
        return eigenvals, eigenvecs, len(eigenvals), method
    
    def train(self):
        """Training: create similarity-based graphs, compute eigendecomposition and setup filters"""
        start = time.time()
        
        print("Creating similarity-based adjacency matrices...")
        
        # Step 1: Create similarity-based adjacency matrix (DySimGCF approach)
        similarity_adj, user_sim_matrix, item_sim_matrix = create_similarity_based_adjacency(
            self.adj_mat, self.k_u, self.k_i, self.similarity_type
        )
        
        # Step 2: Apply attention mechanism if enabled
        if self.use_attention:
            print("Applying symmetric softmax attention...")
            try:
                similarity_adj = apply_symmetric_softmax_attention(similarity_adj)
            except Exception as e:
                print(f"Warning: Attention mechanism failed ({e}), using raw similarities")
                # Continue without attention if it fails
        else:
            print("Skipping attention mechanism (disabled)")
        
        # Step 3: Store different similarity matrices for spectral analysis
        self.similarity_matrices = {}
        
        if self.use_user_similarity:
            # User similarity submatrix
            user_block = similarity_adj[:self.n_users, :self.n_users]
            # Ensure user_block is not empty
            if user_block.nnz > 0 or not sp.issparse(user_block):
                self.similarity_matrices['user_similarity'] = user_block
            else:
                print("Warning: Empty user similarity matrix, skipping")
                
        if self.use_item_similarity:
            # Item similarity submatrix
            item_block = similarity_adj[self.n_users:, self.n_users:]
            # Ensure item_block is not empty
            if item_block.nnz > 0 or not sp.issparse(item_block):
                self.similarity_matrices['item_similarity'] = item_block
            else:
                print("Warning: Empty item similarity matrix, skipping")
        
        if self.use_original_interactions:
            # Original user-item interactions for comparison
            # Create gram matrices from original interactions
            item_gram = self.adj_mat.T @ self.adj_mat
            user_gram_projected = self.adj_mat.T @ (self.adj_mat @ self.adj_mat.T) @ self.adj_mat
            
            self.similarity_matrices['item_gram_original'] = item_gram
            self.similarity_matrices['user_gram_original'] = user_gram_projected
        
        # Ensure we have at least one matrix
        if not self.similarity_matrices:
            print("Warning: No similarity matrices created, using original interactions")
            item_gram = self.adj_mat.T @ self.adj_mat
            self.similarity_matrices['item_gram_fallback'] = item_gram
        
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
            
            eigenvals, eigenvecs, n_eigen, method = self._compute_eigendecomposition(sim_matrix, matrix_type)
            
            print(f"  {name}: {method} ({n_eigen} eigenvalues)")
            
            self.eigendata[name] = {
                'eigenvals': torch.tensor(eigenvals, dtype=torch.float32, device=self.device),
                'eigenvecs': torch.tensor(eigenvecs, dtype=torch.float32, device=self.device)
            }
            
            self.filters[name] = ChebyshevFilter(self.filter_order).to(self.device)
        
        # Combination weights
        n_matrices = len(self.similarity_matrices)
        self.combination_weights = nn.Parameter(torch.ones(n_matrices, device=self.device))
        
        # Base adjacency for predictions
        self.base_adj = torch.tensor(
            self.adj_mat.toarray() if sp.issparse(self.adj_mat) else self.adj_mat, 
            dtype=torch.float32, device=self.device
        )
        
        print(f'DySimSpectralCF training completed in {time.time() - start:.2f}s')
        print(f'Similarity matrices: {list(self.similarity_matrices.keys())}')
    
    def getUsersRating(self, batch_users, ds_name=None):
        """Get user ratings using similarity-enhanced spectral filtering"""
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


# For backward compatibility
UniversalSpectralCF = SimplifiedSpectralCF
DynamicSimilaritySpectralCF = DySimSpectralCF