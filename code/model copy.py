'''
Simplified Spectral CF with PolyCF-style Multi-view Normalization
Creates multiple differently normalized adjacency matrices first, then computes gram matrices
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


class SimplifiedSpectralCF(object):
    """Simplified Spectral CF with PolyCF-style Multi-view Normalization"""
    
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat.astype(np.float32) if sp.issparse(adj_mat) else adj_mat
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.use_full_eigen = self.config.get('use_full_eigen', True)  # True for PolyCF style
        self.n_eigen_item = self.config.get('n_eigen_item', 128)  # For item grams when not using full
        self.n_eigen_user = self.config.get('n_eigen_user', 128)  # For user grams when not using full
        self.filter_order = self.config.get('filter_order', 6)
        self.use_user_gram = self.config.get('use_user_gram', True)
        self.use_item_gram = self.config.get('use_item_gram', True)
        
        # PolyCF-style gamma values for different adjacency views
        self.gamma_values = self.config.get('gamma_values', [0.0, 0.5, 1.0])
        
        self.n_users, self.n_items = self.adj_mat.shape
        
        print(f"Simplified Spectral CF with PolyCF Normalization: {self.n_users:,} users, {self.n_items:,} items")
        print(f"User gram: {self.use_user_gram}, Item gram: {self.use_item_gram}")
        print(f"Gamma values for adjacency views: {self.gamma_values}")
        if self.use_full_eigen:
            print(f"Eigenspace: Full spectrum (PolyCF style)")
        else:
            print(f"Eigenspace: Item grams={self.n_eigen_item}, User grams={self.n_eigen_user}")
    
    def _compute_eigendecomposition(self, gram_matrix, matrix_type='item'):
        """Compute eigendecomposition for a gram matrix"""
        if self.use_full_eigen:
            # Use full eigenspace (PolyCF style)
            try:
                eigenvals, eigenvecs = eigsh(gram_matrix, k=self.n_items-2, which='LM')
                eigenvals = np.maximum(eigenvals, 0.0)
                n_eigen = len(eigenvals)
                method = "Full eigenspace (sparse)"
            except:
                # Fallback: use dense eigendecomposition for full spectrum
                try:
                    gram_dense = gram_matrix.toarray() if sp.issparse(gram_matrix) else gram_matrix
                    eigenvals, eigenvecs = np.linalg.eigh(gram_dense)
                    # Sort by descending eigenvalues
                    idx = np.argsort(eigenvals)[::-1]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
                    eigenvals = np.maximum(eigenvals, 0.0)
                    n_eigen = len(eigenvals)
                    method = "Full eigenspace (dense)"
                except:
                    # Last resort fallback
                    k = min(128, self.n_items - 2)
                    eigenvals = np.linspace(0.1, 1.0, k)
                    eigenvecs = np.eye(self.n_items, k)
                    n_eigen = k
                    method = "Fallback"
        else:
            # Use truncated eigenspace
            if matrix_type == 'item':
                k = min(self.n_eigen_item, self.n_items - 2)
            else:  # user
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
        """Training: create multiple adjacency views, compute gram matrices, eigendecomposition and setup filters"""
        start = time.time()
        
        # Step 1: Create multiple adjacency views with different gamma values (PolyCF approach)
        print(f"Creating {len(self.gamma_values)} adjacency views with gamma values: {self.gamma_values}")
        adjacency_views = create_polycf_adjacency_views(self.adj_mat, self.gamma_values)
        
        # Store gram matrices from different adjacency views
        self.gram_matrices = {}
        
        # Step 2: For each adjacency view, create gram matrices
        for view_name, norm_adj in adjacency_views.items():
            
            # Item Gram Matrix: R_γ^T @ R_γ (captures item-item relationships via shared users)
            if self.use_item_gram:
                item_gram = norm_adj.T @ norm_adj
                key = f'item_gram_{view_name}'
                self.gram_matrices[key] = item_gram
            
            # User Gram Matrix: R_γ @ R_γ^T (captures user-user relationships via shared items)
            # Project to item space for consistency: R_γ^T @ (R_γ @ R_γ^T) @ R_γ = R_γ^T @ R_γ @ R_γ^T @ R_γ
            if self.use_user_gram:
                user_gram = norm_adj @ norm_adj.T
                # Project user gram to item space
                projected_user_gram = norm_adj.T @ user_gram @ norm_adj
                key = f'user_gram_{view_name}'
                self.gram_matrices[key] = projected_user_gram
        
        print(f"Created {len(self.gram_matrices)} gram matrices from {len(adjacency_views)} adjacency views")
        
        # Step 3: Convert all matrices to tensors and compute eigendecompositions
        self.eigendata = {}
        self.filters = {}
        
        for name, gram_matrix in self.gram_matrices.items():
            # Determine if this is item or user gram
            matrix_type = 'item' if 'item_gram' in name else 'user'
            
            # Compute eigendecomposition
            eigenvals, eigenvecs, n_eigen, method = self._compute_eigendecomposition(gram_matrix, matrix_type)
            
            print(f"  {name}: {method} ({n_eigen} eigenvalues)")
            
            eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32, device=self.device)
            eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32, device=self.device)
            
            self.eigendata[name] = {
                'eigenvals': eigenvals_tensor,
                'eigenvecs': eigenvecs_tensor
            }
            
            # Create learnable Chebyshev filter for each gram matrix
            self.filters[name] = ChebyshevFilter(self.filter_order).to(self.device)
        
        # Step 4: Combination weights for different gram matrices
        n_matrices = len(self.gram_matrices)
        self.combination_weights = nn.Parameter(torch.ones(n_matrices, device=self.device))
        
        # Step 5: For prediction, we'll use the symmetric view (gamma=0.5) as default
        # or create a weighted combination of views
        if 'symmetric' in adjacency_views:
            prediction_adj = adjacency_views['symmetric']
        else:
            # Fallback to first view
            prediction_adj = list(adjacency_views.values())[0]
        
        self.norm_adj = torch.tensor(prediction_adj.toarray(), dtype=torch.float32, device=self.device)
        
        print(f'Training completed in {time.time() - start:.2f}s')
        print(f'Using {len(self.gram_matrices)} gram matrices with Chebyshev filters')
        print(f'Gram matrices: {list(self.gram_matrices.keys())}')
    
    def getUsersRating(self, batch_users, ds_name=None):
        """Get user ratings using multi-view spectral filtering"""
        if isinstance(batch_users, np.ndarray):
            batch_users = torch.LongTensor(batch_users).to(self.device)
        
        # Get base scores from adjacency (using symmetric view for prediction)
        base_scores = self.norm_adj[batch_users]
        
        all_filtered_scores = []
        
        # Apply each gram matrix with its filter
        for name, eigen_data in self.eigendata.items():
            eigenvals = eigen_data['eigenvals']
            eigenvecs = eigen_data['eigenvecs']
            
            # Apply Chebyshev filter
            filter_response = self.filters[name](eigenvals)
            filter_matrix = eigenvecs @ torch.diag(filter_response) @ eigenvecs.t()
            
            # Apply filtering
            filtered_scores = base_scores @ filter_matrix
            all_filtered_scores.append(filtered_scores)
        
        # Combine filtered scores with learned weights
        if len(all_filtered_scores) > 1:
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = sum(w * s for w, s in zip(weights, all_filtered_scores))
        else:
            predicted = all_filtered_scores[0]
        
        return predicted.detach().cpu().numpy()


# For backward compatibility with existing code
UniversalSpectralCF = SimplifiedSpectralCF