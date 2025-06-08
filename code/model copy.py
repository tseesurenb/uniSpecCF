'''
Enhanced Simple Universal Spectral CF with Adaptive Multi-hop Support
Addresses over-smoothing in higher hops and allows combining multiple hop patterns
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time


class SpectralFilter(nn.Module):
    """Enhanced learnable spectral filter with better initialization"""
    
    def __init__(self, filter_order=6, hop_count=2):
        super().__init__()
        # Hop-aware initialization - higher hops need less aggressive filtering
        if hop_count <= 2:
            init_coeffs = [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015]
        elif hop_count == 3:
            # Gentler filtering for 3-hop
            init_coeffs = [1.0, -0.3, 0.05, -0.01, 0.002, -0.0004, 0.00008]
        else:  # 4-hop
            # Very gentle filtering for 4-hop to preserve signal
            init_coeffs = [1.0, -0.1, 0.02, -0.004, 0.0008, -0.00016, 0.000032]
        
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(init_coeffs[:filter_order + 1]):
            coeffs_data[i] = val
        self.coeffs = nn.Parameter(coeffs_data)
    
    def forward(self, eigenvalues):
        """Apply polynomial spectral filter"""
        result = self.coeffs[0] * torch.ones_like(eigenvalues)
        if len(self.coeffs) > 1:
            eigen_power = eigenvalues.clone()
            for i in range(1, len(self.coeffs)):
                result += self.coeffs[i] * eigen_power
                if i < len(self.coeffs) - 1:
                    eigen_power = eigen_power * eigenvalues
        return torch.sigmoid(result) + 1e-6


class UniversalSpectralCF(object):
    """Enhanced Simple Universal Spectral CF with Adaptive Multi-hop Support"""
    
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat.astype(np.float32) if sp.issparse(adj_mat) else adj_mat
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.n_eigen = self.config.get('n_eigen', 128)
        self.filter_order = self.config.get('filter_order', 6)
        self.n_hops = self.config.get('n_hops', 2)
        self.combine_hops = self.config.get('combine_hops', False)  # New: combine multiple hops
        self.n_users, self.n_items = self.adj_mat.shape
        
        # Validate hop count
        if self.n_hops not in [1, 2, 3, 4]:
            raise ValueError(f"n_hops must be 1, 2, 3, or 4, got {self.n_hops}")
        
        print(f"Enhanced Spectral CF: {self.n_users:,} users, {self.n_items:,} items, {self.n_eigen} eigenvalues, {self.n_hops}-hop")
        if self.combine_hops:
            print(f"Combining hops 1 through {self.n_hops}")
    
    def _compute_hop_matrices(self, norm_adj):
        """Compute multi-hop adjacency matrices with normalization"""
        hop_matrices = {}
        
        if self.n_hops >= 1:
            # 1-hop: Direct adjacency A
            hop_matrices[1] = norm_adj
        
        if self.n_hops >= 2:
            # 2-hop: A A^T A (User → Item → User → Item)
            hop_matrices[2] = norm_adj @ norm_adj.T @ norm_adj
        
        if self.n_hops >= 3:
            # 3-hop: A A^T A A^T A (User → Item → User → Item → User → Item)
            hop_matrices[3] = hop_matrices[2] @ norm_adj.T @ norm_adj
        
        if self.n_hops >= 4:
            # 4-hop: A A^T A A^T A A^T A (extended propagation)
            hop_matrices[4] = hop_matrices[3] @ norm_adj.T @ norm_adj
        
        # Apply row normalization to prevent magnitude explosion in higher hops
        for hop in hop_matrices:
            matrix = hop_matrices[hop]
            row_sums = np.array(matrix.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            norm_matrix = matrix / row_sums[:, np.newaxis]
            hop_matrices[hop] = norm_matrix
        
        return hop_matrices
    
    def train(self):
        """Training: compute eigendecomposition and setup filters"""
        adj_mat = self.adj_mat
        start = time.time()
        
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum + 1e-8, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_row = sp.diags(d_inv_sqrt)
        
        colsum = np.array(adj_mat.sum(axis=0)).flatten()
        d_inv_sqrt = np.power(colsum + 1e-8, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_col = sp.diags(d_inv_sqrt)
        
        norm_adj = d_row @ adj_mat @ d_col
        self.norm_adj = torch.tensor(norm_adj.toarray(), dtype=torch.float32, device=self.device)
        
        # Compute multi-hop matrices
        self.hop_matrices = {}
        hop_matrices_np = self._compute_hop_matrices(norm_adj)
        
        for hop, matrix in hop_matrices_np.items():
            if hop <= self.n_hops:
                self.hop_matrices[hop] = torch.tensor(matrix.toarray(), dtype=torch.float32, device=self.device)
        
        # Choose matrix for eigendecomposition based on hop count
        if self.n_hops == 1:
            gram_matrix = norm_adj.T @ norm_adj
        elif self.n_hops == 2:
            two_hop_matrix = hop_matrices_np[2]
            gram_matrix = two_hop_matrix.T @ two_hop_matrix
        elif self.n_hops == 3:
            three_hop_matrix = hop_matrices_np[3]
            gram_matrix = three_hop_matrix.T @ three_hop_matrix
        else:  # 4-hop
            four_hop_matrix = hop_matrices_np[4]
            gram_matrix = four_hop_matrix.T @ four_hop_matrix
        
        k = min(self.n_eigen, self.n_items - 2)
        
        try:
            eigenvals, eigenvecs = eigsh(gram_matrix, k=k, which='LM')
            eigenvals = np.maximum(eigenvals, 0.0)
        except:
            eigenvals = np.linspace(0.1, 1.0, k)
            eigenvecs = np.eye(self.n_items, k)
        
        self.eigenvals = torch.tensor(eigenvals, dtype=torch.float32, device=self.device)
        self.eigenvecs = torch.tensor(eigenvecs, dtype=torch.float32, device=self.device)
        
        # Setup learnable filters - hop-aware initialization
        self.spectral_filter = SpectralFilter(self.filter_order, self.n_hops).to(self.device)
        
        if self.combine_hops:
            # Weights for combining multiple hops + filtered version
            n_components = self.n_hops + 1  # Each hop + filtered version
            init_weights = torch.ones(n_components, device=self.device)
            # Give more weight to lower hops initially
            for i in range(self.n_hops):
                init_weights[i] = 1.0 / (i + 1)
            init_weights[-1] = 0.5  # Filtered version
            self.combination_weights = nn.Parameter(init_weights)
        else:
            # Just base + filtered combination
            self.combination_weights = nn.Parameter(torch.tensor([0.6, 0.4], device=self.device))
        
        print(f'Training completed in {time.time() - start:.2f}s')
        print(f'Using {self.n_hops}-hop propagation pattern')
        if self.combine_hops:
            print(f'Combining multiple hop patterns')
    
    def getUsersRating(self, batch_users, ds_name=None):
        """Get user ratings with enhanced multi-hop support"""
        if isinstance(batch_users, np.ndarray):
            batch_users = torch.LongTensor(batch_users).to(self.device)
        
        scores = []
        
        if self.combine_hops:
            # Combine all hop patterns
            for hop in range(1, self.n_hops + 1):
                if hop in self.hop_matrices:
                    hop_scores = self.hop_matrices[hop][batch_users]
                    scores.append(hop_scores)
            
            # Add filtered version of the highest hop
            filter_response = self.spectral_filter(self.eigenvals)
            filter_matrix = self.eigenvecs @ torch.diag(filter_response) @ self.eigenvecs.t()
            filtered_scores = self.hop_matrices[self.n_hops][batch_users] @ filter_matrix
            scores.append(filtered_scores)
            
            # Combine with learned weights
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = sum(w * s for w, s in zip(weights, scores))
        else:
            # Original approach: base + filtered
            base_scores = self.hop_matrices[self.n_hops][batch_users]
            scores.append(base_scores)
            
            # Filtered scores
            filter_response = self.spectral_filter(self.eigenvals)
            filter_matrix = self.eigenvecs @ torch.diag(filter_response) @ self.eigenvecs.t()
            filtered_scores = base_scores @ filter_matrix
            scores.append(filtered_scores)
            
            # Combine with learned weights
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * scores[0] + weights[1] * scores[1]
        
        return predicted.detach().cpu().numpy()