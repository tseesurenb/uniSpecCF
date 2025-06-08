'''
Fixed SVD-Enhanced Multi-hop Spectral CF - Handles negative strides issue
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds as sparsesvd
from scipy.sparse.linalg import eigsh
import time


class SpectralFilter(nn.Module):
    """Learnable spectral filter"""
    
    def __init__(self, filter_order=6, hop_count=2):
        super().__init__()
        # Hop-aware initialization
        if hop_count <= 2:
            init_coeffs = [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015]
        elif hop_count == 3:
            init_coeffs = [1.0, -0.3, 0.05, -0.01, 0.002, -0.0004, 0.00008]
        else:  # 4-hop
            init_coeffs = [1.0, -0.1, 0.02, -0.004, 0.0008, -0.00016, 0.000032]
        
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(init_coeffs[:filter_order + 1]):
            coeffs_data[i] = val
        self.coeffs = nn.Parameter(coeffs_data)
    
    def forward(self, values):
        """Apply polynomial filter to eigenvalues or singular values"""
        result = self.coeffs[0] * torch.ones_like(values)
        if len(self.coeffs) > 1:
            value_power = values.clone()
            for i in range(1, len(self.coeffs)):
                result += self.coeffs[i] * value_power
                if i < len(self.coeffs) - 1:
                    value_power = value_power * values
        return torch.sigmoid(result) + 1e-6


class SVDEnhancedSpectralCF(object):
    """Fixed SVD-Enhanced Multi-hop Spectral CF"""
    
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat.astype(np.float32) if sp.issparse(adj_mat) else adj_mat
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.n_eigen = self.config.get('n_eigen', 128)
        self.n_svd = self.config.get('n_svd', 256)
        self.filter_order = self.config.get('filter_order', 6)
        self.n_hops = self.config.get('n_hops', 2)
        self.combine_hops = self.config.get('combine_hops', False)
        self.use_svd = self.config.get('use_svd', False)
        self.svd_weight = self.config.get('svd_weight', 0.3)
        self.n_users, self.n_items = self.adj_mat.shape
        
        print(f"SVD-Enhanced Spectral CF: {self.n_users:,} users, {self.n_items:,} items")
        print(f"Config: {self.n_hops}-hop" + (" combined" if self.combine_hops else "") + 
              (f", SVD-{self.n_svd}" if self.use_svd else ""))
    
    def _compute_hop_matrices(self, norm_adj):
        """Compute multi-hop adjacency matrices"""
        hop_matrices = {}
        
        if self.n_hops >= 1:
            hop_matrices[1] = norm_adj
        
        if self.n_hops >= 2:
            hop_matrices[2] = norm_adj @ norm_adj.T @ norm_adj
        
        if self.n_hops >= 3:
            hop_matrices[3] = hop_matrices[2] @ norm_adj.T @ norm_adj
        
        if self.n_hops >= 4:
            hop_matrices[4] = hop_matrices[3] @ norm_adj.T @ norm_adj
        
        # Apply row normalization for higher hops
        for hop in hop_matrices:
            if hop > 2:
                matrix = hop_matrices[hop]
                row_sums = np.array(matrix.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1
                norm_matrix = matrix / row_sums[:, np.newaxis]
                hop_matrices[hop] = norm_matrix
        
        return hop_matrices
    
    def _compute_svd_components(self, norm_adj):
        """Compute SVD components with proper array copying to handle negative strides"""
        if not self.use_svd:
            return
            
        start_svd = time.time()
        
        try:
            # Convert to CSC for efficient SVD
            norm_adj_csc = norm_adj.tocsc()
            
            # Compute truncated SVD
            k = min(self.n_svd, min(self.n_users, self.n_items) - 1)
            ut, s, vt = sparsesvd(norm_adj_csc, k=k)
            
            print(f"SVD computed: {k} components in {time.time() - start_svd:.2f}s")
            
            # FIX: Make copies to handle negative strides
            ut_copy = np.array(ut.T, copy=True)  # Transpose and copy
            s_copy = np.array(s, copy=True)      # Copy singular values
            vt_copy = np.array(vt, copy=True)    # Copy V^T
            
            # Store SVD components with proper copies
            self.svd_ut = torch.tensor(ut_copy, dtype=torch.float32, device=self.device)  # users x k
            self.svd_s = torch.tensor(s_copy, dtype=torch.float32, device=self.device)    # k
            self.svd_vt = torch.tensor(vt_copy, dtype=torch.float32, device=self.device)  # k x items
            
            # Compute item degree matrices (similar to GF-CF)
            colsum = np.array(self.adj_mat.sum(axis=0)).flatten()
            d_inv = np.power(colsum + 1e-8, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            
            # Create diagonal matrices as dense (easier for small matrices)
            d_mat_i_diag = np.zeros(self.n_items)
            d_mat_i_diag[:len(d_inv)] = d_inv
            self.d_mat_i_diag = torch.tensor(d_mat_i_diag, dtype=torch.float32, device=self.device)
            
            d_inv_inv = np.power(colsum + 1e-8, 0.5).flatten()
            d_inv_inv[np.isinf(d_inv_inv)] = 0.
            d_mat_i_inv_diag = np.zeros(self.n_items)
            d_mat_i_inv_diag[:len(d_inv_inv)] = d_inv_inv
            self.d_mat_i_inv_diag = torch.tensor(d_mat_i_inv_diag, dtype=torch.float32, device=self.device)
            
        except Exception as e:
            print(f"SVD computation failed: {e}")
            print("Disabling SVD for this run")
            self.use_svd = False
    
    def train(self):
        """Training: compute hop matrices, eigendecomposition, and SVD"""
        start = time.time()
        
        # Symmetric normalization
        rowsum = np.array(self.adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum + 1e-8, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_row = sp.diags(d_inv_sqrt)
        
        colsum = np.array(self.adj_mat.sum(axis=0)).flatten()
        d_inv_sqrt = np.power(colsum + 1e-8, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_col = sp.diags(d_inv_sqrt)
        
        norm_adj = d_row @ self.adj_mat @ d_col
        self.norm_adj = torch.tensor(norm_adj.toarray(), dtype=torch.float32, device=self.device)
        
        # Store original adjacency for SVD reconstruction
        self.orig_adj = torch.tensor(self.adj_mat.toarray(), dtype=torch.float32, device=self.device)
        
        # Compute multi-hop matrices
        self.hop_matrices = {}
        hop_matrices_np = self._compute_hop_matrices(norm_adj)
        
        for hop, matrix in hop_matrices_np.items():
            if hop <= self.n_hops:
                self.hop_matrices[hop] = torch.tensor(matrix.toarray(), dtype=torch.float32, device=self.device)
        
        # Compute SVD components if enabled
        if self.use_svd:
            self._compute_svd_components(norm_adj)
        
        # Eigendecomposition for spectral filtering
        if self.n_hops == 1:
            gram_matrix = norm_adj.T @ norm_adj
        else:
            target_matrix = hop_matrices_np[self.n_hops]
            gram_matrix = target_matrix.T @ target_matrix
        
        k = min(self.n_eigen, self.n_items - 2)
        
        try:
            eigenvals, eigenvecs = eigsh(gram_matrix, k=k, which='LM')
            eigenvals = np.maximum(eigenvals, 0.0)
            # FIX: Copy eigenvalues and eigenvectors to handle negative strides
            eigenvals_copy = np.array(eigenvals, copy=True)
            eigenvecs_copy = np.array(eigenvecs, copy=True)
        except:
            eigenvals_copy = np.linspace(0.1, 1.0, k)
            eigenvecs_copy = np.eye(self.n_items, k)
        
        self.eigenvals = torch.tensor(eigenvals_copy, dtype=torch.float32, device=self.device)
        self.eigenvecs = torch.tensor(eigenvecs_copy, dtype=torch.float32, device=self.device)
        
        # Setup learnable filters
        self.spectral_filter = SpectralFilter(self.filter_order, self.n_hops).to(self.device)
        
        if self.use_svd:
            # Filter for SVD singular values
            self.svd_filter = SpectralFilter(self.filter_order, 1).to(self.device)
        
        # Setup combination weights
        n_components = 2  # base + filtered
        if self.combine_hops:
            n_components = self.n_hops + 1  # each hop + filtered
        if self.use_svd:
            n_components += 1  # add SVD component
            
        init_weights = torch.ones(n_components, device=self.device)
        if self.combine_hops:
            # Initialize based on experimental findings
            for i in range(self.n_hops):
                init_weights[i] = 1.0 / (i + 1)
            init_weights[self.n_hops] = 0.4  # filtered version
            if self.use_svd:
                init_weights[-1] = self.svd_weight  # SVD component
        else:
            init_weights[0] = 0.6  # base hop
            init_weights[1] = 0.4  # filtered hop
            if self.use_svd:
                init_weights[0] = 0.5  # reduce base
                init_weights[1] = 0.3  # reduce filtered  
                init_weights[2] = self.svd_weight  # SVD
        
        self.combination_weights = nn.Parameter(init_weights)
        
        print(f'Training completed in {time.time() - start:.2f}s')
        print(f'Components: {"hop-combined" if self.combine_hops else "single-hop"}'
              f'{" + SVD" if self.use_svd else ""}')
    
    def _compute_svd_scores(self, batch_users):
        """Compute SVD-based scores similar to GF-CF"""
        if not self.use_svd:
            return torch.zeros(len(batch_users), self.n_items, device=self.device)
            
        # Get original user-item interactions for batch
        user_items = self.orig_adj[batch_users]  # batch_size x n_items
        
        # SVD reconstruction with diagonal multiplication
        # Apply learnable filtering to singular values
        filtered_s = self.svd_filter(self.svd_s)
        
        # Reconstruct: batch_users -> items via filtered SVD
        # user_items @ D_i @ V^T @ diag(filtered_s) @ V @ D_i^(-1)
        step1 = user_items * self.d_mat_i_diag.unsqueeze(0)  # Apply D_i
        step2 = step1 @ self.svd_vt.T                        # @ V^T
        step3 = step2 * filtered_s.unsqueeze(0)              # @ diag(filtered_s)
        step4 = step3 @ self.svd_vt                          # @ V
        svd_scores = step4 * self.d_mat_i_inv_diag.unsqueeze(0)  # Apply D_i^(-1)
        
        return svd_scores
    
    def getUsersRating(self, batch_users, ds_name=None):
        """Get user ratings with SVD enhancement"""
        if isinstance(batch_users, np.ndarray):
            batch_users = torch.LongTensor(batch_users).to(self.device)
        
        scores = []
        
        if self.combine_hops:
            # Add all hop patterns
            for hop in range(1, self.n_hops + 1):
                if hop in self.hop_matrices:
                    hop_scores = self.hop_matrices[hop][batch_users]
                    scores.append(hop_scores)
            
            # Add filtered version of highest hop
            filter_response = self.spectral_filter(self.eigenvals)
            filter_matrix = self.eigenvecs @ torch.diag(filter_response) @ self.eigenvecs.t()
            filtered_scores = self.hop_matrices[self.n_hops][batch_users] @ filter_matrix
            scores.append(filtered_scores)
            
        else:
            # Standard: base + filtered
            base_scores = self.hop_matrices[self.n_hops][batch_users]
            scores.append(base_scores)
            
            filter_response = self.spectral_filter(self.eigenvals)
            filter_matrix = self.eigenvecs @ torch.diag(filter_response) @ self.eigenvecs.t()
            filtered_scores = base_scores @ filter_matrix
            scores.append(filtered_scores)
        
        # Add SVD component if enabled
        if self.use_svd:
            svd_scores = self._compute_svd_scores(batch_users)
            scores.append(svd_scores)
        
        # Combine all components with learned weights
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * s for w, s in zip(weights, scores))
        
        return predicted.detach().cpu().numpy()