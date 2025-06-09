# SimplifiedSpectralCF with PolyCF Normalization - Usage Examples

## 🚀 PolyCF Multi-View Approach

This implementation follows the **PolyCF normalization approach**: create multiple differently normalized adjacency matrices **first**, then compute gram matrices from each view. This avoids double-normalization and captures diverse collaborative patterns.

### Mathematical Foundation

**PolyCF Adjacency Views:**
- **R_γ = D_r^(-γ) R D_c^(-(1-γ))**

**Different γ values capture different patterns:**
- **γ=0.0**: Column normalization (emphasizes popular items)
- **γ=0.5**: Symmetric normalization (balanced view)  
- **γ=1.0**: Row normalization (user-centric view)

## 📊 Basic Usage Examples

### Default PolyCF (Recommended)
```bash
# Classic PolyCF: Raw + Symmetric + Row views (γ = 0.0, 0.5, 1.0)
python main.py --dataset ml-100k --epochs 50

# With full eigenspace (most accurate)
python main.py --dataset ml-100k --use_full_eigen --epochs 50
```

### Single Gamma Ablation Studies
```bash
# Raw view only (γ=0.0) - emphasizes popular items
python main.py --dataset ml-100k --single_gamma 0.0 --epochs 50

# Symmetric view only (γ=0.5) - balanced normalization
python main.py --dataset ml-100k --single_gamma 0.5 --epochs 50

# Row-normalized view only (γ=1.0) - user-centric
python main.py --dataset ml-100k --single_gamma 1.0 --epochs 50
```

### Custom Gamma Combinations
```bash
# Fine-grained around symmetric
python main.py --dataset ml-100k --gamma_values 0.25 0.5 0.75 --epochs 50

# Popular + balanced view
python main.py --dataset ml-100k --gamma_values 0.0 0.5 --epochs 50

# Balanced + user-centric
python main.py --dataset ml-100k --gamma_values 0.5 1.0 --epochs 50

# Extended range
python main.py --dataset ml-100k --gamma_values 0.0 0.3 0.6 0.9 --epochs 50
```

## 🔬 Comprehensive Gamma Study

### Run Complete Ablation Study
```bash
# Test all gamma combinations on ML-100K
python polycf_gamma_study.py --dataset ml-100k --runs_per_config 3

# Quick study
python polycf_gamma_study.py --dataset ml-100k --runs_per_config 1 --epochs 20

# Different dataset
python polycf_gamma_study.py --dataset gowalla --epochs 30
```

## 🎯 Dataset-Specific Recommendations

### Small Datasets (ML-100K, LastFM)
```bash
# Use full eigenspace for maximum accuracy
python main.py --dataset ml-100k --use_full_eigen --epochs 50
python main.py --dataset lastfm --use_full_eigen --epochs 40

# Test different gamma ranges
python main.py --dataset ml-100k --gamma_values 0.1 0.3 0.5 0.7 0.9 --use_full_eigen --epochs 50
```

### Medium Datasets (Gowalla)
```bash
# Balanced approach
python main.py --dataset gowalla --n_eigen_item 256 --n_eigen_user 128 --epochs 30

# Custom gamma for collaborative patterns
python main.py --dataset gowalla --gamma_values 0.0 0.4 0.8 --n_eigen_item 256 --epochs 30
```

### Large Datasets (Yelp2018, Amazon-Book)
```bash
# Efficiency-focused with truncated eigenspace
python main.py --dataset yelp2018 --n_eigen_item 256 --n_eigen_user 256 --epochs 30

# Popular items focus for large datasets
python main.py --dataset amazon-book --gamma_values 0.0 0.3 --n_eigen_item 512 --epochs 30
```

## ⚡ Efficiency Controls

### Gram Matrix Selection
```bash
# Item grams only (faster, good for item-item patterns)
python main.py --dataset yelp2018 --no_user_gram --n_eigen_item 256 --epochs 30

# User grams only (user-user patterns)
python main.py --dataset gowalla --no_item_gram --n_eigen_user 256 --epochs 30

# Both grams with custom gamma
python main.py --dataset ml-100k --gamma_values 0.2 0.6 --use_full_eigen --epochs 50
```

### Eigenspace Control
```bash
# Memory efficient
python main.py --dataset yelp2018 --n_eigen_item 128 --n_eigen_user 64 --epochs 30

# High accuracy
python main.py --dataset gowalla --n_eigen_item 400 --n_eigen_user 300 --epochs 30

# Full spectrum (PolyCF style)
python main.py --dataset ml-100k --use_full_eigen --epochs 50
```

## 🔍 Comparison Studies

### PolyCF vs Single Views
```bash
# Compare all approaches
python main.py --dataset ml-100k --gamma_values 0.0 0.5 1.0 --epochs 50    # PolyCF combination
python main.py --dataset ml-100k --single_gamma 0.0 --epochs 50             # Raw only  
python main.py --dataset ml-100k --single_gamma 0.5 --epochs 50             # Symmetric only
python main.py --dataset ml-100k --single_gamma 1.0 --epochs 50             # Row-norm only
```

### Gram Matrix Importance
```bash
# Test different configurations with best gamma
python main.py --dataset ml-100k --no_user_gram --single_gamma 0.5 --epochs 50    # Item grams only
python main.py --dataset ml-100k --no_item_gram --single_gamma 0.5 --epochs 50    # User grams only  
python main.py --dataset ml-100k --single_gamma 0.5 --epochs 50                   # Both grams
```

### Filter Order Impact
```bash
# Different Chebyshev filter orders with PolyCF
python main.py --dataset ml-100k --filter_order 4 --gamma_values 0.0 0.5 1.0 --epochs 50
python main.py --dataset ml-100k --filter_order 8 --gamma_values 0.0 0.5 1.0 --epochs 50
```

## 🧪 Advanced Experimental Setups

### Fine-grained Gamma Analysis
```bash
# Test many gamma values to find optimal range
python main.py --dataset ml-100k --gamma_values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --epochs 50

# Focus on best-performing region  
python main.py --dataset ml-100k --gamma_values 0.3 0.4 0.5 0.6 0.7 --use_full_eigen --epochs 50
```

### Domain-specific Patterns
```bash
# Music dataset (LastFM) - user taste similarity important
python main.py --dataset lastfm --gamma_values 0.6 0.8 1.0 --use_full_eigen --epochs 40

# Location dataset (Gowalla) - geographic patterns
python main.py --dataset gowalla --gamma_values 0.0 0.3 0.6 --n_eigen_item 256 --epochs 30

# E-commerce (Amazon) - popular items matter
python main.py --dataset amazon-book --gamma_values 0.0 0.2 0.4 --n_eigen_item 512 --epochs 30
```

## 📈 Performance Optimization

### Quick Testing
```bash
# Fast convergence testing
python main.py --dataset ml-100k --gamma_values 0.0 0.5 1.0 --epochs 10 --n_epoch_eval 2

# Single gamma quick test
python main.py --dataset ml-100k --single_gamma 0.5 --epochs 15 --n_epoch_eval 3
```

### Production Settings
```bash
# Balanced accuracy/efficiency for production
python main.py --dataset gowalla --gamma_values 0.0 0.5 1.0 --n_eigen_item 200 --n_eigen_user 150 --epochs 30

# High-accuracy production
python main.py --dataset ml-100k --use_full_eigen --gamma_values 0.0 0.3 0.6 0.9 --epochs 50
```

## 🎯 Key Insights from PolyCF Approach

**Expected Results:**
1. **γ=0.0 (Raw)**: Good for capturing popular item patterns
2. **γ=0.5 (Symmetric)**: Balanced performance, often best single view
3. **γ=1.0 (Row-norm)**: Good for personalization, user-centric
4. **Combinations**: Should outperform single views by capturing complementary patterns

**Why This Works Better:**
- **No double-normalization**: Each view is properly normalized once
- **Diverse patterns**: Different γ values capture different collaborative signals
- **Learned combination**: Model learns optimal weighting of different views
- **Spectral diversity**: Each view contributes different eigenspace information