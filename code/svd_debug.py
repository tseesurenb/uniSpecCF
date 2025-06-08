#!/usr/bin/env python3
"""
Debug SVD-specific issues
"""

import subprocess
import sys

def debug_svd_failure():
    """Debug why SVD experiments are failing"""
    
    cmd = [
        sys.executable, "main.py",
        "--dataset", "ml-100k",
        "--n_eigen", "65",
        "--n_hops", "2",
        "--use_svd",
        "--epochs", "5"  # Quick test
    ]
    
    print(f"Debugging SVD failure with command: {' '.join(cmd)}")
    print("="*80)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"Return code: {result.returncode}")
        print("\nSTDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        
        if "Final Results:" in result.stdout:
            print("\n✓ Found Final Results - parsing should work")
        else:
            print("\n✗ No Final Results found - this is the issue")
            
        # Look for SVD-specific errors
        if "SVD" in result.stderr:
            print("✗ SVD-related error found in stderr")
        elif "sparse" in result.stderr.lower():
            print("✗ Sparse matrix related error")
        elif "import" in result.stderr.lower():
            print("✗ Import error detected")
        
    except subprocess.TimeoutExpired:
        print("Command timed out!")
    except Exception as e:
        print(f"Error running command: {e}")

def test_svd_imports():
    """Test if SVD-related imports work"""
    test_cmd = [sys.executable, "-c", """
import sys
try:
    from scipy.sparse.linalg import svds
    print('✓ scipy.sparse.linalg.svds import successful')
except ImportError as e:
    print(f'✗ svds import failed: {e}')

try:
    import scipy.sparse as sp
    import numpy as np
    # Test creating a simple sparse matrix and running SVD
    data = np.random.random((10, 5))
    sparse_mat = sp.csr_matrix(data)
    ut, s, vt = svds(sparse_mat, k=3)
    print('✓ Basic SVD computation successful')
except Exception as e:
    print(f'✗ Basic SVD test failed: {e}')
    import traceback
    traceback.print_exc()

try:
    # Test if our model can be instantiated
    import world
    import model
    print('✓ Model imports successful')
    
    # Check if SVD parameters are in config
    config = world.config
    svd_params = ['use_svd', 'n_svd', 'svd_weight']
    missing = [p for p in svd_params if p not in config]
    if missing:
        print(f'✗ Missing SVD parameters in config: {missing}')
    else:
        print('✓ All SVD parameters found in config')
        
except Exception as e:
    print(f'✗ Model/config error: {e}')
    import traceback
    traceback.print_exc()
"""]
    
    result = subprocess.run(test_cmd, capture_output=True, text=True)
    print("SVD Import Test Results:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

if __name__ == "__main__":
    print("=== SVD DEBUG ANALYSIS ===\n")
    
    print("1. Testing SVD-related imports...")
    test_svd_imports()
    
    print("\n2. Testing SVD experiment failure...")
    debug_svd_failure()