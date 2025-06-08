#!/usr/bin/env python3
"""
Debug version to see what's going wrong with experiments
"""

import subprocess
import sys
import time

def debug_single_experiment():
    """Run a single experiment and show full output"""
    cmd = [
        sys.executable, "main.py",
        "--dataset", "ml-100k",
        "--n_eigen", "65", 
        "--n_hops", "2"
    ]
    
    print(f"Running debug command: {' '.join(cmd)}")
    print("="*80)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        # Try to find the final results line
        lines = result.stdout.split('\n')
        print(f"\nLooking for 'Final Results:' in {len(lines)} lines...")
        
        for i, line in enumerate(lines):
            if "Final Results:" in line:
                print(f"Found at line {i}: {line}")
                break
            elif "NDCG" in line or "Recall" in line:
                print(f"Related line {i}: {line}")
        else:
            print("No 'Final Results:' line found!")
            print("Last 10 lines:")
            for line in lines[-10:]:
                if line.strip():
                    print(f"  {line}")
                    
    except subprocess.TimeoutExpired:
        print("Command timed out!")
    except Exception as e:
        print(f"Error running command: {e}")

def test_import():
    """Test if we can import the model"""
    try:
        print("Testing imports...")
        
        # Test if the enhanced model can be imported
        test_cmd = [sys.executable, "-c", """
try:
    import world
    import model
    print('Basic imports successful')
    
    # Test if SVDEnhancedSpectralCF exists
    if hasattr(model, 'SVDEnhancedSpectralCF'):
        print('SVDEnhancedSpectralCF found')
    else:
        print('SVDEnhancedSpectralCF NOT found')
        print('Available in model module:')
        print([attr for attr in dir(model) if not attr.startswith('_')])
        
except Exception as e:
    print(f'Import error: {e}')
    import traceback
    traceback.print_exc()
"""]
        
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        print("Import test output:")
        print(result.stdout)
        if result.stderr:
            print("Import test errors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Import test failed: {e}")

def check_file_structure():
    """Check if all required files exist"""
    from pathlib import Path
    
    required_files = [
        "main.py",
        "model.py", 
        "world.py",
        "parse.py",
        "procedure.py",
        "dataloader.py",
        "register.py",
        "utils.py"
    ]
    
    print("Checking file structure:")
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"✓ {file} exists ({path.stat().st_size} bytes)")
        else:
            print(f"✗ {file} MISSING")
    
    # Check if SVD model is in the current model.py
    try:
        with open("model.py", "r") as f:
            content = f.read()
            if "SVDEnhancedSpectralCF" in content:
                print("✓ SVDEnhancedSpectralCF found in model.py")
            else:
                print("✗ SVDEnhancedSpectralCF NOT found in model.py")
                if "UniversalSpectralCF" in content:
                    print("  - UniversalSpectralCF found instead")
                else:
                    print("  - No SpectralCF classes found")
    except Exception as e:
        print(f"Error reading model.py: {e}")

if __name__ == "__main__":
    print("=== DEBUG EXPERIMENT RUNNER ===\n")
    
    print("1. Checking file structure...")
    check_file_structure()
    
    print("\n2. Testing imports...")
    test_import()
    
    print("\n3. Running single experiment...")
    debug_single_experiment()