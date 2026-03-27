# Run once: python fix_cupy.py
# Tests if CuPy works, and if not, patches config.py to fall back gracefully

import os
print("Testing CuPy compilation...")

try:
    import cupy as cp
    # This triggers JIT compilation — if it fails, we know
    a = cp.array([1.0, 2.0, 3.0])
    b = cp.mean(a)
    print(f"CuPy works! mean = {float(b)}")
    print("No fix needed.")
except Exception as e:
    print(f"CuPy compilation failed: {type(e).__name__}")
    print("Patching config.py to force CPU mode until CUDA is fixed...")
    
    config_path = os.path.join("eris", "config.py")
    with open(config_path, "r") as f:
        content = f.read()
    
    # Add a force-CPU flag at the top
    patch = '''# CUDA 13.2 FP8 header bug workaround — force CPU until fixed
# Remove this line when CuPy/CUDA is updated
FORCE_CPU = True

'''
    if "FORCE_CPU" not in content:
        # Replace the try block to check FORCE_CPU first
        old = "try:\n    import cupy as cp"
        new = "try:\n    if FORCE_CPU:\n        raise ImportError('Forced CPU mode')\n    import cupy as cp"
        content = patch + content.replace(old, new)
        
        with open(config_path, "w") as f:
            f.write(content)
        print("Patched! System will run on CPU (NumPy) until CUDA headers are fixed.")
        print("To re-enable GPU: remove FORCE_CPU = True from eris/config.py")