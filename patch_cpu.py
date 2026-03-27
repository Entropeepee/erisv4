path = 'eris/config.py'
with open(path, 'r') as f:
    content = f.read()

old = 'try:\n    import cupy as cp\n    from cupy import fft as cupyfft'
new = 'try:\n    raise ImportError("Forced CPU: CUDA 13.2 FP8 headers broken on Blackwell")\n    import cupy as cp\n    from cupy import fft as cupyfft'

if 'Forced CPU' not in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('Patched config.py - CPU mode enabled')
else:
    print('Already patched')