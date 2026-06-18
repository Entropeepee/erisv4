import os
import re

def fix(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        code = f.read()
    orig = code
    # Replace import
    if 'from eris.config import' not in code:
        code = code.replace('import numpy as np', 'import numpy as np\nfrom eris.config import to_numpy, xp')
    # Replace specific bad conversions
    code = code.replace('np.asarray(self.field', 'to_numpy(self.field')
    code = code.replace('np.asarray(f1.phi)', 'to_numpy(f1.phi)')
    code = code.replace('np.asarray(f2.phi)', 'to_numpy(f2.phi)')
    code = code.replace('np.asarray(field.phi)', 'to_numpy(field.phi)')
    code = code.replace('np.asarray(restored.phi)', 'to_numpy(restored.phi)')
    code = code.replace('np.asarray(f.phi)', 'to_numpy(f.phi)')
    code = code.replace('np.asarray(f.theta)', 'to_numpy(f.theta)')
    code = code.replace('np.asarray(davidian_weight', 'to_numpy(davidian_weight')
    # Fix tests trying to convert CuPy arrays directly
    code = code.replace('np.mean(second_deriv ** 2)', 'xp.mean(second_deriv ** 2)')
    code = code.replace('np.sum(energy_arr)', 'xp.sum(energy_arr)')
    code = code.replace('np.mean(energy_arr)', 'xp.mean(energy_arr)')
    code = code.replace('np.sum((elastic - plastic) * weights)', 'xp.sum((elastic - plastic) * weights)')
    code = code.replace('np.sum(total_coupling * weights)', 'xp.sum(total_coupling * weights)')
    code = code.replace('np.mean(total_coupling)', 'xp.mean(total_coupling)')
    code = code.replace('np.mean(integrand)', 'xp.mean(integrand)')
    
    if orig != code:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(code)

for d in ['eris', 'tests']:
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith('.py'):
                fix(os.path.join(root, file))
