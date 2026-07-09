import os
import numpy as np

# number of independent samplers to prepare (each will generate a set of ini conds)
n_samplers = 10
dyn_seeds = np.random.randint(10**6, size=n_samplers)
sampler_seeds = np.random.randint(10**6, size=n_samplers)

for i in range(n_samplers):
    dname = f'FV_replica_{i}'
    os.makedirs(dname, exist_ok=True)
    os.system(f'cp job_topaze run_ini_conds.py {dname}/')
    os.system(f'cp job {dname}/')
    # inject seeds into the copied run_ini_conds.py
    with open('run_ini_conds.py', 'rt') as f:
        txt = f.read()
    txt = txt.replace('sampler_seed = None', f'sampler_seed = {sampler_seeds[i]}')
    txt = txt.replace('dyn_seed = None', f'dyn_seed = {dyn_seeds[i]}')
    txt = txt.replace('walker_index = 0', f'walker_index = {i}')
    with open(f'{dname}/run_ini_conds.py', 'wt') as f:
        f.write(txt)

print(f'Prepared {n_samplers} sampling jobs in FV_replica_0..{n_samplers-1}')
