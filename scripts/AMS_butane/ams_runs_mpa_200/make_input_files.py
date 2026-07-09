import os
import numpy as np

# number of AMS jobs to prepare
n_ams = 70
offset = 0
dyn_seeds = np.random.randint(10**6, size=n_ams)
ams_seeds = np.random.randint(10**6, size=n_ams)

for i in range(offset, offset + n_ams):
    dirname = f"ams_{i}"
    os.makedirs(dirname, exist_ok=True)
    # copy input files into per-run directory
    os.system(f'cp job job_topaze {dirname}/')
    os.system(f'cp run_ams.py {dirname}/')

    # replace seed placeholders in the copied run_ams.py
    with open('run_ams.py', 'rt') as f_in:
        text = f_in.read()
    text = text.replace('ams_seed = None', f'ams_seed = {ams_seeds[i - offset]}')
    text = text.replace('dyn_seed = None', f'dyn_seed = {dyn_seeds[i - offset]}')
    with open(f'{dirname}/run_ams.py', 'wt') as f_out:
        f_out.write(text)

print(f"Prepared {n_ams} AMS run directories (ams_0..ams_{n_ams-1}).")
