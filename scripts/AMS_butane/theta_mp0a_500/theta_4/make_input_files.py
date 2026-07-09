import os
import numpy as np

# AMS jobs to prepare for this theta.
theta_index = 77
n_ams = 70
offset = 0
dyn_seeds = np.random.randint(10**6, size=n_ams)
ams_seeds = np.random.randint(10**6, size=n_ams)

for i in range(offset, offset + n_ams):
    dirname = f"ams_{i}"
    os.makedirs(dirname, exist_ok=True)
    os.system(f'cp job job_topaze {dirname}/')
    os.system(f'cp run_ams.py {dirname}/')

    with open('run_ams.py', 'rt') as f_in:
        text = f_in.read()
    text = text.replace('ams_seed = None', f'ams_seed = {ams_seeds[i - offset]}')
    text = text.replace('dyn_seed = None', f'dyn_seed = {dyn_seeds[i - offset]}')
    text = text.replace('theta_index = None', f'theta_index = {theta_index}')
    text = text.replace(
        'ams.set_ini_cond_dir("../../../ams_runs_mp0a_500/sample_ini_conds/ini_conds")',
        f'ams.set_ini_cond_dir("../../../ams_runs_mp0a_500/sample_ini_conds/ini_conds/{i}/")',
    )
    with open(f'{dirname}/run_ams.py', 'wt') as f_out:
        f_out.write(text)

print(f"Prepared {n_ams} AMS run directories for theta_{theta_index}.")
