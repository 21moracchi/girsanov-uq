import numpy as np
import shutil
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units as units
from ase.io import read
import os
from girsanov_uq.integrators.langevinOBABO import LangevinOBABO
from aseams.cvs import CollectiveVariables
from aseams.inicondssamplers import SingleWalkerSampler, MultiWalkerSampler

# Parameters, atoms, potentials, and integrator (edit in common/model_block.py)
n_walkers = 10
n_conditions = None
# >>> COMMON: common/model_block.py
# <<< COMMON

# CV definition (edit in common/cv_block.py)
# >>> COMMON: common/cv_block.py
# <<< COMMON

## Recurrent parameters
sampler_seed = None 
dyn_seed = None 
rng_sampler, rng_dyn = [np.random.default_rng(s) for s in [sampler_seed, dyn_seed]]

## Recurrent workflow
dyn = LangevinOBABO(atoms, timestep=dt, friction=gamma, temperature_K=temperature_K, rng=rng_dyn, fixcm = fixcm)
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

def _wrap_pbc_positions():
    if np.any(inicondsampler.dyn.atoms.get_pbc()):
        inicondsampler.dyn.atoms.wrap()

inicondsampler = MultiWalkerSampler(dyn, cv, n_walkers=n_walkers, walker_index=0, rng=rng_sampler,fixcm = fixcm)
dyn.attach(_wrap_pbc_positions, interval=1)
inicondsampler.set_ini_cond_dir(f"../ini_conds/{str(inicondsampler.w_i)}")
if os.path.exists('../FV_replica_' + str(inicondsampler.w_i) + "/ini_fv_" + str(inicondsampler.w_i) + "_checkpoint.txt"):
    md_fnames = [fname for fname in os.listdir() if fname.startswith('md_traj')]
    ini_atoms = read('md_traj_' + str(len(md_fnames)-1) + '.traj', format='traj', index=-1)
    if np.any(ini_atoms.get_pbc()):
        ini_atoms.wrap()
    inicondsampler.dyn.atoms = ini_atoms
    inicondsampler.dyn.atoms.calc = calc
    inicondsampler.set_run_dir('../FV_replica_', append_traj=False)
    inicondsampler._read_checkpoint()
else: 
    MaxwellBoltzmannDistribution(inicondsampler.dyn.atoms, temperature_K=temperature_K)
    inicondsampler.set_run_dir('../FV_replica_', append_traj=False)

inicondsampler.first_in_r = False
if np.any(inicondsampler.dyn.atoms.get_pbc()):
    inicondsampler.dyn.atoms.wrap()
inicondsampler.sample(n_conditions=n_conditions)
inicondsampler._write_checkpoint()
