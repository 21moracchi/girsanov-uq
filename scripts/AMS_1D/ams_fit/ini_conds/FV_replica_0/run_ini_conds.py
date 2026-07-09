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

# parameters, atoms, potentials and integrator (TO ADAPT in common/model_block.py)
n_walkers = 10
n_conditions = 250
from pathlib import Path

from girsanov_uq.potentials import OneDimensionalPotentialCalculator, Polynomial1DPotential
R = -0.56756757
P  = 1.677
temperature_K = 1000
dt = 0.01
gamma = 5

atoms = Atoms("H", positions=[[R, 0.0, 0.0]])
atoms.set_cell((20.0, 20.0, 20.0))


weights_path = "/ifpengpfs/scratch/work/r11/moracchl/girsAMS/scripts/AMS_pops_toysystem/weights.npy"


theta_vec = np.load(weights_path)
potential_1d = Polynomial1DPotential(theta_vec=theta_vec)
calc = OneDimensionalPotentialCalculator(potential_1d=potential_1d)
atoms.calc = calc
# Definition of the CV (TO ADAPT in common/cv_block.py)
from aseams.cvs import CollectiveVariables

def get_x(atoms):
    return float(atoms.positions[0, 0])

R = -0.56756757
P  = 1.677
cv = CollectiveVariables(get_x, get_x, get_x)
cv.set_r_crit("below")
cv.set_in_r_boundary(-0.5)
cv.set_sigma_r_level(-0.48)
cv.set_out_of_r_zone(0.0)
cv.set_p_crit("above")
cv.set_in_p_boundary(P-0.1)
## Recurrent parameters
sampler_seed = 412467 
dyn_seed = 414342 
rng_sampler, rng_dyn = [np.random.default_rng(s) for s in [sampler_seed, dyn_seed]]

## Recurrent workflow
dyn = LangevinOBABO(atoms, timestep=dt, friction=gamma, temperature_K=temperature_K, rng=rng_dyn)
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

inicondsampler = MultiWalkerSampler(dyn, cv, n_walkers=n_walkers, walker_index=0, rng=rng_sampler)
inicondsampler.set_ini_cond_dir(f"../ini_conds/{str(inicondsampler.w_i)}")
if os.path.exists('../FV_replica_' + str(inicondsampler.w_i) + "/ini_fv_" + str(inicondsampler.w_i) + "_checkpoint.txt"):
    md_fnames = [fname for fname in os.listdir() if fname.startswith('md_traj')]
    ini_atoms = read('md_traj_' + str(len(md_fnames)-1) + '.traj', format='traj', index=-1)
    inicondsampler.dyn.atoms = ini_atoms
    inicondsampler.dyn.atoms.calc = calc
    inicondsampler.set_run_dir('../FV_replica_', append_traj=False)
    inicondsampler._read_checkpoint()
else: 
    MaxwellBoltzmannDistribution(inicondsampler.dyn.atoms, temperature_K=temperature_K)
    inicondsampler.set_run_dir('../FV_replica_', append_traj=False)

inicondsampler.first_in_r = False
inicondsampler.sample(n_conditions=n_conditions)
inicondsampler._write_checkpoint()
