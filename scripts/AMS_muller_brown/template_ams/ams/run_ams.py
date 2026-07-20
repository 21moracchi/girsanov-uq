import sys
import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units as units
from girsanov_uq.integrators.langevinOBABO import LangevinOBABO
from aseams.ams import AMS

# Parameters, atoms, potentials, and integrator (edit in common/model_block.py)
n_reps = None
ams_id = None
# >>> COMMON: common/model_block.py
# <<< COMMON

# CV definition (edit in common/cv_block.py)
# >>> COMMON: common/cv_block.py
# <<< COMMON

## Recurrent parameters
ams_seed = None
dyn_seed = None
rng_ams, rng_dyn = [np.random.default_rng(s) for s in [ams_seed, dyn_seed]]

dyn_ams = LangevinOBABO(atoms, fixcm=fixcm, timestep=dt, temperature_K=temperature_K, friction=gamma, logfile=None, trajectory=None, rng=rng_dyn, record_noise=True)
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

ams = AMS(n_rep=n_reps, k_min=1, dyn=dyn_ams, xi=cv, fixcm=fixcm, save_all=False, rc_threshold=1e-6, verbose=False, rng=rng_ams)
ams.set_ini_cond_dir(f"../../ini_conds/ini_conds/{ams_id}")
def _wrap_pbc_positions():
    if np.any(ams.dyn.atoms.get_pbc()):
        ams.dyn.atoms.wrap()
ams.dyn.attach(_wrap_pbc_positions, interval=1)


# ams._reuse_ini_conds()
ams.set_ams_dir("./", clean=False)
ams._initialize()
ams.run()

p = ams.p_ams()
np.savetxt('result_proba.txt', [p])