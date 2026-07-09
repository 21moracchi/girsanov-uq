import numpy as np
import shutil
from ase.build import molecule
from ase.io import Trajectory
import ase.units as units
from girsanov_uq.integrators.langevinOBABO import LangevinOBABO
from girsanov_uq.descriptors.mace import MACELinear
from aseams.cvs import CollectiveVariables
from aseams.inicondssamplers import MultiWalkerSampler

sampler_seed = None
dyn_seed = None

rng_sampler = np.random.default_rng(sampler_seed)
rng_dyn = np.random.default_rng(dyn_seed)

atoms = molecule('trans-butane')
atoms.set_cell((20.0, 20.0, 20.0))
atoms.center()

mace_model_path = '../../../models/mace_mp0a_ft.model'
device = 'cuda'
calc = MACELinear(model_path=mace_model_path, device=device, default_dtype='float32')
atoms.calc = calc

temperature_K = 500
dt = 1.0 * units.fs
gamma = 0.01 / units.fs

dyn = LangevinOBABO(atoms, timestep=dt, temperature_K=temperature_K, friction=gamma, fixcm=True, rng=rng_dyn)

def get_torsion(atoms_obj):
    p = atoms_obj.get_positions(wrap=False)
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    b0 = - (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1_norm) * b1_norm
    w = b2 - np.dot(b2, b1_norm) * b1_norm
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_norm, v), w)
    angle = np.arctan2(y, x)
    return angle % (2 * np.pi)

def cv_torsion(atoms):
    phi = get_torsion(atoms)
    return abs(phi-np.pi)

cv = CollectiveVariables(get_torsion, [get_torsion, get_torsion], cv_torsion)
cv.set_r_crit("between")
cv.set_in_r_boundary([np.radians(175), np.radians(185.0)])
cv.set_sigma_r_level([np.radians(170), np.radians(190.0)])
cv.set_out_of_r_zone([np.radians(120), np.radians(240.0)])
cv.set_p_crit(["below", "above"])
cv.set_in_p_boundary([np.radians(60), np.radians(290.0)])

n_rep = 10000

# multi-walker setup: each replica (FV_replica_i) runs one walker (walker_index)
n_walkers = 10
walker_index = 0
if 'WALKER_INDEX' in __import__('os').environ:
    walker_index = int(__import__('os').environ['WALKER_INDEX'])

# shutil.rmtree("../ini_conds", ignore_errors=True)
inicondsampler = MultiWalkerSampler(dyn, cv, n_walkers=n_walkers, walker_index=walker_index, rng=rng_sampler)
inicondsampler.set_ini_cond_dir("../ini_conds")

# if a checkpoint for this walker exists, resume
if __import__('os').path.exists('../FV_replica_' + str(inicondsampler.w_i) + "/ini_fv_" + str(inicondsampler.w_i) + "_checkpoint.txt"):
    md_fnames = [fname for fname in __import__('os').listdir() if fname.startswith('md_traj')]
    if md_fnames:
        from ase.io import read
        ini_atoms = read('md_traj_' + str(len(md_fnames)-1) + '.traj', format='traj', index=-1)
        inicondsampler.dyn.atoms = ini_atoms
        inicondsampler.dyn.atoms.calc = calc
    inicondsampler.set_run_dir('../FV_replica_', append_traj=False)
    inicondsampler._read_checkpoint()
else:
    inicondsampler.set_run_dir('../FV_replica_', append_traj=False)

inicondsampler.first_in_r = False
inicondsampler.sample(n_conditions=n_rep)
inicondsampler._write_checkpoint()

print("Done: initial conditions written in ../ini_conds")
