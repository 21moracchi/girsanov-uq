import numpy as np
import shutil
from ase.build import molecule
from ase.io import Trajectory
import ase.units as units
from girsanov_uq.integrators.langevinOBABO import LangevinOBABO
from girsanov_uq.descriptors.mace import MACELinear
from aseams.ams import AMS
from aseams.cvs import CollectiveVariables
from aseams.inicondssamplers import SingleWalkerSampler

ams_seed = None
dyn_seed = None

# random number generators used for AMS runs (set by make_input_files if desired)
rng_ams = np.random.default_rng(ams_seed)
rng_dyn_ams = np.random.default_rng(dyn_seed)

atoms = molecule('trans-butane')
atoms.set_cell((20.0, 20.0, 20.0))
atoms.center()

# mace_model_path = '/ifpengpfs/scratch/work/r11/moracchl/girsAMS/scripts/AMS_butane/FT_mace/mace_small_ft.model'
# mace_model_path = '/ifpengpfs/scratch/work/r11/moracchl/divers/mace_small.model'
mace_model_path = '../../models/mace-mpa-0-medium.model'
device = 'cuda'
calc = MACELinear(model_path=mace_model_path, device=device, default_dtype='float32')
atoms.calc = calc

temperature_K = 300
dt = 1.0 * units.fs
gamma = 0.01 / units.fs

times = []
torsion_rad = []

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

cv = CollectiveVariables(get_torsion, [get_torsion,get_torsion],cv_torsion)
cv.set_r_crit("between")
cv.set_in_r_boundary([np.radians(175),np.radians(185.0)])
cv.set_sigma_r_level([np.radians(170),np.radians(190.0)])
cv.set_out_of_r_zone([np.radians(150),np.radians(210.0)])
cv.set_p_crit(["below","above"])
cv.set_in_p_boundary([np.radians(60),np.radians(290.0)])

# number of initial conditions expected in the ini conds directory
n_rep = 100

# Single AMS run per folder (one AMS is launched per prepared folder)
# draw RNGs from provided seeds (make_input_files can write `ams_seed`/`dyn_seed` into run_ams.py)
rng_ams_k = np.random.default_rng(ams_seed)
rng_dyn_ams_k = np.random.default_rng(dyn_seed)

dyn_ams = LangevinOBABO(atoms,
                        fixcm=True,
                        timestep=dt,
                        temperature_K=temperature_K,
                        friction=gamma,
                        logfile=None,
                        trajectory=None,
                        rng=rng_dyn_ams_k,
                        record_noise=True)

ams = AMS(n_rep=n_rep,
          k_min=1,
          dyn=dyn_ams,
          xi=cv,
          fixcm=True,
          save_all=False,
          rc_threshold=1e-6,
          verbose=False,
          rng=rng_ams_k)

# expect initial conditions to be created by sample_ini_conds/run_ini_conds.py
ams.set_ini_cond_dir("../sample_ini_conds/ini_conds")
# ams._reuse_ini_conds()
ams.set_ams_dir('./', clean=False)

ams._initialize()
ams.run()
print(ams.p_ams())
