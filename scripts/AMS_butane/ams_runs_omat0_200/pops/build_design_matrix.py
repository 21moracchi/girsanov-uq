import numpy as np
from ase.io import read
from sklearn.linear_model import Ridge
from tqdm import tqdm
from girsanov_uq.descriptors.mace import MACELinear

train_on_forces = True
device = 'cuda'
trajectories = ['../../FT_mace/saved_datasets/train_set.xyz','../../FT_mace/saved_datasets/test_set.xyz','../../FT_mace/saved_datasets/valid_set.xyz']
model = 'omat0'
model_path = f"../../models/mace_{model}_ft.model"
sampling_rate = 1
calc = MACELinear(model_paths=model_path, device=device, default_dtype = 'float32')

design_matrix_energy = []
design_matrix_forces = []
targets_energy = []
targets_forces = []
mace_energy = []
mace_forces = []
from copy import deepcopy
for traj in tqdm(trajectories, desc="Processing trajectories"): 
    atoms_list = read(traj, index=':')
    for atoms in atoms_list[::sampling_rate]:
        
        atoms1 = deepcopy(atoms)
        e_dft = atoms1.get_potential_energy()
        f_dft = atoms1.get_forces() if train_on_forces else None


        atoms2 = deepcopy(atoms)
        atoms2.set_calculator(calc)
        if train_on_forces:
            derivatives = calc.get_descriptors_jacobian(atoms2)
            descriptor = calc.get_descriptors(atoms2)
        else:
            descriptor = calc.get_descriptors(atoms2)

        e_mace0 = atoms2.get_potential_energy()
        f_mace0 = atoms2.get_forces() if train_on_forces else None

        delta_e = e_dft - e_mace0
        delta_f = (f_dft - f_mace0) if train_on_forces else None
        
        design_matrix_energy.append(descriptor)
        targets_energy.append(delta_e)
        mace_energy.append(e_mace0)

        if train_on_forces:
            design_matrix_forces.append(derivatives.reshape(descriptor.shape[0], -1).T)
            targets_forces.append(delta_f.flatten())
            mace_forces.append(f_mace0.flatten())

design_matrix_energy = np.array(design_matrix_energy)
design_matrix_forces = np.array(design_matrix_forces) if train_on_forces else None
targets_energy = np.array(targets_energy)
targets_forces = np.array(targets_forces) if train_on_forces else None
mace_energy = np.array(mace_energy)
mace_forces = np.array(mace_forces) if train_on_forces else None
np.save(f'binary_data/design_matrix_energy_{model}.npy', design_matrix_energy)
np.save(f'binary_data/targets_energy_{model}.npy', targets_energy)
np.save(f'binary_data/mace_energy_{model}.npy', mace_energy)
if train_on_forces:
    np.save(f'binary_data/design_matrix_forces_{model}.npy', design_matrix_forces)
    np.save(f'binary_data/targets_forces_{model}.npy', targets_forces)
    np.save(f'binary_data/mace_forces_{model}.npy', mace_forces)
