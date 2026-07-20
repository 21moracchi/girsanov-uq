from girsanov_uq.potentials.dimer_potential import SolvatedDimer

n_solvent = 30
L = 10

# Potential parameters
calc = SolvatedDimer()
calc.parameters = {
    'epsilon': 1.0,
    'h': 5.0,
    'w': 0.25,
    'sigma': 1.0
}

# Dimer definition (distance P)
r0 = 2**(1/6) * calc.parameters['sigma']
P = r0 + 2 * calc.parameters['w']
pos_dimer = np.array([
    [L/2, L/2, L/2],
    [L/2 + P, L/2, L/2]
])

# Generate a grid of solvent candidate positions
# Use enough spacing to avoid overlaps: d > sigma
n_points_axis = int(np.ceil((n_solvent + 2)**(1/3))) + 1
grid_coords = np.linspace(0.5, L - 0.5, n_points_axis)
x, y, z = np.meshgrid(grid_coords, grid_coords, grid_coords)
possible_positions = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

# Exclude sites too close to the dimer (safety distance > sigma)
min_dist_threshold = calc.parameters['sigma'] * 1.1
distances_to_d1 = np.linalg.norm(possible_positions - pos_dimer[0], axis=1)
distances_to_d2 = np.linalg.norm(possible_positions - pos_dimer[1], axis=1)
mask = (distances_to_d1 > min_dist_threshold) & (distances_to_d2 > min_dist_threshold)

valid_positions = possible_positions[mask]

# Randomly select n_solvent valid grid positions
if len(valid_positions) < n_solvent:
    raise ValueError("La grille est trop lâche ou L est trop petit pour n_solvent.")

idx = np.random.choice(len(valid_positions), n_solvent, replace=False)
solvent_positions = valid_positions[idx]

# Assemblage final
symbols = 'N2' + 'H' * n_solvent
atoms = Atoms(symbols)
atoms.set_masses([1.0] * len(atoms))
atoms.set_cell([L, L, L])
atoms.set_pbc([True, True, True])

final_positions = np.vstack([pos_dimer, solvent_positions])
atoms.set_positions(final_positions)
atoms.calc = calc

temperature_K = 1 / units.kB
dt = 0.005
gamma = 1.0
fixcm = True