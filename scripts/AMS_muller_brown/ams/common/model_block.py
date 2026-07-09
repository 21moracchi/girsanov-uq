from girsanov_uq.potentials.muller_brown import RuggedMullerBrown

N_dim = 10
N_atoms = int(np.ceil(N_dim/3))


positions = np.zeros((N_atoms, 3))
positions[0, 0:2] = [-0.558,  1.442] 

atoms = Atoms('Ar' * N_atoms, positions=positions)
masses = [100] * N_atoms
atoms.set_masses(masses)
atoms.set_cell((10.0, 10.0, 10.0))

calc = RuggedMullerBrown(
    N_dim=N_dim,
    A=np.array([-200.0, -100.0, -170.0, 15.0]),
    gamma_=10.0,
    K=np.ones(N_dim - 2),
    omega=10.0
)
atoms.calc = calc

temperature_K = 0.7e5

dt = 1 * units.fs
gamma = 0.1 / units.fs





fixcm = False
masses = atoms.get_masses()