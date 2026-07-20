from pathlib import Path

from girsanov_uq.potentials import OneDimensionalPotentialCalculator, Polynomial1DPotential
R = -0.56756757
P  = 1.677
temperature_K = 1000
dt = 0.01
gamma = 5

atoms = Atoms("H", positions=[[R, 0.0, 0.0]])
atoms.set_cell((20.0, 20.0, 20.0))


weights_path = next(parent / "weights.npy" for parent in Path(__file__).resolve().parents if (parent / "weights.npy").exists())


theta_vec = np.load(weights_path)
potential_1d = Polynomial1DPotential(theta_vec=theta_vec)
calc = OneDimensionalPotentialCalculator(potential_1d=potential_1d)
atoms.calc = calc
