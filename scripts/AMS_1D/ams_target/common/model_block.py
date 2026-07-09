from girsanov_uq.potentials import OneDimensionalPotentialCalculator, TargetFunction1DPotential

temperature_K = 500.0
dt = 1 * units.fs
gamma = 0.01 / units.fs
masses = np.array([1.0])

atoms = Atoms("H", positions=[[-8.0, 0.0, 0.0]])
atoms.set_cell((20.0, 20.0, 20.0))
atoms.set_masses(masses)

potential_1d = TargetFunction1DPotential()
calc = OneDimensionalPotentialCalculator(potential_1d=potential_1d)
atoms.calc = calc
