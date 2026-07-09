import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class Polynomial1DPotential:
	"""1D polynomial potential with free coefficients.

	The potential is defined as:
	V(x) = sum_k theta_k x^k

	where ``theta_vec[k]`` is the weight of ``x^k``.
	"""

	def __init__(self, theta_vec):
		theta = np.asarray(theta_vec, dtype=float).reshape(-1)
		if theta.size == 0:
			raise ValueError("theta_vec must contain at least one coefficient")
		self.theta_vec = theta

	@classmethod
	def from_sklearn_linear_model(cls, model):
		"""Build a polynomial potential from a fitted sklearn linear model.

		Works with estimators exposing ``coef_`` and optionally ``intercept_``.
		If ``fit_intercept=True``, the intercept is prepended to the weights.
		"""

		if not hasattr(model, "coef_"):
			raise AttributeError("Model must expose a fitted coef_ attribute")

		coef = np.asarray(model.coef_, dtype=float).reshape(-1)
		fit_intercept = bool(getattr(model, "fit_intercept", False))

		if fit_intercept:
			intercept = np.asarray(getattr(model, "intercept_", 0.0), dtype=float).reshape(-1)
			intercept_scalar = float(intercept[0]) if intercept.size else 0.0
			theta = np.concatenate(([intercept_scalar], coef))
		else:
			theta = coef

		return cls(theta)

	def energy(self, x):
		x_arr = np.asarray(x, dtype=float)
		return np.polynomial.polynomial.polyval(x_arr, self.theta_vec)

	def derivative(self, x):
		x_arr = np.asarray(x, dtype=float)
		grad_coeffs = np.polynomial.polynomial.polyder(self.theta_vec)
		return np.polynomial.polynomial.polyval(x_arr, grad_coeffs)

	def force(self, x):
		return -self.derivative(x)

	def get_descriptors(self, x_1d):
		"""Return polynomial descriptors and their Jacobian at one point.

		Parameters
		----------
		x_1d : float or array-like
			Scalar or array containing one coordinate value.

		Returns
		-------
		D : np.ndarray, shape (M,)
			Descriptor vector with ``D[k] = x^k``.
		J_D : np.ndarray, shape (M, 1)
			Jacobian of descriptors with respect to ``x``.
		"""

		x_arr = np.asarray(x_1d, dtype=float)
		x = float(x_arr.reshape(-1)[0])

		powers = np.arange(self.theta_vec.size, dtype=float)
		D = x ** powers

		J_D = np.zeros((self.theta_vec.size, 1), dtype=float)
		if self.theta_vec.size > 1:
			J_D[1:, 0] = powers[1:] * (x ** (powers[1:] - 1.0))

		return D, J_D


class TargetFunction1DPotential:
	"""Non-linear 1D target potential used in the POPS toy example.

	By default, this matches ``train_pops.ipynb``:
	V(x) = 0.5 * x^2 + 0.5 * sin(3x) + 0.5 * cos(5x)

	An optional quadratic confinement can be added outside ``[x_min, x_max]``.
	"""

	def __init__(
		self,
		x_min=-2.5,
		x_max=4.0,
		kappa=0.0,
		quadratic_coeff=0.5,
		sin_amp=0.5,
		sin_freq=3.0,
		cos_amp=0.5,
		cos_freq=5.0,
	):
		self.theta_vec = np.array([1.0], dtype=float)
		self.x_min = float(x_min)
		self.x_max = float(x_max)
		self.kappa = float(kappa)
		self.quadratic_coeff = float(quadratic_coeff)
		self.sin_amp = float(sin_amp)
		self.sin_freq = float(sin_freq)
		self.cos_amp = float(cos_amp)
		self.cos_freq = float(cos_freq)

		if self.x_min >= self.x_max:
			raise ValueError("x_min must be strictly smaller than x_max")

	def _confinement_energy(self, x_arr):
		upper = np.maximum(0.0, x_arr - self.x_max)
		lower = np.maximum(0.0, self.x_min - x_arr)
		return self.kappa * (upper**2 + lower**2)

	def _confinement_derivative(self, x_arr):
		dconf = np.zeros_like(x_arr, dtype=float)
		mask_upper = x_arr > self.x_max
		mask_lower = x_arr < self.x_min
		dconf[mask_upper] = 2.0 * self.kappa * (x_arr[mask_upper] - self.x_max)
		dconf[mask_lower] = 2.0 * self.kappa * (x_arr[mask_lower] - self.x_min)
		return dconf

	def energy(self, x):
		x_arr = np.asarray(x, dtype=float)
		base = (
			self.quadratic_coeff * x_arr**2
			+ self.sin_amp * np.sin(self.sin_freq * x_arr)
			+ self.cos_amp * np.cos(self.cos_freq * x_arr)
		)
		return base + self._confinement_energy(x_arr)

	def derivative(self, x):
		x_arr = np.asarray(x, dtype=float)
		base = 2.0 * self.quadratic_coeff * x_arr + self.sin_amp * self.sin_freq * np.cos(
			self.sin_freq * x_arr
		) - self.cos_amp * self.cos_freq * np.sin(
			self.cos_freq * x_arr
		)
		return base + self._confinement_derivative(x_arr)

	def force(self, x):
		return -self.derivative(x)

	def get_descriptors(self, x_1d):
		"""Return a descriptor view compatible with linear potential interfaces."""

		x_arr = np.asarray(x_1d, dtype=float)
		x = float(x_arr.reshape(-1)[0])
		D = np.array([self.energy(x)], dtype=float)
		J_D = np.array([[self.derivative(x)]], dtype=float)
		return D, J_D


class OneDimensionalPotentialCalculator(Calculator):
	"""ASE calculator for a 1D potential defined through descriptors.

	The first atom x-coordinate is used as the scalar coordinate ``x``.
	Potential models are expected to provide:
	- ``theta_vec``
	- ``get_descriptors(x_1d) -> (D, J_D)``
	"""

	implemented_properties = [
		"energy",
		"forces",
		"stress",
		"descriptors",
		"grad_descriptors",
	]

	def __init__(self, potential_1d, atom_index=0, **kwargs):
		super().__init__(**kwargs)
		self.potential_1d = potential_1d
		self.atom_index = int(atom_index)

	def calculate(
		self,
		atoms=None,
		properties=["energy", "forces", "stress", "descriptors", "grad_descriptors"],
		system_changes=all_changes,
	):
		super().calculate(atoms, properties, system_changes)

		n_atoms = len(self.atoms)
		if self.atom_index < 0 or self.atom_index >= n_atoms:
			raise IndexError("atom_index is out of bounds for the atoms object")

		x = float(self.atoms.positions[self.atom_index, 0])
		D, J_D = self.potential_1d.get_descriptors(np.array([x]))
		theta = np.asarray(self.potential_1d.theta_vec, dtype=float)

		energy = float(np.dot(theta, D))

		dVdx = float(np.dot(theta, J_D[:, 0]))
		forces = np.zeros((n_atoms, 3), dtype=float)
		forces[self.atom_index, 0] = -dVdx

		grad_descriptors = np.zeros((n_atoms, D.shape[0], 3), dtype=float)
		grad_descriptors[self.atom_index, :, 0] = J_D[:, 0]

		self.results["energy"] = energy
		self.results["forces"] = forces
		self.results["descriptors"] = np.asarray(D, dtype=float)
		self.results["grad_descriptors"] = grad_descriptors

		if "stress" in properties:
			volume = self.atoms.get_volume()
			if abs(volume) < 1e-12:
				self.results["stress"] = np.zeros(6, dtype=float)
			else:
				stress_tensor = np.zeros((3, 3), dtype=float)
				for i in range(n_atoms):
					stress_tensor += np.outer(self.atoms.positions[i], forces[i])
				stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)
				res = stress_tensor / volume
				self.results["stress"] = np.array(
					[res[0, 0], res[1, 1], res[2, 2], res[1, 2], res[0, 2], res[0, 1]],
					dtype=float,
				)

	def get_descriptors_jacobian(self, atoms):
		self.calculate(atoms, properties=["descriptors", "grad_descriptors"], system_changes="all")
		return self.results["grad_descriptors"]
