import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic

class SolvatedDimer(Calculator):
    implemented_properties = [
        'energy',
        'forces',
        'stress',
        'descriptors',
        'grad_descriptors'
    ]

    default_parameters = {
        'epsilon': 1.0,
        'h': 1.0,
        'w': 1.0,
        'sigma': 1.0
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate(
        self,
        atoms=None,
        properties=['energy', 'forces', 'stress', 'descriptors', 'grad_descriptors'],
        system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)

        epsilon = self.parameters['epsilon']
        h = self.parameters['h']
        w = self.parameters['w']
        sigma = self.parameters['sigma']

        r0 = (2.0 ** (1.0 / 6.0)) * sigma

        pos = self.atoms.positions
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        has_pbc = pbc.any()
        N = len(pos)
        vol = cell.volume

        D_h = 0.0
        D_eps = 0.0
        D_w = 0.0
        D_sigma = 0.0

        grad_D_h = np.zeros((N, 3))
        grad_D_eps = np.zeros((N, 3))
        grad_D_w = np.zeros((N, 3))
        grad_D_sigma = np.zeros((N, 3))
        stress_mat = np.zeros((3, 3))

        v12 = pos[0] - pos[1]
        if has_pbc:
            v12 = find_mic(np.array([v12]), cell, pbc)[0][0]
            
        r12 = np.linalg.norm(v12)
        u12 = v12 / r12

        u_val = (r12 - r0 - w) / w
        u2 = u_val**2
        u3 = u_val**3
        u4 = u_val**4

        D_h = (1.0 - u2)**2
        D_w = h * 4.0 * (u2 + u_val - u4 - u3) / w
        D_sigma = h * 4.0 * (2.0**(1.0/6.0)) * (u_val - u3) / w

        dDh_dr = -(4.0 / w) * u_val * (1.0 - u2)
        dDw_dr = h * (4.0 / w**2) * (-4.0 * u3 - 3.0 * u2 + 2.0 * u_val + 1.0)
        dDsigmaS_dr = h * (4.0 * (2.0**(1.0/6.0)) / w**2) * (1.0 - 3.0 * u2)

        grad_D_h[0] = dDh_dr * u12
        grad_D_h[1] = -dDh_dr * u12

        grad_D_w[0] = dDw_dr * u12
        grad_D_w[1] = -dDw_dr * u12

        grad_D_sigma[0] = dDsigmaS_dr * u12
        grad_D_sigma[1] = -dDsigmaS_dr * u12

        stress_mat += (h * dDh_dr / r12) * np.outer(v12, v12)

        for i in range(N):
            for j in range(i + 1, N):
                if i == 0 and j == 1:
                    continue

                vij = pos[i] - pos[j]
                if has_pbc:
                    vij = find_mic(np.array([vij]), cell, pbc)[0][0]
                    
                rij = np.linalg.norm(vij)

                if rij <= r0:
                    sr = sigma / rij
                    sr6 = sr**6
                    sr12 = sr6**2

                    D_eps += 4.0 * (sr12 - sr6) + 1.0
                    D_sigma += epsilon * (48.0 * sr12 - 24.0 * sr6) / sigma

                    dDeps_dr = -24.0 / rij * (2.0 * sr12 - sr6)
                    dDsigmaWCA_dr = -epsilon * (144.0 / (rij * sigma)) * (4.0 * sr12 - sr6)
                    
                    uij = vij / rij

                    grad_D_eps[i] += dDeps_dr * uij
                    grad_D_eps[j] -= dDeps_dr * uij

                    grad_D_sigma[i] += dDsigmaWCA_dr * uij
                    grad_D_sigma[j] -= dDsigmaWCA_dr * uij

                    stress_mat += (epsilon * dDeps_dr / rij) * np.outer(vij, vij)

        D = np.array([D_h, D_eps, D_w, D_sigma])
        energy = h * D_h + epsilon * D_eps
        grad_D = np.array([grad_D_h, grad_D_eps, grad_D_w, grad_D_sigma]).transpose(1, 0, 2)
        forces = -(h * grad_D_h + epsilon * grad_D_eps)

        if vol > 0:
            stress = np.array([
                stress_mat[0, 0],
                stress_mat[1, 1],
                stress_mat[2, 2],
                stress_mat[1, 2],
                stress_mat[0, 2],
                stress_mat[0, 1]
            ]) / vol
        else:
            stress = np.zeros(6)

        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress
        self.results['descriptors'] = D
        self.results['grad_descriptors'] = grad_D

    def get_descriptors_jacobian(self, atoms):
        self.calculate(
            atoms, 
            properties=['descriptors', 'grad_descriptors'], 
            system_changes='all'
        )
        return self.results['grad_descriptors']