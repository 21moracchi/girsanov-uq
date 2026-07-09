import numpy as np
import ase.units as units

class Reweightor:
    def __init__(self, calculator, integrator="OBABO", mode='linear', temperature_K=500, dt=1*units.fs, friction=1.0, masses=1, target_calculator=None):

        self.calculator = calculator
        self.integrator = integrator
        self.mode = mode
        self.temperature_K = temperature_K
        self.dt = dt
        self.friction = friction
        self.masses = masses 
        self.target_calculator = target_calculator

    def compute_girsanov_score_fim_one_step(self, xi, grad_descriptors_1, grad_descriptors_2):
        if self.integrator == "OBABO":
            xi = np.asarray(xi)
            N = xi.shape[1]
            grad_descriptors_1 = self._as_reweighting_jacobian(grad_descriptors_1, N)
            grad_descriptors_2 = self._as_reweighting_jacobian(grad_descriptors_2, N)
            _, D, _ = grad_descriptors_1.shape
            
            beta = 1.0 / (units.kB * self.temperature_K)
            c1 = 1.0 + (self.friction * self.dt) / 4.0
            c2 = 1.0 - (self.friction * self.dt) / 4.0
            
            m = np.array(self.masses)
            if m.ndim == 0:
                m = np.full(N, m)
            
            c3 = np.sqrt((m * self.friction * self.dt) / beta)
            c3_3N = np.repeat(c3, 3)[:, None] 
            
            coeff_1 = (c1 * self.dt) / (2.0 * c3_3N) 
            coeff_2 = (c2 * self.dt) / (2.0 * c3_3N)

            xi_k = xi[0].flatten()
            eta_k = xi[1].flatten()

            J_k_1 = grad_descriptors_1.transpose(1, 0, 2).reshape(D, -1)
            J_k_2 = grad_descriptors_2.transpose(1, 0, 2).reshape(D, -1)

            Y_xi = coeff_1 * J_k_1.T
            Y_eta = coeff_2 * J_k_2.T

            score = -(xi_k @ Y_xi + eta_k @ Y_eta)
            
            fim = (Y_xi.T @ Y_xi) + (Y_eta.T @ Y_eta)

        return score, fim

    def _as_reweighting_jacobian(self, grad_descriptors, n_atoms):
        grad_descriptors = np.asarray(grad_descriptors)

        if grad_descriptors.ndim != 3:
            raise ValueError(
                "grad_descriptors must have 3 dimensions, got "
                f"{grad_descriptors.shape}"
            )

        if grad_descriptors.shape[0] == n_atoms and grad_descriptors.shape[2] == 3:
            return grad_descriptors

        if grad_descriptors.shape[1] == n_atoms and grad_descriptors.shape[2] == 3:
            return grad_descriptors.transpose(1, 0, 2)

        raise ValueError(
            "Unable to interpret grad_descriptors shape "
            f"{grad_descriptors.shape} for n_atoms={n_atoms}; expected "
            "(N_atoms, D, 3) or (D, N_atoms, 3)."
        )

    def _global_descriptors(self, atoms):
        descriptors = np.asarray(self.calculator.get_descriptors(atoms))
        if descriptors.ndim == 2:
            return descriptors.sum(axis=0)
        return descriptors

    def reweight_one_trajectory_linear(self, trajectory: list, as_list = False):
        beta = 1.0 / (units.kB * self.temperature_K)
        initial_atoms = trajectory[0]
        initial_descriptors = self._global_descriptors(initial_atoms)
        
        score = -beta * initial_descriptors
        fim = 0.0
        
        score_array = []
        fim_array = []
        if as_list:
            score_array.append(score.copy())
            fim_array.append(np.zeros((score.shape[0], score.shape[0])))
            
        grad_descriptors_prev = None

        for atoms in trajectory:
            grad_descriptors = self.calculator.get_descriptors_jacobian(atoms)
            if grad_descriptors_prev is not None:
                xi = atoms.info.get('noise')
                if xi is not None:
                    xi = np.array(xi) 
                    marginal_score, marginal_fim = self.compute_girsanov_score_fim_one_step(xi, grad_descriptors_prev, grad_descriptors)
                    score += marginal_score
                    fim += marginal_fim
                    if as_list:
                        score_array.append(score.copy())
                        fim_array.append(fim.copy())

            grad_descriptors_prev = grad_descriptors
        
        if as_list:
            return np.array(score_array), np.array(fim_array)
        return score, fim
    
    def _get_noise_coefficients(self, N):
        beta = 1.0 / (units.kB * self.temperature_K)
        c1 = 1.0 + (self.friction * self.dt) / 4.0
        c2 = 1.0 - (self.friction * self.dt) / 4.0
        
        m = np.array(self.masses)
        if m.ndim == 0:
            m = np.full(N, m)
            
        c3 = np.sqrt((m * self.friction * self.dt) / beta)
        c3_N3 = c3[:, None] 
        
        coeff_1 = (c1 * self.dt) / (2.0 * c3_N3)
        coeff_2 = (c2 * self.dt) / (2.0 * c3_N3)
        
        return coeff_1, coeff_2

    def compute_girsanov_directional_score_one_step(self, directional_grad_descriptors_1, directional_grad_descriptors_2):
        if self.integrator == "OBABO":
            score = -(directional_grad_descriptors_1 + directional_grad_descriptors_2)
            return score

    def reweight_one_trajectory_score_only(self, trajectory: list, as_list = False):
        beta = 1.0 / (units.kB * self.temperature_K)
        initial_atoms = trajectory[0]
        initial_descriptors = self._global_descriptors(initial_atoms)
        
        score = -beta * initial_descriptors
        score_array = []
        if as_list:
            score_array.append(score.copy())
            
        atoms_prev = None

        for atoms in trajectory:
            if atoms_prev is not None:
                noise = atoms.info.get('noise')
                if noise is not None:
                    noise = np.array(noise)
                    xi, eta = noise[0], noise[1]
                    
                    N = len(atoms)
                    coeff_1, coeff_2 = self._get_noise_coefficients(N)
                    
                    xi_scaled = xi * coeff_1
                    eta_scaled = eta * coeff_2

                    dir_grad_1 = self.calculator.get_descriptors_directional(atoms_prev, xi_scaled)
                    dir_grad_2 = self.calculator.get_descriptors_directional(atoms, eta_scaled)

                    marginal_score = self.compute_girsanov_directional_score_one_step(dir_grad_1, dir_grad_2)
                    score += marginal_score
                    
                    if as_list:
                        score_array.append(score.copy())

            atoms_prev = atoms
        
        if as_list:
            return np.array(score_array)
        return score
    
    def reweight_one_trajectory_nonlinear(self, trajectory: list, as_list = False):
        beta = 1.0 / (units.kB * self.temperature_K)
        initial_atoms = trajectory[0]
        v_ref = initial_atoms.get_potential_energy()
        
        perturbed_initial = initial_atoms.copy()
        perturbed_initial.set_calculator(self.target_calculator)
        v_target = perturbed_initial.get_potential_energy()
        
        log_girsanov_weight = -beta * (v_target - v_ref)
        log_girsanov_weight_array = []
        if as_list:
            log_girsanov_weight_array.append(log_girsanov_weight)
            
        delta_forces_prev = None
        for atoms in trajectory:
            forces = atoms.get_forces().copy()
            atoms_perturbed = atoms.copy()
            atoms_perturbed.set_calculator(self.target_calculator)
            forces_perturbed = atoms_perturbed.get_forces().copy()
            delta_force_1 = delta_forces_prev 
            delta_force_2 = forces_perturbed - forces 
            if delta_forces_prev is not None:
                xi = atoms.info.get('noise')
                if xi is not None:
                    xi = np.array(xi) 
                    log_marginal_weight = self.compute_girsanov_weight_one_step(xi, delta_force_1, delta_force_2)
                    log_girsanov_weight += log_marginal_weight
                    if as_list:
                        log_girsanov_weight_array.append(log_girsanov_weight)

            delta_forces_prev = delta_force_2
        if as_list:
            return np.exp(np.array(log_girsanov_weight_array))
        return np.exp(log_girsanov_weight)                                                                 
    
    def reweight_one_trajectory(self, trajectory: list, as_list = False):
        if self.mode == 'linear':
            return self.reweight_one_trajectory_linear(trajectory, as_list)
        elif self.mode == 'nonlinear':
            return self.reweight_one_trajectory_nonlinear(trajectory, as_list)
        elif self.mode == 'score_only':
            return self.reweight_one_trajectory_score_only(trajectory, as_list)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented yet.")
        
    def compute_girsanov_weight_one_step(self, xi, delta_force_1, delta_force_2):
        if self.integrator == "OBABO":
            N, _ = delta_force_1.shape
            
            beta = 1.0 / (units.kB * self.temperature_K)
            c1 = 1.0 + (self.friction * self.dt) / 4.0
            c2 = 1.0 - (self.friction * self.dt) / 4.0
            
            m = np.array(self.masses)
            if m.ndim == 0:
                m = np.full(N, m)
            
            c3 = np.sqrt((m * self.friction * self.dt) / beta)
            c3_3N = np.repeat(c3, 3)
            
            coeff_1 = (c1 * self.dt) / (2.0 * c3_3N) 
            coeff_2 = (c2 * self.dt) / (2.0 * c3_3N)

            xi_k = xi[0].flatten()
            eta_k = xi[1].flatten()
            delta_force_1_flat = delta_force_1.flatten()
            delta_force_2_flat = delta_force_2.flatten()

            Y_xi = coeff_1 * delta_force_1_flat
            Y_eta = coeff_2 * delta_force_2_flat

            stoch_integral = (xi_k @ Y_xi + eta_k @ Y_eta)
            
            det_integral = (Y_xi.T @ Y_xi) + (Y_eta.T @ Y_eta)

        log_weight = stoch_integral - 0.5 * det_integral
        return log_weight
    
    def reweight_trajectories(self, trajectories: list, as_list = False):
        if self.mode == 'linear':
            return self.reweight_trajectories_linear(trajectories, as_list)
        elif self.mode == 'nonlinear':
            return self.reweight_trajectories_nonlinear(trajectories, as_list)
        elif self.mode == 'score_only':
            return self.reweight_trajectories_score_only(trajectories, as_list)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented yet.")
    
    def reweight_trajectories_linear(self, trajectories: list, as_list = False):
        scores = []
        fims = []

        for traj in trajectories:
            score, fim = self.reweight_one_trajectory(traj, as_list)
            scores.append(score)
            fims.append(fim)
        if as_list:
            return scores, fims
        return np.array(scores), np.array(fims)
    
    def reweight_trajectories_nonlinear(self, trajectories: list, as_list = False):
        weights = []
        for traj in trajectories:
            weight = self.reweight_one_trajectory(traj, as_list)
            weights.append(weight)
        if as_list:
            return weights
        return np.array(weights)

    def reweight_trajectories_score_only(self, trajectories: list, as_list = False):
        scores = []
        for traj in trajectories:
            score = self.reweight_one_trajectory(traj, as_list)
            scores.append(score)
        if as_list:
            return scores
        return np.array(scores)
