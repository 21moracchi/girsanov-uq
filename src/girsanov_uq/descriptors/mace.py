import torch
from mace.calculators import MACECalculator
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools.torch_geometric.dataloader import DataLoader
from torch.autograd.functional import jacobian
from ase.calculators.calculator import all_changes
from e3nn import o3
from mace.modules.utils import extract_invariant
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
class MACELinear(MACECalculator):
    
    def __init__(self, linear_weights=torch.zeros(256, dtype=torch.float32), **kwargs):
        super().__init__(**kwargs)
        self.linear_weights = torch.tensor(linear_weights, device=self.device, dtype=torch.float32)
        if "descriptors" not in self.implemented_properties:
                    self.implemented_properties.append("descriptors")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Calculator differing from the original MACECalculator as it :
        1) Computes the node descriptors during the forward pass and stores them in self.results["descriptors"].
        2) Computes the energy and forces using the generalized linear form. 
        """
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)

        if self.model_type in ["MACE", "EnergyDipoleMACE", "PolarMACE"]:
            compute_stress = not self.use_compile
        else:
            compute_stress = False

        ret_tensors = None
        node_e0 = None
        descriptors_list = []

        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            model_dtype = next(model.parameters()).dtype
            for key in batch.keys:
                value = batch[key]
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    if key == "positions":
                        batch[key] = value.to(dtype=model_dtype).detach().clone().requires_grad_(True)
                    else:
                        batch[key] = value.to(dtype=model_dtype)
            
            out = model(
                batch.to_dict(),
                compute_stress=compute_stress,
                training=True,
                compute_edge_forces=self.compute_atomic_stresses,
                compute_atomic_stresses=self.compute_atomic_stresses,
            )
            
            node_feats = out["node_feats"]
            irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
            l_max = irreps_out.lmax
            num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
            num_interactions = int(model.num_interactions)

            desc = extract_invariant(
                node_feats,
                num_layers=num_interactions,
                num_features=num_invariant_features,
                l_max=l_max,
            )

            per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
            per_layer_features[-1] = num_invariant_features
            to_keep = np.sum(per_layer_features[:num_interactions])
            
            node_desc_kept = desc[:, :to_keep]
            global_desc = node_desc_kept.sum(dim=0)
            
            e_base = out["energy"].view(-1)
            
            
            if self.linear_weights.shape[0] == global_desc.shape[0]:
                e_tot = torch.dot(self.linear_weights, global_desc) + e_base
            else:
                e_tot = torch.dot(self.linear_weights[:to_keep], global_desc) + e_base

            forces = -torch.autograd.grad(
                outputs=e_tot,
                inputs=batch["positions"],
                create_graph=False,
                retain_graph=compute_stress
            )[0]
            
            out["energy"] = e_tot.unsqueeze(0).detach()
            out["forces"] = forces.detach()

            if i == 0:
                ret_tensors, node_e0 = self._create_result_tensors(
                    self.num_models, len(atoms), batch, out
                )
            for key, val in ret_tensors.items():
                if out.get(key) is not None:
                    val[i] = out[key].detach()

            descriptors_list.append(node_desc_kept.detach().cpu().numpy())

        self.results = {}
        scalar_tensors = set(["energy"])
        results_store_ensemble = set(["energy", "forces", "stress", "dipole"])
        results_map = [
            ("energy", "energy", self.energy_units_to_eV),
            ("node_energy", "node_energy", self.energy_units_to_eV),
            ("forces", "forces", self.energy_units_to_eV / self.length_units_to_A),
            ("stress", "stress", self.energy_units_to_eV / self.length_units_to_A**3),
            ("stresses", "atomic_stresses", self.energy_units_to_eV / self.length_units_to_A**3),
            ("virials", "atomic_virials", self.energy_units_to_eV / self.length_units_to_A**3),
            ("dipole", "dipole", 1.0),
            ("charges", "charges", 1.0),
            ("polarizability", "polarizability", 1.0),
            ("polarizability_sh", "polarizability_sh", 1.0),
        ]
        
        if self.model_type == "PolarMACE":
            results_map.extend([
                ("interaction_energy", "interaction_energy", self.energy_units_to_eV),
                ("electrostatic_energy", "electrostatic_energy", self.energy_units_to_eV),
                ("electron_energy", "electron_energy", self.energy_units_to_eV),
                ("spins", "spins", 1.0),
                ("density_coefficients", "density_coefficients", 1.0),
                ("spin_charge_density", "spin_charge_density", 1.0),
            ])
            
        for results_key, ret_key, unit_conv in results_map:
            if ret_tensors.get(ret_key) is not None:
                data = torch.mean(ret_tensors[ret_key], dim=0).cpu()
                if ret_key in scalar_tensors:
                    data = data.item()
                else:
                    data = data.numpy()
                self.results[results_key] = data * unit_conv

                if self.num_models > 1 and results_key in results_store_ensemble:
                    data = ret_tensors[results_key].cpu().numpy()
                    data *= unit_conv
                    self.results[results_key + "_comm"] = data

                    data = torch.var(ret_tensors[results_key], dim=0, unbiased=False).cpu()
                    if ret_key in scalar_tensors:
                        data = data.item()
                    else:
                        data = data.numpy()
                    data *= unit_conv
                    self.results[results_key + "_var"] = data

        if self.results.get("energy") is not None:
            self.results["free_energy"] = self.results["energy"]
        if self.results.get("node_energy") is not None:
            self.results["energies"] = self.results["node_energy"].copy()
            self.results["node_energy"] -= node_e0
        if self.results.get("stress") is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])
        if self.results.get("stresses") is not None:
            self.results["stresses"] = np.asarray([
                full_3x3_to_voigt_6_stress(stress) for stress in self.results["stresses"]
            ])

        if self.num_models == 1:
            self.results["descriptors"] = descriptors_list[0]
        else:
            self.results["descriptors"] = descriptors_list
    
    def get_descriptors_jacobian(self, atoms):
        """
        Compute $D$ and $\partial D / \partial r$ using the adjoint force method.
        
        $E_{corr} = \theta^\top D(r)$, 
        the force $\mathbf{F} = -\nabla_r E_{corr}$ satisfies:
        $$\frac{\partial D}{\partial r} = -\frac{\partial \mathbf{F}}{\partial \theta}$$

        

        Args:
            atoms (ase.Atoms): Atomic system of $N$ atoms.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - descriptor: Shape (P,)
                - jacobian: Shape (P, N, 3) reconstructed from $-\nabla_\theta \mathbf{F}$
        """
        batch = self._prepare_custom_batch(atoms).to(self.device)
        batch_dict_base = batch.to_dict()
        
        positions = batch_dict_base["positions"].detach().clone().requires_grad_(True)
        theta = self.linear_weights.detach().clone().requires_grad_(True)
        def force_fn(theta_var):
            batch_dict = batch_dict_base.copy()
            batch_dict["positions"] = positions
            
            node_desc = self._get_descriptors_from_batch(batch_dict)
            global_desc = node_desc.sum(dim=0)
            
            e_corr = torch.dot(global_desc, theta_var)
            
            force = -torch.autograd.grad(
                outputs=e_corr,
                inputs=positions,
                create_graph=True,
            )[0]
            return force
        jac_f_theta = jacobian(force_fn, theta, create_graph=False, vectorize=False)
        
        jac_d_r = -jac_f_theta.permute(2, 0, 1)

        batch_dict = batch_dict_base.copy()
        batch_dict["positions"] = positions

        return  jac_d_r.detach().cpu().numpy()
   
    
    def get_descriptors_directional(self, atoms, direction):
        """Compute the directional derivative of the descriptors in a given direction using the adjoint force method."""
        
        batch = self._prepare_custom_batch(atoms).to(self.device)
        batch_dict_base = batch.to_dict()
        
        positions = batch_dict_base["positions"].detach().clone().requires_grad_(True)
        theta = self.linear_weights.detach().clone().requires_grad_(True)
        directions= torch.tensor(direction, device=self.device, dtype=torch.float32).reshape_as(positions)
        def forces_direction_fn(theta_var):
            batch_dict = batch_dict_base.copy()
            batch_dict["positions"] = positions
            
            node_desc = self._get_descriptors_from_batch(batch_dict)
            global_desc = node_desc.sum(dim=0)
            
            e_corr = torch.dot(global_desc, theta_var)
            
            force = -torch.autograd.grad(
                outputs=e_corr,
                inputs=positions,
                create_graph=True,
            )[0]
            return torch.dot(force.flatten(), directions.flatten())
        directional_derivative = -jacobian(forces_direction_fn, theta, create_graph=False, vectorize=False)
        
        

        return directional_derivative.detach().cpu().numpy()
    
    def _prepare_custom_batch(self, atoms):
        config = config_from_atoms(atoms)
        data = AtomicData.from_config(config, z_table=self.z_table, cutoff=self.r_max)
        dl = DataLoader([data], batch_size=1, shuffle=False)
        return next(iter(dl))

    def _get_descriptors_from_batch(self, batch_dict, invariants_only=True, num_layers=-1):
        """Homemade function that returns the node descriptors as a torch tensor."""

        num_interactions = int(self.models[0].num_interactions)
        if num_layers == -1:
            num_layers = num_interactions
        
        out = self.models[0](batch_dict, training=False, compute_force=False)
        descriptor = out["node_feats"]
        
        irreps_out = o3.Irreps(str(self.models[0].products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
        per_layer_features[-1] = num_invariant_features

        if invariants_only:
            descriptor = extract_invariant(
                descriptor,
                num_layers=num_layers,
                num_features=num_invariant_features,
                l_max=l_max,
            )
        
        to_keep = sum(per_layer_features[:num_layers])
        descriptor = descriptor[:, :to_keep]
        
        return descriptor
    def check_state(self, atoms):
        # Avoid the array comparison bug in atoms.info
        return Calculator.check_state(self, atoms)