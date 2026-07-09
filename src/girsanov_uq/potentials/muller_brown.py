import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

class RuggedMullerBrown(Calculator):
    implemented_properties = [
        'energy',
        'forces',
        'stress',
        'descriptors',
        'grad_descriptors'
    ]

    default_parameters = {
        'N_dim': 3,
        'A': [-200.0, -100.0, -170.0, 15.0],
        'gamma_': 1.0,
        'K': [1.0],
        'omega': 10.0
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

        N_dim = self.parameters['N_dim']
        A = np.asarray(self.parameters['A'], dtype=float)
        gamma = self.parameters['gamma_']
        K = np.asarray(self.parameters['K'], dtype=float)
        omega = self.parameters['omega']

        pos = self.atoms.positions
        N_atoms = len(pos)
        r = pos.flatten()
        
        if len(r) < N_dim:
            raise ValueError(f"Le système ne possède pas assez de degrés de liberté ({len(r)}) pour la dimension N={N_dim}.")
        if len(K) != N_dim - 2:
            raise ValueError(f"Le tableau K doit être de taille N_dim - 2 (attendu {N_dim - 2}, reçu {len(K)}).")

        x = r[0]
        y = r[1]

        a = np.array([-1.0, -1.0, -6.5, 0.7])
        b = np.array([0.0, 0.0, 11.0, 0.6])
        c = np.array([-10.0, -10.0, -6.5, 0.7])
        X = np.array([1.0, 0.0, -0.5, -1.0])
        Y = np.array([0.0, 0.5, 1.5, 1.0])

        N_desc = N_dim + 3
        D = np.zeros(N_desc)
        grad_D_flat = np.zeros((N_desc, len(r)))

        dx = x - X
        dy = y - Y
        
        for i in range(4):
            D[i] = np.exp(a[i] * dx[i]**2 + b[i] * dx[i] * dy[i] + c[i] * dy[i]**2)
            grad_D_flat[i, 0] = D[i] * (2.0 * a[i] * dx[i] + b[i] * dy[i])
            grad_D_flat[i, 1] = D[i] * (b[i] * dx[i] + 2.0 * c[i] * dy[i])

        D[4] = np.sin(omega * x) * np.sin(omega * y)
        grad_D_flat[4, 0] = omega * np.cos(omega * x) * np.sin(omega * y)
        grad_D_flat[4, 1] = omega * np.sin(omega * x) * np.cos(omega * y)

        for j in range(2, N_dim):
            idx = j + 3
            D[idx] = 0.5 * r[j]**2
            grad_D_flat[idx, j] = r[j]

        params = np.concatenate((A, [gamma], K))
        
        energy = np.dot(params, D)
        
        forces_flat = -np.dot(params, grad_D_flat)
        forces = forces_flat.reshape(N_atoms, 3)
        
        grad_D = grad_D_flat.reshape(N_desc, N_atoms, 3)

        self.results['energy'] = energy
        self.results['forces'] = forces
        # Contrainte virielle (Voigt 6 composantes) pour compatibilité ASE.
        volume = float(self.atoms.get_volume())
        if volume > 0.0:
            stress_tensor = -np.einsum('ia,ib->ab', pos, forces) / volume
            self.results['stress'] = full_3x3_to_voigt_6_stress(stress_tensor)
        else:
            self.results['stress'] = np.zeros(6)
        self.results['descriptors'] = D
        self.results['grad_descriptors'] = grad_D

    def get_descriptors_jacobian(self, atoms):
        self.calculate(
            atoms, 
            properties=['descriptors', 'grad_descriptors'], 
            system_changes='all'
        )
        grad_descriptors = self.results['grad_descriptors']
        # Convention used by the reweighting code: (N_atoms, D, 3)
        # Internal storage here is (D, N_atoms, 3).
        return grad_descriptors.transpose(1, 0, 2)

    def plot_2d_potential(
        self,
        xlim=(-1.8, 1.2),
        ylim=(-0.2, 2.2),
        n_points=300,
        r_fixed=None,
        levels=60,
        clip_percentiles=(1.0, 99.0),
        filled=True,
        cmap='viridis',
        ax=None,
        colorbar=True,
        points=None,
        point_kwargs=None,
        show=True
    ):
        """
        Trace une coupe 2D du potentiel en fonction de (x, y).

        Paramètres
        ----------
        xlim, ylim : tuple(float, float)
            Bornes des axes x et y.
        n_points : int
            Nombre de points par axe pour la grille.
        r_fixed : array-like ou None
            Valeurs fixées des coordonnées supplémentaires r[2:],
            de longueur N_dim - 2. Si None, elles sont fixées à zéro.
        levels : int
            Nombre de niveaux de contour.
        clip_percentiles : tuple(float, float) ou None
            Percentiles (pmin, pmax) utilisés pour clipper les valeurs
            de V avant affichage afin d'améliorer le contraste.
            Mettre None pour désactiver le clipping.
        filled : bool
            Si True, utilise contourf; sinon contour.
        cmap : str
            Colormap matplotlib.
        ax : matplotlib.axes.Axes ou None
            Axe sur lequel tracer. Si None, crée une nouvelle figure.
        colorbar : bool
            Ajoute une barre de couleur.
        points : array-like ou None
            Liste/array de points 2D de forme (N, 2) à superposer sur la carte.
            Les points sont tracés sans label.
        point_kwargs : dict ou None
            Options passées à ax.scatter pour personnaliser l'apparence des points.
        show : bool
            Appelle plt.show() si True.

        Retour
        ------
        ax : matplotlib.axes.Axes
            Axe contenant le tracé.
        """
        import matplotlib.pyplot as plt

        N_dim = int(self.parameters['N_dim'])
        A = np.asarray(self.parameters['A'], dtype=float)
        gamma = float(self.parameters['gamma_'])
        K = np.asarray(self.parameters['K'], dtype=float)
        omega = float(self.parameters['omega'])

        if len(A) != 4:
            raise ValueError(f"Le tableau A doit être de taille 4 (reçu {len(A)}).")
        if len(K) != N_dim - 2:
            raise ValueError(
                f"Le tableau K doit être de taille N_dim - 2 (attendu {N_dim - 2}, reçu {len(K)})."
            )

        if r_fixed is None:
            r_fixed = np.zeros(max(0, N_dim - 2), dtype=float)
        else:
            r_fixed = np.asarray(r_fixed, dtype=float)
            if len(r_fixed) != N_dim - 2:
                raise ValueError(
                    f"r_fixed doit être de taille N_dim - 2 (attendu {N_dim - 2}, reçu {len(r_fixed)})."
                )

        x = np.linspace(xlim[0], xlim[1], int(n_points))
        y = np.linspace(ylim[0], ylim[1], int(n_points))
        Xg, Yg = np.meshgrid(x, y, indexing='xy')

        a = np.array([-1.0, -1.0, -6.5, 0.7])
        b = np.array([0.0, 0.0, 11.0, 0.6])
        c = np.array([-10.0, -10.0, -6.5, 0.7])
        X0 = np.array([1.0, 0.0, -0.5, -1.0])
        Y0 = np.array([0.0, 0.5, 1.5, 1.0])

        V_mb = np.zeros_like(Xg, dtype=float)
        for i in range(4):
            dx = Xg - X0[i]
            dy = Yg - Y0[i]
            V_mb += A[i] * np.exp(a[i] * dx**2 + b[i] * dx * dy + c[i] * dy**2)

        V_rugged = gamma * np.sin(omega * Xg) * np.sin(omega * Yg)

        V_extra = np.zeros_like(Xg, dtype=float)
        if N_dim > 2:
            V_extra = 0.5 * np.sum(K * r_fixed**2)

        V = V_mb + V_rugged + V_extra

        V_plot = V
        if clip_percentiles is not None:
            if len(clip_percentiles) != 2:
                raise ValueError(
                    f"clip_percentiles doit contenir 2 valeurs (reçu {len(clip_percentiles)})."
                )
            pmin, pmax = float(clip_percentiles[0]), float(clip_percentiles[1])
            if not (0.0 <= pmin < pmax <= 100.0):
                raise ValueError(
                    "clip_percentiles doit vérifier 0 <= pmin < pmax <= 100."
                )
            vmin, vmax = np.percentile(V, [pmin, pmax])
            V_plot = np.clip(V, vmin, vmax)

        created_fig = False
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
            created_fig = True

        if filled:
            contour = ax.contourf(Xg, Yg, V_plot, levels=levels, cmap=cmap)
        else:
            contour = ax.contour(Xg, Yg, V_plot, levels=levels, cmap=cmap)

        if points is not None:
            pts = np.asarray(points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError(
                    f"points doit être de forme (N, 2) (reçu {pts.shape})."
                )
            kwargs = {
                's': 24,
                'c': 'white',
                'edgecolors': 'black',
                'linewidths': 0.6,
                'zorder': 3,
            }
            if point_kwargs is not None:
                kwargs.update(point_kwargs)
            ax.scatter(pts[:, 0], pts[:, 1], **kwargs)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Rugged Muller-Brown potential (2D cut)')

        if colorbar:
            plt.colorbar(contour, ax=ax, label='V(x, y)')

        if show and created_fig:
            plt.tight_layout()
            plt.show()

        return ax