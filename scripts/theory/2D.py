# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import sys, os

sys.path.insert(0, os.path.abspath('scripts/theory'))
from utils import (solve_committor_upwind, q_func_factory, weighted_diffusion_system, build_symbolic_helpers)
from ase import units

# %%
gamma_trap = 0    

D = ['(x**2-1)**2', 'y**2']
theta = np.array([0.1, 0.1])
theta_len = len(theta)
temperature_K = 200
beta = 1 / (units.kB * temperature_K)
gamma = 0.1 / units.fs

helpers = build_symbolic_helpers(D, theta_len=theta_len)
    
# %%
gradV = helpers['gradV_func']

# %%
x_min, x_max = -1.5, 1.5
y_min, y_max = -2.0, 2.0
dx = 0.005
dy = 0.005
discretisation = int((x_max - x_min) / dx) + 1

q = solve_committor_upwind(discretisation, discretisation, (x_min, x_max), (y_min, y_max), theta, gradV, beta=beta, gamma=gamma)
X, Y = np.meshgrid(np.linspace(x_min, x_max, discretisation), np.linspace(y_min, y_max, discretisation), indexing='ij')

V = helpers['V_func'](X, Y, theta)
V_clipped = np.clip(V, 0, 5)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].contourf(X, Y, V_clipped, levels=50, cmap='RdBu_r')
fig.colorbar(im0, ax=axes[0], label='Potential V(x,y)')
axes[0].set_title('Double well potential')

log_q = np.log(q)
im1 = axes[1].contourf(X, Y, log_q, levels=50, cmap='RdBu_r')
fig.colorbar(im1, ax=axes[1], label='Log Committor function')
axes[1].set_title('Log Committor function')
plt.tight_layout()
plt.savefig('potential_and_committor.png', dpi=300)
# %%
q_func = q_func_factory(x_min, y_min, dx, dy, q)
q_func(-0.8, 0)

# %%
x = np.linspace(x_min, x_max, discretisation)
y = np.linspace(y_min, y_max, discretisation)
X, Y = np.meshgrid(x, y, indexing='ij')
radius_P = 0.1
mask_P = np.sqrt((X - 1.0)**2 + Y**2) <= radius_P
eps_q = 1e-12
q_pos = np.clip(q, eps_q, 1.0)
V_field = helpers['V_func'](X, Y, theta)
M_field = np.exp(-beta * V_field) * (q_pos**2)

gradVx, gradVy = helpers['gradV_func'](X, Y, theta)
gradV_field = np.stack([gradVx, gradVy], axis=-1)
gradD_field = helpers['gradientD_func'](X, Y)
lapD_field = helpers['laplacianD_func'](X, Y)
g_field = 0.5 * lapD_field - 0.5 * beta * np.sum(gradD_field * gradV_field[:, :, None, :], axis=-1)

A_u_base, rhs_u_base = weighted_diffusion_system(M_field, mask_P, dx, dy)
nx, ny = discretisation, discretisation
interior_mask = ~(mask_P | (X == x_min) | (X == x_max) | (Y == y_min) | (Y == y_max))
u = np.zeros((nx, ny, theta_len))

for k in range(theta_len):
    A_u = A_u_base.tolil()
    rhs_u = rhs_u_base.copy()
    source_u = -beta * (M_field * g_field[:, :, k])
    rhs_u[interior_mask.ravel()] += source_u.ravel()[interior_mask.ravel()]
    u[:, :, k] = spsolve(A_u.tocsr(), rhs_u).reshape((nx, ny))

A_w_base, rhs_w_base = weighted_diffusion_system(M_field, mask_P, dx, dy)
w = np.zeros((nx, ny, theta_len, theta_len))

for k in range(theta_len):
    for l in range(theta_len):
        A_w = A_w_base.tolil()
        rhs_w = rhs_w_base.copy()
        source_w = -beta * (M_field * (g_field[:, :, k] * u[:, :, l] + g_field[:, :, l] * u[:, :, k]))
        rhs_w[interior_mask.ravel()] += source_w.ravel()[interior_mask.ravel()]
        w[:, :, k, l] = spsolve(A_w.tocsr(), rhs_w).reshape((discretisation, discretisation))

# %%
A_T_base, rhs_T_base = weighted_diffusion_system(M_field, mask_P, dx, dy)
rhs_T = rhs_T_base.copy()
source_T = -beta * gamma * M_field
rhs_T[interior_mask.ravel()] += source_T.ravel()[interior_mask.ravel()]
T_reactive = spsolve(A_T_base, rhs_T).reshape((discretisation, discretisation))

A_tau2_base, rhs_tau2_base = weighted_diffusion_system(M_field, mask_P, dx, dy)
rhs_tau2 = rhs_tau2_base.copy()
source_tau2 = -2.0 * beta * gamma * (M_field * T_reactive)
rhs_tau2[interior_mask.ravel()] += source_tau2.ravel()[interior_mask.ravel()]
tau2_reactive = spsolve(A_tau2_base, rhs_tau2).reshape((discretisation, discretisation))

Var_T_reactive = np.maximum(tau2_reactive - T_reactive**2, 0.0)
sigma_T_reactive = np.sqrt(Var_T_reactive)
x0 = (-0.8, 0.0)
i0 = int((x0[0] - x_min) / dx)
j0 = int((x0[1] - y_min) / dy)
i0 = np.clip(i0, 0, discretisation - 1)
j0 = np.clip(j0, 0, discretisation - 1)

print(f"T(x0) = E*[tau | x0={x0}] = {T_reactive[i0, j0] / units.fs:.6f} fs")
print(f"sigma(T(x0)) = sqrt(Var*[tau | x0={x0}]) = {sigma_T_reactive[i0, j0] / units.fs:.6f} fs")
fig, ax = plt.subplots()
im0 = ax.contourf(X, Y, T_reactive / units.fs, levels=50, cmap='viridis')
fig.colorbar(im0, ax=ax, label='T(x,y) [fs]')
ax.set_title('Temps moyen reactif T (fs)')

# %%
D_vals = helpers['D_func'](X, Y)
x_cible = (1.0, 0.0)
D_on_boundary_mean = helpers['D_func'](np.array([[x_cible[0]]]), np.array([[x_cible[1]]]))[0, 0, :]

x0 = (-0.8, 0.0)
i0 = int((x0[0] - x_min) / dx)
j0 = int((x0[1] - y_min) / dy)
i0 = np.clip(i0, 0, discretisation - 1)
j0 = np.clip(j0, 0, discretisation - 1)

D_x0 = helpers['D_func'](np.array([[x0[0]]]), np.array([[x0[1]]]))[0, 0, :]
u_x0 = u[i0, j0, :]
w_x0 = w[i0, j0, :, :]

E_a = 0.5 * beta * (D_x0 - D_on_boundary_mean) + u_x0
Cov_a = w_x0 - np.outer(u_x0, u_x0)

print('D_on_boundary_mean =', D_on_boundary_mean)
print('D_x0 =', D_x0)
print('u(x0) by component =', u_x0)
print('w(x0) matrix =\n', w_x0)
print('E[a] by component =', E_a)
print('Cov[a] matrix =\n', Cov_a)
print('probability of transition (q at x0) =', q_func(x0[0], x0[1]))

# %%
n_rep = int(np.max(np.diag(Cov_a)))
print('n_rep =', n_rep)
# %%
mu_field = np.exp(-beta * V_field)

A_T_mc_base, rhs_T_mc_base = weighted_diffusion_system(mu_field, mask_P, dx, dy)
rhs_T_mc = rhs_T_mc_base.copy()
source_T_mc = -beta * gamma * mu_field
rhs_T_mc[interior_mask.ravel()] += source_T_mc.ravel()[interior_mask.ravel()]
T_mc = spsolve(A_T_mc_base, rhs_T_mc).reshape((discretisation, discretisation))

A_tau2_mc_base, rhs_tau2_mc_base = weighted_diffusion_system(mu_field, mask_P, dx, dy)
rhs_tau2_mc = rhs_tau2_mc_base.copy()
source_tau2_mc = -2.0 * beta * gamma * (mu_field * T_mc)
rhs_tau2_mc[interior_mask.ravel()] += source_tau2_mc.ravel()[interior_mask.ravel()]
tau2_mc = spsolve(A_tau2_mc_base, rhs_tau2_mc).reshape((discretisation, discretisation))

Var_T_mc = np.maximum(tau2_mc - T_mc**2, 0.0)
sigma_T_mc = np.sqrt(Var_T_mc)

u_mc = np.zeros((nx, ny, theta_len))
A_u_mc_base, rhs_u_mc_base = weighted_diffusion_system(mu_field, mask_P, dx, dy)

for k in range(theta_len):
    A_u_mc = A_u_mc_base.tolil()
    rhs_u_mc = rhs_u_mc_base.copy()
    source_u_mc = -beta * (mu_field * g_field[:, :, k])
    rhs_u_mc[interior_mask.ravel()] += source_u_mc.ravel()[interior_mask.ravel()]
    u_mc[:, :, k] = spsolve(A_u_mc.tocsr(), rhs_u_mc).reshape((nx, ny))

w_mc = np.zeros((nx, ny, theta_len, theta_len))
A_w_mc_base, rhs_w_mc_base = weighted_diffusion_system(mu_field, mask_P, dx, dy)

for k in range(theta_len):
    for l in range(theta_len):
        A_w_mc = A_w_mc_base.tolil()
        rhs_w_mc = rhs_w_mc_base.copy()
        source_w_mc = -beta * (mu_field * (g_field[:, :, k] * u_mc[:, :, l] + g_field[:, :, l] * u_mc[:, :, k]))
        rhs_w_mc[interior_mask.ravel()] += source_w_mc.ravel()[interior_mask.ravel()]
        w_mc[:, :, k, l] = spsolve(A_w_mc.tocsr(), rhs_w_mc).reshape((nx, ny))

u_mc_x0 = u_mc[i0, j0, :]
w_mc_x0 = w_mc[i0, j0, :, :]
E_a_mc = 0.5 * beta * (D_x0 - D_on_boundary_mean) + u_mc_x0
Cov_a_mc = w_mc_x0 - np.outer(u_mc_x0, u_mc_x0)

print("\n--- Comparaison : Trajectoires Standard (MC) vs Réactives ---")
print(f"T_MC(x0) = {T_mc[i0, j0] / units.fs:.6e} fs")
print(f"sigma_T_MC(x0) = {sigma_T_mc[i0, j0] / units.fs:.6e} fs")
print('E_MC[a] =', E_a_mc)
print('Cov_MC[a] matrix =\n', Cov_a_mc)

# %% 
n_rep_mc = int(np.max(np.diag(Cov_a_mc)))
print('n_rep_MC =', n_rep_mc)


# %% Calcul exact de la variance du poids de Girsanov total M(X)
# 1. Définition d'une perturbation arbitraire (ex: delta_theta)
delta_theta = np.array([-0, 0.0140]) 

# 2. Calcul de U et de son gradient sur la grille
U_field = helpers['V_func'](X, Y, delta_theta)
gradUx, gradUy = helpers['gradV_func'](X, Y, delta_theta)
gradU_field = np.stack([gradUx, gradUy], axis=-1)

# 3. Calcul de la fonction scalaire h(x)
# h(x) = g_U(x) - (beta/4)*||grad U||^2 = sum(delta_theta * g) - (beta/4)*||grad U||^2
h_field = np.sum(g_field * delta_theta, axis=-1) - 0.25 * beta * np.sum(gradU_field**2, axis=-1)

# 4. Résolution des EDPs pour v1 (k=1) et v2 (k=2)
v = np.zeros((nx, ny, 2))

for k_idx, k_val in enumerate([1, 2]):
    A_v = A_u_base.tolil() # On repart de la matrice de diffusion de base
    rhs_v = np.zeros(nx * ny)
    
    # Condition aux limites: v_k = 1 sur l'état produit P
    rhs_v[mask_P.ravel()] = 1.0
    
    # Terme source proportionnel à v_k: vient s'ajouter sur la diagonale
    # L* v + k*h*v = 0  =>  div(M grad v) + (gamma * beta * M * k * h) * v = 0
    diag_addition = (gamma * beta * M_field * k_val * h_field).ravel()
    
    for idx in np.where(interior_mask.ravel())[0]:
        
        A_v[idx, idx] += diag_addition[idx]
    v[:, :, k_idx] = spsolve(A_v.tocsr(), rhs_v).reshape((nx, ny))

v1 = v[:, :, 0]
v2 = v[:, :, 1]

# 5. Évaluation en x0
U_x0 = U_field[i0, j0]
U_P_mean = U_field[mask_P].mean() # U(P)

E_M = np.exp(0.5 * beta * (U_x0 - U_P_mean)) * v1[i0, j0]
Var_M = np.exp(beta * (U_x0 - U_P_mean)) * (v2[i0, j0] - v1[i0, j0]**2)

print("\n--- Analyse du Poids de Girsanov Total M(X) ---")
print(f"Perturbation delta_theta = {delta_theta}")
print(f"E*[M(X) | x0]   = {E_M:.6e} (Théoriquement proche de 1 si perturbation faible)")
print(f"Var*[M(X) | x0] = {Var_M:.6e}")
print(f"Ecart-type      = {np.sqrt(Var_M):.6e}")


v_mc = np.zeros((nx, ny, 2))

for k_idx, k_val in enumerate([1, 2]):
    A_v_mc = A_u_mc_base.tolil()
    rhs_v_mc = np.zeros(nx * ny)
    
    rhs_v_mc[mask_P.ravel()] = 1.0
    
    diag_addition_mc = (gamma * beta * mu_field * k_val * h_field).ravel()
    
    for idx in np.where(interior_mask.ravel())[0]:
        A_v_mc[idx, idx] += diag_addition_mc[idx]
        
    v_mc[:, :, k_idx] = spsolve(A_v_mc.tocsr(), rhs_v_mc).reshape((nx, ny))

v1_mc = v_mc[:, :, 0]
v2_mc = v_mc[:, :, 1]

E_M_mc = np.exp(0.5 * beta * (U_x0 - U_P_mean)) * v1_mc[i0, j0]
Var_M_mc = np.exp(beta * (U_x0 - U_P_mean)) * (v2_mc[i0, j0] - v1_mc[i0, j0]**2)

print("\n--- Analyse du Poids de Girsanov Total M(X) (Trajectoires Standards) ---")
print(f"E_MC*[M(X) | x0]   = {E_M_mc:.6e}")
print(f"Var_MC*[M(X) | x0] = {Var_M_mc:.6e}")
print(f"Ecart-type_MC      = {np.sqrt(Var_M_mc):.6e}")