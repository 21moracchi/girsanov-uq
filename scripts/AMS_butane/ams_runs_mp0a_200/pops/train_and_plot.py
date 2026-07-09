import numpy as np
from popsregression import POPSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


use_forces = True
lambda_f = 50
lambda_e = 2
data_dir = '.'
test_size = 0.2
def run_model(model):
    phi_e_raw = np.load(os.path.join(data_dir, f'binary_data/design_matrix_energy_{model}.npy'))
    y_e_raw = np.load(os.path.join(data_dir, f'binary_data/targets_energy_{model}.npy'))
    energy_mace = np.load(os.path.join(data_dir, f'binary_data/mace_energy_{model}.npy'))
    forces_mace = np.load(os.path.join(data_dir, f'binary_data/mace_forces_{model}.npy'))
    if len(phi_e_raw.shape) == 3:
        phi_e_raw = phi_e_raw.sum(axis=1)

    n_total_conf = phi_e_raw.shape[0]
    n_desc = phi_e_raw.shape[1]

    indices = np.arange(n_total_conf)
    idx_train, idx_test = train_test_split(indices, test_size=test_size, random_state=42)

    phi_e_train = phi_e_raw[idx_train]
    y_e_train = y_e_raw[idx_train]
    phi_e_test = phi_e_raw[idx_test]
    y_e_test = y_e_raw[idx_test]

    if use_forces:
        dphi_dr_raw = np.load(os.path.join(data_dir, f'binary_data/design_matrix_forces_{model}.npy'))
        y_f_raw = np.load(os.path.join(data_dir, f'binary_data/targets_forces_{model}.npy'))
        n_dof = y_f_raw.shape[1]

        dphi_dr_reshaped = dphi_dr_raw.reshape(n_total_conf, n_desc, n_dof)
        A_f_all = -dphi_dr_reshaped.transpose(0, 2, 1).reshape(n_total_conf, n_dof, n_desc)

        A_f_train = A_f_all[idx_train].reshape(-1, n_desc)
        y_f_train = y_f_raw[idx_train].flatten()

        A_f_test = A_f_all[idx_test].reshape(-1, n_desc)
        y_f_test = y_f_raw[idx_test].flatten()

        A_train = np.vstack([phi_e_train, A_f_train])
        y_train = np.concatenate([y_e_train, y_f_train])

        energy_mace_train = energy_mace[idx_train]
        E_ref_train = y_e_train + energy_mace_train

        forces_mace_train = forces_mace[idx_train].flatten()
        F_ref_train = y_f_train + forces_mace_train

        E_min = np.min(E_ref_train)
        w_e_raw = np.exp(-(E_ref_train - E_min) / lambda_e)
        w_f_raw = np.exp(-np.abs(F_ref_train) / lambda_f)

        N_E = len(y_e_train)
        w_e_norm = w_e_raw * (N_E / np.sum(w_e_raw))
        w_f_norm = w_f_raw * (N_E / np.sum(w_f_raw))

        train_weights = np.concatenate([w_e_norm, w_f_norm])
    else:
        A_train = phi_e_train
        y_train = y_e_train

        energy_mace_train = energy_mace[idx_train]
        E_ref_train = y_e_train + energy_mace_train
        E_min = np.min(E_ref_train)

        w_e_raw = np.exp(-(E_ref_train - E_min) / lambda_e)
        N_E = len(y_e_train)
        train_weights = w_e_raw * (N_E / np.sum(w_e_raw))

        A_f_train = None
        y_f_train = None
        A_f_test = None
        y_f_test = None

    regressor = POPSRegression(fit_intercept=False, posterior='hypercube')
    regressor.fit(A_train, y_train, sample_weight=train_weights)
    w = regressor.coef_

    def evaluate(phi_e, y_e, A_f=None, y_f=None, label="Train"):
        p_e = phi_e @ w
        m_e = mean_absolute_error(y_e, p_e)
        r_e = np.sqrt(mean_squared_error(y_e, p_e))

        print(f"--- {label} ---")
        print(f"MAE Energie: {m_e:.6f} eV")
        print(f"RMSE Energie: {r_e:.6f} eV")

        if A_f is not None:
            p_f = A_f @ w
            m_f = mean_absolute_error(y_f, p_f)
            r_f = np.sqrt(mean_squared_error(y_f, p_f))
            print(f"MAE Forces:  {m_f:.6f} eV/A")
            print(f"RMSE Forces:  {r_f:.6f} eV/A")
        print("")

    evaluate(
        phi_e_train,
        y_e_train,
        A_f_train if use_forces else None,
        y_f_train if use_forces else None,
        label=f"TRAIN {model}",
    )

    evaluate(
        phi_e_test,
        y_e_test,
        A_f_test if use_forces else None,
        y_f_test if use_forces else None,
        label=f"TEST {model}",
    )

    np.save(f'binary_data/w_corrector_{model}.npy', w)

    print(f"Norme de w: {np.linalg.norm(w)}")
    print(f"Conditionnement de A: {np.linalg.cond(A_train)}")

    regressor.coef_[0], (regressor.misspecification_sigma_[0][0] + regressor.sigma_[0][0])

    y_pred, y_std, y_max, y_min = regressor.predict(
        phi_e_test, return_std=True, return_bounds=True
    )

    theta_samples = regressor.posterior_samples_
    np.save(f'binary_data/posterior_samples_{model}.npy', theta_samples)

    energy_mace_test = energy_mace[idx_test]

    E_pred_tot = y_pred + energy_mace_test
    E_ref_tot = y_e_test + energy_mace_test
    E_min_tot = y_min + energy_mace_test
    E_max_tot = y_max + energy_mace_test

    true_errors = E_pred_tot - E_ref_tot
    true_mae = mean_absolute_error(E_ref_tot, E_pred_tot)
    normalized_true_errors = true_errors / true_mae

    y_ens = phi_e_test @ theta_samples
    ens_errors = y_ens - y_pred[:, np.newaxis]
    normalized_ens_errors = (ens_errors / true_mae).flatten()

    out_of_bounds = (E_ref_tot < E_min_tot) | (E_ref_tot > E_max_tot)
    ev_percentage = np.mean(out_of_bounds) * 100

    ens_mae_relative = np.mean(np.abs(ens_errors))
    mae_percentage = (ens_mae_relative / true_mae) * 100

    return {
        "model": model,
        "E_pred_tot": E_pred_tot,
        "E_ref_tot": E_ref_tot,
        "E_min_tot": E_min_tot,
        "E_max_tot": E_max_tot,
        "normalized_true_errors": normalized_true_errors,
        "normalized_ens_errors": normalized_ens_errors,
        "mae_percentage": mae_percentage,
        "ev_percentage": ev_percentage,
    }


results = [run_model("omat0"), run_model("mp0a")]

# %%
n_points_plot = 200

default_fontsize = 18
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for row_idx, result in enumerate(results):
    ax1 = axes[row_idx, 0]
    ax2 = axes[row_idx, 1]

    bins = np.linspace(-5, 5, 30)
    ax1.hist(
        result["normalized_true_errors"],
        bins=bins,
        density=True,
        histtype='step',
        color='black',
        linewidth=2,
    )

    label_uq = r"$\pi_{\mathcal{H}}^*$,  EV: %.1f%%" % (
        
        result["ev_percentage"],
    )
    ax1.hist(
        result["normalized_ens_errors"],
        bins=bins,
        density=True,
        histtype='step',
        color='tab:green',
        linewidth=2,
        label=label_uq,
    )

    ax1.set_yscale('log')
    ax1.set_xlabel('Error / True MAE', fontsize=default_fontsize)
    ax1.set_ylabel('Density', fontsize=default_fontsize)
    ax1.legend(loc='upper left')
    ax1.set_ylim(bottom=1e-3)
    
    idx_plot = np.random.choice(len(result["E_pred_tot"]), size=min(n_points_plot, len(result["E_pred_tot"])), replace=False)
    sort_idx = np.argsort(result["E_pred_tot"][idx_plot])
    E_pred_sorted = result["E_pred_tot"][idx_plot][sort_idx]
    E_min_sorted = result["E_min_tot"][idx_plot][sort_idx]
    E_max_sorted = result["E_max_tot"][idx_plot][sort_idx]

    ax2.fill_between(E_pred_sorted, E_min_sorted, E_max_sorted, color='tab:green', alpha=0.4)
    ax2.scatter(result["E_pred_tot"][idx_plot], result["E_ref_tot"][idx_plot], color='black', s=15, zorder=3)

    min_val = min(result["E_pred_tot"][idx_plot].min(), result["E_ref_tot"][idx_plot].min())
    max_val = max(result["E_pred_tot"][idx_plot].max(), result["E_ref_tot"][idx_plot].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    ax2.set_xlabel('Total Energy MLE Prediction [eV]', fontsize=default_fontsize)
    ax2.set_ylabel('Total Reference Energy [eV]', fontsize=default_fontsize)
row_titles = ['OMAT-0', 'MP-0a']
axes[0, 0].set_title(row_titles[0])
axes[1, 0].set_title(row_titles[1])
plt.tight_layout()
plt.savefig('uq_pointwise_total_energy_2x2.png', dpi=300)
plt.show()
