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
test_size = 0.6
n_max_hist_plot = 4000
n_max_scatter_plot = 150
n_posterior_samples = 2000
plot_rng_seed = 42
histogram_error_units = "ev"  # "mae" for Error / True MAE, "ev" for raw errors.
histogram_n_bins = 50
prediction_interval_mode = "95"  # "95" for MLE +/- 2 std, "minmax" for POPS sample min/max.

def sample_indices(n_available, n_samples, rng):
    return rng.choice(n_available, size=min(n_samples, n_available), replace=False)

def share_histogram_ylim(axes):
    y_min = min(ax.get_ylim()[0] for ax in axes)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(y_min, y_max)

def get_interval_bounds(prediction, std, baseline, ensemble_prediction=None):
    if prediction_interval_mode == "95":
        return (prediction - 2 * std) + baseline, (prediction + 2 * std) + baseline

    if prediction_interval_mode == "minmax":
        if ensemble_prediction is None:
            raise ValueError("ensemble_prediction is required for minmax intervals")
        ensemble_prediction_total = ensemble_prediction + baseline[:, np.newaxis]
        return (
            np.min(ensemble_prediction_total, axis=1),
            np.max(ensemble_prediction_total, axis=1),
        )

    raise ValueError('prediction_interval_mode must be "95" or "minmax"')

def get_histogram_plot_settings(results, error_key, ensemble_key, norm_error_key,
                                norm_ensemble_key, mae_xlabel, ev_xlabel):
    if histogram_error_units == "mae":
        return {
            "error_key": norm_error_key,
            "ensemble_key": norm_ensemble_key,
            "bins": np.linspace(-6.5, 6.5, histogram_n_bins),
            "xlim": (-6, 6),
            "xlabel": mae_xlabel,
        }
    if histogram_error_units == "ev":
        return {
            "error_key": error_key,
            "ensemble_key": ensemble_key,
            "bins": np.linspace(-0.055, 0.055, histogram_n_bins),
            "xlim": (-0.05, 0.05),
            "xlabel": ev_xlabel,
        }
    if histogram_error_units != "ev":
        raise ValueError('histogram_error_units must be "mae" or "ev"')

    histogram_values = []
    for result in results:
        histogram_values.extend([result[error_key], result[ensemble_key]])
    finite_values = np.concatenate(histogram_values)
    finite_values = finite_values[np.isfinite(finite_values)]
    max_abs = np.max(np.abs(finite_values)) if finite_values.size else 1.0
    if max_abs == 0:
        max_abs = 1.0

    bin_limit = max_abs * 1.05
    return {
        "error_key": error_key,
        "ensemble_key": ensemble_key,
        "bins": np.linspace(-bin_limit, bin_limit, histogram_n_bins),
        "xlim": (-max_abs, max_abs),
        "xlabel": ev_xlabel,
    }

def run_model(model):
    phi_e_raw = np.load(os.path.join(data_dir, f'../ams_runs_{model}/pops/binary_data/design_matrix_energy_{model}.npy'))
    y_e_raw = np.load(os.path.join(data_dir, f'../ams_runs_{model}/pops/binary_data/targets_energy_{model}.npy'))
    energy_mace = np.load(os.path.join(data_dir, f'../ams_runs_{model}/pops/binary_data/mace_energy_{model}.npy'))
    forces_mace = np.load(os.path.join(data_dir, f'../ams_runs_{model}/pops/binary_data/mace_forces_{model}.npy'))
    if len(phi_e_raw.shape) == 3:
        phi_e_raw = phi_e_raw.sum(axis=1)

    n_total_conf = phi_e_raw.shape[0]
    n_desc = phi_e_raw.shape[1]

    indices = np.arange(n_total_conf)
    idx_train, idx_test = train_test_split(indices, test_size=test_size, random_state=42,shuffle = False)

    phi_e_train = phi_e_raw[idx_train]
    y_e_train = y_e_raw[idx_train]
    phi_e_test = phi_e_raw[idx_test]
    y_e_test = y_e_raw[idx_test]

    if use_forces:
        dphi_dr_raw = np.load(os.path.join(data_dir, f'../ams_runs_{model}/pops/binary_data/design_matrix_forces_{model}.npy'))
        y_f_raw = np.load(os.path.join(data_dir, f'../ams_runs_{model}/pops/binary_data/targets_forces_{model}.npy'))
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

    regressor =  POPSRegression(
    fit_intercept=False, 
    posterior='hypercube',

)
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

    np.save(f'../ams_runs_{model}/pops/binary_data/w_corrector_{model}.npy', w)

    print(f"Norme de w: {np.linalg.norm(w)}")
    print(f"Conditionnement de A: {np.linalg.cond(A_train)}")

    y_pred, y_std = regressor.predict(
        phi_e_test, return_std=True, return_bounds=False
    )

    theta_samples,cov_matrix = regressor._sample_hypercube(n_posterior_samples)
    np.save(f'../ams_runs_{model}/pops/binary_data/posterior_samples_{model}.npy', theta_samples)
    np.save(f'../ams_runs_{model}/pops/binary_data/cov_matrix_{model}.npy', cov_matrix)

    energy_mace_test = energy_mace[idx_test]

    E_pred_tot = y_pred + energy_mace_test
    E_ref_tot = y_e_test + energy_mace_test

    true_errors = E_pred_tot - E_ref_tot
    true_mae = mean_absolute_error(E_ref_tot, E_pred_tot)
    normalized_true_errors = true_errors / true_mae

    plot_rng = np.random.default_rng(plot_rng_seed + sum(ord(char) for char in model))
    energy_hist_idx = sample_indices(len(E_pred_tot), n_max_hist_plot, plot_rng)
    energy_scatter_idx = sample_indices(len(E_pred_tot), n_max_scatter_plot, plot_rng)

    y_ens = phi_e_test[energy_hist_idx] @ theta_samples
    ens_errors = y_ens - y_pred[energy_hist_idx, np.newaxis]
    normalized_ens_errors = (ens_errors / true_mae).flatten()

    y_ens_scatter = phi_e_test[energy_scatter_idx] @ theta_samples
    E_pred_plot = E_pred_tot[energy_scatter_idx]
    E_ref_plot = E_ref_tot[energy_scatter_idx]
    E_min_plot, E_max_plot = get_interval_bounds(
        y_pred[energy_scatter_idx],
        y_std[energy_scatter_idx],
        energy_mace_test[energy_scatter_idx],
        ensemble_prediction=y_ens_scatter,
    )

    out_of_bounds = (E_ref_plot < E_min_plot) | (E_ref_plot > E_max_plot)
    ev_percentage = np.mean(out_of_bounds) * 100

    ens_mae_relative = np.mean(np.abs(ens_errors))
    mae_percentage = (ens_mae_relative / true_mae) * 100

    force_results = {}
    if use_forces and A_f_test is not None:
        forces_mace_test = forces_mace[idx_test].flatten()
        y_f_pred = A_f_test @ w
        F_pred_tot = y_f_pred + forces_mace_test
        F_ref_tot = y_f_test + forces_mace_test

        force_errors = F_pred_tot - F_ref_tot
        force_mae = mean_absolute_error(F_ref_tot, F_pred_tot)
        normalized_force_errors = force_errors / force_mae

        force_hist_idx = sample_indices(len(F_pred_tot), n_max_hist_plot, plot_rng)
        force_scatter_idx = sample_indices(
            len(F_pred_tot), n_max_scatter_plot, plot_rng
        )
        y_f_pred_plot, y_f_std_plot = regressor.predict(
            A_f_test[force_scatter_idx], return_std=True, return_bounds=False
        )

        F_pred_plot = y_f_pred_plot + forces_mace_test[force_scatter_idx]
        F_ref_plot = F_ref_tot[force_scatter_idx]
        y_f_ens_scatter = A_f_test[force_scatter_idx] @ theta_samples
        F_min_plot, F_max_plot = get_interval_bounds(
            y_f_pred_plot,
            y_f_std_plot,
            forces_mace_test[force_scatter_idx],
            ensemble_prediction=y_f_ens_scatter,
        )

        force_out_of_bounds = (F_ref_plot < F_min_plot) | (F_ref_plot > F_max_plot)
        force_ev_percentage = np.mean(force_out_of_bounds) * 100

        y_f_ens_plot = A_f_test[force_hist_idx] @ theta_samples
        force_ens_errors = y_f_ens_plot - y_f_pred[force_hist_idx, np.newaxis]
        normalized_force_ens_errors = (force_ens_errors / force_mae).flatten()

        force_results = {
            "F_pred_plot": F_pred_plot,
            "F_ref_plot": F_ref_plot,
            "F_min_plot": F_min_plot,
            "F_max_plot": F_max_plot,
            "force_errors_plot": force_errors[force_hist_idx],
            "force_ens_errors": force_ens_errors.flatten(),
            "normalized_force_errors_plot": normalized_force_errors[force_hist_idx],
            "normalized_force_ens_errors": normalized_force_ens_errors,
            "force_mae": force_mae,
            "force_ev_percentage": force_ev_percentage,
        }

    return {
        "model": model,
        "E_pred_tot": E_pred_tot,
        "E_ref_tot": E_ref_tot,
        "E_pred_plot": E_pred_plot,
        "E_ref_plot": E_ref_plot,
        "E_min_plot": E_min_plot,
        "E_max_plot": E_max_plot,
        "E_scatter_idx": energy_scatter_idx,
        "true_errors": true_errors[energy_hist_idx],
        "ens_errors": ens_errors.flatten(),
        "normalized_true_errors": normalized_true_errors[energy_hist_idx],
        "normalized_ens_errors": normalized_ens_errors,
        "mae_percentage": mae_percentage,
        "ev_percentage": ev_percentage,
        **force_results,
    }

results = [run_model("omat0"), run_model("mp0a")]
# %%
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

energy_hist_settings = get_histogram_plot_settings(
    results,
    error_key="true_errors",
    ensemble_key="ens_errors",
    norm_error_key="normalized_true_errors",
    norm_ensemble_key="normalized_ens_errors",
    mae_xlabel="Error / True MAE",
    ev_xlabel="Energy Error [eV]",
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[1, 0].sharex(axes[0, 0])
for row_idx, result in enumerate(results):
    ax1 = axes[row_idx, 0]
    ax2 = axes[row_idx, 1]

    ax1.hist(
        result[energy_hist_settings["error_key"]],
        bins=energy_hist_settings["bins"],
        density=True,
        histtype='step',
        color='black',
        linewidth=2,
    )

    label_uq = r"$\pi_{\mathrm{POPS}}$,  EV: %.1f%%" % (
        result["ev_percentage"],
    )
    ax1.hist(
        result[energy_hist_settings["ensemble_key"]],
        bins=energy_hist_settings["bins"],
        density=True,
        histtype='step',
        color='tab:green',
        linewidth=2,
        
    )
    
    ax1.set_yscale('log')
    ax1.set_xlabel(energy_hist_settings["xlabel"], fontsize=default_fontsize)
    ax1.set_ylabel('Density', fontsize=default_fontsize)
    # ax1.set_ylim(bottom=1e-6)
    ax1.set_xlim(*energy_hist_settings["xlim"])
    sort_idx = np.argsort(result["E_pred_plot"])
    E_pred_sorted = result["E_pred_plot"][sort_idx]
    E_min_sorted = result["E_min_plot"][sort_idx]
    E_max_sorted = result["E_max_plot"][sort_idx]

    ax2.fill_between(E_pred_sorted, E_min_sorted, E_max_sorted, color='tab:green', alpha=0.4,label=label_uq)
    ax2.scatter(result["E_pred_plot"], result["E_ref_plot"], color='black', s=15, zorder=3)

    min_val = min(result["E_pred_plot"].min(), result["E_ref_plot"].min())
    max_val = max(result["E_pred_plot"].max(), result["E_ref_plot"].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax2.legend(loc='lower right')

    ax2.set_xlabel('Total Energy MLE Prediction [eV]', fontsize=default_fontsize)
    ax2.set_ylabel('Total Reference Energy [eV]', fontsize=default_fontsize)

row_titles = ['OMAT-0', 'MP-0a']
share_histogram_ylim(axes[:, 0])
axes[0, 0].set_title(row_titles[0])
axes[1, 0].set_title(row_titles[1])
plt.tight_layout()

axes[0, 0].text(-0.1, 1.05, '(a)', transform=axes[0, 0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
axes[0,1].text(-0.1, 1.05, '(b)', transform=axes[0, 1].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
axes[1, 0].text(-0.1, 1.05, '(c)', transform=axes[1, 0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
axes[1, 1].text(-0.1, 1.05, '(d)', transform=axes[1, 1].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
plt.savefig('uq_pointwise_total_energy_2x2.png', dpi=300)

plt.show()

if use_forces:
    force_hist_settings = get_histogram_plot_settings(
        results,
        error_key="force_errors_plot",
        ensemble_key="force_ens_errors",
        norm_error_key="normalized_force_errors_plot",
        norm_ensemble_key="normalized_force_ens_errors",
        mae_xlabel="Force Error / True MAE",
        ev_xlabel="Force Error [eV/A]",
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[1, 0].sharex(axes[0, 0])
    for row_idx, result in enumerate(results):
        ax1 = axes[row_idx, 0]
        ax2 = axes[row_idx, 1]

        ax1.hist(
            result[force_hist_settings["error_key"]],
            bins=force_hist_settings["bins"],
            density=True,
            histtype='step',
            color='black',
            linewidth=2,
        )

        label_uq = r"$\pi_{\mathrm{POPS}}$,  EV: %.1f%%" % (
            result["force_ev_percentage"],
        )
        ax1.hist(
            result[force_hist_settings["ensemble_key"]],
            bins=force_hist_settings["bins"],
            density=True,
            histtype='step',
            color='tab:green',
            linewidth=2,
            label=label_uq,
        )

        ax1.set_yscale('log')
        ax1.set_xlabel(force_hist_settings["xlabel"], fontsize=default_fontsize)
        ax1.set_ylabel('Density', fontsize=default_fontsize)
        ax1.legend(loc='upper left')
        # ax1.set_ylim(bottom=1e-6)
        ax1.set_xlim(*force_hist_settings["xlim"])


        sort_idx = np.argsort(result["F_pred_plot"])
        F_pred_sorted = result["F_pred_plot"][sort_idx]
        F_min_sorted = result["F_min_plot"][sort_idx]
        F_max_sorted = result["F_max_plot"][sort_idx]

        ax2.fill_between(F_pred_sorted, F_min_sorted, F_max_sorted, color='tab:green', alpha=0.4)
        ax2.scatter(result["F_pred_plot"], result["F_ref_plot"], color='black', s=15, zorder=3)

        min_val = min(result["F_pred_plot"].min(), result["F_ref_plot"].min())
        max_val = max(result["F_pred_plot"].max(), result["F_ref_plot"].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        ax2.set_xlabel('Total Force MLE Prediction [eV/A]', fontsize=default_fontsize)
        ax2.set_ylabel('Total Reference Force [eV/A]', fontsize=default_fontsize)

    axes[0, 0].set_title(row_titles[0])
    axes[1, 0].set_title(row_titles[1])
    share_histogram_ylim(axes[:, 0])
    plt.tight_layout()
    plt.savefig('uq_pointwise_total_forces_2x2.png', dpi=300)
    plt.show()
