"""Plot direct AMS and reweighted probabilities along three theta directions.

This script needs only ``results/lambdas.npy``, ``directions.npy`` and
``theta_l.npy``. Therefore it plots the reweighted curves even before any AMS
result has completed. Run ``python get_results.py`` to add direct AMS points.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
REWEIGHTING_DIR = ROOT.parent / "ams_runs_mp0a_500/reweighting/aggregated_results"
OUTPUT_PATH = ROOT / "reweighting_vs_lambda.png"
SCATTER_OUTPUT_PATH = ROOT / "reweighting_vs_lambda_scatter.png"
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261006


def mean_and_sem_by_lambda(values):
    means = np.full(values.shape[0], np.nan)
    errors = np.zeros(values.shape[0])
    for index, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            means[index] = np.mean(finite)
        if finite.size > 1:
            errors[index] = np.std(finite, ddof=1) / np.sqrt(finite.size)
    return means, errors


def main():
    lambdas = np.load(RESULTS_DIR / "lambdas.npy")
    directions = np.load(RESULTS_DIR / "directions.npy")
    theta_l = np.load(RESULTS_DIR / "theta_l.npy")
    theta_indices = np.load(RESULTS_DIR / "theta_indices.npy")
    scores = np.load(REWEIGHTING_DIR / "final_scores.npy")
    reference_probs = np.load(REWEIGHTING_DIR / "final_probs.npy")

    if directions.shape != (len(theta_indices), theta_l.size):
        raise ValueError("directions.npy is incompatible with theta_l.npy.")
    if scores.shape[-1] != theta_l.size:
        raise ValueError("Score dimensions are incompatible with theta_l.npy.")

    if reference_probs.ndim != 1 or scores.shape[0] != reference_probs.size:
        raise ValueError(
            "final_probs and final_scores must have the same number of independent AMS runs "
            "along their first axis."
        )

    p0 = np.mean(reference_probs)
    mean_score_per_ams = np.mean(scores, axis=1)

    # theta(lambda) - theta_L = lambda * delta_theta.
    deltas = lambdas[None, :, None] * directions[:, None, :]
    mean_scores = np.mean(mean_score_per_ams, axis=0)
    log_ratios = np.einsum("sld,d->sl", deltas, mean_scores)
    reweighted = p0 * np.exp(log_ratios)

    # Bootstrap AMS as paired blocks: each draw carries its reference
    # probability and its score contribution together, preserving covariance.
    n_ams = reference_probs.size
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_probs = np.empty((BOOTSTRAP_REPLICATES, len(theta_indices), len(lambdas)))
    for bootstrap_id in range(BOOTSTRAP_REPLICATES):
        sampled_ams = rng.integers(0, n_ams, size=n_ams)
        p0_boot = np.mean(reference_probs[sampled_ams])
        scores_boot = np.mean(mean_score_per_ams[sampled_ams], axis=0)
        log_ratio_boot = np.einsum("sld,d->sl", deltas, scores_boot)
        bootstrap_probs[bootstrap_id] = p0_boot * np.exp(log_ratio_boot)

    reweighted_2sigma_lower, reweighted_2sigma_upper = np.quantile(
        bootstrap_probs, [0.02275, 0.97725], axis=0
    )
    reweighted_1sigma_lower, reweighted_1sigma_upper = np.quantile(
        bootstrap_probs, [0.15865, 0.84135], axis=0
    )
    bootstrap_sem = np.std(bootstrap_probs, axis=0, ddof=1)

    direct_path = RESULTS_DIR / "probs.npy"
    direct = None
    if direct_path.exists():
        direct = np.load(direct_path)
        expected_shape = (len(theta_indices), len(lambdas))
        if direct.ndim != 3 or direct.shape[:2] != expected_shape:
            raise ValueError(f"probs.npy must start with shape {expected_shape}.")

    fig, axes = plt.subplots(
        len(theta_indices),
        1,
        figsize=(8, 3.6 * len(theta_indices)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    all_central_curves = []
    zone_centers = []
    for sample_id, ax in enumerate(axes):
        ax.plot(
            lambdas,
            reweighted[sample_id],
            "-",
            alpha=0.7,
            label=r"$\widehat{p}_C$",
            zorder=3,
        )
        ax.fill_between(
            lambdas,
            reweighted_2sigma_lower[sample_id],
            reweighted_2sigma_upper[sample_id],
            color="tab:blue",
            alpha=0.16,
            label=r"$\widehat{p}_C\pm2\sigma$",
            zorder=1,
        )
        # ax.fill_between(
        #     lambdas,
        #     reweighted_1sigma_lower[sample_id],
        #     reweighted_1sigma_upper[sample_id],
        #     color="tab:blue",
        #     alpha=0.36,
        #     label=r"Reweighted $\sim 1\sigma$",
        #     zorder=2,
        # )

        central_curves = [reweighted[sample_id]]
        zone_values = list(reweighted[sample_id, np.abs(lambdas) <= 1])
        if direct is not None:
            direct_mean, direct_sem = mean_and_sem_by_lambda(direct[sample_id])
            available = np.isfinite(direct_mean)
            if np.any(available):
                central_curves.append(direct_mean[available])
                direct_in_zone = available & (np.abs(lambdas) <= 1)
                zone_values.extend(direct_mean[direct_in_zone])
                direct_ci = 1.96 * direct_sem[available]
                # Keep the lower end positive for the logarithmic y-axis.
                direct_ci = np.minimum(direct_ci, 0.999 * direct_mean[available])
                direct_lower = direct_mean[available] - direct_ci
                direct_upper = direct_mean[available] + direct_ci
                ax.fill_between(
                    lambdas[available],
                    direct_lower,
                    direct_upper,
                    color="tab:orange",
                    alpha=0.3,
                    label=r"$\widehat{p}_{\mathrm{AMS}}\pm2\sigma$",
                    zorder=10,
                )
                ax.plot(
                    lambdas[available],
                    direct_mean[available],
                    "s",
                    alpha=0.7,
                    label=r"$\widehat{p}_{\mathrm{AMS}}$",
                    zorder=11,
                )
                mae_p = np.mean(np.abs(direct_mean[available] - reweighted[sample_id, available]))
                mae_log = np.mean(np.abs(
                    np.log10(direct_mean[available]) - np.log10(reweighted[sample_id, available])
                ))
                print(
                    f"sample_{sample_id} (posterior index {theta_indices[sample_id]}): "
                    f"MAE p={mae_p:.4e}, MAE log10 p={mae_log:.4f} decades"
                )

        # Determine one common log-scale range from all central curves. The
        # confidence ribbons remain visible but do not set the shared limits.
        all_central_curves.extend(central_curves)

        print(
            f"sample_{sample_id}: bootstrap SE(p_rew) aux lambdas "
            f"{lambdas[0]:.0f}/{lambdas[-1]:.0f} = "
            f"{bootstrap_sem[sample_id, 0]:.3e}/{bootstrap_sem[sample_id, -1]:.3e}; "
            "IC percentile 95% au dernier lambda = "
            f"[{reweighted_2sigma_lower[sample_id, -1]:.3e}, "
            f"{reweighted_2sigma_upper[sample_id, -1]:.3e}]"
        )

        ax.set_yscale("log")
        ax.set_ylabel("Probability")
        ax.set_title(f"Sample {sample_id} (posterior index {theta_indices[sample_id]})")
        ax.grid(alpha=0.25, which="both")
        # Center the square highlight on the central curves in lambda [-1, 1].
        zone_values = np.asarray(zone_values)
        zone_values = zone_values[np.isfinite(zone_values) & (zone_values > 0)]
        y_low = zone_values.min() / 1.15
        y_high = zone_values.max() * 1.15
        zone_centers.append(np.sqrt(y_low * y_high))
        if sample_id == len(axes) // 2:
            handles, labels = ax.get_legend_handles_labels()
            desired_order = [
                r"$\widehat{p}_C$",
                r"$\widehat{p}_C\pm2\sigma$",
                r"$\widehat{p}_{\mathrm{AMS}}$",
                r"$\widehat{p}_{\mathrm{AMS}}\pm2\sigma$",
            ]
            by_label = dict(zip(labels, handles))
            ordered_labels = [label for label in desired_order if label in by_label]
            ax.legend([by_label[label] for label in ordered_labels], ordered_labels)
    
    central_positive = np.concatenate(all_central_curves)
    central_positive = central_positive[
        np.isfinite(central_positive) & (central_positive > 0)
    ]
    log_min = np.log10(central_positive.min())
    log_max = np.log10(central_positive.max())
    margin = max(0.12, 0.12 * (log_max - log_min))
    axes[0].set_ylim(10 ** (log_min - margin), 10 ** (log_max + margin))

    axes[-1].set_xlabel(r"$\lambda$ in $\theta(\lambda)=\theta_\mathcal{{L}}+\lambda\Delta\theta_i$")
    axes[-1].xaxis.set_major_locator(MultipleLocator(1))
    fig.tight_layout()
    # Use data coordinates horizontally and axes coordinates vertically so
    # the highlighted region is a true square on screen, with a small x margin.
    fig.canvas.draw()
    x_left, x_right = -1.1, 1.1
    for ax, y_center in zip(axes, zone_centers):
        bbox = ax.get_window_extent()
        x0_px = ax.transData.transform((x_left, y_center))[0]
        x1_px = ax.transData.transform((x_right, y_center))[0]
        square_height_axes = abs(x1_px - x0_px) / bbox.height
        center_y_axes = ax.transAxes.inverted().transform(
            ax.transData.transform((0, y_center))
        )[1]
        ax.add_patch(
            Rectangle(
                (x_left, center_y_axes - square_height_axes / 2),
                x_right - x_left,
                square_height_axes,
                transform=ax.get_xaxis_transform(),
                fill=False,
                edgecolor="crimson",
                linewidth=1.8,
                zorder=5,
            )
        )
    fig.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved {OUTPUT_PATH}")

    # Compare direct and reweighted estimates at every available lambda value.
    direct_means = np.full(direct.shape[:2], np.nan)
    direct_sems = np.zeros(direct.shape[:2])
    for sample_id in range(direct.shape[0]):
        direct_means[sample_id], direct_sems[sample_id] = mean_and_sem_by_lambda(
            direct[sample_id]
        )

    available = (
        np.isfinite(direct_means)
        & (direct_means > 0)
        & np.isfinite(reweighted)
        & (reweighted > 0)
        & np.isfinite(reweighted_2sigma_lower)
        & np.isfinite(reweighted_2sigma_upper)
    )
    x = np.log10(direct_means[available])
    y = np.log10(reweighted[available])
    direct_lower = np.maximum(
        direct_means - 1.96 * direct_sems,
        np.finfo(float).tiny,
    )
    direct_upper = direct_means + 1.96 * direct_sems
    x_error_lower = np.log10(direct_means[available]) - np.log10(direct_lower[available])
    x_error_upper = np.log10(direct_upper[available]) - np.log10(direct_means[available])
    y_error_lower = np.maximum(
        0.0,
        y - np.log10(reweighted_2sigma_lower[available]),
    )
    y_error_upper = np.maximum(
        0.0,
        np.log10(reweighted_2sigma_upper[available]) - y,
    )

    scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
    scatter_ax.errorbar(
        x,
        y,
        xerr=np.vstack((x_error_lower, x_error_upper)),
        yerr=np.vstack((y_error_lower, y_error_upper)),
        fmt="none",
        ecolor="gray",
        alpha=0.55,
        capsize=2,
        zorder=1,
    )
    scatter_ax.scatter(x, y, s=35, label=r"Samples $\theta_i$", zorder=2)
    log_p0 = np.log10(p0)
    scatter_ax.scatter(
        log_p0,
        log_p0,
        s=40,
        color="tab:orange",
        zorder=3,
        label=r"Reference $\theta_\mathcal{L}$",
    )

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        scatter_ax.plot(
            x_fit,
            slope * x_fit + intercept,
            "g--",
            lw=2,
            label="Fit OLS",
        )

    scatter_extent = max(
        np.max(np.abs(x - log_p0) + np.maximum(x_error_lower, x_error_upper)),
        np.max(np.abs(y - log_p0) + np.maximum(y_error_lower, y_error_upper)),
        0.05,
    ) * 1.08
    scatter_limits = (log_p0 - scatter_extent, log_p0 + scatter_extent)
    scatter_ax.plot(scatter_limits, scatter_limits, "k--", lw=1.5, label=r"$y=x$")
    scatter_ax.set_xlim(scatter_limits)
    scatter_ax.set_ylim(scatter_limits)
    scatter_ax.set_xlabel(
        r"Direct log-probabilities $\log_{10}(\hat p_{\mathrm{AMS}}(\theta))$"
    )
    scatter_ax.set_ylabel(
        r"Reweighted log-probabilities $\log_{10}(\hat p_C(\theta))$"
    )
    scatter_ax.legend()
    scatter_fig.tight_layout()
    scatter_fig.savefig(SCATTER_OUTPUT_PATH, dpi=300)
    print(f"Saved {SCATTER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
