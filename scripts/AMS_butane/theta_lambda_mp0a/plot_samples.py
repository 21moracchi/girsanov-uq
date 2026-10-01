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


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
REWEIGHTING_DIR = ROOT.parent / "ams_runs_mp0a_500/reweighting/aggregated_results"
OUTPUT_PATH = ROOT / "reweighting_vs_lambda.png"


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
        raise ValueError("directions.npy incompatible avec theta_l.npy.")
    if scores.shape[-1] != theta_l.size:
        raise ValueError("Dimension des scores incompatible avec theta_l.npy.")

    p0 = np.mean(reference_probs)
    sem_p0 = np.std(reference_probs, ddof=1) / np.sqrt(len(reference_probs))
    mean_scores = np.mean(scores, axis=(0, 1))
    mean_score_per_ams = np.mean(scores, axis=1)

    # theta(lambda) - theta_L = lambda * delta_theta.
    deltas = lambdas[None, :, None] * directions[:, None, :]
    log_ratios = np.einsum("sld,d->sl", deltas, mean_scores)
    reweighted = p0 * np.exp(log_ratios)
    log_ratio_per_ams = np.einsum("sld,ad->sla", deltas, mean_score_per_ams)
    sem_log_ratio = np.std(log_ratio_per_ams, axis=2, ddof=1) / np.sqrt(mean_score_per_ams.shape[0])
    # Form the interval in log probability and exponentiate its endpoints.
    # A symmetric interval in p can cross zero for large lambda, which is
    # invalid on a logarithmic probability axis and caused the apparent 1e-308
    # bands in the plot.
    sem_log_p0 = sem_p0 / p0
    sem_log_reweighted = np.sqrt(sem_log_p0**2 + sem_log_ratio**2)
    ci_factor = 1.96 * sem_log_reweighted
    reweighted_lower = reweighted * np.exp(-ci_factor)
    reweighted_upper = reweighted * np.exp(ci_factor)

    direct_path = RESULTS_DIR / "probs.npy"
    direct = None
    if direct_path.exists():
        direct = np.load(direct_path)
        expected_shape = (len(theta_indices), len(lambdas))
        if direct.ndim != 3 or direct.shape[:2] != expected_shape:
            raise ValueError(f"probs.npy doit commencer par la forme {expected_shape}.")

    fig, axes = plt.subplots(
        len(theta_indices),
        1,
        figsize=(8, 3.6 * len(theta_indices)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    all_central_curves = []
    for sample_id, ax in enumerate(axes):
        ax.plot(
            lambdas,
            reweighted[sample_id],
            "o-",
            alpha=0.7,
            label="Reweighted",
            zorder=2,
        )
        ax.fill_between(
            lambdas,
            reweighted_lower[sample_id],
            reweighted_upper[sample_id],
            alpha=0.2,
            label="95% CI reweighting",
        )

        central_curves = [reweighted[sample_id]]
        if direct is not None:
            direct_mean, direct_sem = mean_and_sem_by_lambda(direct[sample_id])
            available = np.isfinite(direct_mean)
            if np.any(available):
                central_curves.append(direct_mean[available])
                direct_ci = 1.96 * direct_sem[available]
                # Keep the lower end positive for the logarithmic y-axis.
                direct_ci = np.minimum(direct_ci, 0.999 * direct_mean[available])
                direct_lower = direct_mean[available] - direct_ci
                direct_upper = direct_mean[available] + direct_ci
                ax.fill_between(
                    lambdas[available],
                    direct_lower,
                    direct_upper,
                    alpha=0.2,
                    label="95% CI direct AMS",
                    zorder=1,
                )
                ax.plot(
                    lambdas[available],
                    direct_mean[available],
                    "s",
                    alpha=0.7,
                    label="Direct AMS",
                    zorder=3,
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
            f"sample_{sample_id}: SE(log p_rew) a lambda={lambdas[0]:.0f}/"
            f"{lambdas[-1]:.0f} = {sem_log_reweighted[sample_id, 0]:.3f}/"
            f"{sem_log_reweighted[sample_id, -1]:.3f}; "
            "facteur multiplicatif 95% = "
            f"x{np.exp(1.96 * sem_log_reweighted[sample_id, -1]):.1f}"
        )

        ax.set_yscale("log")
        ax.set_ylabel("Probability")
        ax.set_title(f"Sample {sample_id} (posterior index {theta_indices[sample_id]})")
        ax.grid(alpha=0.25, which="both")
        ax.legend()

    central_positive = np.concatenate(all_central_curves)
    central_positive = central_positive[
        np.isfinite(central_positive) & (central_positive > 0)
    ]
    log_min = np.log10(central_positive.min())
    log_max = np.log10(central_positive.max())
    margin = max(0.12, 0.12 * (log_max - log_min))
    axes[0].set_ylim(10 ** (log_min - margin), 10 ** (log_max + margin))

    axes[-1].set_xlabel(r"$\lambda$ in $\theta(\lambda)=\theta_\mathcal{{L}}+\lambda\Delta\theta_i$")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
