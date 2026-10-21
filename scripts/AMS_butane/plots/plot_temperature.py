"""Compare reference MPA probabilities and reweighted UQ across temperatures.

The left panel shows posterior-sample uncertainty only. The right panel adds
the AMS estimator uncertainty to each posterior probability estimate.
Run from any directory with ``python plot_temperature.py`` (or its full path).
"""

from pathlib import Path

import matplotlib
import numpy as np
import argparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOTS_DIR = Path(__file__).resolve().parent
AMS_DIR = PLOTS_DIR.parent
GENERAL_DIR = PLOTS_DIR / "general"
TEMPERATURES = (200, 300, 400, 500)
MODELS = ("mp0a", "omat0")
POSTERIOR_TEMPERATURE = 300
OUTPUT_PATH = GENERAL_DIR / "probability_vs_temperature.png"
RATE_OUTPUT_PATH = GENERAL_DIR / "k_vs_temperature.png"
N_TOTAL_DRAWS = 100_000
N_AMS_BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_BATCH_SIZE = 250
RANDOM_SEED = 42
KB_KCAL_PER_MOL_K = 0.0019872041
# Select "central_95", "pops_95" (central 95% of POPS samples), or
# "pops_minmax" (full POPS range); both POPS modes include AMS 95% intervals.
TOTAL_UQ_ENVELOPE = "central_95"


def load_reference_probabilities(temperature):
    """Read the independent direct MPA AMS estimates at one temperature."""
    path = AMS_DIR / f"ams_runs_mpa_{temperature}" / "results_prob.txt"
    if not path.exists():
        raise FileNotFoundError(f"Résultats AMS de référence absents : {path}")
    values = np.atleast_1d(np.loadtxt(path, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any(values <= 0):
        raise ValueError(f"Probabilités de référence invalides dans {path}")
    return values


def load_loop_times(model, temperature):
    """Read positive AMS loop times in femtoseconds."""
    path = AMS_DIR / f"ams_runs_{model}_{temperature}" / "t_loop.txt"
    if not path.exists():
        raise FileNotFoundError(f"Temps de boucle AMS absents : {path}")
    values = np.atleast_1d(np.loadtxt(path, dtype=float))
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        raise ValueError(f"Pas assez de temps de boucle positifs dans {path}")
    return values


def positive_lognormal_draws(mean, std, rng, n_draws):
    """Draw positive values from a lognormal distribution with given moments."""
    mean = max(float(mean), np.finfo(float).tiny)
    log_variance = np.log1p((float(std) / mean) ** 2)
    log_sigma = np.sqrt(log_variance)
    log_mean = np.log(mean) - 0.5 * log_variance
    return rng.lognormal(log_mean, log_sigma, size=n_draws)


def mean_flux(loop_times):
    """Return the fixed AMS flux from mean loop time, converting fs to seconds."""
    return 1.0 / (float(np.mean(loop_times)) * 1e-15)


def probability_by_posterior_sample(model, temperature):
    """Return p(theta) and its AMS standard error for posterior samples.

    This follows the loglinear, probability-weighted estimator and delta-method
    variance used by ``compute_probability_violin``.
    """
    run_dir = AMS_DIR / f"ams_runs_{model}_{temperature}"
    aggregated_dir = run_dir / "reweighting" / "aggregated_results"
    probs = np.load(aggregated_dir / "final_probs.npy")
    scores = np.load(aggregated_dir / "final_scores.npy")
    initial_descriptors = np.load(aggregated_dir / "ini_D.npy")
    if initial_descriptors.ndim == 3:
        initial_descriptors = initial_descriptors.mean(axis=1)
    posterior_path = (
        AMS_DIR
        / f"ams_runs_{model}_{POSTERIOR_TEMPERATURE}"
        / "pops"
        / "binary_data"
        / f"posterior_samples_{model}.npy"
    )
    posterior_samples = np.load(posterior_path)

    if probs.ndim != 1 or scores.ndim != 3:
        raise ValueError(f"Formes inattendues des données AMS pour {model}, {temperature} K")
    if scores.shape[0] != probs.size:
        raise ValueError(f"Nombre de scores et de probabilités incohérent pour {model}, {temperature} K")
    if scores.shape[-1] != posterior_samples.shape[0]:
        raise ValueError(f"Dimension des scores et des échantillons POPS incompatible pour {model}")
    if initial_descriptors.shape[-1] != posterior_samples.shape[0]:
        raise ValueError(f"Dimension de ini_D et des échantillons POPS incompatible pour {model}")
    if np.any(probs <= 0) or not np.all(np.isfinite(probs)):
        raise ValueError(f"Probabilités AMS invalides pour {model}, {temperature} K")

    beta = 1.0 / (temperature * KB_KCAL_PER_MOL_K)
    mean_scores_by_run = np.mean(scores, axis=1)
    projected_scores = mean_scores_by_run @ posterior_samples

    p0 = np.mean(probs)
    weighted_log_slope = np.mean(probs[:, None] * projected_scores, axis=0) / p0
    normalizer = np.mean(
        np.exp(-beta * (initial_descriptors @ posterior_samples)), axis=0
    )
    posterior_probabilities = p0 * np.exp(weighted_log_slope) / normalizer

    # Delta-method standard error of the estimator at each posterior draw,
    # including the covariance between AMS probability and score projection.
    variance_p0 = np.var(probs) / probs.size
    sigma = np.zeros_like(posterior_probabilities)
    for sample_index in range(posterior_probabilities.size):
        weighted_score = probs * projected_scores[:, sample_index]
        variance_g = np.var(weighted_score) / probs.size
        covariance_pg = np.cov(probs, weighted_score, ddof=1)[0, 1] / probs.size
        exp_factor = np.exp(weighted_log_slope[sample_index]) / normalizer[sample_index]
        derivative_p = exp_factor * (1.0 - weighted_log_slope[sample_index])
        derivative_g = exp_factor
        variance = (
            derivative_p**2 * variance_p0
            + derivative_g**2 * variance_g
            + 2.0 * derivative_p * derivative_g * covariance_pg
        )
        sigma[sample_index] = np.sqrt(max(0.0, variance))

    if (
        not np.all(np.isfinite(posterior_probabilities))
        or not np.all(np.isfinite(sigma))
        or np.any(posterior_probabilities <= 0)
    ):
        raise ValueError(f"Repondération invalide pour {model}, {temperature} K")
    return posterior_probabilities, sigma, probs, probs[:, None] * projected_scores, normalizer


def interval_summary(values):
    """Median and central 95% interval."""
    return np.quantile(values, (0.5, 0.025, 0.975))


def pops_minmax_summary(posterior_probabilities, sigma, median):
    """Envelope the per-POPS AMS 95% intervals using their min/max bounds."""
    lower, upper = per_pops_ams_bounds(posterior_probabilities, sigma)
    return median, float(np.min(lower)), float(np.max(upper))


def pops_quantile_summary(posterior_probabilities, sigma, median):
    """Use central 95% POPS quantiles of the per-sample AMS 95% bounds."""
    lower, upper = per_pops_ams_bounds(posterior_probabilities, sigma)
    return median, float(np.quantile(lower, 0.025)), float(np.quantile(upper, 0.975))


def per_pops_ams_bounds(posterior_probabilities, sigma):
    """Return each POPS sample's lower and upper AMS 95% probability bounds."""
    centers = np.clip(posterior_probabilities, 1e-12, 1.0 - 1e-12)
    log_variances = np.log1p((sigma / centers) ** 2)
    log_sigmas = np.sqrt(log_variances)
    log_means = np.log(centers) - 0.5 * log_variances
    z_975 = 1.959963984540054
    lower = np.exp(log_means - z_975 * log_sigmas)
    upper = np.exp(log_means + z_975 * log_sigmas)
    return lower, upper


def reference_band_fraction_in_uq(reference_summary, uq_summary):
    """Fraction of the reference 95% interval covered by the UQ 95% interval."""
    reference_lower, reference_upper = reference_summary[1:]
    uq_lower, uq_upper = uq_summary[1:]
    reference_width = reference_upper - reference_lower
    if reference_width <= 0:
        return float(uq_lower <= reference_lower <= uq_upper)
    overlap = max(
        0.0,
        min(reference_upper, uq_upper) - max(reference_lower, uq_lower),
    )
    return overlap / reference_width


def total_uq_draws(posterior_probabilities, sigma, rng):
    """Draw from the posterior mixture with estimator uncertainty included."""
    sample_indices = rng.integers(
        0, posterior_probabilities.size, size=N_TOTAL_DRAWS
    )
    centers = np.clip(posterior_probabilities[sample_indices], 1e-12, 1.0 - 1e-12)
    variances = sigma[sample_indices] ** 2
    log_variances = np.log1p(variances / centers**2)
    log_sigmas = np.sqrt(log_variances)
    log_means = np.log(centers) - 0.5 * log_variances
    return rng.lognormal(mean=log_means, sigma=log_sigmas)


def ams_only_bootstrap_draws(probs, weighted_scores, normalizer, rng):
    """Bootstrap AMS runs while holding the posterior sample fixed.

    Each bootstrap replicate recomputes the posterior-averaged probability.
    Its spread therefore reflects AMS-run uncertainty without adding the
    variation between posterior parameter samples.
    """
    n_ams = probs.size
    counts = rng.multinomial(
        n_ams,
        np.full(n_ams, 1.0 / n_ams),
        size=N_AMS_BOOTSTRAP_DRAWS,
    )
    posterior_average = np.empty(N_AMS_BOOTSTRAP_DRAWS)
    for start in range(0, N_AMS_BOOTSTRAP_DRAWS, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, N_AMS_BOOTSTRAP_DRAWS)
        batch_counts = counts[start:stop]
        p0_bootstrap = batch_counts @ probs / n_ams
        weighted_score_bootstrap = batch_counts @ weighted_scores / n_ams
        log_slope = weighted_score_bootstrap / p0_bootstrap[:, None]
        probability_by_bootstrap_and_theta = (
            p0_bootstrap[:, None]
            * np.exp(log_slope)
            / normalizer[None, :]
        )
        posterior_average[start:stop] = np.mean(
            probability_by_bootstrap_and_theta, axis=1
        )
    if not np.all(np.isfinite(posterior_average)) or np.any(posterior_average <= 0):
        raise ValueError("Le bootstrap AMS a produit des probabilités invalides.")
    return posterior_average


def plot_summary(ax, temperatures, summaries, label, color, marker):
    medians = np.asarray([summary[0] for summary in summaries])
    lower = np.asarray([summary[1] for summary in summaries])
    upper = np.asarray([summary[2] for summary in summaries])
    ax.plot(
        temperatures,
        medians,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=6,
        label=label,
        zorder=3,
    )
    ax.fill_between(temperatures, lower, upper, color=color, alpha=0.18, linewidth=0)


def plot_temperature_axes(axes, models=MODELS):
    """Draw the two-model temperature comparison into a supplied 2x2 axes array."""
    rng = np.random.default_rng(RANDOM_SEED)
    reference_summaries = []
    uq_ams_only = {model: [] for model in MODELS}
    uq_total = {model: [] for model in MODELS}
    coverage_ams_only = {model: [] for model in MODELS}
    coverage_total = {model: [] for model in MODELS}

    for temperature in TEMPERATURES:
        ref = load_reference_probabilities(temperature)
        ref_mean = np.mean(ref)
        ref_sem = np.std(ref, ddof=1) / np.sqrt(ref.size) if ref.size > 1 else 0.0
        reference_summaries.append(
            (ref_mean, max(ref_mean - 1.96 * ref_sem, np.finfo(float).tiny), ref_mean + 1.96 * ref_sem)
        )

        for model in MODELS:
            posterior_probabilities, sigma, probs, weighted_scores, normalizer = (
                probability_by_posterior_sample(model, temperature)
            )
            ams_draws = ams_only_bootstrap_draws(
                probs, weighted_scores, normalizer, rng
            )
            ams_summary = interval_summary(ams_draws)
            uq_ams_only[model].append(ams_summary)
            ams_covered = reference_band_fraction_in_uq(
                reference_summaries[-1], ams_summary
            )
            coverage_ams_only[model].append(ams_covered)
            draws = total_uq_draws(posterior_probabilities, sigma, rng)
            total_summary = interval_summary(draws)
            if TOTAL_UQ_ENVELOPE == "pops_minmax":
                total_summary = pops_minmax_summary(
                    posterior_probabilities, sigma, total_summary[0]
                )
            elif TOTAL_UQ_ENVELOPE == "pops_95":
                total_summary = pops_quantile_summary(
                    posterior_probabilities, sigma, total_summary[0]
                )
            elif TOTAL_UQ_ENVELOPE != "central_95":
                raise ValueError(
                    "TOTAL_UQ_ENVELOPE doit être 'central_95' ou 'pops_minmax'."
                )
            uq_total[model].append(total_summary)
            total_covered = reference_band_fraction_in_uq(
                reference_summaries[-1], total_summary
            )
            coverage_total[model].append(total_covered)

            print(
                f"{model}, {temperature} K: "
                f"p moyen sur theta={np.mean(posterior_probabilities):.4e}, "
                f"IC95 AMS seul=[{np.quantile(ams_draws, 0.025):.4e}, "
                f"{np.quantile(ams_draws, 0.975):.4e}], "
                f"IC95 total=[{np.quantile(draws, 0.025):.4e}, "
                f"{np.quantile(draws, 0.975):.4e}]"
            )

        print(
            f"MPA référence, {temperature} K: "
            f"p={ref_mean:.4e}, IC95=[{reference_summaries[-1][1]:.4e}, "
            f"{reference_summaries[-1][2]:.4e}] (n={ref.size})"
        )

    colors = {"mp0a": "tab:blue", "omat0": "tab:orange"}
    labels = {"mp0a": "MP-0a", "omat0": "OMAT-0"}
    markers = {"mp0a": "o", "omat0": "s"}
    uq_columns = (
        (uq_ams_only, coverage_ams_only, "AMS variance"),
        (
            uq_total,
            coverage_total,
            "AMS variance + model UQ"
            + {
                "pops_95": " (POPS 95%)",
                "pops_minmax": " (POPS min–max)",
                "central_95": "",
            }[TOTAL_UQ_ENVELOPE],
        ),
    )

    for row, model in enumerate(models):
        for col, (model_summaries, model_coverage, uq_title) in enumerate(uq_columns):
            ax = axes[row, col]
            global_coverage = float(np.mean(model_coverage[model]))
            envelope_violation = 1.0 - global_coverage
            uq_color = "tab:blue" if col == 0 else "tab:green"
            plot_summary(
                ax,
                TEMPERATURES,
                model_summaries[model],
                (
                    f"{labels[model]}, EV: {envelope_violation:.1%}"
                    if col == 0
                    else f"{labels[model]} + UQ, EV: {envelope_violation:.1%}"
                ),
                uq_color,
                markers[model],
            )
            plot_summary(
                ax,
                TEMPERATURES,
                reference_summaries,
                "MPA reference",
                "black",
                "D",
            )
            ax.set_title(uq_title, fontsize=18)
            ax.set_xticks(TEMPERATURES)
            ax.set_yscale("log")
            ax.set_ylim(2e-5, 3e-2)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=13)
            print(
                f"Global envelope violation, {labels[model]} / {uq_title}: "
                f"{envelope_violation:.1%}"
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("Temperature [K]", fontsize=18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Rare Event Probability", fontsize=13)
    for ax in axes.flat:
        ax.tick_params(labelsize=12)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    return reference_summaries, uq_ams_only, uq_total


def plot_rate_temperature_axes(axes):
    """Draw model rates with probability UQ and fixed, theta-independent flux."""
    rng = np.random.default_rng(RANDOM_SEED)
    reference_summaries = []
    uq_ams_only = {model: [] for model in MODELS}
    uq_total = {model: [] for model in MODELS}
    coverage_ams_only = {model: [] for model in MODELS}
    coverage_total = {model: [] for model in MODELS}

    for temperature in TEMPERATURES:
        reference_probabilities = load_reference_probabilities(temperature)
        reference_sem = (
            np.std(reference_probabilities, ddof=1) / np.sqrt(reference_probabilities.size)
            if reference_probabilities.size > 1
            else 0.0
        )
        reference_probability_draws = positive_lognormal_draws(
            np.mean(reference_probabilities),
            reference_sem,
            rng,
            N_TOTAL_DRAWS,
        )
        reference_flux = mean_flux(load_loop_times("mpa", temperature))
        reference_rate_draws = reference_probability_draws * reference_flux
        reference_summary = interval_summary(reference_rate_draws)
        reference_summaries.append(reference_summary)

        for model in MODELS:
            posterior_probabilities, sigma, probs, weighted_scores, normalizer = (
                probability_by_posterior_sample(model, temperature)
            )
            model_loop_times = load_loop_times(model, temperature)

            ams_probability_draws = ams_only_bootstrap_draws(
                probs, weighted_scores, normalizer, rng
            )
            model_flux = mean_flux(model_loop_times)
            ams_rate_draws = ams_probability_draws * model_flux
            ams_summary = interval_summary(ams_rate_draws)
            uq_ams_only[model].append(ams_summary)
            coverage_ams_only[model].append(
                reference_band_fraction_in_uq(reference_summary, ams_summary)
            )

            total_probability_draws = total_uq_draws(
                posterior_probabilities, sigma, rng
            )
            total_rate_draws = total_probability_draws * model_flux
            total_summary = interval_summary(total_rate_draws)
            uq_total[model].append(total_summary)
            coverage_total[model].append(
                reference_band_fraction_in_uq(reference_summary, total_summary)
            )

            print(
                f"{model}, {temperature} K: k médian (AMS seul)="
                f"{ams_summary[0]:.4e} s⁻¹, k médian (UQ totale)="
                f"{total_summary[0]:.4e} s⁻¹"
            )
        print(
            f"MPA référence, {temperature} K: k médian="
            f"{reference_summary[0]:.4e} s⁻¹ (IC95 "
            f"[{reference_summary[1]:.4e}, {reference_summary[2]:.4e}])"
        )

    colors = {"mp0a": "tab:blue", "omat0": "tab:orange"}
    labels = {"mp0a": "MP-0a", "omat0": "OMAT-0"}
    markers = {"mp0a": "o", "omat0": "s"}
    uq_columns = (
        (uq_ams_only, coverage_ams_only, "AMS variance"),
        (uq_total, coverage_total, "AMS variance + model UQ"),
    )
    for row, model in enumerate(MODELS):
        for col, (summaries, coverage, uq_title) in enumerate(uq_columns):
            ax = axes[row, col]
            envelope_violation = 1.0 - float(np.mean(coverage[model]))
            plot_summary(
                ax,
                TEMPERATURES,
                summaries[model],
                (
                    f"{labels[model]}, EV: {envelope_violation:.1%}"
                    if col == 0
                    else f"{labels[model]} + UQ, EV: {envelope_violation:.1%}"
                ),
                "tab:blue" if col == 0 else "tab:green",
                markers[model],
            )
            plot_summary(
                ax,
                TEMPERATURES,
                reference_summaries,
                "MPA reference",
                "black",
                "D",
            )
            ax.set_title(uq_title, fontsize=18)
            ax.set_xticks(TEMPERATURES)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=13)

    for ax in axes[-1, :]:
        ax.set_xlabel("Temperature [K]", fontsize=18)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Rate $k$ [s$^{-1}$]", fontsize=13)
    for ax in axes.flat:
        ax.tick_params(labelsize=12)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    return reference_summaries, uq_ams_only, uq_total


def main():
    GENERAL_DIR.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description="Plot probability or rate versus temperature.")
    parser.add_argument(
        "--quantity",
        choices=("probability", "rate"),
        default="probability",
        help="Plot rare event probability (default) or rate k.",
    )
    quantity = parser.parse_args().quantity
    if quantity == "probability":
        fig, axes = plt.subplots(len(MODELS), 2, figsize=(11, 8), sharex=True, sharey=True)
        plot_temperature_axes(axes)
        output_path = OUTPUT_PATH
    else:
        fig, axes = plt.subplots(len(MODELS), 2, figsize=(11, 8), sharex=True, sharey=True)
        plot_rate_temperature_axes(axes)
        output_path = RATE_OUTPUT_PATH
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Figure enregistrée : {output_path}")


if __name__ == "__main__":
    main()
