from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
    }
)


def find_repo_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "scripts/AMS_1D").exists() and (
            candidate / "scripts/AMS_butane_3"
        ).exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find the repository root.")


def load_1d_case(repo_root):
    base_dir = repo_root / "scripts/AMS_1D"
    fit = np.load(base_dir / "data_1D/reweighting_aggregate_fit_40_500.npz")
    theta_samples = np.load(base_dir / "pops_data/posterior_samples.npy")
    return fit["all_scores"], theta_samples


def load_butane_case(repo_root, model):
    base_dir = repo_root / "scripts/AMS_butane_3" / f"ams_runs_{model}_300"
    scores = np.load(base_dir / "reweighting/aggregated_results/final_scores.npy")
    theta_samples = np.load(
        base_dir / f"pops/binary_data/posterior_samples_{model}.npy"
    )
    return scores, theta_samples


def compute_mean_L(scores, theta_samples):
    scores_ams = scores.mean(axis=1)
    delta = theta_samples.T
    L_by_theta_and_ams = delta @ scores_ams.T

    L_mean = L_by_theta_and_ams.mean(axis=1)
    L_sem = L_by_theta_and_ams.std(axis=1, ddof=1) / np.sqrt(
        L_by_theta_and_ams.shape[1]
    )
    return L_mean, L_sem


def plot_case(ax, label, scores, theta_samples, color):
    L_mean, L_sem = compute_mean_L(scores, theta_samples)
    idx = np.argsort(L_mean)
    y = L_mean[idx]
    yerr = 1.96 * L_sem[idx]
    x = np.linspace(0.0, 100.0, len(y))

    ax.plot(x, y, linewidth=2.0, color=color)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.5, color=color, linewidth=0)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title(label)
    ax.set_xlabel(r"$\theta$ sorted by $L( \theta )$ (%)")
    ax.tick_params(axis="both", which="major", labelsize=15)


def main():
    repo_root = find_repo_root()

    cases = [
        ("1D", *load_1d_case(repo_root), "tab:green"),
        ("omat0", *load_butane_case(repo_root, "omat0"), "tab:blue"),
        ("mp0a", *load_butane_case(repo_root, "mp0a"), "tab:orange"),
    ]

    fig, axes = plt.subplots(1, len(cases), figsize=(12, 4.5), sharey=True)
    for ax, (label, scores, theta_samples, color) in zip(axes, cases):
        plot_case(ax, label, scores, theta_samples, color)
    axes[0].set_title('One-dimensional')
    axes[1].set_title(r'Butane: OMAT-0')
    axes[2].set_title(r'Butane: MP-0a')
    axes[0].set_ylabel(r"Mean $L(\theta)$")
    axes[0].text(-0.18, 1.08, "(a)", transform=axes[0].transAxes, fontsize=18, fontweight="bold", va="top", ha="right")
    axes[1].text(-0.18, 1.08, "(b)", transform=axes[1].transAxes, fontsize=18, fontweight="bold", va="top", ha="right")
    axes[2].text(-0.18, 1.08, "(c)", transform=axes[2].transAxes, fontsize=18, fontweight="bold", va="top", ha="right")
    # fig.suptitle(r"Mean score projection $L = \delta\theta^T \bar{s}$", y=1.02)
    
    fig.tight_layout()

    output_path = Path(__file__).with_name("L_three_cases.png")
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
