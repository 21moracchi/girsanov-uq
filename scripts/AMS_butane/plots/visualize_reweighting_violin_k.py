# %%
import argparse

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from girsanov_uq.utils.utils import compute_log_rate_violin


models = ['omat0', 'mp0a']
parser = argparse.ArgumentParser(description="Plot AMS log-rate uncertainty at one temperature.")
parser.add_argument("temperature", nargs="?", type=int, default=300)
parser.add_argument("--style", choices=("violin", "raincloud"), default="violin")
args = parser.parse_args()
temperature = args.temperature

fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

for i, model in enumerate(models):
    probs = np.load(f"../ams_runs_{model}_{temperature}/reweighting/aggregated_results/final_probs.npy")
    scores = np.load(f"../ams_runs_{model}_{temperature}/reweighting/aggregated_results/final_scores.npy")
    ref_probs = np.loadtxt(f"../ams_runs_mpa_{temperature}/results_prob.txt")
    loop_times = np.loadtxt(f"../ams_runs_{model}_{temperature}/t_loop.txt")
    ref_loop_times = np.loadtxt(f"../ams_runs_{model}_{temperature}/t_loop.txt")
    posterior_samples = np.load(f"../ams_runs_{model}_300/pops/binary_data/posterior_samples_{model}.npy")

    compute_log_rate_violin(
        scores[:],
        probs[:],
        loop_times=loop_times,
        thetas_sample=posterior_samples,
        ref_probs=ref_probs,
        ref_loop_times=ref_loop_times,
        ax=axs[i],
        n_samples=10000,
        n_theta_plot=10,
        rng=42 + i,
        plot_style=args.style,
    )

axs[0].set_title('OMAT-0')
axs[1].set_title('MP-0a')
axs[0].set_xlabel('')
axs[0].text(-0.1, 1.05, '(a)', transform=axs[0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
axs[1].text(-0.1, 1.05, '(b)', transform=axs[1].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

fig.tight_layout()
output_dir = Path(f'{temperature}K')
if args.style == "raincloud":
    output_dir /= "raincloud"
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / f'uq_{temperature}_{args.style}_k_log.png', dpi=300, bbox_inches='tight')
