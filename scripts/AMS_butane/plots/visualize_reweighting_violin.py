# %%
import numpy as np
import matplotlib.pyplot as plt
from girsanov_uq.utils.utils import compute_probability_violin


models = [ 'omat0','mp0a']
temperature = 200

fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

for i, model in enumerate(models):
    probs = np.load(f"../ams_runs_{model}_{temperature}/reweighting/aggregated_results/final_probs.npy")
    scores = np.load(f"../ams_runs_{model}_{temperature}/reweighting/aggregated_results/final_scores.npy")
    ini_conds_D = np.load(f"../ams_runs_{model}_{temperature}/reweighting/aggregated_results/ini_D.npy")
    ref_probs = np.loadtxt(f'../ams_runs_mpa_{temperature}/results_prob.txt')
    posterior_samples = np.load(f"../ams_runs_{model}_300/pops/binary_data/posterior_samples_{model}.npy")
    ini_conds_D = np.mean(ini_conds_D, axis=1)
    compute_probability_violin(
        scores[:],
        probs[:],
        thetas_sample=posterior_samples,
        ref_probs=ref_probs,
        method='loglinear',
        variance='estimator',
        prior_dist='lognormal',
        ax=axs[i],
        use_Ahat=True,
        n_samples=10000,
        n_theta_plot=10,
        rng=42 + i,
        initial_descriptors=ini_conds_D, beta = 1.0 / (temperature * 0.0019872041)
    )

axs[0].set_title('OMAT-0')
axs[1].set_title('MP-0a')
axs[0].set_xlabel('')
axs[0].text(-0.1, 1.05, '(a)', transform=axs[0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
axs[1].text(-0.1, 1.05, '(b)', transform=axs[1].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

fig.tight_layout()
fig.savefig(f'uq_{temperature}_violin.png', dpi=300)
