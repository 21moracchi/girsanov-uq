import numpy as np
import matplotlib.pyplot as plt
# %%
# -----------------------
# Load data
# -----------------------
direct_probs_array = np.load('../theta_mp0a_500/results/probs.npy')
direct_probs_theta_l = np.load('../ams_runs_mp0a_500/reweighting/aggregated_results/final_probs.npy')
scores = np.load('../ams_runs_mp0a_500/reweighting/aggregated_results/final_scores.npy')
thetas = np.load('../theta_mp0a_500/results/thetas.npy')

# -----------------------
# Means
# -----------------------
p0 = np.mean(direct_probs_theta_l)
direct_probs = np.mean(direct_probs_array, axis=1)

mean_scores = np.mean(scores, axis=(0, 1))
L_theta = np.einsum('ij,j->i', thetas, mean_scores)

reweighted_probs = p0 * np.exp(L_theta)


# -----------------------
# SEM direct probs
# -----------------------
n_direct = direct_probs_array.shape[1]

sem_direct_probs = np.std(
    direct_probs_array,
    axis=1,
    ddof=1
) / np.sqrt(n_direct)

ci95_direct_probs = 1.96 * sem_direct_probs

# -----------------------
# SEM p0
# -----------------------
n_p0 = len(direct_probs_theta_l)

sem_p0 = np.std(
    direct_probs_theta_l,
    ddof=1
) / np.sqrt(n_p0)

# -----------------------
# SEM L_theta
# -----------------------
# mean score per AMS run
mean_score_per_ams = np.mean(scores, axis=1)

L_theta_per_ams = np.einsum(
    'ij,kj->ik',
    thetas,
    mean_score_per_ams
)

L_theta_std = np.std(L_theta_per_ams, axis=1, ddof=1)
n_ams = mean_score_per_ams.shape[0]
L_theta_sem = L_theta_std / np.sqrt(n_ams)

# -----------------------
# Delta method SEM reweighted probs
# p_rw = p0 * exp(L)
# -----------------------
sem_reweighted_probs = np.sqrt(
    (np.exp(L_theta) * sem_p0)**2
    + (reweighted_probs * L_theta_sem)**2
)

ci95_reweighted_probs = 1.96 * sem_reweighted_probs

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8, 6))

alphas = np.exp(-L_theta_std / L_theta_std.max())

plt.errorbar(
    np.log10(direct_probs),
    np.log10(reweighted_probs),
    # xerr=ci95_direct_probs,
    # yerr=ci95_reweighted_probs,
    fmt='o',
    ms=5,
    alpha=0.6,
    capsize=2,
    elinewidth=0.8,
    label='95% CI'
)

plt.scatter(
    np.log10(p0),
    np.log10(p0),  # Assuming you want to plot the first reweighted probability
    color='red',
    s=70,
    zorder=10,
    label=r'$\theta_\mathcal{L}$'
)

# Trend line
order = np.argsort(np.log10(direct_probs))
trend = np.polyfit(np.log10(direct_probs), np.log10(reweighted_probs), 1)

plt.plot(
    np.log10(direct_probs[order]),
    np.polyval(trend, np.log10(direct_probs[order])),
    'r--',
    lw=2,
    label='linear fit'
)

# y = x
x_min = -2.2
x_max = -1.5


plt.plot(
    [x_min, x_max],
    [x_min, x_max],
    'k--',
    lw=2,
    label=r'$y=x$'
)

plt.xlabel('Direct log-probabilities')
plt.ylabel('Reweighted log-probabilities')
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.legend()
plt.tight_layout()
plt.show()