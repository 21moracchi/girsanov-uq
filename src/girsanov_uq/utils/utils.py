from matplotlib import ticker
from networkx import sigma
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas

# Increase default plot font sizes for better readability
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 20
})

def compute_derivative_check(scores, fims, probs, delta_theta, alpha_range=0.1, n_points=200, samples=None, ax=None, theta_reference=None):
    alphas = np.linspace(-alpha_range, alpha_range, n_points)
    n_traj = len(probs)
    z_score = 1.96 
    if theta_reference is not None and theta_reference == 0.0:
        raise ValueError("theta_reference must be non-zero to plot relative variations.")
    
    mean_scores = np.mean(scores, axis=1) 
    projected_scores = np.dot(mean_scores, delta_theta)
    
    y_linear = probs[None, :] * (1.0 + alphas[:, None] * projected_scores[None, :])
    p_linear = np.mean(y_linear, axis=1)
    sem_linear = np.sqrt(np.var(y_linear, axis=1) / n_traj)
    p_linear_ci = z_score * sem_linear

    # y_loglinear = probs[None, :] * np.exp(alphas[:, None] * projected_scores[None, :])
    # p_loglinear = np.mean(y_loglinear, axis=1)
    # sem_loglinear = np.sqrt(np.var(y_loglinear, axis=1) / n_traj)
    # p_loglinear_ci = z_score * sem_loglinear

    #aggregated version with bootstrap for confidence intervals
    P_m = probs
    G_m = projected_scores * probs
    
    bar_P = np.mean(P_m)
    bar_G = np.mean(G_m)
    
    p_loglinear = bar_P * np.exp(alphas * (bar_G / bar_P))
    
    cov_matrix = np.cov(P_m, G_m) / n_traj
    var_P = cov_matrix[0, 0]
    var_G = cov_matrix[1, 1]
    cov_PG = cov_matrix[0, 1]
    
    p_loglinear_ci = []
    for a in alphas:
        exp_factor = np.exp(a * bar_G / bar_P)
        df_dP = exp_factor * (1.0 - a * bar_G / bar_P)
        df_dG = a * exp_factor
        var_f = (df_dP**2) * var_P + (df_dG**2) * var_G + 2.0 * df_dP * df_dG * cov_PG
        p_loglinear_ci.append(z_score * np.sqrt(max(0.0, var_f)))
        
    p_loglinear_ci = np.array(p_loglinear_ci)
    
    
    p_exact = []
    p_ci = []
    
    lin_base = np.einsum('ikl,l->ik', scores, delta_theta)
    quad_base = 0.5 * np.einsum('iklm,l,m->ik', fims, delta_theta, delta_theta)
    
    for a in alphas:
        exponent = a * lin_base - (a**2) * quad_base
        weights = np.mean(np.exp(exponent), axis=1)
        weighted_probs = probs * weights
        p_exact.append(np.mean(weighted_probs))
        p_ci.append(z_score * np.sqrt(np.var(weighted_probs) / n_traj))
    
    p_exact = np.array(p_exact)
    p_ci = np.array(p_ci)
    if theta_reference is None:
        x_values = alphas
        x_label = r'Step size $\alpha$'
    else:
        x_values = alphas / theta_reference
        x_label = r'Step size $\alpha$'
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    
    # ax.fill_between(alphas,
    #                 p_linear - p_linear_ci,
    #                 p_linear + p_linear_ci,
    #                 color='orange', alpha=0.15)
    # ax.plot(alphas, p_linear,  label='Linear estimator', color='orange',linestyle='solid', lw=1.5)

    ax.fill_between(x_values, 
                    p_exact - p_ci, 
                    p_exact + p_ci, 
                    alpha=0.15, color='darkblue')
    # Thinner dotted line in the foreground
    ax.plot(x_values, p_exact, label='Full Girsanov estimator', color='darkblue', linestyle='solid', lw=2.5)
    ax.fill_between(x_values,
                    p_loglinear - p_loglinear_ci,
                    p_loglinear + p_loglinear_ci,
                    color='orange', alpha=0.15)
    # Thicker solid line in the background
    ax.plot(x_values, p_loglinear, label='Cumulant estimator', color='orange', linestyle='--', lw=1.5)


  
        
    if samples is not None:
        sample_alphas = []
        sample_means = []
        sample_errs = []
        
        for a_s, p_s in samples:
            m_p = np.mean(p_s)
            sem_p_s = np.sqrt(np.var(p_s) / len(p_s))
            
            if theta_reference is None:
                sample_alphas.append(a_s)
            else:
                sample_alphas.append( a_s / theta_reference)
            sample_means.append(m_p)
            sample_errs.append(z_score * sem_p_s)
            
        ax.errorbar(sample_alphas, sample_means, yerr=sample_errs, 
                    fmt='o', color='black', capsize=4, elinewidth=1.5, 
                    label='Direct Samples (95% CI)', markersize=5)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'$p(\theta_0 + \alpha)$')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    if theta_reference is None:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    else:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    return fig, ax


def compute_relative_cumulant_error(
    scores,
    fims,
    probs,
    delta_theta,
    theta_reference,
    variation_percent_range=10.0,
    n_points=200,
    ax=None,
):
    """Plot the relative error between full Girsanov and cumulant estimates."""
    if theta_reference == 0.0:
        raise ValueError("theta_reference must be non-zero to compute percentages.")

    variation_percent = np.linspace(
        -variation_percent_range,
        variation_percent_range,
        n_points,
    )
    theta_offsets = theta_reference * variation_percent / 100.0
    delta_theta = np.asarray(delta_theta, dtype=float)
    probs = np.asarray(probs, dtype=float)

    mean_scores = np.mean(scores, axis=1)
    projected_scores = mean_scores @ delta_theta
    p_0 = np.mean(probs)
    weighted_score_mean = np.mean(probs * projected_scores)
    p_cumulant = p_0 * np.exp(
        theta_offsets * weighted_score_mean / p_0
    )

    linear_term = np.einsum("ikl,l->ik", scores, delta_theta)
    quadratic_term = 0.5 * np.einsum(
        "iklm,l,m->ik",
        fims,
        delta_theta,
        delta_theta,
    )

    p_girsanov = np.empty_like(theta_offsets)
    for index, theta_offset in enumerate(theta_offsets):
        exponent = (
            theta_offset * linear_term
            - theta_offset**2 * quadratic_term
        )
        trajectory_weights = np.mean(np.exp(exponent), axis=1)
        p_girsanov[index] = np.mean(probs * trajectory_weights)

    absolute_relative_error = np.abs(np.divide(
        p_girsanov - p_cumulant,
        p_girsanov,
        out=np.full_like(p_girsanov, np.nan),
        where=p_girsanov != 0.0,
    ))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(
        variation_percent,
        absolute_relative_error,
        color="tab:blue",
        linewidth=2.0,
    )
    ax.set_xlabel(
        r"$100\,(\theta - \theta_{\mathcal{L}}) / \theta_{\mathcal{L}}$ (\%)"
    )
    ax.set_ylabel(r"$\left|(p_G(\theta) - p_C(\theta)) / p_G(\theta)\right|$")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    return fig, ax, variation_percent, absolute_relative_error


import numpy as np
from scipy.stats import norm, lognorm, beta
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def compute_probability_pdf(scores, model_probs, thetas_sample, fims=None, ref_probs=None,
                            method='loglinear', variance='heteroskedastic', prior_dist='normal',
                            xlim=None, plot_bounds=False, ax=None, samples_id=None,
                            log_scale=False, use_Ahat=False):

    fontsize = 15
    eps = 1e-12

    n_traj = model_probs.shape[0]
    mean_prob = np.mean(model_probs)
    sigma_model = np.std(model_probs) / np.sqrt(n_traj)

    if ref_probs is not None:
        mean_prob_ref = np.mean(ref_probs)
        sigma_model_ref = np.std(ref_probs) / np.sqrt(ref_probs.shape[0])

    if method in ['linear', 'loglinear']:
        mean_scores = np.mean(scores, axis=1)
        projected = np.einsum('il,lm->im', mean_scores, thetas_sample)

        if method == 'linear':
            p_theta = np.mean(model_probs[:, None] * (1.0 + projected), axis=0)

            if variance == 'estimator':
                var_theta = np.var(model_probs[:, None] * (1.0 + projected), axis=0) / n_traj
                sigma_theta = np.sqrt(var_theta)

        elif method == 'loglinear':
            p_0 = np.mean(model_probs)
            G_all = model_probs[:, None] * projected
            bar_G = np.mean(G_all, axis=0)

            log_slope = bar_G / p_0
            p_theta = p_0 * np.exp(log_slope)

            if variance == 'estimator':
                var_P = np.var(model_probs) / n_traj
                sigma_theta = np.zeros_like(p_theta)

                for m in range(thetas_sample.shape[1]):
                    G_m = G_all[:, m]
                    var_G_m = np.var(G_m) / n_traj
                    cov_P_G = np.cov(model_probs, G_m)[0, 1] / n_traj

                    exp_factor = np.exp(log_slope[m])
                    df_dP = exp_factor * (1.0 - log_slope[m])
                    df_dG = exp_factor

                    var_f = (
                        (df_dP**2) * var_P
                        + (df_dG**2) * var_G_m
                        + 2.0 * df_dP * df_dG * cov_P_G
                    )

                    sigma_theta[m] = np.sqrt(max(0.0, var_f))

    elif method == 'full':
        lin = np.einsum('ikl,lm->ikm', scores, thetas_sample)
        quad = 0.5 * np.einsum('iklm,ld,md->ikd', fims, thetas_sample, thetas_sample)

        exponent = lin - quad
        weights = np.mean(np.exp(exponent), axis=1)

        p_theta = np.mean(model_probs[:, None] * weights, axis=0)

        if variance == 'estimator':
            var_theta = np.var(model_probs[:, None] * weights, axis=0) / n_traj
            sigma_theta = np.sqrt(var_theta)

    else:
        raise ValueError("Method must be 'linear', 'loglinear', or 'full'")

    if variance == 'homoscedastic':
        sigma_theta = np.full_like(p_theta, sigma_model)

    elif variance == 'heteroskedastic':
        sigma_theta = p_theta * (sigma_model / mean_prob)

    if prior_dist == 'normal':

        def cdf_mixture(x):
            return np.mean(norm.cdf(x, loc=p_theta, scale=sigma_theta))

        p_min_search = np.min(p_theta) - 5 * np.max(sigma_theta)
        p_max_search = np.max(p_theta) + 5 * np.max(sigma_theta)

    elif prior_dist == 'lognormal':
        var_theta = sigma_theta**2

        s_ln = np.sqrt(np.log(1.0 + var_theta / (p_theta**2)))
        scale_ln = p_theta / np.sqrt(1.0 + var_theta / (p_theta**2))

        def cdf_mixture(x):
            if x <= 0:
                return 0.0
            return np.mean(lognorm.cdf(x, s=s_ln, scale=scale_ln))

        p_min_search = max(eps, np.min(p_theta) / 10.0)
        p_max_search = np.max(p_theta) * 5.0

    elif prior_dist == 'beta':
        p_theta_clipped = np.clip(p_theta, eps, 1.0 - eps)
        var_theta = sigma_theta**2

        var_max = p_theta_clipped * (1.0 - p_theta_clipped) - eps
        var_theta_clipped = np.minimum(var_theta, var_max)
        var_theta_clipped = np.maximum(var_theta_clipped, eps)

        v = (p_theta_clipped * (1.0 - p_theta_clipped) / var_theta_clipped) - 1.0

        alpha_param = p_theta_clipped * v
        beta_param = (1.0 - p_theta_clipped) * v

        def cdf_mixture(x):
            if x <= 0.0:
                return 0.0
            if x >= 1.0:
                return 1.0
            return np.mean(beta.cdf(x, alpha_param, beta_param))

        p_min_search = eps
        p_max_search = 1.0 - eps

    else:
        raise ValueError("prior_dist must be 'normal', 'lognormal', or 'beta'")

    ci_marg = [
        brentq(lambda x: cdf_mixture(x) - 0.025, p_min_search, p_max_search),
        brentq(lambda x: cdf_mixture(x) - 0.975, p_min_search, p_max_search)
    ]

    ci_model = [
        mean_prob - 1.96 * sigma_model,
        mean_prob + 1.96 * sigma_model
    ]

    ci_ref = (
        [
            mean_prob_ref - 1.96 * sigma_model_ref,
            mean_prob_ref + 1.96 * sigma_model_ref
        ]
        if ref_probs is not None else (None, None)
    )

    max_sigma = np.max(sigma_theta)

    p_min = min(
        ci_model[0],
        ci_marg[0],
        ci_ref[0] if ci_ref[0] is not None else np.inf
    ) - 3 * max_sigma

    p_max = max(
        ci_model[1],
        ci_marg[1],
        ci_ref[1] if ci_ref[1] is not None else -np.inf
    ) + 3 * max_sigma

    p_min_grid = max(eps, p_min)
    p_max_grid = min(1.0 - eps, p_max)

    prob_grid = np.linspace(p_min_grid, p_max_grid, 1000)

    if prior_dist == 'normal':
        pdf_vals = np.mean(
            norm.pdf(
                prob_grid[:, None],
                loc=p_theta[None, :],
                scale=sigma_theta[None, :]
            ),
            axis=1
        )

    elif prior_dist == 'lognormal':
        pdf_vals = np.mean(
            lognorm.pdf(
                prob_grid[:, None],
                s=s_ln[None, :],
                scale=scale_ln[None, :]
            ),
            axis=1
        )

    elif prior_dist == 'beta':
        pdf_vals = np.mean(
            beta.pdf(
                prob_grid[:, None],
                alpha_param[None, :],
                beta_param[None, :]
            ),
            axis=1
        )

    pdf_model = norm.pdf(
        prob_grid,
        loc=mean_prob,
        scale=sigma_model
    )

    if ref_probs is not None:
        pdf_ref = norm.pdf(
            prob_grid,
            loc=mean_prob_ref,
            scale=sigma_model_ref
        )
    else:
        pdf_ref = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    color_rho_pi = "tab:green"
    color_ref = 'black'
    color_model = 'tab:orange'

    # ============================================================
    # Changement de variable :
    #
    #     A_hat = -log10(p)
    #
    # If f_p(p) is the density of p, then:
    #
    #     f_A(A) = f_p(10**(-A)) * ln(10) * 10**(-A)
    #
    # Comme p = 10**(-A), le jacobien vaut ln(10) * p.
    # ============================================================

    if use_Ahat:
        x_plot = -np.log10(np.maximum(prob_grid, eps))

        jacobian = np.log(10.0) * prob_grid
        pdf_vals_plot = pdf_vals * jacobian
        pdf_model_plot = pdf_model * jacobian

        if pdf_ref is not None:
            pdf_ref_plot = pdf_ref * jacobian
        else:
            pdf_ref_plot = None

        # Reverse so A_hat increases from left to right
        x_plot = x_plot[::-1]
        pdf_vals_plot = pdf_vals_plot[::-1]
        pdf_model_plot = pdf_model_plot[::-1]

        if pdf_ref_plot is not None:
            pdf_ref_plot = pdf_ref_plot[::-1]

        xlabel = r'$-\log_{10}(\langle \hat{p}_{A \to B} \rangle _{\lambda_{\partial A}})$'
        ylabel = r'Density'

    else:
        x_plot = prob_grid
        pdf_vals_plot = pdf_vals
        pdf_model_plot = pdf_model
        pdf_ref_plot = pdf_ref

        xlabel = r'$p = p_{A \to B}(\partial A)$'
        ylabel = r'Density'

    ax.plot(
        x_plot,
        pdf_vals_plot,
        color=color_rho_pi,
        lw=2,
        label=r'$\rho(\cdot\mid\pi)$'
    )

    if use_Ahat:
        ci_left = -np.log10(max(ci_marg[1], eps))
        ci_right = -np.log10(max(ci_marg[0], eps))

        ax.axvspan(
            ci_left,
            ci_right,
            color=color_rho_pi,
            alpha=0.2
        )
    else:
        ax.axvspan(
            ci_marg[0],
            ci_marg[1],
            color=color_rho_pi,
            alpha=0.2
        )

    indices_random = (
        samples_id
        if samples_id is not None
        else np.random.choice(
            len(p_theta),
            size=min(10, len(p_theta)),
            replace=False
        )
    )

    for i, idx in enumerate(indices_random):

        if prior_dist == 'normal':
            pdf_indiv = norm.pdf(
                prob_grid,
                loc=p_theta[idx],
                scale=sigma_theta[idx]
            )

        elif prior_dist == 'lognormal':
            pdf_indiv = lognorm.pdf(
                prob_grid,
                s=s_ln[idx],
                scale=scale_ln[idx]
            )

        elif prior_dist == 'beta':
            pdf_indiv = beta.pdf(
                prob_grid,
                alpha_param[idx],
                beta_param[idx]
            )

        if use_Ahat:
            pdf_indiv_plot = pdf_indiv * jacobian
            pdf_indiv_plot = pdf_indiv_plot[::-1]
        else:
            pdf_indiv_plot = pdf_indiv

        label = r'$\rho(\cdot \mid \Theta_{\text{sample}})$' if i == 0 else None

        ax.plot(
            x_plot,
            pdf_indiv_plot,
            color='gray',
            lw=1,
            alpha=0.7,
            linestyle=':',
            label=label
        )

    ax.plot(
        x_plot,
        pdf_model_plot,
        color=color_model,
        lw=2,
        label=r'$\rho(\cdot\mid \Theta_\mathcal{L})$'
    )

    if ref_probs is not None:
        ax.plot(
            x_plot,
            pdf_ref_plot,
            color=color_ref,
            lw=2,
            label=r'${\rho}_{\text{ref}}$'
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True, linestyle=':', alpha=0.6)

    if log_scale and not use_Ahat:
        ax.set_xscale('log')

    if xlim is not None:
        ax.set_xlim(xlim)

    y_max = max(
        np.max(pdf_vals_plot),
        np.max(pdf_model_plot)
    )

    if pdf_ref_plot is not None:
        y_max = max(y_max, np.max(pdf_ref_plot))

    ax.set_ylim(0, y_max * 1.2)

    if not use_Ahat:
        formatter_x = ticker.ScalarFormatter(useMathText=True)
        formatter_x.set_scientific(True)
        formatter_x.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(formatter_x)

    formatter_y = ticker.ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter_y)

    ax.legend()

    return fig, ax, p_theta


def compute_probability_violin(scores, model_probs, thetas_sample, fims=None, ref_probs=None,
                               method='loglinear', variance='heteroskedastic',
                               prior_dist='normal', xlim=None, ax=None,
                               n_samples=2000, 
                               rng=None, use_Ahat=False, n_theta_plot=5, theta_plot_indices=None,
                               initial_descriptors=None, beta=None):
    fontsize = 15
    eps = 1e-12

    rng = np.random.default_rng(rng)

    n_traj = model_probs.shape[0]
    mean_prob = np.mean(model_probs)
    sigma_model = np.std(model_probs) / np.sqrt(n_traj)

    if ref_probs is not None:
        mean_prob_ref = np.mean(ref_probs)
        sigma_model_ref = np.std(ref_probs) / np.sqrt(ref_probs.shape[0])

    if initial_descriptors is not None:
        if beta is not None:
            Z = np.mean(np.exp(-beta * (initial_descriptors @ thetas_sample)), axis=0)
        else:
            Z = np.mean(np.exp(initial_descriptors @ thetas_sample), axis=0)
    else:
        Z = np.ones(thetas_sample.shape[1])

    if method in ['linear', 'loglinear', 'order2']:
        mean_scores = np.mean(scores, axis=1)
        projected = np.einsum('il,lm->im', mean_scores, thetas_sample)

        if method == 'linear':
            p_theta = np.mean(model_probs[:, None] * (1.0 + projected), axis=0) / Z

            if variance == 'estimator':
                var_theta = np.var(model_probs[:, None] * (1.0 + projected), axis=0) / n_traj
                sigma_theta = np.sqrt(var_theta)

        elif method == 'loglinear':
            p_0 = np.mean(model_probs)
            G_all = model_probs[:, None] * projected
            bar_G = np.mean(G_all, axis=0)

            log_slope = bar_G / p_0
            p_theta = (p_0 * np.exp(log_slope)) / Z

            if variance == 'estimator':
                var_P = np.var(model_probs) / n_traj
                sigma_theta = np.zeros_like(p_theta)

                for m in range(thetas_sample.shape[1]):
                    G_m = G_all[:, m]
                    var_G_m = np.var(G_m) / n_traj
                    cov_P_G = np.cov(model_probs, G_m)[0, 1] / n_traj

                    exp_factor = np.exp(log_slope[m]) / Z[m]
                    df_dP = exp_factor * (1.0 - log_slope[m])
                    df_dG = exp_factor

                    var_f = (
                        (df_dP**2) * var_P
                        + (df_dG**2) * var_G_m
                        + 2.0 * df_dP * df_dG * cov_P_G
                    )

                    sigma_theta[m] = np.sqrt(max(0.0, var_f))

        elif method == 'order2':
            p_0 = np.mean(model_probs)
            
            E_YS = np.mean(model_probs[:, None, None] * scores, axis=(0, 1))
            E_YFIM = np.mean(model_probs[:, None, None, None] * fims, axis=(0, 1))
            
            scores_flat = scores.reshape(-1, scores.shape[-1])
            weights_flat = np.repeat(model_probs, scores.shape[1])[:, None]
            E_YSS = (scores_flat * weights_flat).T @ scores_flat / (n_traj * scores.shape[1])
            
            t1 = np.einsum('l,lm->m', E_YS / p_0, thetas_sample)
            t2 = 0.5 * np.einsum('ld,lm,dm->m', E_YFIM / p_0, thetas_sample, thetas_sample)
            
            cov_Y = E_YSS / p_0 - np.outer(E_YS, E_YS) / (p_0**2)
            t3 = 0.5 * np.einsum('ld,lm,dm->m', cov_Y, thetas_sample, thetas_sample)
            
            log_factor = t1 - t2 + t3
            p_theta = (p_0 * np.exp(log_factor)) / Z

            if variance == 'estimator':
                sigma_theta = p_theta * np.sqrt(
                    (sigma_model / mean_prob) ** 2
                    + np.var(np.einsum('ikl,lm->ikm', scores, thetas_sample), axis=(0, 1)) / n_traj
                )

    elif method == 'full':
        lin = np.einsum('ikl,lm->ikm', scores, thetas_sample)
        quad = 0.5 * np.einsum('iklm,ld,md->ikd', fims, thetas_sample, thetas_sample)

        exponent = lin - quad
        weights = np.mean(np.exp(exponent), axis=1)

        p_theta = np.mean(model_probs[:, None] * weights, axis=0) / Z

        if variance == 'estimator':
            var_theta = np.var(model_probs[:, None] * weights, axis=0) / n_traj
            sigma_theta = np.sqrt(var_theta)
    
    else:
        raise ValueError("Method must be 'linear', 'loglinear', or 'full'")

    if variance == 'homoscedastic':
        sigma_theta = np.full_like(p_theta, sigma_model)

    elif variance == 'heteroskedastic':
        sigma_theta = p_theta * (sigma_model / mean_prob)

    p_theta_clipped = np.clip(p_theta, eps, 1.0 - eps)
    var_theta = np.maximum(sigma_theta**2, eps)

    if prior_dist == 'normal':
        theta_indices = rng.integers(0, len(p_theta), size=n_samples)
        rho_pi_samples = rng.normal(
            loc=p_theta[theta_indices],
            scale=sigma_theta[theta_indices]
        )

    elif prior_dist == 'lognormal':
        s_ln = np.sqrt(np.log(1.0 + var_theta / (p_theta_clipped**2)))
        scale_ln = p_theta_clipped / np.sqrt(1.0 + var_theta / (p_theta_clipped**2))

        theta_indices = rng.integers(0, len(p_theta), size=n_samples)
        rho_pi_samples = rng.lognormal(
            mean=np.log(scale_ln[theta_indices]),
            sigma=s_ln[theta_indices]
        )

    elif prior_dist == 'beta':
        var_max = p_theta_clipped * (1.0 - p_theta_clipped) - eps
        var_theta_clipped = np.minimum(var_theta, var_max)
        var_theta_clipped = np.maximum(var_theta_clipped, eps)

        v = (p_theta_clipped * (1.0 - p_theta_clipped) / var_theta_clipped) - 1.0
        alpha_param = p_theta_clipped * v
        beta_param = (1.0 - p_theta_clipped) * v

        theta_indices = rng.integers(0, len(p_theta), size=n_samples)
        rho_pi_samples = rng.beta(
            alpha_param[theta_indices],
            beta_param[theta_indices]
        )

    else:
        raise ValueError("prior_dist must be 'normal', 'lognormal', or 'beta'")

    if theta_plot_indices is not None:
        theta_plot_indices = theta_plot_indices
    else:   
        theta_plot_indices = rng.choice(
            len(p_theta),
            size=min(n_theta_plot, len(p_theta)),
            replace=False
        )

    theta_plot_samples = []
    for idx in theta_plot_indices:
        if prior_dist == 'normal':
            samples_theta = rng.normal(
                loc=p_theta[idx],
                scale=sigma_theta[idx],
                size=n_samples
            )
        elif prior_dist == 'lognormal':
            samples_theta = rng.lognormal(
                mean=np.log(scale_ln[idx]),
                sigma=s_ln[idx],
                size=n_samples
            )
        elif prior_dist == 'beta':
            samples_theta = rng.beta(
                alpha_param[idx],
                beta_param[idx],
                size=n_samples
            )
        theta_plot_samples.append(np.clip(samples_theta, eps, 1.0 - eps))

    model_samples = rng.normal(mean_prob, sigma_model, size=n_samples)
    sample_groups = [rho_pi_samples, model_samples]
    labels = [r'UQ $\pi$', r'MLE $\theta_\mathcal{L}$']
    colors = ["tab:green", "tab:orange"]

    if ref_probs is not None:
        ref_samples = rng.normal(mean_prob_ref, sigma_model_ref, size=n_samples)
        sample_groups.append(ref_samples)
        labels.append(r'Reference')
        colors.append("black")

    sample_groups = [
        np.clip(samples, eps, 1.0 - eps)
        for samples in sample_groups
    ]

    if use_Ahat:
        sample_groups = [-np.log10(samples) for samples in sample_groups]
        theta_plot_samples = [-np.log10(samples) for samples in theta_plot_samples]
        xlabel = r'$-\log_{10}(p)$'
    else:
        xlabel = r'$p = p_{A \to B}(\partial A)$'

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    positions = np.arange(len(sample_groups), 0, -1)
    violins = ax.violinplot(
        sample_groups,
        positions=positions,
        vert=False,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    for body, color in zip(violins['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)

    theta_position = len(sample_groups) + 1
    theta_violins = ax.violinplot(
        theta_plot_samples,
        positions=np.full(len(theta_plot_samples), theta_position, dtype=float),
        vert=False,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in theta_violins['bodies']:
        body.set_facecolor('gray')
        body.set_edgecolor('gray')
        body.set_alpha(0.1)

    violins['cmedians'].set_color('black')
    violins['cmedians'].set_linewidth(1.5)

    for samples, position, color in zip(sample_groups, positions, colors):
        q025, q25, q50, q75, q975 = np.quantile(
            samples,
            [0.025, 0.25, 0.5, 0.75, 0.975]
        )
        ax.hlines(position, q025, q975, color=color, lw=1.2, alpha=0.9)
        ax.hlines(position, q25, q75, color=color, lw=4.0, alpha=0.9)
        ax.plot(q50, position, marker='o', color='black', markersize=4)

    ax.set_yticks(np.r_[theta_position, positions])
    ax.set_yticklabels([r'Samples $\theta_i \sim \pi$'] + labels)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)

    if xlim is not None:
        ax.set_xlim(xlim)

    if not use_Ahat:
        formatter_x = ticker.ScalarFormatter(useMathText=True)
        formatter_x.set_scientific(True)
        formatter_x.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(formatter_x)

    return fig, ax, p_theta, sample_groups

def _lognormal_params_from_mean_std(mean, std, eps=1e-12):
    mean = np.maximum(mean, eps)
    var = np.maximum(std**2, eps)
    sigma = np.sqrt(np.log1p(var / mean**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma


def _draw_lognormal(mean, std, rng, size, eps=1e-12):
    mu, sigma = _lognormal_params_from_mean_std(mean, std, eps=eps)
    return rng.lognormal(mu, sigma, size=size)


def _draw_flux_samples(loop_times, rng, n_samples, eps=1e-12, fs_to_s=1e-15):
    loop_times = np.asarray(loop_times, dtype=float)
    loop_times = loop_times[np.isfinite(loop_times) & (loop_times > 0.0)]
    if loop_times.size < 2:
        raise ValueError("Need at least two positive t_loop values to estimate phi_A.")

    mean_loop = np.mean(loop_times)
    sem_loop = np.std(loop_times, ddof=1) / np.sqrt(loop_times.size)
    mean_flux = 1.0 / (mean_loop * fs_to_s)
    std_flux = sem_loop / mean_loop**2 / fs_to_s
    return _draw_lognormal(mean_flux, std_flux, rng, n_samples, eps=eps)


def _loglinear_probability_stats(scores, model_probs, thetas_sample, eps=1e-12):
    n_traj = model_probs.shape[0]
    p_0 = np.mean(model_probs)
    mean_scores = np.mean(scores, axis=1)
    projected = np.einsum('il,lm->im', mean_scores, thetas_sample)

    G_all = model_probs[:, None] * projected
    bar_G = np.mean(G_all, axis=0)
    log_slope = bar_G / p_0
    p_theta = p_0 * np.exp(log_slope)

    var_P = np.var(model_probs) / n_traj
    sigma_theta = np.zeros_like(p_theta)

    for m in range(thetas_sample.shape[1]):
        G_m = G_all[:, m]
        var_G_m = np.var(G_m) / n_traj
        cov_P_G = np.cov(model_probs, G_m)[0, 1] / n_traj
        exp_factor = np.exp(log_slope[m])
        df_dP = exp_factor * (1.0 - log_slope[m])
        df_dG = exp_factor
        var_f = (
            df_dP**2 * var_P
            + df_dG**2 * var_G_m
            + 2.0 * df_dP * df_dG * cov_P_G
        )
        sigma_theta[m] = np.sqrt(max(0.0, var_f))

    return np.clip(p_theta, eps, 1.0 - eps), np.maximum(sigma_theta, eps)


def _draw_probability_samples(means, stds, rng, n_samples, eps=1e-12):
    indices = rng.integers(0, means.size, size=n_samples)
    samples = _draw_lognormal(means[indices], stds[indices], rng, n_samples, eps=eps)
    return np.clip(samples, eps, 1.0 - eps)


def _draw_single_probability_samples(probs, rng, n_samples, eps=1e-12):
    mean = np.mean(probs)
    sem = np.std(probs) / np.sqrt(probs.shape[0])
    samples = _draw_lognormal(mean, sem, rng, n_samples, eps=eps)
    return np.clip(samples, eps, 1.0 - eps)


def compute_log_rate_violin(scores, model_probs, loop_times, thetas_sample,
                            ref_probs=None, ref_loop_times=None, ax=None,
                            n_samples=2000, rng=None, n_theta_plot=5,
                            theta_plot_indices=None, xlim=None):
    eps = 1e-12
    rng = np.random.default_rng(rng)

    flux_samples = _draw_flux_samples(loop_times, rng, n_samples, eps=eps)
    p_theta, sigma_theta = _loglinear_probability_stats(
        scores,
        model_probs,
        thetas_sample,
        eps=eps,
    )

    sample_groups = [
        np.log10(flux_samples * _draw_probability_samples(p_theta, sigma_theta, rng, n_samples, eps=eps)),
        np.log10(flux_samples * _draw_single_probability_samples(model_probs, rng, n_samples, eps=eps)),
    ]
    labels = [r'UQ $\pi$', r'MLE $\theta_\mathcal{L}$']
    colors = ["tab:green", "tab:orange"]

    if ref_probs is not None:
        if ref_loop_times is None:
            raise ValueError("ref_loop_times is required when ref_probs is provided.")
        ref_flux_samples = _draw_flux_samples(ref_loop_times, rng, n_samples, eps=eps)
        sample_groups.append(
            np.log10(
                ref_flux_samples
                * _draw_single_probability_samples(ref_probs, rng, n_samples, eps=eps)
            )
        )
        labels.append(r'Reference')
        colors.append("black")

    if theta_plot_indices is None:
        theta_plot_indices = rng.choice(
            len(p_theta),
            size=min(n_theta_plot, len(p_theta)),
            replace=False,
        )

    theta_plot_samples = [
        np.log10(
            flux_samples
            * _draw_lognormal(p_theta[idx], sigma_theta[idx], rng, n_samples, eps=eps)
        )
        for idx in theta_plot_indices
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    positions = np.arange(len(sample_groups), 0, -1)
    violins = ax.violinplot(
        sample_groups,
        positions=positions,
        vert=False,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for body, color in zip(violins['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)

    theta_position = len(sample_groups) + 1
    theta_violins = ax.violinplot(
        theta_plot_samples,
        positions=np.full(len(theta_plot_samples), theta_position, dtype=float),
        vert=False,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in theta_violins['bodies']:
        body.set_facecolor('gray')
        body.set_edgecolor('gray')
        body.set_alpha(0.1)

    violins['cmedians'].set_color('black')
    violins['cmedians'].set_linewidth(1.5)

    for samples, position, color in zip(sample_groups, positions, colors):
        q025, q25, q50, q75, q975 = np.quantile(
            samples,
            [0.025, 0.25, 0.5, 0.75, 0.975],
        )
        ax.hlines(position, q025, q975, color=color, lw=1.2, alpha=0.9)
        ax.hlines(position, q25, q75, color=color, lw=4.0, alpha=0.9)
        ax.plot(q50, position, marker='o', color='black', markersize=4)

    ax.set_yticks(np.r_[theta_position, positions])
    ax.set_yticklabels([r'Samples $\theta_i \sim \pi$'] + labels)
    ax.set_xlabel(r'$\log_{10}(k_{AB}) \ (\mathrm{s}^{-1})$')
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)

    if xlim is not None:
        ax.set_xlim(xlim)

    return fig, ax, p_theta, sample_groups
