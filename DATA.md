# Data inventory and availability

This repository does not version large computational artifacts (trajectories, `.npy/.npz` aggregates, trained models). This page documents what the scripts expect.

## Observed versioning rules

- `.gitignore` explicitly excludes `*.traj`, `*.npy`, `*.npz`.
- The `.model` files required for butane experiments are not present in the repository.

## Expected models (absent from the repository)

### Butane (`scripts/AMS_butane/models/`)

The `run_ams.py`, `sample_ini_conds/run_ini_conds.py`, and `reweighting/compute_ini_D.py` scripts in the `ams_runs_*` folders expect:

- `mace-mpa-0-medium.model`
- `mace_mp0a_ft.model`
- `mace_omat0_ft.model`

References:
- `scripts/AMS_butane/ams_runs_mpa_300/run_ams.py`
- `scripts/AMS_butane/ams_runs_mp0a_300/run_ams.py`
- `scripts/AMS_butane/ams_runs_omat0_300/run_ams.py`
- same patterns in `*_200`, `*_500`, and `theta_mp0a_500/theta_*/run_ams.py`.

## Expected/generated trajectories (not versioned)

These trajectories are consumed by reweighting scripts and produced during AMS/MD runs:

- `rep_*.traj` in `ams/ams_*` folders (1D, Müller-Brown, dimers)
- `md_traj_*.traj` in `ini_conds/` (initial condition sampling)
- `.traj` files in butane folders for `reweighting/reweight.py` and `reweighting_full/reweight.py`

References:
- `scripts/AMS_1D/ams_fit/ams/run_reweight.py`
- `scripts/AMS_1D/ams_target/ams/run_reweight.py`
- `scripts/AMS_muller_brown/ams/ams/run_reweight.py`
- `scripts/AMS_dimers/ams/ams/run_reweight.py`
- `scripts/AMS_butane/ams_runs_*/reweighting/reweight.py`

## `.npy/.npz` files expected by notebooks (absent from the repository)

### AMS_1D

- `scripts/AMS_1D/data_1D/reweighting_aggregate_fit_40_500.npz`
- `scripts/AMS_1D/data_1D/reweighting_aggregate_target_40_500.npz`
- `scripts/AMS_1D/pops_data/misspecification_sigma.npy`
- `scripts/AMS_1D/pops_data/posterior_samples.npy`

Reference: `scripts/AMS_1D/post_process.ipynb`.

### AMS_muller_brown

- `scripts/AMS_muller_brown/data_muller_brown/reweighting_aggregate.npz`
- `scripts/AMS_muller_brown/data_muller_brown/*_ref_probs.npy`

Reference: `scripts/AMS_muller_brown/post_process.ipynb`.

### AMS_dimers

- `scripts/AMS_dimers/data_dimer/reweighting_aggregate.npz`
- `scripts/AMS_dimers/data_dimer/probas_h-.npy`
- `scripts/AMS_dimers/data_dimer/probas_h--.npy`
- `scripts/AMS_dimers/data_dimer/probas_epsilon-.npy`
- `scripts/AMS_dimers/data_dimer/probas_epsilon--.npy`

Reference: `scripts/AMS_dimers/post_process.ipynb`.

### Butane (non-versioned pipeline outputs)

Examples of outputs written by the scripts:

- `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/final_scores.npy`
- `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/final_probs.npy`
- `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/ini_D.npy`
- `scripts/AMS_butane/theta_mp0a_500/results/probs.npy`
- `scripts/AMS_butane/theta_mp0a_500/results/thetas.npy`

References: `aggregate.py`, `compute_ini_D.py`, `get_results.py`, `plots/` scripts.

## License status

Current state:

- No `LICENSE`/`LICENCE` file is present at the repository root.
- `pyproject.toml` does not declare a distributable license.
- `CITATION.cff` is provided with `license: NOASSERTION` to reflect this state without arbitrarily choosing a license.

Recommended action (outside the scope of this section): explicitly choose a license and add a `LICENSE` file validated by the maintainers.
