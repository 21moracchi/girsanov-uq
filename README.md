# Girsanov UQ

Code accompanying our paper on parametric uncertainty propagation for rare-event probabilities using Adaptive Multilevel Splitting (AMS) and Girsanov reweighting.

The reusable Python package is `girsanov_uq` (`src/girsanov_uq`). The remainder of the repository contains the experimental pipelines (1D, Müller–Brown, dimers, and butane), notebooks, and post-processing scripts.

If you use this repository, please cite:

> L. Moracchini, T. Pigeon, M. Menz, T. Faney, T. D. Swinburne, and M.-C. Marinica, *Girsanov Reweighting for Uncertainty Propagation in Rare-Event Kinetics*, arXiv:2607.13757 (2026). https://arxiv.org/abs/2607.13757

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Minimal verification:

```bash
python -m unittest tests.test_imports
```

## Quick CPU example

Run the local 1D demonstration notebook (CPU):

```bash
PYTHONPATH=src jupyter lab notebooks/pops_uncertainty_pipeline_1d.ipynb
```

This notebook demonstrates the complete workflow on a simplified 1D potential, including POPS training, AMS simulations, Girsanov reweighting, and uncertainty aggregation.

## Repository structure

- `src/girsanov_uq/`: Python package (integrators, potentials, reweighting)
- `scripts/AMS_1D/`: 1D experiments (target and fitted potentials)
- `scripts/AMS_muller_brown/`: Müller–Brown experiments
- `scripts/AMS_dimers/`: dimer experiments
- `scripts/AMS_butane/`: butane experiments together with POPS scripts and plotting utilities
- `notebooks/`: demonstration notebook
- `tests/`: lightweight import tests
- `DATA.md`: inventory of non-versioned datasets and trained models

## Reproducing the experiments

The experimental pipelines are primarily designed for execution on HPC clusters using batch schedulers (`sbatch` or `ccc_msub`). Outputs are generated in the `ams_*`, `ini_conds/*`, and `reweighting*` directories.

Example entry points:

```bash
python scripts/AMS_1D/ams_fit/run_pipeline.py --n-reps 500
python scripts/AMS_1D/ams_target/run_pipeline.py --n-reps 500
python scripts/AMS_muller_brown/ams/run_pipeline.py --n-reps 500
python scripts/AMS_dimers/ams/run_pipeline.py --n-reps 500
```

For the butane system, the repository is organized by experiment (`ams_runs_mp0a_*`, `ams_runs_mpa_*`, `ams_runs_omat0_*`). Local orchestration is handled through `scripts/AMS_butane/ams_runs_*/make_input_files.py`, `scripts/AMS_butane/ams_runs_*/run_all_topaze`, and `scripts/AMS_butane/ams_runs_*/reweighting/*`.

## Mapping between paper systems and repository structure

| System | Main scripts | Main inputs | Main outputs |
| --- | --- | --- | --- |
| 1D (fitted potential) | `scripts/AMS_1D/ams_fit/run_pipeline.py`, `scripts/AMS_1D/ams_fit/aggregate_reweighting.py` | CLI parameters (`--n-reps`, `--n-ams`, `--n-walkers`), `scripts/AMS_1D/ams_fit/ini_conds/run_ini_conds.py` | `scripts/AMS_1D/ams_fit/ams/ams_*/reweighting_results.npz`, `scripts/AMS_1D/ams_fit/reweighting_aggregate.npz` |
| 1D (target potential) | `scripts/AMS_1D/ams_target/run_pipeline.py`, `scripts/AMS_1D/ams_target/aggregate_reweighting.py` | CLI parameters (`--n-reps`, `--n-ams`, `--n-walkers`), `scripts/AMS_1D/ams_target/ini_conds/run_ini_conds.py` | `scripts/AMS_1D/ams_target/ams/ams_*/reweighting_results.npz`, `scripts/AMS_1D/ams_target/reweighting_aggregate.npz` |
| Müller–Brown | `scripts/AMS_muller_brown/ams/run_pipeline.py`, `scripts/AMS_muller_brown/ams/aggregate_reweighting.py` | Pipeline CLI parameters, `scripts/AMS_muller_brown/ams/ini_conds/run_ini_conds.py` | `scripts/AMS_muller_brown/ams/ams/ams_*/reweighting_results.npz`, `scripts/AMS_muller_brown/ams/reweighting_aggregate.npz` |
| Dimer | `scripts/AMS_dimers/ams/run_pipeline.py`, `scripts/AMS_dimers/ams/aggregate_reweighting.py` | Pipeline CLI parameters, `scripts/AMS_dimers/ams/ini_conds/run_ini_conds.py` | `scripts/AMS_dimers/ams/ams/ams_*/reweighting_results.npz`, `scripts/AMS_dimers/ams/reweighting_aggregate.npz` |
| Butane | `scripts/AMS_butane/ams_runs_*/run_ams.py`, `scripts/AMS_butane/ams_runs_*/reweighting/{reweight.py,aggregate.py}`| MACE models in `scripts/AMS_butane/models/` (not versioned), AMS trajectories and initial conditions generated through batch jobs | `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/final_scores.npy`, `final_probs.npy`, `ini_D.npy` |

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
