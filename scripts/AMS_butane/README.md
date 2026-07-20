# Butane AMS Workflow

This directory contains the butane experiments (using the `mpa`, `mp0a`, and `omat0` models) together with the associated post-processing scripts.

## Prerequisites

The following model files are expected in `scripts/AMS_butane/models/` (they are **not** versioned in this repository):

- `mace-mpa-0-medium.model`
- `mace_mp0a_ft.model`
- `mace_omat0_ft.model`

## Directory organization

- `ams_runs_mpa_{200,300,500}/`
- `ams_runs_mp0a_{200,300,500}/`
- `ams_runs_omat0_{200,300,500}/`
- `scripts/AMS_butane/theta_mp0a_500/`
- `scripts/AMS_butane/plots/`

Each `ams_runs_*` directory typically contains:

- `scripts/AMS_butane/ams_runs_*/make_input_files.py` – generates input folders and batch jobs
- `scripts/AMS_butane/ams_runs_*/run_ams.py` – runs the AMS simulations
- `scripts/AMS_butane/ams_runs_*/sample_ini_conds/` – generates and organizes initial conditions
- `scripts/AMS_butane/ams_runs_*/reweighting/` – Girsanov reweighting and result aggregation
- `scripts/AMS_butane/ams_runs_*/reweighting_full/` – full reweighting workflow (when applicable)

## Execution order (for each `ams_runs_*` case)

```sh
python make_input_files.py
./run_all_topaze
cd reweighting
python make_input_files.py
./run_all_topaze
python aggregate.py
```

The aggregated outputs are written to `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/`, including files such as:

- `final_scores.npy`
- `final_probs.npy`
- `ini_D.npy`

## Theta campaigns

For the `scripts/AMS_butane/theta_mp0a_500/` directory, run:

```sh
ccc_msub submit_theta_batches_topaze.sh
```

Then aggregate the results with:

```sh
python get_results.py
```
