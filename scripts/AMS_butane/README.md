# AMS butane 3 workflow

This folder is a clean restart of the butane workflow with an explicit balanced
fine-tuning dataset:

- `A`: 4000 sampled configurations
- `B_plus`: 2000 sampled configurations
- `B_minus`: 2000 sampled configurations

Expected model files in `models/`:

- `mace-mpa-0-medium.model`
- `mace_mp0a_small.model`
- `mace_mp0a_ft.model` after fine tuning
- `mace_omat0_ft.model` after fine tuning, if using the OMAT branch

## Launch order

From `scripts/AMS_butane_3/FT_mace/samples`:

```sh
ccc_msub job_A_topaze
ccc_msub job_B_plus_topaze
ccc_msub job_B_minus_topaze
```

From `scripts/AMS_butane_3/FT_mace`:

```sh
ccc_msub job_prepare_datasets_topaze
ccc_msub job_ft_topaze
```

`prepare_datasets.py` writes:

- `samples/A/points.xyz`
- `samples/B_plus/points.xyz`
- `samples/B_minus/points.xyz`
- `dataset_8000.xyz`
- `train_set.xyz`
- `valid_set.xyz`
- `test_set.xyz`

From `scripts/AMS_butane_3/train_linear_mace`:

```sh
ccc_msub matrix_job_topaze
python train_and_plot_poster.py
```

Then prepare and launch the AMS runs from each `ams_runs_*` directory:

```sh
python make_input_files.py
./run_all_topaze
```
