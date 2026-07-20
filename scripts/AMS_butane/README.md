# Workflow AMS butane

Ce dossier contient les campagnes butane (modèles `mpa`, `mp0a`, `omat0`) et les scripts de post-traitement.

## Pré-requis

Les fichiers suivants sont attendus dans `scripts/AMS_butane/models/` (non versionnés dans le dépôt) :

- `mace-mpa-0-medium.model`
- `mace_mp0a_ft.model`
- `mace_omat0_ft.model`

## Organisation

- `ams_runs_mpa_{200,300,500}/`
- `ams_runs_mp0a_{200,300,500}/`
- `ams_runs_omat0_{200,300,500}/`
- `scripts/AMS_butane/theta_mp0a_500/`
- `scripts/AMS_butane/plots/`

Chaque dossier `ams_runs_*` contient typiquement :

- `scripts/AMS_butane/ams_runs_*/make_input_files.py` (génération des dossiers/jobs)
- `scripts/AMS_butane/ams_runs_*/run_ams.py` (exécution AMS)
- `scripts/AMS_butane/ams_runs_*/sample_ini_conds/` (génération/organisation des conditions initiales)
- `scripts/AMS_butane/ams_runs_*/reweighting/` (reweighting + agrégation)
- `scripts/AMS_butane/ams_runs_*/reweighting_full/` (version complète selon le cas)

## Ordre d'exécution (par cas `ams_runs_*`)

```sh
python make_input_files.py
./run_all_topaze
cd reweighting
python make_input_files.py
./run_all_topaze
python aggregate.py
```

Les sorties agrégées sont écrites sous `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/` (ex. `final_scores.npy`, `final_probs.npy`, `ini_D.npy`).

## Campagnes theta

Pour `scripts/AMS_butane/theta_mp0a_500/`, lancer depuis ce dossier :

```sh
ccc_msub submit_theta_batches_topaze.sh
```

Puis agréger avec :

```sh
python get_results.py
```
