# Data inventory and availability

Ce dépôt ne versionne pas les gros artefacts de calcul (trajectoires, agrégats `.npy/.npz`, modèles entraînés). Cette page documente ce qui est attendu par les scripts.

## Règles de versionnement observées

- `.gitignore` exclut explicitement `*.traj`, `*.npy`, `*.npz`.
- Les fichiers de modèles `.model` nécessaires aux expériences butane ne sont pas présents dans le dépôt.

## Modèles attendus (absents du dépôt)

### Butane (`scripts/AMS_butane/models/`)

Les scripts `run_ams.py`, `sample_ini_conds/run_ini_conds.py` et `reweighting/compute_ini_D.py` des dossiers `ams_runs_*` attendent :

- `mace-mpa-0-medium.model`
- `mace_mp0a_ft.model`
- `mace_omat0_ft.model`

Références :
- `scripts/AMS_butane/ams_runs_mpa_300/run_ams.py`
- `scripts/AMS_butane/ams_runs_mp0a_300/run_ams.py`
- `scripts/AMS_butane/ams_runs_omat0_300/run_ams.py`
- mêmes motifs dans `*_200`, `*_500` et `theta_mp0a_500/theta_*/run_ams.py`.

## Trajectoires attendues/générées (non versionnées)

Ces trajectoires sont consommées par les scripts de reweighting et produites pendant les runs AMS/MD :

- `rep_*.traj` dans les dossiers `ams/ams_*` (1D, Müller-Brown, dimers)
- `md_traj_*.traj` dans `ini_conds/` (échantillonnage des conditions initiales)
- `.traj` dans les dossiers butane pour `reweighting/reweight.py` et `reweighting_full/reweight.py`

Références :
- `scripts/AMS_1D/ams_fit/ams/run_reweight.py`
- `scripts/AMS_1D/ams_target/ams/run_reweight.py`
- `scripts/AMS_muller_brown/ams/ams/run_reweight.py`
- `scripts/AMS_dimers/ams/ams/run_reweight.py`
- `scripts/AMS_butane/ams_runs_*/reweighting/reweight.py`

## Fichiers `.npy/.npz` attendus par les notebooks (absents du dépôt)

### AMS_1D

- `scripts/AMS_1D/data_1D/reweighting_aggregate_fit_40_500.npz`
- `scripts/AMS_1D/data_1D/reweighting_aggregate_target_40_500.npz`
- `scripts/AMS_1D/pops_data/misspecification_sigma.npy`
- `scripts/AMS_1D/pops_data/posterior_samples.npy`

Référence : `scripts/AMS_1D/post_process.ipynb`.

### AMS_muller_brown

- `scripts/AMS_muller_brown/data_muller_brown/reweighting_aggregate.npz`
- `scripts/AMS_muller_brown/data_muller_brown/*_ref_probs.npy`

Référence : `scripts/AMS_muller_brown/post_process.ipynb`.

### AMS_dimers

- `scripts/AMS_dimers/data_dimer/reweighting_aggregate.npz`
- `scripts/AMS_dimers/data_dimer/probas_h-.npy`
- `scripts/AMS_dimers/data_dimer/probas_h--.npy`
- `scripts/AMS_dimers/data_dimer/probas_epsilon-.npy`
- `scripts/AMS_dimers/data_dimer/probas_epsilon--.npy`

Référence : `scripts/AMS_dimers/post_process.ipynb`.

### Butane (sorties de pipeline non versionnées)

Exemples de sorties écrites par les scripts :

- `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/final_scores.npy`
- `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/final_probs.npy`
- `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/ini_D.npy`
- `scripts/AMS_butane/theta_mp0a_500/results/probs.npy`
- `scripts/AMS_butane/theta_mp0a_500/results/thetas.npy`

Références : `aggregate.py`, `compute_ini_D.py`, `get_results.py`, scripts de `plots/`.

## License status

Constat actuel :

- Aucun fichier `LICENSE`/`LICENCE` n'est présent à la racine du dépôt.
- `pyproject.toml` ne déclare pas de licence distribuable.
- `CITATION.cff` est fourni avec `license: NOASSERTION` pour refléter cet état sans choisir arbitrairement une licence.

Action recommandée (hors de ce passage) : choisir explicitement une licence et ajouter un fichier `LICENSE` validé par les mainteneurs.
