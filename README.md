# Girsanov UQ

Code du papier sur la propagation d'incertitude paramétrique pour des probabilités d'événements rares avec AMS (Adaptive Multilevel Splitting) et repondération de Girsanov.

Le package Python réutilisable est `girsanov_uq` (`src/girsanov_uq`). Le reste du dépôt contient les pipelines d'expériences (1D, Müller-Brown, dimers, butane), notebooks et scripts de post-traitement.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Vérification minimale :

```bash
python -m unittest tests.test_imports
```

## Petit exemple CPU

Exécuter le notebook de démonstration 1D en local (CPU) :

```bash
PYTHONPATH=src jupyter lab notebooks/pops_uncertainty_pipeline_1d.ipynb
```

Ce notebook illustre un pipeline complet sur potentiel 1D simplifié (entraînement POPS, AMS, reweighting, agrégation).

## Structure du dépôt

- `src/girsanov_uq/` : package Python (intégrateurs, potentiels, reweighting)
- `scripts/AMS_1D/` : expériences 1D (cible et potentiel ajusté)
- `scripts/AMS_muller_brown/` : expériences Müller-Brown
- `scripts/AMS_dimers/` : expériences dimer
- `scripts/AMS_butane/` : expériences butane + scripts POPS/plots
- `notebooks/` : notebook de démonstration
- `tests/` : tests légers d'import
- `DATA.md` : inventaire des fichiers de données/modèles non versionnés

## Reproduire les expériences

Les pipelines d'expériences utilisent principalement des jobs batch (`sbatch`/`ccc_msub`) et produisent les sorties sous les dossiers `ams_*`, `ini_conds/*` et `reweighting*`.

Exemples de points d'entrée :

```bash
python scripts/AMS_1D/ams_fit/run_pipeline.py --n-reps 500
python scripts/AMS_1D/ams_target/run_pipeline.py --n-reps 500
python scripts/AMS_muller_brown/ams/run_pipeline.py --n-reps 500
python scripts/AMS_dimers/ams/run_pipeline.py --n-reps 500
```

Pour butane, la structure est organisée par cas (`ams_runs_mp0a_*`, `ams_runs_mpa_*`, `ams_runs_omat0_*`) avec orchestration locale via `scripts/AMS_butane/ams_runs_*/make_input_files.py`, `scripts/AMS_butane/ams_runs_*/run_all_topaze`, `scripts/AMS_butane/ams_runs_*/reweighting/*`.

## Correspondance systèmes du papier ↔ scripts/entrées/sorties

| Système | Scripts principaux | Entrées principales | Sorties principales |
| --- | --- | --- | --- |
| 1D (potentiel ajusté) | `scripts/AMS_1D/ams_fit/run_pipeline.py`, `scripts/AMS_1D/ams_fit/aggregate_reweighting.py` | paramètres CLI (`--n-reps`, `--n-ams`, `--n-walkers`), scripts `scripts/AMS_1D/ams_fit/ini_conds/run_ini_conds.py` | `scripts/AMS_1D/ams_fit/ams/ams_*/reweighting_results.npz`, `scripts/AMS_1D/ams_fit/reweighting_aggregate.npz` |
| 1D (potentiel cible) | `scripts/AMS_1D/ams_target/run_pipeline.py`, `scripts/AMS_1D/ams_target/aggregate_reweighting.py` | paramètres CLI (`--n-reps`, `--n-ams`, `--n-walkers`), scripts `scripts/AMS_1D/ams_target/ini_conds/run_ini_conds.py` | `scripts/AMS_1D/ams_target/ams/ams_*/reweighting_results.npz`, `scripts/AMS_1D/ams_target/reweighting_aggregate.npz` |
| Müller-Brown | `scripts/AMS_muller_brown/ams/run_pipeline.py`, `scripts/AMS_muller_brown/ams/aggregate_reweighting.py` | paramètres CLI du pipeline, scripts `scripts/AMS_muller_brown/ams/ini_conds/run_ini_conds.py` | `scripts/AMS_muller_brown/ams/ams/ams_*/reweighting_results.npz`, `scripts/AMS_muller_brown/ams/reweighting_aggregate.npz` |
| Dimer | `scripts/AMS_dimers/ams/run_pipeline.py`, `scripts/AMS_dimers/ams/aggregate_reweighting.py` | paramètres CLI du pipeline, scripts `scripts/AMS_dimers/ams/ini_conds/run_ini_conds.py` | `scripts/AMS_dimers/ams/ams/ams_*/reweighting_results.npz`, `scripts/AMS_dimers/ams/reweighting_aggregate.npz` |
| Butane | `scripts/AMS_butane/ams_runs_*/run_ams.py`, `scripts/AMS_butane/ams_runs_*/reweighting/{reweight.py,aggregate.py}`, `scripts/AMS_butane/theta_mp0a_500/*` | modèles MACE sous `scripts/AMS_butane/models/` (non versionnés), trajectoires AMS/ini_conds générées en batch | `scripts/AMS_butane/ams_runs_*/reweighting/aggregated_results/final_scores.npy`, `final_probs.npy`, `ini_D.npy`; `scripts/AMS_butane/theta_mp0a_500/results/{probs.npy,thetas.npy}` |

## Licence

Aucun fichier `LICENSE` n'est présent dans ce dépôt à ce stade. Voir la section "License status" de `DATA.md` pour l'état constaté sans choix arbitraire de licence.
