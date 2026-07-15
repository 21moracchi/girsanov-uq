# Girsanov UQ

Repository for uncertainty quantification in rare-event simulations with Girsanov reweighting and Adaptive Multilevel Splitting (AMS).

The reusable Python package is `girsanov_uq`. The rest of the tree contains the experiment scripts, notebooks, and small data products used for the published numerical results.

## Contents

- `src/girsanov_uq/`: reusable Python package
- `scripts/`: experiment pipelines and result-processing scripts
- `notebooks/`: exploratory notebooks
- `tests/`: lightweight import checks

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The project targets Python 3.12+.

## Quick Check

```bash
python -m unittest tests.test_imports
```

## Notebooks

```bash
PYTHONPATH=src jupyter lab notebooks/simple_ams_reweighting_1d.ipynb
```

