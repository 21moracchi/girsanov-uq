# Girsanov UQ

Code, scripts, notebooks, and generated data for uncertainty propagation in
rare-event simulations using Girsanov reweighting and Adaptive Multilevel
Splitting (AMS).

The reusable Python package is named `girsanov_uq`. The repository also keeps
the study scripts and data products used to generate the numerical results.

## Repository Layout

- `src/girsanov_uq/`: reusable Python package
- `scripts/AMS_1D/`: one-dimensional toy model experiments
- `scripts/AMS_muller_brown/`: Muller-Brown experiments
- `scripts/AMS_dimers/`: solvated dimer experiments
- `scripts/AMS_butane/`: butane experiments and plotting scripts
- `scripts/theory/`: theoretical and illustrative plots
- `notebooks/`: exploratory notebooks
- `tests/`: lightweight import checks
- `trajectories.dat`: trajectory data used by the examples

## Installation

This project requires Python `>=3.12`.

Create and activate an environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The package dependencies are declared in `pyproject.toml`:

- `aseams`
- `scikit-learn`
- `mace-torch`

`aseams` is installed from GitHub. If dependency installation fails, check that
the environment has network access and Git available.

If you use `uv`, the equivalent setup is:

```bash
uv sync
source .venv/bin/activate
```

## Quick Check

After installation, run:

```bash
python -m unittest tests.test_imports
```

For a quick check without installing the package, run from the repository root:

```bash
PYTHONPATH=src python -m unittest tests.test_imports
```

## Basic Usage

Import the package as:

```python
from girsanov_uq.integrators import LangevinOBABO
from girsanov_uq.post_processing.reweighting import Reweightor
from girsanov_uq.potentials import RuggedMullerBrown, SolvatedDimer
```

The package contains:

- an OBABO Langevin integrator that can record the random noise used during the
  dynamics;
- reweighting utilities for Girsanov scores and weights;
- potential calculators used in the numerical experiments;
- plotting and post-processing helpers.

## Reproducing Experiments

Most numerical workflows are stored under `scripts/`. The scripts are intended
to be run from their own experiment directories because many of them use local
relative paths.

Typical AMS workflow:

```bash
cd scripts/AMS_muller_brown/ams
python ini_conds/make_input_files.py
./ini_conds/run_all
python ams/make_input_files.py
./ams/run_all
python aggregate_reweighting.py
```

The butane workflows use Topaze/CCC job files in several places. For example,
from an `ams_runs_*` directory:

```bash
python make_input_files.py
./run_all_topaze
python compute_probs.py
```

Reweighting jobs usually follow the same pattern:

```bash
cd reweighting
python make_input_files.py
./run_all_topaze
python aggregate.py
```

Plotting scripts for the butane results are in:

```bash
scripts/AMS_butane/plots
```

## Notebooks

The notebooks are exploratory and assume the package is importable either via
an editable install or `PYTHONPATH=src`.

Available notebooks:

- `notebooks/simple_ams_reweighting_1d.ipynb`: pedagogical AMS and Girsanov
  reweighting example on a one-dimensional double well.
- `notebooks/muller_brown.ipynb`: Muller-Brown exploratory notebook.

Example:

```bash
PYTHONPATH=src jupyter lab notebooks/simple_ams_reweighting_1d.ipynb
```

## Notes For Publication

- Generated Python build artifacts such as `build/`, `*.egg-info/`, and
  `__pycache__/` are ignored and should not be committed.
- Large simulation outputs such as `*.traj`, `*.npy`, `*.npz`, `*out*`,
  `*err*`, and `*.tar.gz` are ignored by default.
- The repository keeps the scripts and compact data needed to document how the
  published results were produced. Large raw trajectories should be archived
  separately if they are required for full reproducibility.
