"""Aggregate available direct AMS results on the three lambda directions."""

import io
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def numeric_values(path):
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", line.strip())
    ]
    if not lines:
        raise ValueError(f"Aucune probabilite dans {path}")
    return np.atleast_1d(np.loadtxt(io.StringIO("\n".join(lines))))


def main():
    lambdas = np.load(RESULTS_DIR / "lambdas.npy")
    indices = np.load(RESULTS_DIR / "theta_indices.npy")
    lookup = {int(value): position for position, value in enumerate(lambdas)}
    collected = {}

    for sample_id in range(len(indices)):
        sample_dir = ROOT / f"sample_{sample_id}"
        for lambda_dir in sample_dir.glob("lambda_*"):
            params_path = lambda_dir / "parameters.json"
            results_path = lambda_dir / "results_prob.txt"
            if not params_path.exists() or not results_path.exists():
                continue
            params = json.loads(params_path.read_text())
            lambda_value = int(params["lambda"])
            if lambda_value not in lookup:
                # Older experiments may still have lambda folders outside the
                # active range; the Topaze submitter intentionally skips them.
                continue
            collected[sample_id, lookup[lambda_value]] = numeric_values(results_path)

    if not collected:
        raise SystemExit("Aucun results_prob.txt trouve.")
    sizes = {values.size for values in collected.values()}
    if len(sizes) != 1:
        raise SystemExit(f"Nombre de repetitions AMS incompatible: {sorted(sizes)}")

    n_rep = sizes.pop()
    probs = np.full((len(indices), len(lambdas), n_rep), np.nan)
    for (sample_id, lambda_position), values in collected.items():
        probs[sample_id, lambda_position] = values
    np.save(RESULTS_DIR / "probs.npy", probs)
    np.save(RESULTS_DIR / "available.npy", np.isfinite(probs).any(axis=2))
    print(f"Saved {RESULTS_DIR / 'probs.npy'} with shape {probs.shape}")
    print(f"Available points: {len(collected)}/{len(indices) * len(lambdas)}")


if __name__ == "__main__":
    main()
