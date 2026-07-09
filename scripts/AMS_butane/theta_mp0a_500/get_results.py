import io
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
POSTERIOR_PATH = ROOT / "../ams_runs_mp0a/pops/binary_data/posterior_samples_mp0a.npy"
RESULTS_DIR = ROOT / "results"


def theta_sort_key(path):
    return int(path.name.split("_")[1])


def theta_dirs():
    return sorted(
        (
            path
            for path in ROOT.iterdir()
            if path.is_dir() and re.fullmatch(r"theta_\d+", path.name)
        ),
        key=theta_sort_key,
    )


def load_prob_values(path):
    numeric_lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", stripped):
            numeric_lines.append(stripped)

    if not numeric_lines:
        raise ValueError(f"Aucune valeur numerique trouvee dans {path}")

    return np.atleast_1d(np.loadtxt(io.StringIO("\n".join(numeric_lines))))


def read_theta_index(theta_dir):
    make_file = theta_dir / "make_input_files.py"
    text = make_file.read_text()
    match = re.search(r"^theta_index\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"theta_index introuvable dans {make_file}")
    return int(match.group(1))


def main():
    posterior_samples = np.load(POSTERIOR_PATH)

    probs = []
    thetas = []
    used_dirs = []

    for theta_dir in theta_dirs():
        results_path = theta_dir / "results_prob.txt"
        if not results_path.exists():
            continue

        values = load_prob_values(results_path)
        theta_index = read_theta_index(theta_dir)
        if theta_index < 0 or theta_index >= posterior_samples.shape[1]:
            raise ValueError(
                f"{theta_dir.name}: theta_index={theta_index} hors limites "
                f"pour posterior_samples.shape={posterior_samples.shape}"
            )

        probs.append(values)
        thetas.append(posterior_samples[:, theta_index])
        used_dirs.append(theta_dir.name)

    if not probs:
        raise SystemExit("Aucun results_prob.txt trouve.")

    n_values = {arr.shape[0] for arr in probs}
    if len(n_values) != 1:
        sizes = ", ".join(
            f"{name}:{arr.shape[0]}" for name, arr in zip(used_dirs, probs)
        )
        raise SystemExit(f"Nombre de valeurs AMS incompatible: {sizes}")

    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / "probs.npy", np.vstack(probs))
    np.save(RESULTS_DIR / "thetas.npy", np.vstack(thetas))

    print(f"Saved {RESULTS_DIR / 'probs.npy'} with shape {np.vstack(probs).shape}")
    print(f"Saved {RESULTS_DIR / 'thetas.npy'} with shape {np.vstack(thetas).shape}")
    print(f"Used theta folders: {' '.join(used_dirs)}")


if __name__ == "__main__":
    main()
