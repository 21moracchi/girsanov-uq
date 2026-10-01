"""Create three posterior directions and AMS folders along each direction."""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BASE_RUN = ROOT.parent / "ams_runs_mp0a_500"
POSTERIOR_PATH = ROOT.parent / "ams_runs_mp0a_300/pops/binary_data/posterior_samples_mp0a.npy"
THETA_L_PATH = ROOT.parent / "ams_runs_mp0a_300/pops/binary_data/w_corrector_mp0a.npy"
RESULTS_DIR = ROOT / "results"
N_AMS = 70
DEFAULT_N_SAMPLES = 3
LAMBDAS = np.arange(-10, 11, dtype=int)


def lambda_name(value):
    return f"lambda_{'p' if value >= 0 else 'm'}{abs(value):02d}"


def require_base_files():
    names = ["job", "job_topaze", "run_ams.py", "run_all_topaze", "compute_probs.py"]
    missing = [name for name in names if not (BASE_RUN / name).is_file()]
    if missing:
        raise SystemExit(f"Fichiers de base manquants dans {BASE_RUN}: {', '.join(missing)}")


def make_run_template():
    text = (BASE_RUN / "run_ams.py").read_text()
    text = text.replace(
        "dyn_seed = None\n",
        "dyn_seed = None\ntheta_index = None\nlambda_value = None\n",
        1,
    )
    text = text.replace(
        "mace_model_path = '../../models/mace_mp0a_ft.model'",
        "mace_model_path = '../../../../models/mace_mp0a_ft.model'",
        1,
    )
    text = text.replace(
        "calc = MACELinear(model_path=mace_model_path, device=device, default_dtype='float32')",
        "posterior_samples = np.load(\n"
        "    '../../../../ams_runs_mp0a_300/pops/binary_data/posterior_samples_mp0a.npy'\n"
        ")\n"
        "theta_l = np.load('../../../../ams_runs_mp0a_300/pops/binary_data/w_corrector_mp0a.npy')\n"
        "delta_theta = posterior_samples[:, theta_index] - theta_l\n"
        "theta = theta_l + lambda_value * delta_theta\n"
        "calc = MACELinear(model_path=mace_model_path, device=device, "
        "default_dtype='float32', linear_weights=theta)",
        1,
    )
    text = text.replace(
        'ams.set_ini_cond_dir("../sample_ini_conds/ini_conds")',
        'ams.set_ini_cond_dir("../../../../ams_runs_mp0a_500/sample_ini_conds/ini_conds")',
        1,
    )
    required = ("theta_index = None", "lambda_value = None", "linear_weights=theta")
    if not all(item in text for item in required):
        raise RuntimeError("Impossible de construire le template run_ams.py")
    return text


def make_input_script(theta_index, lambda_value):
    return f'''import re
import shutil
from pathlib import Path

import numpy as np


theta_index = {theta_index}
lambda_value = {lambda_value}
n_ams = {N_AMS}
rng = np.random.default_rng()
dyn_seeds = rng.integers(10**6, size=n_ams)
ams_seeds = rng.integers(10**6, size=n_ams)

for i in range(n_ams):
    dirname = Path(f"ams_{{i}}")
    dirname.mkdir(exist_ok=True)
    for filename in ("job", "job_topaze"):
        shutil.copy2(filename, dirname / filename)

    text = Path("run_ams.py").read_text()
    text = text.replace("ams_seed = None", f"ams_seed = {{ams_seeds[i]}}")
    text = text.replace("dyn_seed = None", f"dyn_seed = {{dyn_seeds[i]}}")
    text = text.replace("theta_index = None", f"theta_index = {{theta_index}}")
    text = text.replace("lambda_value = None", f"lambda_value = {{lambda_value}}")
    text = re.sub(
        r'(?m)^\\s*ams\\.set_ini_cond_dir\\(.*\\)\\s*$',
        f'ams.set_ini_cond_dir("../../../../ams_runs_mp0a_500/sample_ini_conds/ini_conds/{{i}}")',
        text,
        count=1,
    )
    if "None" in text[text.find("ams_seed ="):text.find("atoms = molecule")]:
        raise RuntimeError(f"Placeholders non remplaces dans {{dirname}}/run_ams.py")
    (dirname / "run_ams.py").write_text(text)

print(f"Prepared {{n_ams}} AMS directories for theta index {{theta_index}}, lambda={{lambda_value}}.")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="seed for selecting posterior samples")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    args = parser.parse_args()

    if args.n_samples != DEFAULT_N_SAMPLES:
        raise SystemExit(f"Ce protocole requiert exactement {DEFAULT_N_SAMPLES} samples.")
    if any(ROOT.glob("sample_*")):
        raise SystemExit("Des dossiers sample_* existent deja; aucun ecrasement n'est effectue.")

    require_base_files()
    posterior = np.load(POSTERIOR_PATH)
    theta_l = np.load(THETA_L_PATH)
    if posterior.ndim != 2 or posterior.shape[0] != theta_l.size:
        raise SystemExit("posterior_samples et w_corrector ont des dimensions incompatibles.")

    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(posterior.shape[1], size=DEFAULT_N_SAMPLES, replace=False))
    directions = (posterior[:, indices].T - theta_l)
    run_template = make_run_template()

    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / "lambdas.npy", LAMBDAS)
    np.save(RESULTS_DIR / "theta_indices.npy", indices)
    np.save(RESULTS_DIR / "theta_l.npy", theta_l)
    np.save(RESULTS_DIR / "directions.npy", directions)

    for sample_id, theta_index in enumerate(indices):
        sample_dir = ROOT / f"sample_{sample_id}"
        sample_dir.mkdir()
        (sample_dir / "metadata.json").write_text(json.dumps({
            "sample_id": int(sample_id),
            "theta_index": int(theta_index),
        }, indent=2) + "\n")
        for lambda_value in LAMBDAS:
            lambda_dir = sample_dir / lambda_name(int(lambda_value))
            lambda_dir.mkdir()
            for name in ("job", "job_topaze", "run_all_topaze", "compute_probs.py"):
                shutil.copy2(BASE_RUN / name, lambda_dir / name)
            (lambda_dir / "run_ams.py").write_text(run_template)
            (lambda_dir / "make_input_files.py").write_text(
                make_input_script(int(theta_index), int(lambda_value))
            )
            (lambda_dir / "parameters.json").write_text(json.dumps({
                "sample_id": int(sample_id),
                "theta_index": int(theta_index),
                "lambda": int(lambda_value),
            }, indent=2) + "\n")

    print(f"Selected posterior indices: {', '.join(map(str, indices))}")
    print(f"Created {DEFAULT_N_SAMPLES} samples x {len(LAMBDAS)} lambda values x {N_AMS} AMS runs.")


if __name__ == "__main__":
    main()
