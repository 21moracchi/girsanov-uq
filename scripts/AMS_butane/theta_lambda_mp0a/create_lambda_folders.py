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
RESULTS_DIR = ROOT / "results"
N_AMS = 70
DEFAULT_N_SAMPLES = 3
LAMBDAS = np.arange(-5, 6, dtype=int)


def lambda_name(value):
    return f"lambda_{'p' if value >= 0 else 'm'}{abs(value):02d}"


def require_base_files():
    names = ["job", "job_topaze", "run_ams.py", "run_all_topaze", "compute_probs.py"]
    missing = [name for name in names if not (BASE_RUN / name).is_file()]
    if missing:
        raise SystemExit(f"Missing base files in {BASE_RUN}: {', '.join(missing)}")


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
        "theta_l = np.zeros(posterior_samples.shape[0], dtype=posterior_samples.dtype)\n"
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
        raise RuntimeError("Could not build the run_ams.py template")
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
        raise RuntimeError(f"Unreplaced placeholders in {{dirname}}/run_ams.py")
    (dirname / "run_ams.py").write_text(text)

print(f"Prepared {{n_ams}} AMS directories for theta index {{theta_index}}, lambda={{lambda_value}}.")
'''


def rebase_existing_runs_at_zero():
    """Set existing theta-lambda folders to the zero-corrector reference."""
    posterior = np.load(POSTERIOR_PATH)
    indices_path = RESULTS_DIR / "theta_indices.npy"
    if not indices_path.is_file():
        raise SystemExit(f"Indices posterior introuvables: {indices_path}")
    indices = np.load(indices_path)
    if indices.ndim != 1 or len(indices) != DEFAULT_N_SAMPLES:
        raise SystemExit(f"Expected exactly {DEFAULT_N_SAMPLES} indices in {indices_path}.")
    if np.any(indices < 0) or np.any(indices >= posterior.shape[1]):
        raise SystemExit("A theta_index is out of range for the posterior.")

    theta_l = np.zeros(posterior.shape[0], dtype=posterior.dtype)
    directions = posterior[:, indices].T
    np.save(RESULTS_DIR / "theta_l.npy", theta_l)
    np.save(RESULTS_DIR / "directions.npy", directions)

    lambdas_path = RESULTS_DIR / "lambdas.npy"
    old_lambdas = np.load(lambdas_path) if lambdas_path.exists() else LAMBDAS
    keep_lambda = np.isin(old_lambdas, LAMBDAS)
    if not np.all(np.isin(LAMBDAS, old_lambdas)):
        raise SystemExit(f"Could not find all values {LAMBDAS} in {old_lambdas}.")
    for filename in ("probs.npy", "available.npy"):
        array_path = RESULTS_DIR / filename
        if not array_path.exists():
            continue
        values = np.load(array_path)
        if values.ndim < 2 or values.shape[1] != len(old_lambdas):
            raise SystemExit(
                f"Unexpected shape for {array_path}: {values.shape}; "
                f"the lambda axis should have {len(old_lambdas)} values."
            )
        np.save(array_path, values[:, keep_lambda, ...])
    np.save(lambdas_path, LAMBDAS)

    run_files = list(ROOT.glob("sample_*/lambda_*/run_ams.py"))
    run_files.extend(ROOT.glob("sample_*/lambda_*/ams_*/run_ams.py"))
    for path in run_files:
        text = path.read_text()
        text = text.replace(
            "../../../ams_runs_mp0a_", "../../../../ams_runs_mp0a_"
        ).replace("../../../models/", "../../../../models/")
        text = re.sub(
            r"(?m)^\s*theta_l\s*=\s*np\.load\(.*w_corrector_mp0a\.npy[\"']\)\s*$",
            "theta_l = np.zeros(posterior_samples.shape[0], dtype=posterior_samples.dtype)",
            text,
        )
        text = re.sub(
            r"(?m)^\s*theta_l = np\.zeros\(posterior_samples\.shape\[0\], dtype=posterior_samples\.dtype\)\s*$",
            "theta_l = np.zeros(posterior_samples.shape[0], dtype=posterior_samples.dtype)",
            text,
        )
        path.write_text(text)

    make_files = list(ROOT.glob("sample_*/lambda_*/make_input_files.py"))
    for path in make_files:
        text = path.read_text()
        text = text.replace(
            "../../../ams_runs_mp0a_", "../../../../ams_runs_mp0a_"
        ).replace("../../../models/", "../../../../models/")
        path.write_text(text)

    for pattern in (
        "sample_*/lambda_*/job_topaze",
        "sample_*/lambda_*/ams_*/job_topaze",
    ):
        for path in ROOT.glob(pattern):
            text = path.read_text().replace("#MSUB -q milan", "#MSUB -q a100", 1)
            path.write_text(text)

    for path in ROOT.glob("sample_*/lambda_*/run_all_topaze"):
        text = path.read_text().replace("#MSUB -q a100", "#MSUB -q milan", 1)
        path.write_text(text)

    print(f"Reference zero appliquee aux {len(run_files)} scripts run_ams.py.")
    print(f"Recomputed directions for indices {', '.join(map(str, indices))}.")
    print(f"Active lambda range: {LAMBDAS[0]} to {LAMBDAS[-1]} ({len(LAMBDAS)} values).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="seed for selecting posterior samples")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument(
        "--rebase-zero-reference",
        action="store_true",
        help="update existing folders/results so theta_L is exactly zero",
    )
    args = parser.parse_args()

    if args.rebase_zero_reference:
        rebase_existing_runs_at_zero()
        return

    if args.n_samples != DEFAULT_N_SAMPLES:
        raise SystemExit(f"Ce protocole requiert exactement {DEFAULT_N_SAMPLES} samples.")
    if any(ROOT.glob("sample_*")):
        raise SystemExit("Some sample_* directories already exist; no files were overwritten.")

    require_base_files()
    posterior = np.load(POSTERIOR_PATH)
    if posterior.ndim != 2:
        raise SystemExit("posterior_samples must be a 2D array.")
    theta_l = np.zeros(posterior.shape[0], dtype=posterior.dtype)

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
            for name in ("job", "compute_probs.py"):
                shutil.copy2(BASE_RUN / name, lambda_dir / name)
            shutil.copy2(BASE_RUN / "job_topaze", lambda_dir / "job_topaze")
            run_all = (BASE_RUN / "run_all_topaze").read_text()
            run_all = run_all.replace("#MSUB -q a100", "#MSUB -q milan", 1)
            (lambda_dir / "run_all_topaze").write_text(run_all)
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
