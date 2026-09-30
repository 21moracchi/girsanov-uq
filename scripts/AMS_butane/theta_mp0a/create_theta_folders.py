import argparse
import random
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "theta_0"
BASE_RUN = ROOT.parent / "ams_runs_mp0a_500"
N_AMS = 70
FILES_TO_COPY = [
    "job",
    "job_topaze",
    "run_ams.py",
    "make_input_files.py",
    "run_all_topaze",
    "compute_probs.py",
]


def theta_dirs():
    return sorted(
        (
            path
            for path in ROOT.iterdir()
            if path.is_dir() and re.fullmatch(r"theta_\d+", path.name)
        ),
        key=lambda path: int(path.name.split("_")[1]),
    )


def read_theta_index(theta_dir):
    make_file = theta_dir / "make_input_files.py"
    if not make_file.exists():
        return None

    match = re.search(r"^theta_index\s*=\s*(\d+)\s*$", make_file.read_text(), re.MULTILINE)
    if match is None:
        return None
    return int(match.group(1))


def write_theta_index(make_file, theta_index):
    text = make_file.read_text()
    text, n_subs = re.subn(
        r"^theta_index\s*=\s*\d+\s*$",
        f"theta_index = {theta_index}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_subs != 1:
        raise RuntimeError(f"Impossible de remplacer theta_index dans {make_file}")
    make_file.write_text(text)


def ensure_template():
    """Rebuild theta_0 from the standard AMS run if it was deleted."""
    if TEMPLATE.is_dir():
        return

    required = [
        "job",
        "job_topaze",
        "run_ams.py",
        "make_input_files.py",
        "run_all_topaze",
        "compute_probs.py",
    ]
    missing = [name for name in required if not (BASE_RUN / name).exists()]
    if missing:
        raise SystemExit(
            "Template theta_0 introuvable et fichiers de base manquants dans "
            f"{BASE_RUN}: {', '.join(missing)}"
        )

    TEMPLATE.mkdir(parents=True)
    shutil.copy2(BASE_RUN / "job", TEMPLATE / "job")
    shutil.copy2(BASE_RUN / "job_topaze", TEMPLATE / "job_topaze")
    shutil.copy2(BASE_RUN / "run_all_topaze", TEMPLATE / "run_all_topaze")
    shutil.copy2(BASE_RUN / "compute_probs.py", TEMPLATE / "compute_probs.py")

    run_text = (BASE_RUN / "run_ams.py").read_text()
    run_text = run_text.replace("import shutil\n", "import shutil\nimport os\n", 1)
    run_text = run_text.replace(
        "dyn_seed = None\n",
        "dyn_seed = None\ntheta_index = None\n"
        "lambda_scale = float(os.environ.get(\"LAMBDA\", \"5.0\"))\n",
        1,
    )
    run_text = run_text.replace(
        "mace_model_path = '../../models/mace_mp0a_ft.model'",
        "mace_model_path = '../../../models/mace_mp0a_ft.model'",
        1,
    )
    run_text = run_text.replace(
        "calc = MACELinear(model_path=mace_model_path, device=device, default_dtype='float32')",
        "posterior_samples = np.load("
        "'../../../ams_runs_mp0a_300/pops/binary_data/posterior_samples_mp0a.npy')\n"
        "theta = lambda_scale * posterior_samples[:, theta_index]\n"
        "calc = MACELinear(model_path=mace_model_path, device=device, "
        "default_dtype='float32', linear_weights=theta)",
        1,
    )
    run_text = run_text.replace(
        'ams.set_ini_cond_dir("../sample_ini_conds/ini_conds")',
        'ams.set_ini_cond_dir("../../../ams_runs_mp0a_500/sample_ini_conds/ini_conds")',
        1,
    )
    (TEMPLATE / "run_ams.py").write_text(run_text)

    make_text = (BASE_RUN / "make_input_files.py").read_text()
    make_text = re.sub(r"^n_ams\s*=\s*\d+\s*$", f"n_ams = {N_AMS}", make_text, count=1, flags=re.MULTILINE)
    make_text = make_text.replace(
        "# number of AMS jobs to prepare\n",
        "# AMS sample selected from the posterior.\n"
        "theta_index = 0\n"
        "# number of AMS jobs to prepare\n",
        1,
    )
    make_text = make_text.replace(
        'ams.set_ini_cond_dir("../sample_ini_conds/ini_conds")',
        'ams.set_ini_cond_dir("../../../ams_runs_mp0a_500/sample_ini_conds/ini_conds/{i}")',
        1,
    )
    make_text = make_text.replace(
        "print(f\"Prepared {n_ams} AMS run directories (ams_0..ams_{n_ams-1}).\")",
        "print(f\"Prepared {n_ams} AMS run directories for theta_{theta_index}.\")",
        1,
    )
    # Inject the selected posterior column into each generated run.
    make_text = make_text.replace(
        "text = text.replace('dyn_seed = None', f'dyn_seed = {dyn_seeds[i - offset]}')\n",
        "text = text.replace('dyn_seed = None', f'dyn_seed = {dyn_seeds[i - offset]}')\n"
        "    text = text.replace('theta_index = None', f'theta_index = {theta_index}')\n",
        1,
    )
    (TEMPLATE / "make_input_files.py").write_text(make_text)

    print(f"Template theta_0 reconstruit depuis {BASE_RUN}")


def main():
    parser = argparse.ArgumentParser(
        description="Create new theta_i folders with random theta_index values below 2000."
    )
    parser.add_argument("n", type=int, help="number of theta folders to add")
    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit("n doit etre strictement positif.")
    ensure_template()

    existing_dirs = theta_dirs()
    for theta_dir in existing_dirs:
        make_file = theta_dir / "make_input_files.py"
        if make_file.exists():
            text = make_file.read_text()
            text = re.sub(
                r"^n_ams\s*=\s*\d+\s*$",
                f"n_ams = {N_AMS}",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            make_file.write_text(text)
    if existing_dirs:
        next_folder_id = max(int(path.name.split("_")[1]) for path in existing_dirs) + 1
    else:
        next_folder_id = 0

    used_theta_ids = {
        theta_id
        for theta_id in (read_theta_index(path) for path in existing_dirs)
        if theta_id is not None
    }
    available_theta_ids = [theta_id for theta_id in range(2000) if theta_id not in used_theta_ids]
    if args.n > len(available_theta_ids):
        raise SystemExit(
            f"Pas assez de theta_id disponibles: {len(available_theta_ids)} restants pour {args.n} demandes."
        )

    selected_theta_ids = random.sample(available_theta_ids, args.n)

    for offset, theta_index in enumerate(selected_theta_ids):
        folder_id = next_folder_id + offset
        theta_dir = ROOT / f"theta_{folder_id}"
        theta_dir.mkdir()

        for filename in FILES_TO_COPY:
            src = TEMPLATE / filename
            if not src.exists():
                raise SystemExit(f"Fichier template manquant: {src}")
            shutil.copy2(src, theta_dir / filename)

        write_theta_index(theta_dir / "make_input_files.py", theta_index)
        print(f"{theta_dir.name}: theta_index = {theta_index}")


if __name__ == "__main__":
    main()
