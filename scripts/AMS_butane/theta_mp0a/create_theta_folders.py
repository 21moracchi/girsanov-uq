import argparse
import random
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "theta_0"
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


def main():
    parser = argparse.ArgumentParser(
        description="Create new theta_i folders with random theta_index values below 2000."
    )
    parser.add_argument("n", type=int, help="number of theta folders to add")
    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit("n doit etre strictement positif.")
    if not TEMPLATE.is_dir():
        raise SystemExit(f"Template introuvable: {TEMPLATE}")

    existing_dirs = theta_dirs()
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
