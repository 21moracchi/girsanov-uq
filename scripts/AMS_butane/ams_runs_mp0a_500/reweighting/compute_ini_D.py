import argparse
import re
from pathlib import Path

import numpy as np
from ase.io import read
from girsanov_uq.descriptors.mace import MACELinear


READABLE_SUFFIXES = {".extxyz", ".xyz", ".traj"}
DEFAULT_INI_COND_DIR = "../sample_ini_conds/ini_conds"
DEFAULT_MODEL_PATH = "../../models/mace_mp0a_ft.model"


def natural_key(path):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    ]


def find_condition_files(ini_cond_dir):
    files = [
        path
        for path in ini_cond_dir.rglob("*")
        if path.is_file() and path.suffix in READABLE_SUFFIXES
    ]
    return sorted(files, key=natural_key)


def iter_initial_conditions(condition_files):
    for path in condition_files:
        atoms_or_list = read(path, index=":")
        atoms_list = atoms_or_list if isinstance(atoms_or_list, list) else [atoms_or_list]
        for atoms in atoms_list:
            yield path, atoms


def compute_descriptors(calc, atoms):
    atoms.calc = calc
    if hasattr(calc, "get_descriptors"):
        descriptor = calc.get_descriptors(atoms)
    else:
        atoms.get_potential_energy()
        descriptor = calc.results["descriptors"]
    descriptor = np.asarray(descriptor)
    if descriptor.ndim == 2:
        descriptor = descriptor.sum(axis=0)
    return descriptor


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute ini_D.npy from the first N initial conditions found recursively "
            "in the initial-condition directory."
        )
    )
    parser.add_argument("N", type=int, help="Number of initial conditions to process.")
    parser.add_argument(
        "--ini-cond-dir",
        default=DEFAULT_INI_COND_DIR,
        help="Initial-condition directory.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="MACE model path.")
    parser.add_argument("--output", default="aggregated_results/ini_D.npy", help="Output .npy file.")
    parser.add_argument("--device", default="cuda", help="Torch device used by MACE.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.N <= 0:
        raise ValueError("N must be strictly positive.")

    ini_cond_dir = Path(args.ini_cond_dir)
    if not ini_cond_dir.is_dir():
        raise FileNotFoundError(f"{ini_cond_dir} is not an existing directory.")

    condition_files = find_condition_files(ini_cond_dir)
    if not condition_files:
        raise FileNotFoundError(f"No initial-condition files found recursively in {ini_cond_dir}.")

    calc = MACELinear(model_path=args.model_path, device=args.device, default_dtype="float32")

    descriptors = []
    sources = []
    for source, atoms in iter_initial_conditions(condition_files):
        descriptors.append(compute_descriptors(calc, atoms))
        sources.append(str(source))
        if len(descriptors) == args.N:
            break

    if len(descriptors) < args.N:
        raise ValueError(
            f"Requested {args.N} initial conditions, found only {len(descriptors)} "
            f"in {ini_cond_dir}."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ini_D = np.asarray(descriptors)
    np.save(output_path, ini_D)
    print(f"Saved {args.output} with shape {ini_D.shape}.")
    print(f"Used {len(sources)} initial conditions from {ini_cond_dir}.")


if __name__ == "__main__":
    main()
