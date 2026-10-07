import argparse
import re
import shutil
from pathlib import Path


N_DIRS = 70


def natural_key(path):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create 70 ini-condition folders and move exactly K .extxyz files into each one."
    )
    parser.add_argument("ini_cond_dir", help="Directory containing the generated .extxyz initial conditions.")
    parser.add_argument("k", type=int, help="Number of initial conditions to move into each subdirectory.")
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.ini_cond_dir)

    if args.k <= 0:
        raise ValueError("K must be strictly positive.")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"{source_dir} is not an existing directory.")

    ini_conds = sorted(
        [
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix == ".extxyz" and "_used" not in path.stem
        ],
        key=natural_key,
    )
    expected = N_DIRS * args.k
    if len(ini_conds) < expected:
        raise ValueError(f"Expected at least {expected} .extxyz files, found {len(ini_conds)}.")

    for i in range(N_DIRS):
        target_dir = source_dir / str(i)
        target_dir.mkdir(exist_ok=True)
        if any(target_dir.iterdir()):
            raise ValueError(f"{target_dir} is not empty.")

    for i in range(N_DIRS):
        target_dir = source_dir / str(i)
        for ini_cond in ini_conds[i * args.k:(i + 1) * args.k]:
            shutil.move(str(ini_cond), str(target_dir / ini_cond.name))

    print(f"Moved {expected} initial conditions into {N_DIRS} folders with K={args.k}.")
    if len(ini_conds) > expected:
        print(f"Left {len(ini_conds) - expected} extra initial conditions in {source_dir}.")


if __name__ == "__main__":
    main()
