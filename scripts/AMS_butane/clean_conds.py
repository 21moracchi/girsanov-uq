"""Restore initial-condition files marked as used by AMS runs."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Restore AMS initial-condition files ending in '_used.extxyz' "
            "recursively under the supplied directory."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Parent directory whose subdirectories contain the initial conditions.",
    )
    root = parser.parse_args().path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Initial-condition directory not found: {root}")

    restores = [
        (path, path.with_name(path.name.removesuffix("_used.extxyz") + ".extxyz"))
        for path in root.rglob("*_used.extxyz")
        if path.is_file()
    ]
    collisions = [
        (source, destination)
        for source, destination in restores
        if destination.exists()
    ]
    if collisions:
        details = "\n".join(
            f"  {source} -> {destination} (destination already exists)"
            for source, destination in collisions
        )
        raise FileExistsError(f"Refusing to overwrite existing files:\n{details}")

    for path, destination in restores:
        path.rename(destination)

    print(f"Restored {len(restores)} initial-condition files under {root}.")


if __name__ == "__main__":
    main()
