"""Restore initial-condition files marked as used by AMS runs."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent / "ams_runs_mp0a_500" / "sample_ini_conds" / "ini_conds"
    if not root.is_dir():
        raise FileNotFoundError(f"Initial-condition directory not found: {root}")

    restored = 0
    for path in root.rglob("*"):
        if not path.is_file() or not path.stem.endswith("_used"):
            continue

        destination = path.with_name(path.stem[:-5] + path.suffix)
        if destination.exists():
            raise FileExistsError(f"Cannot restore {path.name}: {destination} already exists")
        path.rename(destination)
        restored += 1

    print(f"Restored {restored} initial-condition files in {root}.")


if __name__ == "__main__":
    main()
