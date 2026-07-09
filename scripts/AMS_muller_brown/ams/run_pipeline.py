#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
INI_TEMPLATE_DIR = ROOT / "ini_conds"
AMS_TEMPLATE_DIR = ROOT / "ams"
COMMON_BLOCK_PATTERN = re.compile(
    r"(?ms)^([ \t]*)# >>> COMMON:\s*([^\n]+?)\s*$\n^\1# <<< COMMON\s*$"
)


def _ams_index(path: Path) -> int:
    match = re.search(r"ams_(\d+)$", path.name)
    if match is None:
        raise ValueError(f"Nom de dossier AMS inattendu: {path}")
    return int(match.group(1))


def find_existing_ams_dirs(n_ams: int | None) -> list[Path]:
    ams_dirs = sorted([p for p in AMS_TEMPLATE_DIR.glob("ams_*") if p.is_dir()], key=_ams_index)
    if n_ams is not None:
        ams_dirs = [p for p in ams_dirs if _ams_index(p) < n_ams]
    return ams_dirs


def copy_job_files(template_dir: Path, target_dir: Path) -> None:
    for source_file in template_dir.glob("job*"):
        if source_file.is_file():
            shutil.copy2(source_file, target_dir / source_file.name)


def prepare_target_dir(target_dir: Path, force: bool) -> None:
    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def _expand_common_blocks(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        indent = match.group(1)
        rel_path = match.group(2).strip()
        snippet_path = ROOT / rel_path
        if not snippet_path.exists():
            raise FileNotFoundError(f"Bloc commun introuvable: {snippet_path}")

        snippet_text = snippet_path.read_text()
        lines = snippet_text.splitlines()
        return "\n".join(f"{indent}{line}" if line else "" for line in lines)

    return COMMON_BLOCK_PATTERN.sub(_replace, text)


def write_patched_script(template_path: Path, target_path: Path, replacements: list[tuple[str, str]]) -> None:
    text = _expand_common_blocks(template_path.read_text())
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    target_path.write_text(text)


def build_ams_inputs(n_ams: int, n_reps: int, force: bool) -> list[Path]:
    target_dirs: list[Path] = []
    template_script = AMS_TEMPLATE_DIR / "run_ams.py"
    template_reweight_script = AMS_TEMPLATE_DIR / "run_reweight.py"
    rng = np.random.default_rng()
    ams_seeds = rng.integers(10**6, size=n_ams)
    dyn_seeds = rng.integers(10**6, size=n_ams)

    for ams_index in range(n_ams):
        target_dir = AMS_TEMPLATE_DIR / f"ams_{ams_index}"
        prepare_target_dir(target_dir, force)
        copy_job_files(AMS_TEMPLATE_DIR, target_dir)
        write_patched_script(
            template_script,
            target_dir / "run_ams.py",
            [
                (r"ams_seed\s*=\s*None", f"ams_seed = {int(ams_seeds[ams_index])}"),
                (r"dyn_seed\s*=\s*None", f"dyn_seed = {int(dyn_seeds[ams_index])}"),
                (r"ams_id\s*=\s*None", f"ams_id = {int(ams_index)}"),
                (r"n_reps\s*=\s*None", f"n_reps = {int(n_reps)}")
            ],
        )
        write_patched_script(
            template_reweight_script,
            target_dir / "run_reweight.py",
            [
                (r"ams_id\s*=\s*None", f"ams_id = {int(ams_index)}"),
                (r"n_reps\s*=\s*None", f"n_reps = {int(n_reps)}")
            ],
        )
        target_dirs.append(target_dir)

    return target_dirs


def build_ini_conds_inputs(n_walkers: int, force: bool, n_reps: int) -> list[Path]:
    target_dirs: list[Path] = []
    template_script = INI_TEMPLATE_DIR / "run_ini_conds.py"
    rng = np.random.default_rng()
    sampler_seeds = rng.integers(10**6, size=n_walkers)
    dyn_seeds = rng.integers(10**6, size=n_walkers)

    for walker_index in range(n_walkers):
        target_dir = INI_TEMPLATE_DIR / f"FV_replica_{walker_index}"
        prepare_target_dir(target_dir, force)
        copy_job_files(INI_TEMPLATE_DIR, target_dir)
        write_patched_script(
            template_script,
            target_dir / "run_ini_conds.py",
            [
                (r"n_walker\s*=\s*0", f"walker_index={n_walkers}"),
                (r"walker_index\s*=\s*0", f"walker_index={walker_index}"),
                (r"sampler_seed\s*=\s*None", f"sampler_seed = {int(sampler_seeds[walker_index])}"),
                (r"dyn_seed\s*=\s*None", f"dyn_seed = {int(dyn_seeds[walker_index])}"),
                (r"n_conditions\s*=\s*None", f"n_conditions = {int(10e6)}")
            ], 
        )
        target_dirs.append(target_dir)

    return target_dirs


def build_single_ini_conds(n_walkers: int, n_reps: int, force: bool) -> None:
    source_ini = INI_TEMPLATE_DIR / "ini_atoms.xyz"
    if not source_ini.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable pour --single-ini: {source_ini}"
        )

    base_dir = INI_TEMPLATE_DIR / "ini_conds"
    if force and base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for walker_index in range(n_walkers):
        walker_dir = base_dir / str(walker_index)
        if force and walker_dir.exists():
            shutil.rmtree(walker_dir)
        walker_dir.mkdir(parents=True, exist_ok=True)

        for rep_index in range(n_reps):
            target_ini = walker_dir / f"rep_{rep_index + 1}.extxyz"
            shutil.copy2(source_ini, target_ini)


def submit_jobs(target_dirs: list[Path]) -> list[str]:
    job_ids = []
    for target_dir in target_dirs:
        result = subprocess.run(
            ["sbatch", "job"], 
            cwd=target_dir, 
            capture_output=True, 
            text=True, 
            check=True
        )
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_ids.append(match.group(1))
        else:
            raise RuntimeError(f"Impossible de récupérer le Job ID : {result.stdout}")
    return job_ids


def submit_reweight_jobs(target_dirs: list[Path], parent_job_ids: list[str]) -> list[str]:
    if len(target_dirs) != len(parent_job_ids):
        raise ValueError("target_dirs et parent_job_ids doivent avoir la même longueur.")

    reweight_job_ids = []
    for target_dir, parent_job_id in zip(target_dirs, parent_job_ids):
        result = subprocess.run(
            ["sbatch", f"--dependency=afterok:{parent_job_id}", "job_reweight"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            reweight_job_ids.append(match.group(1))
        else:
            raise RuntimeError(f"Impossible de récupérer le Job ID du reweighting : {result.stdout}")

    return reweight_job_ids


def submit_reweight_jobs_independent(target_dirs: list[Path]) -> list[str]:
    reweight_job_ids = []
    for target_dir in target_dirs:
        result = subprocess.run(
            ["sbatch", "job_reweight"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            reweight_job_ids.append(match.group(1))
        else:
            raise RuntimeError(f"Impossible de récupérer le Job ID du reweighting : {result.stdout}")

    return reweight_job_ids


def submit_aggregate_job(reweight_job_ids: list[str], n_ams: int, n_reps: int) -> str | None:
    if len(reweight_job_ids) == 0:
        return None

    dependency = ":".join(reweight_job_ids)
    result = subprocess.run(
        [
            "sbatch",
            f"--dependency=afterok:{dependency}",
            "job_aggregate",
            "--n-ams",
            str(n_ams),
            "--n-reps",
            str(n_reps),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        return match.group(1)
    raise RuntimeError(f"Impossible de récupérer le Job ID de l'agrégation : {result.stdout}")


def cancel_jobs(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel"] + job_ids, check=False)


def wait_for_files_completion(n_walkers: int, n_reps: int, poll_seconds: int, timeout_seconds: int | None) -> None:
    deadline = None if timeout_seconds is None else time.time() + timeout_seconds
    base_dir = INI_TEMPLATE_DIR / "ini_conds"

    while True:
        all_done = True
        for i in range(0, n_walkers):
            target_dir = base_dir / str(i)
            if not target_dir.exists():
                all_done = False
                break
            
            file_count = sum(1 for p in target_dir.iterdir() if p.is_file())
            if file_count < n_reps:
                all_done = False
                break
        
        if all_done:
            return

        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Le délai d'attente a été dépassé avant la création de tous les fichiers.")

        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prépare et lance les étapes ini_conds, AMS et resampling.")
    parser.add_argument("--n-walkers", type=int, default=10)
    parser.add_argument("--n-ams", type=int, default=10)
    parser.add_argument("--n-reps", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=100000000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--single-ini",
        action="store_true",
        help=(
            "N'échantillonne pas les conditions initiales. Copie ini_conds/ini_atoms.xyz "
            "dans ini_conds/ini_conds/<i>/ avec n_reps fichiers par walker."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["all", "ini_conds", "ams", "resampling"],
        default="all",
        help=(
            "Étape à exécuter : "
            "all (pipeline complet), ini_conds (seulement conditions initiales), "
            "ams (seulement AMS), resampling (seulement reweighting+agrégation)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in ("all", "ini_conds"):
        if args.single_ini:
            build_single_ini_conds(args.n_walkers, args.n_reps, args.force)
        else:
            ini_dirs = build_ini_conds_inputs(args.n_walkers, args.force, args.n_reps)
            ini_job_ids = submit_jobs(ini_dirs)

            wait_for_files_completion(args.n_walkers, args.n_reps, args.poll_seconds, args.timeout_seconds)
            cancel_jobs(ini_job_ids)

        if args.stage == "ini_conds":
            return

    if args.stage in ("all", "ams"):
        ams_dirs = build_ams_inputs(args.n_ams, args.n_reps, args.force)
        ams_job_ids = submit_jobs(ams_dirs)

        if args.stage == "ams":
            return

        reweight_job_ids = submit_reweight_jobs(ams_dirs, ams_job_ids)
        submit_aggregate_job(reweight_job_ids, args.n_ams, args.n_reps)
        return

    if args.stage == "resampling":
        ams_dirs = find_existing_ams_dirs(args.n_ams)
        if len(ams_dirs) == 0:
            raise RuntimeError(
                "Aucun dossier ams_* trouvé. Lancez d'abord l'étape AMS ou retirez le filtre --n-ams."
            )
        reweight_job_ids = submit_reweight_jobs_independent(ams_dirs)
        submit_aggregate_job(reweight_job_ids, len(ams_dirs), args.n_reps)


if __name__ == "__main__":
    main()