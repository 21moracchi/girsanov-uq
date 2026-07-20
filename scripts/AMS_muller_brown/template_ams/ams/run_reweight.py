import re
from pathlib import Path

import ase.units as units
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from girsanov_uq.post_processing.reweighting import Reweightor

# Parameters, atoms, potentials, and integrator (edit in common/model_block.py)
n_reps = None
ams_id = None
# >>> COMMON: common/model_block.py
# <<< COMMON

# Reweighting parameters (edit if needed)
reweightor = Reweightor(
    calculator=calc,
    integrator="OBABO",
    mode="linear",
    temperature_K=temperature_K,
    dt=dt,
    friction=gamma,
    masses=masses,
)

## Recurrent helpers


def _replica_index(path: Path) -> int:
    match = re.search(r"rep_(\d+)\.traj$", path.name)
    if match is None:
        raise ValueError(f"Nom de fichier inattendu: {path.name}")
    return int(match.group(1))


def _read_frames(traj_path: Path) -> list:
    frames = []
    with Trajectory(str(traj_path), "r") as traj:
        for atoms_frame in traj:
            frames.append(atoms_frame)
    return frames


def main() -> None:
    traj_paths = sorted(Path(".").glob("rep_*.traj"), key=_replica_index)
    descriptor_dim = int(calc.get_descriptors_jacobian(atoms).shape[1])

    if n_reps is not None:
        traj_paths = [p for p in traj_paths if _replica_index(p) < int(n_reps)]

    if n_reps is not None:
        scores_arr = np.full((int(n_reps), descriptor_dim), np.nan, dtype=float)
        fims_arr = np.full((int(n_reps), descriptor_dim, descriptor_dim), np.nan, dtype=float)
        paths_arr = np.full((int(n_reps),), "", dtype=object)
    else:
        scores = []
        fims = []
        replica_ids = []
        used_paths = []

    for traj_path in traj_paths:
        frames = _read_frames(traj_path)
        if len(frames) < 2:
            continue

        score, fim = reweightor.reweight_one_trajectory_linear(frames, as_list=False)
        score = np.asarray(score)
        fim = np.asarray(fim)

        if score.ndim == 0:
            score = np.zeros(descriptor_dim, dtype=float)
        if fim.ndim == 0:
            fim = np.zeros((descriptor_dim, descriptor_dim), dtype=float)

        rep_id = _replica_index(traj_path)
        if n_reps is not None:
            scores_arr[rep_id] = score.astype(float)
            fims_arr[rep_id] = fim.astype(float)
            paths_arr[rep_id] = str(traj_path)
        else:
            scores.append(score.astype(float))
            fims.append(fim.astype(float))
            replica_ids.append(rep_id)
            used_paths.append(str(traj_path))

    if n_reps is not None:
        replica_ids_arr = np.arange(int(n_reps), dtype=int)
        paths_out = paths_arr
    else:
        scores_arr = np.array(scores) if len(scores) > 0 else np.zeros((0, descriptor_dim), dtype=float)
        fims_arr = np.array(fims) if len(fims) > 0 else np.zeros((0, descriptor_dim, descriptor_dim), dtype=float)
        replica_ids_arr = np.array(replica_ids, dtype=int)
        paths_out = np.array(used_paths, dtype=object)

    np.savez_compressed(
        "reweighting_results.npz",
        ams_id=np.array([-1 if ams_id is None else int(ams_id)]),
        replica_ids=replica_ids_arr,
        scores=scores_arr,
        fims=fims_arr,
        paths=paths_out,
    )

    np.save("scores.npy", scores_arr)
    np.save("fims.npy", fims_arr)


if __name__ == "__main__":
    main()
