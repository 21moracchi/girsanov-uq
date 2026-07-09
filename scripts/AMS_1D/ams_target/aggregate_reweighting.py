#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
AMS_DIR = ROOT / "ams"


def _ams_index(path: Path) -> int:
    match = re.search(r"ams_(\d+)$", path.name)
    if match is None:
        raise ValueError(f"Nom de dossier AMS inattendu: {path}")
    return int(match.group(1))


def _find_ams_dirs(n_ams: int | None) -> list[Path]:
    ams_dirs = sorted([p for p in AMS_DIR.glob("ams_*") if p.is_dir()], key=_ams_index)
    if n_ams is not None:
        ams_dirs = [p for p in ams_dirs if _ams_index(p) < n_ams]
    return ams_dirs


def _infer_shapes(ams_dirs: list[Path], n_reps: int | None) -> tuple[int, int]:
    if n_reps is None:
        inferred_n_reps = 0
    else:
        inferred_n_reps = int(n_reps)

    inferred_dim = None

    for ams_dir in ams_dirs:
        result_file = ams_dir / "reweighting_results.npz"
        if not result_file.exists():
            continue
        data = np.load(result_file, allow_pickle=True)
        scores = np.asarray(data["scores"])
        if scores.ndim == 2 and scores.shape[1] > 0:
            inferred_dim = int(scores.shape[1])
            if n_reps is None:
                inferred_n_reps = max(inferred_n_reps, int(scores.shape[0]))

    if inferred_dim is None:
        raise RuntimeError("Impossible d'inférer dim(a) depuis les fichiers reweighting_results.npz")

    return inferred_n_reps, inferred_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reweighting outputs over all AMS runs.")
    parser.add_argument("--n-ams", type=int, default=None)
    parser.add_argument("--n-reps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "reweighting_aggregate.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ams_dirs = _find_ams_dirs(args.n_ams)
    if len(ams_dirs) == 0:
        raise RuntimeError("Aucun dossier ams_* trouvé dans draft/ams")

    n_ams = len(ams_dirs)
    n_reps, dim = _infer_shapes(ams_dirs, args.n_reps)

    all_scores = np.full((n_ams, n_reps, dim), np.nan, dtype=float)
    all_fims = np.full((n_ams, n_reps, dim, dim), np.nan, dtype=float)
    ams_probabilities = np.full((n_ams,), np.nan, dtype=float)

    for local_i, ams_dir in enumerate(ams_dirs):
        proba_file = ams_dir / "result_proba.txt"
        if proba_file.exists():
            try:
                proba_values = np.loadtxt(proba_file, ndmin=1)
                if np.size(proba_values) > 0:
                    ams_probabilities[local_i] = float(np.ravel(proba_values)[0])
            except Exception:
                pass

        result_file = ams_dir / "reweighting_results.npz"
        if not result_file.exists():
            continue

        data = np.load(result_file, allow_pickle=True)
        scores = np.asarray(data["scores"])
        fims = np.asarray(data["fims"])

        if scores.ndim != 2 or fims.ndim != 3:
            continue

        rep_count = min(n_reps, scores.shape[0])
        if scores.shape[1] != dim:
            continue

        all_scores[local_i, :rep_count, :] = scores[:rep_count]
        all_fims[local_i, :rep_count, :, :] = fims[:rep_count]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        all_scores=all_scores,
        all_fims=all_fims,
        ams_probabilities=ams_probabilities,
    )

    np.save(args.output.with_suffix(".scores.npy"), all_scores)
    np.save(args.output.with_suffix(".fims.npy"), all_fims)
    np.save(args.output.with_suffix(".probas.npy"), ams_probabilities)

    print(f"Saved aggregate reweighting to {args.output}")
    print(
        f"all_scores shape={all_scores.shape} | all_fims shape={all_fims.shape} "
        f"| ams_probabilities shape={ams_probabilities.shape}"
    )


if __name__ == "__main__":
    main()
