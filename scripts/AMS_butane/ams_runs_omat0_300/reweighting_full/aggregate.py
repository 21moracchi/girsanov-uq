import numpy as np
import os
import re
import json
from tqdm import tqdm

def get_idx(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else 0

def final_score(score):
    return score[-1, :] if score.ndim == 2 else score

def final_fim(fim):
    return fim[-1, :, :] if fim.ndim == 3 else fim

root_dir = 'results'
ams_dir_path = '../'

ams_folders = sorted([d for d in os.listdir(root_dir) if d.startswith('ams_')], key=get_idx)
n_ams = len(ams_folders)

if n_ams == 0:
    raise FileNotFoundError("Aucun dossier ams_ trouvé.")

first_folder = os.path.join(root_dir, ams_folders[0])
rep_files_init = sorted([f for f in os.listdir(first_folder) if f.endswith('_score.npy')], key=get_idx)
n_rep = len(rep_files_init)

sample_data = np.load(os.path.join(first_folder, rep_files_init[0]))
p = final_score(sample_data).shape[0]
sample_fim_name = rep_files_init[0].replace('_score.npy', '_fim.npy')
sample_fim = np.load(os.path.join(first_folder, sample_fim_name))
fim_dim = final_fim(sample_fim).shape[0]

scores_tensor = np.empty((n_ams, n_rep, p))
fims_tensor = np.empty((n_ams, n_rep, fim_dim, fim_dim))
probs_list = []

with tqdm(total=n_ams, desc="Traitement des dossiers ams") as pbar:
    for i, ams_folder in enumerate(ams_folders):
        # 1. Extraction des scores (dans root_dir)
        current_score_path = os.path.join(root_dir, ams_folder)
        rep_files = sorted([f for f in os.listdir(current_score_path) if f.endswith('_score.npy')], key=get_idx)
        
        for j, rep_file in enumerate(rep_files):
            score_path = os.path.join(current_score_path, rep_file)
            fim_path = os.path.join(current_score_path, rep_file.replace('_score.npy', '_fim.npy'))
            if not os.path.exists(fim_path):
                raise FileNotFoundError(f"Fichier FIM manquant pour {score_path}: {fim_path}")

            score = np.load(score_path)
            fim = np.load(fim_path)
            scores_tensor[i, j, :] = final_score(score)
            fims_tensor[i, j, :, :] = final_fim(fim)

        # 2. Extract probabilities (inside ams_dir_path)
        checkpoint_path = os.path.join(ams_dir_path, ams_folder, "ams_checkpoint.txt")
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    if "current_p" in data:
                        probs_list.append(float(data["current_p"]))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        pbar.update(1)

np.save('final_scores.npy', scores_tensor)
np.save('final_fims.npy', fims_tensor)

if probs_list:
    probs_array = np.array(probs_list)
    np.save('final_probs.npy', probs_array)
    
    mean_val = np.mean(probs_array)
    std_err = np.std(probs_array) / np.sqrt(len(probs_array))
    
    with open("results_prob.txt", "w") as out:
        out.write(f"Mean: {mean_val}, SEM: {std_err}\n")
